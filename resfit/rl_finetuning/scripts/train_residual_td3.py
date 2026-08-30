# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import os

# Cap all BLAS/OpenMP threadpools (critical to set before importing numpy/torch)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# Stop threads from spin-waiting
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")

import hashlib
import io
import json
import logging
import pprint
import queue
import random
import shutil
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import tensordict
import torch
import torchrl
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch.utils.data import DataLoader
from torchrl.data import (
    LazyMemmapStorage,
    LazyTensorStorage,
    ReplayBuffer,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from tqdm import tqdm

import wandb
from resfit.dexmg.environments.dexmg import create_vectorized_env
from resfit.lerobot.policies.act.configuration_act import ACTConfig
from resfit.lerobot.policies.act.modeling_act import ACTPolicy
from resfit.lerobot.utils.load_policy import download_policy_from_wandb, load_policy
from resfit.rl_finetuning.config.residual_td3 import ResidualTD3DexmgConfig
from resfit.rl_finetuning.off_policy.common_utils import utils
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.utils.dtype import to_uint8
from resfit.rl_finetuning.utils.evaluate_dexmg import run_dexmg_evaluation
from resfit.rl_finetuning.utils.hugging_face import (
    _hf_download_buffer,
    _hf_upload_buffer,
    optimized_replay_buffer_dumps,
    optimized_replay_buffer_loads,
)
from resfit.rl_finetuning.utils.normalization import ActionScaler, StateStandardizer
from resfit.rl_finetuning.utils.rb_transforms import MultiStepTransform, get_multi_step_transform
from resfit.rl_finetuning.wrappers.residual_env_wrapper import BasePolicyVecEnvWrapper
from resfit.rl_finetuning.marvin.env import MarvinACTBasePolicy, MarvinResidualConfig, MarvinResidualVecEnv
from resfit.rl_finetuning.marvin.replay import (
    atomic_write_episode,
    cached_episode_ids,
    critic_action,
    inspect_replay_directory,
    manifest_sha1,
    recover_pending_cache_metadata,
    replay_identity,
    replay_manifest_fingerprint,
)
from resfit.rl_finetuning.marvin.grpc_transport import start_learner_server
from resfit.rl_finetuning.utils.checkpoint import save_checkpoint


# -----------------------------------------------------------------------------
# Timing utility --------------------------------------------------------------
# -----------------------------------------------------------------------------
class TrainingTimer:
    """Simple timing utility for measuring training stage proportions."""

    def __init__(self):
        self.times = defaultdict(list)
        self.reset_time = time.perf_counter()

    @contextmanager
    def time(self, stage_name: str):
        """Context manager to time a specific training stage."""
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self.times[stage_name].append(elapsed)

    def get_timing_stats(self) -> dict[str, float]:
        """Get timing statistics as percentages of total time."""
        if not self.times:
            return {}

        # Calculate total time across all stages
        total_time = sum(sum(times) for times in self.times.values())
        if total_time == 0:
            return {}

        # Calculate percentages and averages
        stats = {}
        for stage_name, times_list in self.times.items():
            stage_total = sum(times_list)
            stage_avg = stage_total / len(times_list) if times_list else 0
            stage_percentage = (stage_total / total_time) * 100

            stats[f"timing/{stage_name}_percentage"] = stage_percentage
            stats[f"timing/{stage_name}_avg_ms"] = stage_avg * 1000  # Convert to ms
            stats[f"timing/{stage_name}_total_s"] = stage_total

        return stats

    def reset(self):
        """Reset all timing data."""
        self.times = defaultdict(list)
        self.reset_time = time.perf_counter()


# -----------------------------------------------------------------------------
# Logging configuration -------------------------------------------------------
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _marvin_binary_reward(raw_reward: float) -> float:
    """Match the user's Marvin task rule for old and new replay data."""
    return 1.0 if float(raw_reward) > 0.8 else 0.0


def _add_marvin_episode(data, replay_buffer) -> int:
    size = len(data["reward"])
    for row in range(size):
        replay_buffer.add(_marvin_transition(data, row))
    return size


def _marvin_transition(data, row: int) -> TensorDict:
    obs = {
        "observation.state": torch.as_tensor(data["state"][row], dtype=torch.float32),
        "observation.base_action": torch.as_tensor(data["base_action"][row], dtype=torch.float32),
    }
    next_obs = {
        "observation.state": torch.as_tensor(data["next_state"][row], dtype=torch.float32),
        "observation.base_action": torch.as_tensor(data["next_base_action"][row], dtype=torch.float32),
    }
    for camera in ("side_policy", "wrist_1buchong", "wrist_1", "wrist_2"):
        obs[f"observation.images.{camera}"] = torch.as_tensor(data[f"image_{camera}"][row])
        next_obs[f"observation.images.{camera}"] = torch.as_tensor(data[f"next_image_{camera}"][row])
    done = torch.tensor(bool(data["terminated"][row] or data["truncated"][row]))
    return TensorDict({
        "obs": TensorDict(obs, batch_size=[]),
        "action": torch.as_tensor(critic_action(data, row), dtype=torch.float32),
        "next": TensorDict({
            "obs": TensorDict(next_obs, batch_size=[]),
            "done": done,
            "reward": torch.tensor(_marvin_binary_reward(data["reward"][row])),
        }, batch_size=[]),
        "_priority": torch.tensor(10.0),
    }, batch_size=[]).unsqueeze(0)

# -----------------------------------------------------------------------------
# Hugging Face buffer cache helpers (global) ---------------------------------
# -----------------------------------------------------------------------------
OFFLINE_HF_REPO = os.environ.get("HF_OFFLINE_BUFFER_REPO", None)
ONLINE_HF_REPO = os.environ.get("HF_ONLINE_BUFFER_REPO", None)

if OFFLINE_HF_REPO is not None:
    logger.info(f"Using offline buffer from {OFFLINE_HF_REPO}")
if ONLINE_HF_REPO is not None:
    logger.info(f"Using online buffer from {ONLINE_HF_REPO}")

# Generic environment variable (shared across algorithms) -------------------
# ``CACHE_DIR`` specifies the root folder for **all** local caches.
# Falls back to the current directory if unset.
_CACHE_ROOT = Path(os.environ.get("CACHE_DIR", ".")).expanduser().resolve()

# Dedicated sub-folders for the different cache types. Optional overrides let
# a new online experiment reuse the large immutable offline cache without
# sharing mutable online replay state.
OFFLINE_CACHE_DIR = Path(
    os.environ.get("OFFLINE_CACHE_DIR", _CACHE_ROOT / "offline_buffer_cache")
).expanduser().resolve()
ONLINE_CACHE_DIR = Path(
    os.environ.get("ONLINE_CACHE_DIR", _CACHE_ROOT / "online_buffer_cache")
).expanduser().resolve()


# -----------------------------------------------------------------------------
# Repository-local imports ------------------------------------------------------
# -----------------------------------------------------------------------------
os.environ["MUJOCO_GL"] = "egl"

if "MUJOCO_EGL_DEVICE_ID" in os.environ:
    del os.environ["MUJOCO_EGL_DEVICE_ID"]


def _add_transitions_to_buffer(
    *,
    obs: dict,
    next_obs: dict,
    actions: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    info: dict,
    device: torch.device,
    image_keys: list[str],
    lowdim_keys: list[str],
    num_envs: int,
    online_rb: TensorDictPrioritizedReplayBuffer,
) -> None:
    """Helper function to create transitions and add them to the replay buffer.

    Handles terminal observations correctly and convert images to uint8 for storage.
    """
    obs_keys_set = set(image_keys) | set(lowdim_keys)
    for i in range(num_envs):
        # Handle terminal observation (same logic as main loop)
        if done[i] and "final_obs" in info and info["final_obs"][i] is not None:
            final_obs_dict = info["final_obs"][i]
            # Online n-step transforms keep their staging buffer on CPU.
            # The agent moves sampled batches to CUDA later.
            next_obs_i = {k: torch.as_tensor(v)[0].cpu() for k, v in final_obs_dict.items()}
        else:
            next_obs_i = {k: v[i] for k, v in next_obs.items()}

        curr_obs_i = {k: v[i] for k, v in obs.items()}

        # Keep only relevant keys & convert images to uint8 for storage
        curr_obs_i = {k: v for k, v in curr_obs_i.items() if k in obs_keys_set}
        next_obs_i = {k: v for k, v in next_obs_i.items() if k in obs_keys_set}
        to_uint8(curr_obs_i, image_keys)
        to_uint8(next_obs_i, image_keys)

        td = TensorDict(
            {
                "obs": TensorDict(curr_obs_i, batch_size=[]),
                "next": TensorDict(
                    {
                        "obs": TensorDict(next_obs_i, batch_size=[]),
                        "done": done[i],
                        "reward": reward[i],
                    },
                    batch_size=[],
                ),
                "action": actions[i],
                "_priority": torch.tensor(10.0, dtype=torch.float32),  # High initial priority for new samples
            },
            batch_size=[],
        ).unsqueeze(0)

        online_rb.add(td)


# -----------------------------------------------------------------------------
# Main training loop -----------------------------------------------------------
# -----------------------------------------------------------------------------
def main(cfg: ResidualTD3DexmgConfig):
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    if cfg.environment_backend == "marvin" and device.type == "cuda":
        gpu_fraction = float(cfg.marvin.get("learner_gpu_memory_fraction", 0.8))
        if not 0.0 < gpu_fraction < 1.0:
            raise ValueError("marvin.learner_gpu_memory_fraction must be in (0, 1)")
        torch.cuda.set_per_process_memory_fraction(gpu_fraction, device=torch.cuda.current_device())
        print(f"Marvin learner CUDA memory fraction: {gpu_fraction:.0%}")

    # Enable performance optimizations
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ---------------------------------------------------------------------
    # Load the behaviour-cloning policy that will serve as the "base" policy
    # for residual learning.
    # ---------------------------------------------------------------------
    marvin_cfg = None
    grpc_server = None
    grpc_shutdown = None
    grpc_parameters = None
    grpc_transitions = None
    if cfg.environment_backend == "marvin":
        if cfg.offline_data is None or not cfg.offline_data.use_base_policy_for_base_actions:
            raise ValueError(
                "Marvin ResFiT requires offline_data.use_base_policy_for_base_actions=true: "
                "the frozen ACT action must be stored as observation.base_action while the "
                "demonstration action remains the executed replay action."
            )
        marvin_cfg = MarvinResidualConfig(**dict(cfg.marvin))
        if marvin_cfg.async_grpc:
            import threading

            grpc_shutdown = threading.Event()
            grpc_parameters = queue.Queue(maxsize=1)
            grpc_transitions = queue.Queue(maxsize=2)
            grpc_server = start_learner_server(
                host=marvin_cfg.grpc_host,
                port=marvin_cfg.grpc_port,
                parameters_queue=grpc_parameters,
                transition_queue=grpc_transitions,
                shutdown_event=grpc_shutdown,
                push_frequency_s=marvin_cfg.grpc_parameter_push_s,
            )
        if not marvin_cfg.base_policy_checkpoint:
            raise ValueError("marvin.base_policy_checkpoint is required")
        base_policy = MarvinACTBasePolicy(marvin_cfg)
        eval_base_policy = base_policy
    elif cfg.environment_backend == "marvin":
        assert "base_policy" in cfg, "Base policy configuration is required"
        policy_dir, _ = download_policy_from_wandb(
            cfg.base_policy.wandb_id,
            step=cfg.base_policy.wt_type,
            artifact_version=cfg.base_policy.wt_version,
        )
        base_policy = load_policy(policy_dir)
        base_policy.to(device)
        base_policy.eval()
        eval_base_policy = load_policy(policy_dir)
        eval_base_policy.to(device)
        eval_base_policy.eval()

    # Extract the configuration from base policy
    base_cfg = getattr(base_policy, "config", None)

    if cfg.environment_backend == "marvin" or isinstance(base_cfg, ACTConfig):
        cfg.actor_name = "residual_act"
    else:
        raise ValueError(f"Unknown base policy type: {type(base_cfg)}")

    # Load dataset and get normalization functions early
    print("Loading dataset metadata and setting up normalization...")
    dataset_kwargs = {"root": cfg.offline_data.root} if cfg.offline_data.root else {}
    # Metadata/stats are sufficient for model construction and cache hashing.
    # Avoid constructing LeRobotDataset (which maps every Arrow shard) until
    # the cache-miss path actually needs to populate the replay buffer.
    dataset_meta = LeRobotDatasetMetadata(cfg.offline_data.name, **dataset_kwargs)
    dataset = None

    # Create action scaler from dataset statistics
    action_scaler = ActionScaler.from_dataset_stats(
        action_stats=dataset_meta.stats["action"],
        action_scale=cfg.agent.actor.action_scale,
        min_range_per_dim=cfg.offline_data.min_action_range,
        absolute_min=-1.0 if cfg.environment_backend == "marvin" else None,
        absolute_max=1.0 if cfg.environment_backend == "marvin" else None,
        device=device,
    )

    # Create state standardizer from dataset statistics
    state_standardizer = StateStandardizer.from_dataset_stats(
        state_stats=dataset_meta.stats["observation.state"],
        min_std=cfg.offline_data.min_state_std,
        device=device,
    )
    marvin_replay_identity = None
    if cfg.environment_backend == "marvin":
        marvin_replay_identity = replay_identity(
            base_policy_checkpoint=marvin_cfg.base_policy_checkpoint,
            base_policy_chunk_size=marvin_cfg.base_policy_chunk_size,
            base_policy_n_action_steps=marvin_cfg.base_policy_n_action_steps,
            offline_dataset_root=cfg.offline_data.root,
            offline_dataset_name=cfg.offline_data.name,
            actor_action_scale=cfg.agent.actor.action_scale,
            min_action_range=cfg.offline_data.min_action_range,
            min_state_std=cfg.offline_data.min_state_std,
        )

    def get_envs(
        env_name: str,
        num_envs: int,
        base_policy: ACTPolicy,
        device: str,
        video_key: str,
        debug: bool,
        action_scaler: ActionScaler,
        state_standardizer: StateStandardizer,
    ):
        assert action_scaler is not None, "action_scaler must be provided for consistent normalization"
        assert state_standardizer is not None, "state_standardizer must be provided for consistent normalization"

        if cfg.environment_backend == "marvin":
            return MarvinResidualVecEnv(
                marvin_cfg,
                base_policy=base_policy,
                action_scaler=action_scaler,
                state_standardizer=state_standardizer,
            )

        # Create the vectorized environment
        vec_env = create_vectorized_env(
            env_name=env_name,
            num_envs=num_envs,
            device=device,
            video_key=video_key,
            debug=debug,
        )

        # Wrap it with the base policy wrapper
        return BasePolicyVecEnvWrapper(
            vec_env=vec_env,
            base_policy=base_policy,
            action_scaler=action_scaler,
            state_standardizer=state_standardizer,
        )

    # ---------------------------------------------------------------------
    # Seeding (must be done before environment creation) ------------------
    # ---------------------------------------------------------------------
    if cfg.seed is None:
        cfg.seed = random.randint(0, 2**32 - 1)

    # Comprehensive seeding for reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # CUDA seeding for multi-GPU reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # Set deterministic behavior
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    print(f"Set random seed to {cfg.seed}")

    # ---------------------------------------------------------------------
    # Environment setup ----------------------------------------------------
    # ---------------------------------------------------------------------
    assert cfg.num_envs == 1, "Only support 1 environment for now because of how n_step is implemented"
    learner_only = bool(cfg.environment_backend == "marvin" and marvin_cfg.learner_only)
    env = None if learner_only else get_envs(
        env_name=cfg.task,
        num_envs=cfg.num_envs,
        base_policy=base_policy,
        device=device_str,
        video_key=cfg.video_key,
        debug=cfg.debug,
        action_scaler=action_scaler,
        state_standardizer=state_standardizer,
    )
    cfg.eval_num_envs = min(cfg.eval_num_envs, cfg.eval_num_episodes)
    num_cpus_available = os.cpu_count() - 1 if os.cpu_count() is not None else 1
    cfg.eval_num_envs = min(num_cpus_available, cfg.eval_num_envs)

    eval_env = None if learner_only else env if cfg.environment_backend == "marvin" else get_envs(
        env_name=cfg.task,
        num_envs=cfg.eval_num_envs,
        base_policy=eval_base_policy,
        device=device_str,
        video_key=cfg.video_key,
        debug=cfg.debug,
        action_scaler=action_scaler,
        state_standardizer=state_standardizer,
    )

    # Seed environments explicitly for reproducibility
    if env is not None and hasattr(env, "seed"):
        env.seed(cfg.seed)
    if eval_env is not None and hasattr(eval_env, "seed"):
        eval_env.seed(cfg.seed + 1)  # Use different seed for eval env to avoid correlation

    # ---------------------------------------------------------------------
    # Observation / action dimensions -------------------------------------
    # ---------------------------------------------------------------------
    # Determine which image keys (camera observations) will be used. The
    # configuration can specify either a single camera name (str) or a list of
    # names.
    if isinstance(cfg.rl_camera, str):
        image_keys: list[str] = [cfg.rl_camera]
    else:
        image_keys = list(cfg.rl_camera)
    assert isinstance(image_keys, list)
    if learner_only:
        lowdim_dim = 24
        img_c, img_h, img_w = (3, 128, 128)
        action_dim = 5
        horizon = int(marvin_cfg.max_steps)
        print("Marvin learner-only mode: Bridge/environment construction is disabled")
    else:
        lowdim_dim = env.observation_space["observation.state"].shape[1]
        img_c, img_h, img_w = env.observation_space[image_keys[0]].shape[1:]
        action_dim = env.action_space.shape[1]
        horizon = env.vec_env.metadata["horizon"]

    lowdim_keys = ["observation.state", "observation.base_action"]

    # ---------------------------------------------------------------------
    # Networks ------------------------------------------------------------
    # ---------------------------------------------------------------------
    print("[Marvin startup] constructing QAgent (encoder/critic/actor)...", flush=True)
    agent = QAgent(
        obs_shape=(img_c, img_h, img_w),
        prop_shape=(lowdim_dim,),
        action_dim=action_dim,
        rl_cameras=image_keys,
        cfg=cfg.agent,
        residual_actor=True,  # Enable residual actor mode
    )
    print("[Marvin startup] QAgent construction completed", flush=True)

    # Set up actor learning rate warmup
    actor_updates = 0
    initial_global_step = 0
    critic_warmup_completed = False
    if cfg.environment_backend == "marvin" and marvin_cfg.resume_checkpoint:
        resume_spec = str(marvin_cfg.resume_checkpoint)
        resume_path = (
            Path(marvin_cfg.training_checkpoint_dir).expanduser().resolve() / "latest.pt"
            if resume_spec == "auto"
            else Path(resume_spec).expanduser().resolve()
        )
        if resume_spec == "auto" and not resume_path.is_file():
            resume_path = None
    else:
        resume_path = None
    if resume_path is not None:
        print(f"[Marvin startup] loading learner checkpoint: {resume_path}", flush=True)
        bundle = torch.load(resume_path, map_location=device, weights_only=False)
        if bundle.get("base_policy_sha256") != marvin_replay_identity["base_policy_sha256"]:
            raise ValueError("Resume checkpoint was trained with a different ACT base policy")
        if bundle.get("action_contract") != marvin_replay_identity["action_contract"]:
            raise ValueError("Resume checkpoint uses a different Marvin action contract")
        agent.load_state_dict(bundle["agent_state_dict"])
        optimizer_states = bundle.get("optimizer_state_dict", {})
        for name, optimizer in (
            ("actor_opt", agent.actor_opt),
            ("critic_opt", agent.critic_opt),
            ("encoder_opt", agent.encoder_opt),
        ):
            if name in optimizer_states:
                optimizer.load_state_dict(optimizer_states[name])
        initial_global_step = int(bundle["global_step"])
        actor_updates = int(bundle.get("actor_updates", 0))
        critic_warmup_completed = bool(bundle.get("critic_warmup_completed", False))
        print(f"Resumed Marvin learner from {resume_path} at global_step={initial_global_step}")
    else:
        print("[Marvin startup] no learner checkpoint; starting from step 0", flush=True)
    if cfg.algo.actor_lr_warmup_steps > 0:
        print(
            f"Actor LR warmup enabled: 0.0 -> {cfg.agent.actor_lr:.2e} "
            f"over {cfg.algo.actor_lr_warmup_steps} actor updates"
        )

    # ---------------------------------------------------------------------
    # Replay buffers -------------------------------------------------------
    # ---------------------------------------------------------------------
    # Prioritized replay needs TorchRL's optional C++ segment-tree extension;
    # the published/default uniform path uses the portable random sampler.
    alpha = cfg.algo.priority_alpha if cfg.algo.sampling_strategy == "prioritized_replay" else 0.0
    beta = cfg.algo.priority_beta if cfg.algo.sampling_strategy == "prioritized_replay" else 0.0

    online_batch_size = int(cfg.algo.batch_size * (1 - cfg.algo.offline_fraction))
    offline_batch_size = int(cfg.algo.batch_size * cfg.algo.offline_fraction)

    if cfg.algo.offline_fraction == 0.0:
        print("Online-only training mode: offline_fraction=0.0")

    def _make_replay_buffer(
        max_size: int, batch_size: int, storage=None
    ) -> ReplayBuffer:
        common_kwargs = {
            "storage": storage
            if storage is not None
            else LazyTensorStorage(max_size=max_size, device="cpu"),
            "transform": MultiStepTransform(n_steps=cfg.algo.n_step, gamma=cfg.algo.gamma),
            "pin_memory": True,
            "prefetch": cfg.algo.prefetch_batches,
            "batch_size": batch_size,
        }
        if cfg.algo.sampling_strategy == "prioritized_replay":
            return TensorDictPrioritizedReplayBuffer(
                alpha=alpha,
                beta=beta,
                eps=1e-6,
                priority_key="_priority",
                **common_kwargs,
            )
        return TensorDictReplayBuffer(**common_kwargs)

    online_rb = _make_replay_buffer(cfg.algo.buffer_size, online_batch_size)

    # ------------------------------------------------------------------
    # Caching layer for online replay buffer ----------------------------
    # ------------------------------------------------------------------
    online_cache_meta = {
        "task": cfg.task,
        "image_keys": image_keys,
        "n_step": cfg.algo.n_step,
        "gamma": cfg.algo.gamma,
        "horizon": horizon,
        "size": cfg.algo.learning_starts,
        "sampling_strategy": cfg.algo.sampling_strategy,
        "buffer_size": cfg.algo.buffer_size,
        "batch_size": online_batch_size,
        # Include random action noise scale to prevent mixing data from different noise levels
        "random_action_noise_scale": cfg.algo.random_action_noise_scale,
        # Normalization parameters for consistency
        "min_action_range": cfg.offline_data.min_action_range,
        "min_state_std": cfg.offline_data.min_state_std,
        "normalized_actions": True,
        "critic_action_source": "applied_then_actual_then_command_v1",
        # Library versions for compatibility
        "torchrl_version": torchrl.__version__,
        "tensordict_version": tensordict.__version__,
    }
    if cfg.algo.sampling_strategy == "prioritized_replay":
        online_cache_meta["priority_alpha"] = cfg.algo.priority_alpha
        online_cache_meta["priority_beta"] = cfg.algo.priority_beta

    pprint.pprint(online_cache_meta)
    _online_meta_str = json.dumps(online_cache_meta, sort_keys=True)
    online_cache_hash = hashlib.sha1(_online_meta_str.encode()).hexdigest()[:8]  # noqa: S324
    # Base local path for the online buffer ------------------------------
    online_cache_dir = ONLINE_CACHE_DIR / online_cache_hash

    # Attempt to download/extract from HF every run (no-op if already cached)
    dl_dir = None
    if ONLINE_HF_REPO is not None and cfg.environment_backend != "marvin":
        print(f"Attempting to download online buffer {online_cache_hash} from {ONLINE_HF_REPO}...")
        dl_dir = _hf_download_buffer(ONLINE_HF_REPO, online_cache_hash, ONLINE_CACHE_DIR)
    if dl_dir is not None:
        online_cache_dir = dl_dir

    loaded_online_from_cache = False
    known_marvin_episode_ids: set[int] = set()
    marvin_cached_manifest_sha1: dict[int, str] = {}
    marvin_cached_raw_transitions = 0
    marvin_online_cache_dir = None
    marvin_online_cache_storage = None
    replay_root = None
    marvin_episode_count = 0
    marvin_transition_count = 0
    marvin_success_count = 0
    marvin_intervention_steps = 0
    marvin_intervention_episodes = 0
    if cfg.environment_backend != "marvin" and online_cache_dir.exists():
        print(f"{online_cache_dir} found on disk. Attempting to load...")
        online_rb.sampler._empty()
        optimized_replay_buffer_loads(online_rb, online_cache_dir)
        loaded_online_from_cache = True
        print(f"Loaded online buffer from cache at {online_cache_dir} (size={len(online_rb)})")

    # Marvin collector writes an auditable NPZ interchange format.  Import it
    # only for the Marvin backend; published DexMimicGen runs keep their cache
    # behavior unchanged.
    if cfg.environment_backend == "marvin" and not loaded_online_from_cache:
        replay_dir = str(marvin_cfg.replay_dir or "")
        if not replay_dir:
            raise ValueError("marvin.replay_dir is required")
        replay_root = Path(replay_dir)
        replay_episodes, raw_replay_transitions, next_replay_episode_id = inspect_replay_directory(
            replay_root, marvin_replay_identity
        )
        replay_fingerprint = replay_manifest_fingerprint(
            [manifest for _path, manifest in replay_episodes]
        )
        marvin_online_cache_dir = ONLINE_CACHE_DIR / f"marvin_{online_cache_hash}_incremental"
        marvin_online_cache_storage = marvin_online_cache_dir / "storage"
        marvin_online_cache_complete = (
            marvin_online_cache_storage / "storage_metadata.json"
        ).is_file()

        cache_metadata = {}
        if marvin_online_cache_complete:
            recovery = recover_pending_cache_metadata(marvin_online_cache_dir)
            if recovery == "invalid":
                stale_cache = marvin_online_cache_dir.with_name(
                    f"{marvin_online_cache_dir.name}.invalid_{int(time.time())}"
                )
                os.replace(marvin_online_cache_dir, stale_cache)
                print(
                    f"[Marvin startup] quarantined interrupted online cache as {stale_cache}",
                    flush=True,
                )
                marvin_online_cache_complete = False
            elif recovery != "none":
                print(
                    f"[Marvin startup] recovered pending online cache commit: {recovery}",
                    flush=True,
                )

        if marvin_online_cache_complete:
            metadata_path = marvin_online_cache_dir / "user_metadata.json"
            if metadata_path.is_file():
                cache_metadata = json.loads(metadata_path.read_text())
            validated_ids = cached_episode_ids(cache_metadata, replay_episodes)
            if validated_ids is None:
                stale_cache = marvin_online_cache_dir.with_name(
                    f"{marvin_online_cache_dir.name}.invalid_{int(time.time())}"
                )
                os.replace(marvin_online_cache_dir, stale_cache)
                print(
                    "[Marvin startup] incremental cache mismatched authoritative NPZ replay; "
                    f"quarantined as {stale_cache}",
                    flush=True,
                )
                marvin_online_cache_complete = False

        if marvin_online_cache_complete:
            print(
                f"[Marvin startup] loading incremental online replay memmap cache "
                f"{marvin_online_cache_dir}...",
                flush=True,
            )
            online_rb = _make_replay_buffer(
                cfg.algo.buffer_size,
                max(online_batch_size, 1),
                storage=LazyMemmapStorage(
                    max_size=cfg.algo.buffer_size,
                    scratch_dir=marvin_online_cache_storage,
                    device="cpu",
                    existsok=True,
                ),
            )
            online_rb.sampler._empty()
            optimized_replay_buffer_loads(online_rb, marvin_online_cache_dir)
            loaded_online_from_cache = len(online_rb) > 0
            known_marvin_episode_ids.update(validated_ids)
            marvin_cached_manifest_sha1.update(
                {
                    episode_id: manifest_sha1(manifest)
                    for _path, manifest in replay_episodes
                    if (episode_id := int(manifest["episode_id"])) in validated_ids
                }
            )
            marvin_cached_raw_transitions = sum(
                int(manifest["transitions"])
                for _path, manifest in replay_episodes
                if int(manifest["episode_id"]) in validated_ids
            )
            transform_name = cache_metadata.get("transform_buffer")
            if transform_name:
                multi_step_transform = get_multi_step_transform(online_rb)
                transform_bundle = torch.load(
                    marvin_online_cache_dir / transform_name,
                    map_location="cpu",
                    weights_only=True,
                )
                if int(transform_bundle["buffer_size"]) != len(online_rb):
                    raise RuntimeError("Incremental online cache transform context is stale")
                multi_step_transform._buffer = TensorDict.from_dict(
                    transform_bundle["buffer"],
                    batch_size=torch.Size(transform_bundle["batch_size"]),
                )
                multi_step_transform._validated = True
            elif validated_ids:
                # Legacy caches did not persist MultiStepTransform's short
                # look-back window. Reconstruct it from the authoritative tail
                # so the first appended episode does not lose n_step-1 rows.
                last_cached_id = max(validated_ids)
                last_path = next(
                    path
                    for path, manifest in replay_episodes
                    if int(manifest["episode_id"]) == last_cached_id
                )
                with np.load(last_path, allow_pickle=False) as data:
                    start = max(0, len(data["reward"]) - int(cfg.algo.n_step))
                    multi_step_transform = get_multi_step_transform(online_rb)
                    multi_step_transform._buffer = torch.cat(
                        [_marvin_transition(data, row) for row in range(start, len(data["reward"]))],
                        dim=-1,
                    )
                    multi_step_transform._validated = True
            print(
                f"[Marvin startup] loaded incremental online replay memmap cache "
                f"(size={len(online_rb)}, cached_episodes={len(validated_ids)})",
                flush=True,
            )
        else:
            marvin_online_cache_dir.mkdir(parents=True, exist_ok=True)
            online_rb = _make_replay_buffer(
                cfg.algo.buffer_size,
                max(online_batch_size, 1),
                storage=LazyMemmapStorage(
                    max_size=cfg.algo.buffer_size,
                    scratch_dir=marvin_online_cache_storage,
                    device="cpu",
                    existsok=True,
                ),
            )

        missing_episodes = 0
        for episode_path, _manifest in replay_episodes:
            episode_id = int(_manifest["episode_id"])
            marvin_episode_count += 1
            marvin_transition_count += int(_manifest["transitions"])
            marvin_success_count += int(bool(_manifest.get("success", False)))
            intervention_steps = int(_manifest.get("intervention_steps", 0))
            marvin_intervention_steps += intervention_steps
            marvin_intervention_episodes += int(intervention_steps > 0)
            if episode_id not in known_marvin_episode_ids:
                with np.load(episode_path, allow_pickle=False) as data:
                    _add_marvin_episode(data, online_rb)
                known_marvin_episode_ids.add(episode_id)
                marvin_cached_manifest_sha1[episode_id] = manifest_sha1(_manifest)
                marvin_cached_raw_transitions += int(_manifest["transitions"])
                missing_episodes += 1

        def _persist_marvin_online_cache() -> None:
            if marvin_online_cache_dir is None:
                return
            multi_step_transform = get_multi_step_transform(online_rb)
            transform_name = f"transform_buffer_{len(online_rb):08d}.pt"
            transform_path = marvin_online_cache_dir / transform_name
            transform_tmp = marvin_online_cache_dir / f".{transform_name}.tmp"
            torch.save(
                {
                    "buffer_size": len(online_rb),
                    "buffer": multi_step_transform._buffer.to_dict(),
                    "batch_size": list(multi_step_transform._buffer.batch_size),
                },
                transform_tmp,
            )
            os.replace(transform_tmp, transform_path)
            metadata = {
                "schema_version": 1,
                "replay_dir": str(replay_root),
                "replay_manifest_fingerprint": replay_fingerprint,
                "raw_replay_transitions": marvin_cached_raw_transitions,
                "buffer_size": len(online_rb),
                "transform_buffer": transform_name,
                "episode_manifest_sha1": {
                    str(episode_id): digest
                    for episode_id, digest in sorted(marvin_cached_manifest_sha1.items())
                },
            }
            pending_path = marvin_online_cache_dir / ".user_metadata.pending.json"
            metadata_path = marvin_online_cache_dir / "user_metadata.json"
            pending_path.write_text(json.dumps(metadata, indent=2) + "\n")
            optimized_replay_buffer_dumps(online_rb, marvin_online_cache_dir)
            os.replace(pending_path, metadata_path)
            for stale_transform in marvin_online_cache_dir.glob("transform_buffer_*.pt"):
                if stale_transform != transform_path:
                    stale_transform.unlink()

        if missing_episodes:
            print(
                f"[Marvin startup] persisting {missing_episodes} newly imported NPZ episodes "
                "to incremental online cache...",
                flush=True,
            )
            _persist_marvin_online_cache()
        loaded_online_from_cache = True
        print(
            f"Loaded Marvin NPZ replay from {replay_root} "
            f"(raw_transitions={raw_replay_transitions}, buffer_size={len(online_rb)})"
        )
        required_replay = max(
            int(cfg.algo.learning_starts),
            int(marvin_cfg.minimum_replay_transitions),
            int(online_batch_size),
        )
        if len(online_rb) < required_replay:
            raise RuntimeError(
                f"Marvin replay is too small after n-step conversion: {len(online_rb)} < {required_replay}. "
                "Continue the formal zero-residual collection before starting the learner."
            )

    # Offline data is required for normalization, but can be unused for training if offline_fraction=0
    assert cfg.offline_data is not None and cfg.offline_data.num_episodes is not None

    # Dataset and normalization already loaded above - use existing dataset

    # Use actual dataset metadata for precise buffer sizing
    if cfg.offline_data.num_episodes is not None:
        # Only use subset of episodes if specified
        total_frames = sum(
            dataset_meta.episodes[ep_idx]["length"]
            for ep_idx in range(min(cfg.offline_data.num_episodes, dataset_meta.total_episodes))
        )
        num_episodes = cfg.offline_data.num_episodes
    else:
        # Use entire dataset
        total_frames = dataset_meta.total_frames
        num_episodes = dataset_meta.total_episodes

    # Calculate transitions: each episode contributes (episode_length - 1) transitions
    estimated_transitions = max(0, total_frames - num_episodes)

    print("Dataset buffer sizing:")
    print(f"  Total frames to process: {total_frames}")
    print(f"  Number of episodes: {num_episodes}")
    print(f"  Estimated transitions: {estimated_transitions}")

    # Calculate buffer size for simplified approach (1 transition per frame pair)
    max_offline_transitions = (
        estimated_transitions if cfg.algo.offline_fraction > 0.0 else 1
    )  # Minimum size for online-only mode
    if cfg.algo.offline_fraction > 0.0:
        print(f"Offline buffer sized for GT-as-base approach: {max_offline_transitions} transitions")
    else:
        print("Online-only mode: creating minimal offline buffer (unused)")

    offline_rb = _make_replay_buffer(max_offline_transitions, max(offline_batch_size, 1))

    # Normalization functions already defined above - use them

    # ------------------------------------------------------------------
    # Convert offline dataset episodes into transitions and fill buffer
    # ------------------------------------------------------------------
    def _populate_offline_buffer(
        dataset: LeRobotDataset,
        rb: ReplayBuffer,
        image_keys: list[str],
        num_episodes: int | None = None,
        use_base_policy_for_base_actions: bool = False,
        base_policy: ACTPolicy | None = None,
    ) -> int:
        """
        Iterates through *dataset* sequentially, converts consecutive frames
        into residual RL transitions and pushes them into *rb*.

        Two modes:
        1. GT-as-base (use_base_policy_for_base_actions=False):
           Uses GT actions as both the base action (in observations) and the target action
           (in transitions). Teaches residual policy to output zero: residual = GT - GT = 0

        2. Base-policy-as-base (use_base_policy_for_base_actions=True):
           Uses base policy to generate base actions and GT actions as targets.
           More consistent with online training: residual = GT - base_policy_action

        Returns the number of transitions added.
        """
        if use_base_policy_for_base_actions and base_policy is None:
            raise ValueError("base_policy must be provided when use_base_policy_for_base_actions=True")

        # Populate buffer from pre-loaded dataset
        print("Populating offline buffer from dataset...")
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        episode_cache: dict[int, dict] = {}
        transitions = 0
        step_id = 0
        last_ep_idx: int | None = None

        for sample in tqdm(loader, desc="Processing offline dataset"):
            ep_idx = int(sample["episode_index"].item())
            if num_episodes is not None and ep_idx == num_episodes:
                break
            if use_base_policy_for_base_actions and ep_idx != last_ep_idx:
                # Match online execution: an ACT action queue never crosses an
                # episode boundary.
                base_policy.reset()
            last_ep_idx = ep_idx

            # ------------------------------------------------------------------
            # Build observation and action directly for replay buffer ----------
            # ------------------------------------------------------------------
            # Extract data and keep on CPU (replay buffer uses CPU storage)
            _gt_action: torch.Tensor = sample["action"].float().squeeze(0)
            gt_action_scaled = action_scaler.scale(_gt_action)
            done_flag = bool(sample["next.done"].item())
            raw_reward_value = sample.get("rl.reward", sample.get("reward", None))
            if raw_reward_value is None:
                raw_reward = float(done_flag)
            else:
                raw_reward = float(torch.as_tensor(raw_reward_value).reshape(-1)[0].item())
            binary_reward = _marvin_binary_reward(raw_reward) if cfg.environment_backend == "marvin" else float(done_flag)

            # Generate base action based on the selected mode
            if use_base_policy_for_base_actions:
                # Use base policy to generate base action from current observation
                # Build raw observation first for base policy inference
                raw_obs = {}
                for k in sample:
                    if "observation" in k:
                        raw_obs[k] = sample[k].to(device)  # Keep batch dimension for base policy

                # Get base action from base policy
                with torch.no_grad():
                    base_action = base_policy.select_action(raw_obs)
                base_action_scaled = action_scaler.scale(base_action.squeeze(0).cpu())
            else:
                # Use GT action as base action (original behavior)
                base_action_scaled = gt_action_scaled

            # Build observation dict directly in target format
            curr_obs = {
                "observation.state": state_standardizer.standardize(sample["observation.state"].float().squeeze(0)),
                "observation.base_action": base_action_scaled,
            }
            for k in image_keys:
                curr_obs[k] = sample[k].squeeze(0)

            # Convert images to uint8 for memory-efficient storage
            to_uint8(curr_obs, image_keys)

            # ------------------------------------------------------------------
            # If we already cached the *previous* frame for this episode we can
            # create transitions now.
            # ------------------------------------------------------------------
            if ep_idx in episode_cache:
                # Create transitions for each combination of prev and current variants
                prev_obs = episode_cache[ep_idx]["obs"]
                prev_action_scaled = episode_cache[ep_idx]["action"]
                transition = TensorDict(
                    {
                        "obs": TensorDict(prev_obs, batch_size=[]),
                        "action": prev_action_scaled,
                        "next": TensorDict(
                            {
                                "obs": TensorDict(curr_obs, batch_size=[]),
                                "done": torch.tensor(done_flag, dtype=torch.bool),
                                "reward": torch.tensor(binary_reward, dtype=torch.float32),
                            },
                            batch_size=[],
                        ),
                        "_priority": torch.tensor(10.0, dtype=torch.float32),  # High initial priority for new samples
                    },
                    batch_size=[],
                ).unsqueeze(0)

                rb.add(transition)
                transitions += 1

                step_id += 1
            else:
                step_id = 0

            # Cache current frame for pairing with the next one ---------------
            episode_cache[ep_idx] = {
                "obs": curr_obs,
                "action": gt_action_scaled,
                "done": done_flag,
                "step_id": step_id,
            }

        # Log final statistics
        print(f"Added {transitions} transitions")

        return transitions

    # ------------------------------------------------------------------
    # Caching layer for offline replay buffer ---------------------------
    # ------------------------------------------------------------------
    # Build a metadata dictionary that uniquely identifies the buffer
    offline_cache_meta = {
        "task": cfg.task,
        "dataset_name": cfg.offline_data.name,
        "num_episodes": cfg.offline_data.num_episodes,
        "use_base_policy_for_base_actions": cfg.offline_data.use_base_policy_for_base_actions,
        "min_action_range": cfg.offline_data.min_action_range,
        "min_state_std": cfg.offline_data.min_state_std,
        "image_keys": image_keys,
        "n_step": cfg.algo.n_step,
        "gamma": cfg.algo.gamma,
        "base_policy_wandb_id": cfg.base_policy.wandb_id,
        "sampling_strategy": cfg.algo.sampling_strategy,
        "normalized_actions": True,
        "batch_size": offline_batch_size,
        # Library versions for compatibility
        "torchrl_version": torchrl.__version__,
        "tensordict_version": tensordict.__version__,
    }
    if cfg.environment_backend == "marvin":
        checkpoint_root = Path(marvin_cfg.base_policy_checkpoint).expanduser().resolve()
        if (checkpoint_root / "policy" / "config.json").is_file():
            checkpoint_root = checkpoint_root / "policy"
        weight_candidates = [checkpoint_root / "model.safetensors", checkpoint_root / "pytorch_model.bin"]
        weight_path = next((path for path in weight_candidates if path.is_file()), None)
        if weight_path is None:
            raise FileNotFoundError(f"No ACT weight file found in {checkpoint_root}")
        offline_cache_meta["base_policy_checkpoint"] = str(checkpoint_root)
        offline_cache_meta["base_policy_sha256"] = _sha256_file(weight_path)
        offline_cache_meta["action_contract"] = marvin_replay_identity["action_contract"]
        offline_cache_meta["command_bounds"] = marvin_replay_identity["command_bounds"]
        offline_cache_meta["offline_dataset_stats_sha256"] = marvin_replay_identity[
            "offline_dataset_stats_sha256"
        ]
    if cfg.algo.sampling_strategy == "prioritized_replay":
        offline_cache_meta["priority_alpha"] = cfg.algo.priority_alpha
        offline_cache_meta["priority_beta"] = cfg.algo.priority_beta

    pprint.pprint(offline_cache_meta)
    print("[Marvin startup] offline cache hash computed; checking local/HF cache...", flush=True)

    # Deterministically hash the metadata to create a short cache directory name
    meta_str = json.dumps(offline_cache_meta, sort_keys=True)
    cache_hash = hashlib.sha1(meta_str.encode()).hexdigest()[:8]  # noqa: S324

    # Base local path for this buffer ---------------------------------------
    cache_dir = OFFLINE_CACHE_DIR / cache_hash

    # Try to download/extract from the Hub (will no-op if file not there)
    downloaded_dir = None
    if OFFLINE_HF_REPO is not None:
        print(f"Attempting to download offline buffer {cache_hash} from {OFFLINE_HF_REPO}...")
        downloaded_dir = _hf_download_buffer(OFFLINE_HF_REPO, cache_hash, OFFLINE_CACHE_DIR)
    if downloaded_dir is not None:
        cache_dir = downloaded_dir  # use extracted location

    loaded_from_cache = False
    added = 0

    if cfg.algo.offline_fraction > 0.0:
        # Only populate offline buffer if we're using offline data
        cache_storage_dir = cache_dir / "storage"
        cache_is_complete = (cache_storage_dir / "storage_metadata.json").is_file()
        if cache_dir.exists() and cache_is_complete:
            print(f"{cache_dir} found on disk. Attempting to load...")
            # The cache was dumped as memmaps. Re-open those files directly on
            # cache hits instead of loading ~tens of GB into LazyTensorStorage
            # and copying it back out. This makes subsequent starts fast and
            # keeps image tensors out of anonymous RAM.
            offline_rb = _make_replay_buffer(
                max_offline_transitions,
                max(offline_batch_size, 1),
                storage=LazyMemmapStorage(
                    max_size=max_offline_transitions,
                    scratch_dir=cache_storage_dir,
                    device="cpu",
                    existsok=True,
                ),
            )
            offline_rb.sampler._empty()
            optimized_replay_buffer_loads(offline_rb, cache_dir)
            loaded_from_cache = True
            print(
                f"Loaded offline buffer from memmap cache at {cache_dir} "
                f"(size={len(offline_rb)}; no RAM copy)"
            )

        if not loaded_from_cache:
            print("[Marvin startup] offline cache miss/incomplete; loading Arrow dataset and building replay...", flush=True)
            # Only the cache-miss path needs frame-level Arrow access.
            dataset = LeRobotDataset(cfg.offline_data.name, **dataset_kwargs)
            added = _populate_offline_buffer(
                dataset=dataset,
                rb=offline_rb,
                image_keys=image_keys,
                num_episodes=cfg.offline_data.num_episodes,
                use_base_policy_for_base_actions=cfg.offline_data.use_base_policy_for_base_actions,
                base_policy=base_policy if cfg.offline_data.use_base_policy_for_base_actions else None,
            )

            print(f"Added {added} offline transitions to buffer (size={len(offline_rb)})")

            # Save buffer to disk for future runs + upload to Hub ----------------
            cache_dir.mkdir(parents=True, exist_ok=True)
            optimized_replay_buffer_dumps(offline_rb, cache_dir)

            with open(cache_dir / "user_metadata.json", "w") as f:
                json.dump(offline_cache_meta, f, indent=2)

            if OFFLINE_HF_REPO is not None:
                _hf_upload_buffer(OFFLINE_HF_REPO, cache_dir, cache_hash)
        else:
            added = len(offline_rb)
    else:
        print("Skipping offline buffer population for online-only training")

    # ------------------------------------------------------------------
    # Warm-up phase (random policy) --------------------------------------
    # ------------------------------------------------------------------

    if len(online_rb) < cfg.algo.learning_starts and not loaded_online_from_cache:
        if cfg.environment_backend == "marvin":
            raise RuntimeError(
                "Marvin training refuses automatic random warm-up. Collect a bounded ACT/residual replay "
                "with the Marvin collector and place/load it before starting the learner."
            )
        print(f"Warm-up: filling online buffer with {cfg.algo.learning_starts - len(online_rb)} random steps…")
        obs, _ = env.reset()
        # --------------------------------------------------------------
        # Logging helper: print progress every 1 000 collected transitions
        # --------------------------------------------------------------
        next_log_threshold = 1000  # first threshold for progress message

        reward_sum = 0
        episode_count = 0

        while len(online_rb) < cfg.algo.learning_starts:
            if cfg.algo.use_base_policy_for_warmup:
                # Use base policy action + noise (residual exploration)
                # Since the environment wrapper always adds base_action to residual_action,
                # we just need to provide the noise as the residual action
                rand_actions = (
                    torch.rand((cfg.num_envs, action_dim), device=device) * 2 - 1
                ) * cfg.algo.random_action_noise_scale
            else:
                # Pure uniform random actions - need to cancel out the base policy action
                # Since env does: combined = base_action + residual_action
                # To get pure random: residual_action = random - base_action
                base_action = obs["observation.base_action"]  # Already normalized to [-1, 1]
                pure_random = (
                    torch.rand((cfg.num_envs, action_dim), device=device) * 2 - 1
                ) * cfg.algo.random_action_noise_scale
                rand_actions = pure_random - base_action

            next_obs, reward, terminated, truncated, info = env.step(rand_actions)
            done = terminated | truncated

            reward_sum += reward.sum().item()
            episode_count += done.float().sum().item()

            # Use the executed combined action returned by the environment
            combined_action = info["scaled_action"]
            _add_transitions_to_buffer(
                obs=obs,
                next_obs=next_obs,
                actions=combined_action,
                reward=reward,
                done=done,
                info=info,
                device=device,
                image_keys=image_keys,
                lowdim_keys=lowdim_keys,
                num_envs=cfg.num_envs,
                online_rb=online_rb,
            )

            # ----------------------------------------------------------
            # Progress logging (every ~1 000 transitions) --------------
            # ----------------------------------------------------------
            if len(online_rb) >= next_log_threshold:
                success_rate = reward_sum / episode_count if episode_count > 0 else 0.0
                print(
                    f"[Warm-up] {len(online_rb)} / {cfg.algo.learning_starts} "
                    f"transitions collected, reward_sum={reward_sum:.2f}, "
                    f"success_rate={success_rate:.3f} ({reward_sum}/{episode_count})"
                )
                next_log_threshold += 1000

            obs = next_obs  # roll state

        # Persist freshly-collected buffer (local + HF) --------------------
        online_cache_dir.mkdir(parents=True, exist_ok=True)
        optimized_replay_buffer_dumps(online_rb, online_cache_dir)
        with open(online_cache_dir / "user_metadata.json", "w") as f:
            json.dump(online_cache_meta, f, indent=2)
        if ONLINE_HF_REPO is not None:
            _hf_upload_buffer(ONLINE_HF_REPO, online_cache_dir, online_cache_hash)
        print(f"Warm-up done. Online buffer size = {len(online_rb)} transitions")

        loaded_online_from_cache = True  # treat as cached going forward

    _hp_parts: list[str] = [
        cfg.task,  # e.g. "TwoArmBoxCleanup"
        f"n{cfg.algo.n_step}",  # n-step horizon
        f"utd{cfg.algo.num_updates_per_iteration}",  # updates-to-data ratio
        f"buf{cfg.algo.buffer_size}",  # replay buffer size
    ]

    # Offline dataset statistics (if any)
    if cfg.offline_data is not None and cfg.offline_data.num_episodes is not None and cfg.algo.offline_fraction > 0.0:
        _hp_parts.append(f"off{cfg.offline_data.num_episodes}ep")
    elif cfg.algo.offline_fraction == 0.0:
        _hp_parts.append("online_only")

    # Learning-rate, expressed in scientific notation for brevity (e.g. 1e-4 → 1e-04)
    _hp_parts.append(f"lr{cfg.agent.actor_lr:.0e}")

    # Additional flags ---------------------------------------------------------
    if cfg.agent.clip_q_target_to_reward_range:
        _hp_parts.append("clipT")

    hp_str = "_".join(_hp_parts)

    run_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}__{hp_str}__seed{cfg.seed}"

    if cfg.wandb.name is not None:
        run_name = f"{cfg.wandb.name}__{run_name}"

    _wandb_config = OmegaConf.to_container(cfg, resolve=True)
    # Remove notes from config if present
    assert isinstance(_wandb_config, dict)
    _wandb_config["wandb"].pop("notes", None)

    # Print a nice summary of the config
    print("Launching run with the following config:")
    pprint.pprint(_wandb_config)

    print("[Marvin startup] entering wandb.init()...", flush=True)
    wandb.init(
        id=cfg.wandb.continue_run_id,
        resume=None if cfg.wandb.continue_run_id is None else "allow",
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=_wandb_config,
        name=run_name,
        mode=cfg.wandb.mode if not cfg.debug else "disabled",
        notes=cfg.wandb.notes,
        group=cfg.wandb.group,
    )
    print("[Marvin startup] wandb.init() completed", flush=True)

    # Log horizon to wandb summary
    wandb.summary["environment/horizon"] = horizon

    # Create a timestamped folder in CACHE_DIR for all outputs
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_cache_dir = _CACHE_ROOT / f"run_{timestamp}_{run_name}"

    # Create subdirectories for models and outputs
    model_save_dir = run_cache_dir / "models"
    outputs_dir = run_cache_dir / "outputs"
    model_save_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    global_step = initial_global_step
    best_eval_success_rate = 0.0
    training_cum_time = 0.0
    episode_count = 0

    train_start_time = time.time()

    # Initialize timing utility
    training_timer = TrainingTimer()

    def _run_critic_warmup(
        agent, online_rb, offline_rb, cfg, device, training_timer, online_batch_size, offline_batch_size
    ):
        """Run critic-only updates for warmup phase."""
        for i in range(cfg.algo.critic_warmup_steps):
            # Sample mixed online/offline batch
            with training_timer.time("batch_sampling"):
                # Sample batches from replay buffers
                online_batch = online_rb.sample(online_batch_size)
                online_batch = online_batch.to(device, non_blocking=True)

                if cfg.algo.offline_fraction > 0.0:
                    # Mixed online/offline training
                    offline_batch = offline_rb.sample(offline_batch_size)
                    offline_batch = offline_batch.to(device, non_blocking=True)
                    batch = torch.cat([online_batch, offline_batch], dim=0)
                else:
                    # Online-only training
                    batch = online_batch

            # Only update critic during warmup (update_actor=False)
            with training_timer.time("gradient_update"):
                metrics = agent.update(batch, stddev=0.0, update_actor=False, bc_batch=None, ref_agent=agent)

            # Update priorities for prioritized experience replay
            if cfg.algo.sampling_strategy == "prioritized_replay" and "_td_errors" in metrics:
                # Update priorities in the batch for priority updates
                td_errors = metrics["_td_errors"]
                batch["_priority"] = td_errors

                if cfg.algo.offline_fraction > 0.0:
                    # Mixed online/offline training - update both buffers
                    online_batch_size_actual = int(cfg.algo.batch_size * (1 - cfg.algo.offline_fraction))

                    # Update online buffer priorities
                    if online_batch_size_actual > 0:
                        online_batch_subset = batch[:online_batch_size_actual]
                        online_rb.update_tensordict_priority(online_batch_subset)

                    # Update offline buffer priorities
                    if online_batch_size_actual < len(batch):
                        offline_batch_subset = batch[online_batch_size_actual:]
                        offline_rb.update_tensordict_priority(offline_batch_subset)
                else:
                    # Online-only training - update only online buffer
                    online_rb.update_tensordict_priority(batch)

            # Progress logging
            if i % 100 == 0:
                print(
                    f"Critic warmup: {i} / {cfg.algo.critic_warmup_steps}, "
                    f"train/critic_qt={metrics['train/critic_qt']:.4f} "
                    f"train/critic_loss={metrics['train/critic_loss']:.4f}"
                )
                wandb.log({
                    "warmup/update": i,
                    "warmup/critic_qt": metrics["train/critic_qt"],
                    "warmup/critic_loss": metrics["train/critic_loss"],
                })

    checkpoint_dir = None
    if cfg.environment_backend == "marvin":
        if not marvin_cfg.training_checkpoint_dir or not marvin_cfg.residual_checkpoint:
            raise ValueError("Formal Marvin training requires checkpoint and deployment output paths")
        checkpoint_dir = Path(marvin_cfg.training_checkpoint_dir).expanduser().resolve()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _publish_marvin_actor(step: int) -> None:
        if cfg.environment_backend != "marvin":
            return
        bundle = {
            "actor": agent.actor.state_dict(),
            "encoders": agent.encoders.state_dict(),
            "obs_shape": (img_c, img_h, img_w),
            "prop_shape": (lowdim_dim,),
            "action_dim": action_dim,
            "rl_cameras": list(image_keys),
            "actor_action_scale": float(cfg.agent.actor.action_scale),
            "global_step": step,
            "base_policy_sha256": marvin_replay_identity["base_policy_sha256"],
            "action_contract": marvin_replay_identity["action_contract"],
        }
        deployment_path = Path(marvin_cfg.residual_checkpoint).expanduser().resolve()
        deployment_path.parent.mkdir(parents=True, exist_ok=True)
        deployment_tmp = deployment_path.with_name(f".{deployment_path.name}.tmp")
        torch.save(bundle, deployment_tmp)
        os.replace(deployment_tmp, deployment_path)
        if grpc_parameters is not None:
            buffer = io.BytesIO()
            torch.save(bundle, buffer)
            try:
                while True:
                    grpc_parameters.get_nowait()
            except queue.Empty:
                pass
            grpc_parameters.put_nowait(buffer.getvalue())

    def _save_marvin_state(step: int) -> None:
        if cfg.environment_backend != "marvin":
            return
        step_path = checkpoint_dir / f"step_{step:08d}.pt"
        tmp_step_path = checkpoint_dir / f".step_{step:08d}.pt.tmp"
        save_checkpoint(
            agent,
            tmp_step_path,
            step,
            config=cfg,
            actor_updates=actor_updates,
            critic_warmup_completed=critic_warmup_completed,
            base_policy_sha256=marvin_replay_identity["base_policy_sha256"],
            action_contract=marvin_replay_identity["action_contract"],
        )
        os.replace(tmp_step_path, step_path)
        latest_tmp = checkpoint_dir / ".latest.pt.tmp"
        shutil.copy2(step_path, latest_tmp)
        os.replace(latest_tmp, checkpoint_dir / "latest.pt")

        keep_last = int(marvin_cfg.training_checkpoint_keep_last)
        if keep_last < 0:
            raise ValueError("marvin.training_checkpoint_keep_last must be non-negative")
        if keep_last > 0:
            numbered_checkpoints = sorted(checkpoint_dir.glob("step_*.pt"))
            for stale_checkpoint in numbered_checkpoints[:-keep_last]:
                stale_checkpoint.unlink()

        _publish_marvin_actor(step)
        print(f"Saved formal Marvin checkpoint step={step}: {step_path}")

    # ------------------------------------------------------------------
    # Critic warmup phase ----------------------------------------------
    # ------------------------------------------------------------------
    critic_warmup_ran = False
    print(
        "[Marvin startup] warmup decision: "
        f"requested={cfg.algo.critic_warmup_steps}, "
        f"checkpoint_completed={critic_warmup_completed}",
        flush=True,
    )
    if cfg.algo.critic_warmup_steps > 0 and not critic_warmup_completed:
        print(f"Critic warmup: running {cfg.algo.critic_warmup_steps} critic-only updates...")
        _run_critic_warmup(
            agent=agent,
            online_rb=online_rb,
            offline_rb=offline_rb,
            cfg=cfg,
            device=device,
            training_timer=training_timer,
            online_batch_size=online_batch_size,
            offline_batch_size=offline_batch_size,
        )
        critic_warmup_completed = True
        critic_warmup_ran = True
        print("Critic warmup completed.")

    # Persist the warmed critic and optimizer state before any hardware RPC.
    # A Bridge startup/reset failure must never force the warmup to be repeated.
    if cfg.environment_backend == "marvin" and (
        critic_warmup_ran or (initial_global_step == 0 and cfg.algo.critic_warmup_steps == 0)
    ):
        _save_marvin_state(global_step)
    else:
        # A resumed learner may intentionally change non-parameter actor
        # settings such as action_scale. Publish immediately so collectors do
        # not start from a stale deployment bundle while waiting for the next
        # periodic publish step.
        _publish_marvin_actor(global_step)
        print(
            f"Published startup Marvin actor step={global_step} "
            f"action_scale={cfg.agent.actor.action_scale}",
            flush=True,
        )
    print("[Marvin startup] warmup/checkpoint stage completed", flush=True)

    if learner_only:
        requested_updates = int(marvin_cfg.learner_updates_per_run)
        if requested_updates <= 0:
            raise ValueError("marvin.learner_updates_per_run must be positive")
        if marvin_cfg.actor_publish_interval <= 0:
            raise ValueError("marvin.actor_publish_interval must be positive")
        continuous = bool(marvin_cfg.async_grpc)
        learner_start_time = time.perf_counter()
        online_episodes_total = marvin_episode_count
        online_transitions_total = marvin_transition_count
        online_successes_total = marvin_success_count
        online_intervention_steps_total = marvin_intervention_steps
        online_intervention_episodes_total = marvin_intervention_episodes
        print(
            "[Marvin startup] entering learner loop: "
            f"global_step={global_step}, online_buffer={len(online_rb)}, "
            f"offline_buffer={len(offline_rb)}, grpc={continuous}",
            flush=True,
        )

        def _drain_grpc_transitions() -> int:
            nonlocal online_episodes_total, online_transitions_total
            nonlocal online_successes_total, marvin_cached_raw_transitions
            nonlocal online_intervention_steps_total, online_intervention_episodes_total
            added = 0
            if grpc_transitions is None:
                return added
            while True:
                try:
                    payload = grpc_transitions.get_nowait()
                except queue.Empty:
                    return added
                episode = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
                episode_id = int(episode.pop("_episode_id").item())
                if episode_id in known_marvin_episode_ids:
                    continue
                manifest_path = replay_root / f"episode_{episode_id:06d}.json"
                manifest = json.loads(manifest_path.read_text())
                episode_transitions = _add_marvin_episode(episode, online_rb)
                added += episode_transitions
                online_episodes_total += 1
                online_transitions_total += episode_transitions
                online_successes_total += int(torch.as_tensor(episode["success"]).any().item())
                intervention_steps = int(torch.count_nonzero(episode.get("intervened", torch.zeros(1))).item())
                online_intervention_steps_total += intervention_steps
                online_intervention_episodes_total += int(intervention_steps > 0)
                known_marvin_episode_ids.add(episode_id)
                marvin_cached_manifest_sha1[episode_id] = manifest_sha1(manifest)
                marvin_cached_raw_transitions += episode_transitions
                _persist_marvin_online_cache()
                print(
                    f"Persisted incremental online cache through episode={episode_id} "
                    f"(buffer_size={len(online_rb)})",
                    flush=True,
                )

        while global_step < cfg.algo.total_timesteps:
            round_updates = min(requested_updates, cfg.algo.total_timesteps - global_step)
            print(
                "Starting Marvin learner round: "
                f"updates={round_updates}, UTD={cfg.algo.num_updates_per_iteration}, "
                f"online_buffer={len(online_rb)}, offline_buffer={len(offline_rb)}, grpc={continuous}"
            )
            for _learner_step in range(round_updates):
                added = _drain_grpc_transitions()
                if added:
                    print(f"Received {added} online transitions over gRPC; online_buffer={len(online_rb)}")
                stddev = utils.schedule(cfg.algo.stddev_schedule, global_step)
                actor_update_cadence = (
                    cfg.algo.num_updates_per_iteration // cfg.algo.actor_updates_per_iteration
                )
                for update_index in range(cfg.algo.num_updates_per_iteration):
                    online_batch = online_rb.sample(online_batch_size).to(device, non_blocking=True)
                    if cfg.algo.offline_fraction > 0.0:
                        offline_batch = offline_rb.sample(offline_batch_size).to(device, non_blocking=True)
                        batch = torch.cat([online_batch, offline_batch], dim=0)
                    else:
                        batch = online_batch
                    update_actor = (update_index + 1) % actor_update_cadence == 0
                    if update_actor:
                        actor_updates += 1
                    metrics = agent.update(batch, stddev, update_actor, bc_batch=None, ref_agent=agent)
                global_step += 1
                if global_step % cfg.log_freq == 0:
                    elapsed = max(time.perf_counter() - learner_start_time, 1e-6)
                    log_dict = {
                        "training/global_step": global_step,
                        "training/actor_updates": actor_updates,
                        "training/actor_lr": agent.actor_opt.param_groups[0]["lr"],
                        "learner/steps_per_second": (global_step - initial_global_step) / elapsed,
                        "buffer/online_size": len(online_rb),
                        "buffer/offline_size": len(offline_rb),
                        "online/episodes_total": online_episodes_total,
                        "online/transitions_total": online_transitions_total,
                        "online/success_rate": (
                            online_successes_total / online_episodes_total
                            if online_episodes_total
                            else 0.0
                        ),
                        "online/intervention_steps_total": online_intervention_steps_total,
                        "online/intervention_episodes_total": online_intervention_episodes_total,
                        "online/intervention_episode_rate": (
                            online_intervention_episodes_total / online_episodes_total
                            if online_episodes_total
                            else 0.0
                        ),
                    }
                    log_dict.update({k: v for k, v in metrics.items() if not k.startswith("_")})
                    if "_actions" in metrics:
                        actions = metrics["_actions"]
                        log_dict["train/residual_l1_magnitude"] = torch.mean(torch.abs(actions)).item()
                        log_dict["train/residual_l2_magnitude"] = torch.mean(torch.square(actions)).item()
                        log_dict["train/residual_saturation_fraction"] = torch.mean(
                            (torch.abs(actions) >= 0.9 * cfg.agent.actor.action_scale).float()
                        ).item()
                    wandb.log(log_dict, step=global_step)
                    print(
                        f"[learner {global_step}] critic_loss={metrics['train/critic_loss']:.4f} "
                        f"actor_loss={metrics.get('train/actor_loss_base', float('nan')):.4f} "
                        f"actor_updates={actor_updates}"
                    )
                if global_step % int(marvin_cfg.actor_publish_interval) == 0:
                    _publish_marvin_actor(global_step)
                if cfg.checkpoint_interval > 0 and global_step % cfg.checkpoint_interval == 0:
                    _save_marvin_state(global_step)
                if marvin_cfg.learner_update_sleep_s > 0:
                    time.sleep(marvin_cfg.learner_update_sleep_s)
            if not continuous:
                if cfg.checkpoint_interval <= 0 or global_step % cfg.checkpoint_interval != 0:
                    _save_marvin_state(global_step)
                break
        print(f"Marvin learner finished: global_step={global_step}, actor_updates={actor_updates}")
        if grpc_shutdown is not None:
            grpc_shutdown.set()
            grpc_server.stop(10)
        return

    # Reset only after the potentially long critic warmup. This guarantees
    # the first command is computed from a fresh robot observation.
    obs, _ = env.reset()

    next_checkpoint_step = (
        ((global_step // cfg.checkpoint_interval) + 1) * cfg.checkpoint_interval
        if cfg.checkpoint_interval > 0
        else None
    )
    pending_replay_rows: list[dict] = []

    while global_step < cfg.algo.total_timesteps:
        iter_start = time.time()
        # ------------------------------------------------------------------
        # (1) Collect action + Environment step ---------------------------
        # ------------------------------------------------------------------
        with training_timer.time("env_step"):
            with torch.no_grad(), utils.eval_mode(agent):
                stddev = utils.schedule(cfg.algo.stddev_schedule, global_step)
                action = agent.act(obs, eval_mode=False, stddev=stddev, cpu=False)

            if cfg.algo.progressive_clipping_steps > 0:
                clip_factor = min(1.0, global_step / cfg.algo.progressive_clipping_steps)
                action = action * clip_factor

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated
            if cfg.environment_backend == "marvin":
                transition_next_obs = (
                    info["final_obs"][0]
                    if bool(done[0]) and info.get("final_obs", [None])[0] is not None
                    else next_obs
                )
                pending_replay_rows.append({
                    "state": obs["observation.state"][0].detach().cpu().numpy(),
                    "next_state": transition_next_obs["observation.state"][0].detach().cpu().numpy(),
                    "base_action": obs["observation.base_action"][0].detach().cpu().numpy(),
                    "next_base_action": transition_next_obs["observation.base_action"][0].detach().cpu().numpy(),
                    "residual_action": info["residual_action"][0].detach().cpu().numpy(),
                    "command_action": info["scaled_action"][0].detach().cpu().numpy(),
                    "applied_action": info["applied_action"][0].detach().cpu().numpy(),
                    "requested_action": np.asarray(info["requested_action"], dtype=np.float32),
                    "applied_command_action": np.asarray(info["applied_command_action"], dtype=np.float32),
                    "actual_action": np.asarray(info["actual_action"], dtype=np.float32),
                    "intervened": bool(info["intervened"]),
                    "raw_reward": float(info.get("raw_reward", reward[0])),
                    "reward": float(reward[0]),
                    "success": bool(float(reward[0]) > 0.8),
                    "terminated": bool(terminated[0]),
                    "truncated": bool(truncated[0]),
                    **{
                        f"image_{camera}": obs[f"observation.images.{camera}"][0].detach().cpu().numpy()
                        for camera in ("side_policy", "wrist_1buchong", "wrist_1", "wrist_2")
                    },
                    **{
                        f"next_image_{camera}": transition_next_obs[f"observation.images.{camera}"][0]
                        .detach().cpu().numpy()
                        for camera in ("side_policy", "wrist_1buchong", "wrist_1", "wrist_2")
                    },
                })
        if done.any():
            episode_count += done.float().sum().item()
            # Extract episode information from final_info
            final_info = info["final_info"]
            episode_steps = final_info["episode_steps"]
            episode_indices = final_info["_episode_steps"]

            # Calculate discounted episode return
            discount_factor = cfg.algo.gamma ** episode_steps[episode_indices]
            episode_rewards = reward.cpu().numpy()[episode_indices]
            episode_return = np.mean(discount_factor * episode_rewards)

            wandb.log(
                {
                    "training/episode_return": episode_return,
                    "training/episode_steps": episode_steps,
                    "training/episode_count": episode_count,
                },
                step=global_step,
            )

        # Add to online replay buffer --------------------------------------
        # Use the executed combined action returned by the environment
        combined_action = info["scaled_action"]
        _add_transitions_to_buffer(
            obs=obs,
            next_obs=next_obs,
            actions=combined_action,
            reward=reward,
            done=done,
            info=info,
            device=device,
            image_keys=image_keys,
            lowdim_keys=lowdim_keys,
            num_envs=cfg.num_envs,
            online_rb=online_rb,
        )

        obs = next_obs  # roll

        # ------------------------------------------------------------------
        # (3) Periodic evaluation ------------------------------------------
        # ------------------------------------------------------------------
        if cfg.environment_backend != "marvin" and global_step % cfg.eval_interval_every_steps == 0 and (cfg.eval_first or global_step > 0):
            with training_timer.time("evaluation"):
                eval_metrics = run_dexmg_evaluation(
                    env=eval_env,
                    agent=agent,
                    num_episodes=cfg.eval_num_episodes,
                    device=device,
                    global_step=global_step,
                    save_video=cfg.save_video,
                    save_q_plots=cfg.save_video,  # Enable Q-plots when video saving is enabled
                    run_name=run_name,
                    output_dir=outputs_dir,
                )

                # Handle model saving when success rate improves
                current_success_rate = eval_metrics["eval/success_rate"]
                if current_success_rate > best_eval_success_rate:
                    print(f"🎉 New best success rate: {current_success_rate:.4f} (prev: {best_eval_success_rate:.4f})")
                    best_eval_success_rate = current_success_rate

        global_step += cfg.num_envs

        # ------------------------------------------------------------------
        # (4) Updates -------------------------------------------------------
        # ------------------------------------------------------------------
        if global_step % cfg.algo.update_every_n_steps == 0 or global_step == cfg.num_envs:
            i = 0
            actor_update_cadence = cfg.algo.num_updates_per_iteration // cfg.algo.actor_updates_per_iteration
            # Normal training loop - critic is already warmed up
            while i < cfg.algo.num_updates_per_iteration:
                # --------------------------------------------------------------
                # Sample mixed online/offline batch
                # --------------------------------------------------------------
                with training_timer.time("batch_sampling"):
                    # Sample batches from replay buffers
                    online_batch = online_rb.sample(online_batch_size)
                    online_batch = online_batch.to(device, non_blocking=True)

                    if cfg.algo.offline_fraction > 0.0:
                        # Mixed online/offline training
                        offline_batch = offline_rb.sample(offline_batch_size)
                        offline_batch = offline_batch.to(device, non_blocking=True)
                        batch = torch.cat([online_batch, offline_batch], dim=0)
                    else:
                        # Online-only training
                        batch = online_batch

                # Update actor on the last iteration of each update cycle
                update_actor = (i + 1) % actor_update_cadence == 0

                # Apply actor learning rate warmup
                if update_actor:
                    if cfg.algo.actor_lr_warmup_steps > 0:
                        # Calculate current LR with linear warmup from 0 to target
                        warmup_progress = min(1.0, actor_updates / cfg.algo.actor_lr_warmup_steps)
                        current_lr = cfg.agent.actor_lr * warmup_progress
                        for param_group in agent.actor_opt.param_groups:
                            param_group["lr"] = current_lr

                    actor_updates += 1

                with training_timer.time("gradient_update"):
                    metrics = agent.update(batch, stddev, update_actor, bc_batch=None, ref_agent=agent)

                # Update priorities for prioritized experience replay
                if cfg.algo.sampling_strategy == "prioritized_replay" and "_td_errors" in metrics:
                    # Update priorities in the batch for priority updates
                    td_errors = metrics["_td_errors"]
                    batch["_priority"] = td_errors

                    if cfg.algo.offline_fraction > 0.0:
                        # Mixed online/offline training - update both buffers
                        online_batch_size_actual = int(cfg.algo.batch_size * (1 - cfg.algo.offline_fraction))

                        # Update online buffer priorities
                        if online_batch_size_actual > 0:
                            online_batch_subset = batch[:online_batch_size_actual]
                            online_rb.update_tensordict_priority(online_batch_subset)

                        # Update offline buffer priorities
                        if online_batch_size_actual < len(batch):
                            offline_batch_subset = batch[online_batch_size_actual:]
                            offline_rb.update_tensordict_priority(offline_batch_subset)
                    else:
                        # Online-only training - update only online buffer
                        online_rb.update_tensordict_priority(batch)

                metrics["data/batch_terminal_R"] = batch["next"]["reward"][~batch["nonterminal"]].mean()
                metrics["data/terminal_share"] = (~batch["nonterminal"]).float().mean()

                i += 1

        if cfg.environment_backend == "marvin" and bool(done[0]):
            arrays = {
                key: np.asarray([row[key] for row in pending_replay_rows])
                for key in pending_replay_rows[0]
            }
            manifest = {
                **marvin_replay_identity,
                "episode_id": next_replay_episode_id,
                "transitions": len(pending_replay_rows),
                "source": "residual_td3_learner",
                "residual_checkpoint": str(Path(marvin_cfg.residual_checkpoint).expanduser().resolve()),
                "residual_scale": marvin_cfg.residual_scale,
                "success": bool(any(row["success"] for row in pending_replay_rows)),
                "training_global_step_end": global_step,
            }
            atomic_write_episode(replay_root, next_replay_episode_id, arrays, manifest)
            print(
                f"Persisted online Marvin episode={next_replay_episode_id} "
                f"transitions={len(pending_replay_rows)}"
            )
            next_replay_episode_id += 1
            pending_replay_rows = []
            if next_checkpoint_step is not None and global_step >= next_checkpoint_step:
                _save_marvin_state(global_step)
                next_checkpoint_step = (
                    (global_step // cfg.checkpoint_interval) + 1
                ) * cfg.checkpoint_interval

        training_cum_time += time.time() - iter_start

        # ------------------------------------------------------------------
        # (6) Logging -------------------------------------------------------
        # ------------------------------------------------------------------
        if global_step % cfg.log_freq == 0:
            sps = int(global_step / training_cum_time) if training_cum_time > 0 else 0

            # Prepare base logging dict
            log_dict = {
                "training/SPS": sps,
                "training/global_step": global_step,
                "buffer/online_size": len(online_rb),
                "buffer/offline_size": len(offline_rb) if offline_rb else 0,
                "timing/training_total_time": time.time() - train_start_time,
                "timing/aggregate_steps_per_second": global_step / (time.time() - train_start_time),
                "training/actor_lr": agent.actor_opt.param_groups[0]["lr"],
            }

            # Add timing statistics
            timing_stats = training_timer.get_timing_stats()
            log_dict.update(timing_stats)

            # Add metrics, filtering out internal data
            filtered_metrics = {k: v for k, v in metrics.items() if not k.startswith("_")}
            log_dict.update(filtered_metrics)

            # Compute residual action statistics only when logging
            if "_actions" in metrics:
                actions = metrics["_actions"]
                # Compute L1/L2 magnitudes (only during logging to save computation)
                residual_l1_magnitude = torch.mean(torch.abs(actions)).item()
                residual_l2_magnitude = torch.mean(torch.square(actions)).item()

                log_dict["train/residual_l1_magnitude"] = residual_l1_magnitude
                log_dict["train/residual_l2_magnitude"] = residual_l2_magnitude
                log_dict["histograms/residual_actions"] = wandb.Histogram(actions.numpy().reshape(-1))
            else:
                residual_l1_magnitude = None
                residual_l2_magnitude = None

            # Add Q values histogram when available
            if "_target_q" in metrics:
                target_q = metrics["_target_q"]
                log_dict["histograms/critic_qt"] = wandb.Histogram(target_q.numpy().reshape(-1))

            if cfg.algo.progressive_clipping_steps > 0:
                log_dict["training/progressive_clipping_factor"] = clip_factor

            wandb.log(log_dict, step=global_step)

            # Enhanced print statement with residual action magnitudes, gradient norms, and actor LR
            current_actor_lr = agent.actor_opt.param_groups[0]["lr"]

            if "train/actor_loss_base" in metrics:
                actor_loss_str = f"actor_loss_base={metrics['train/actor_loss_base']:.4f}"
                print_msg = (
                    f"[{global_step}] {actor_loss_str} "
                    f"critic_loss={metrics['train/critic_loss']:.4f} "
                    f"actor_lr={current_actor_lr:.2e}"
                )
            else:
                # During critic warmup, actor might not be updated
                print_msg = (
                    f"[{global_step}] critic_loss={metrics['train/critic_loss']:.4f} "
                    f"actor_lr={current_actor_lr:.2e} (actor not updated)"
                )
            if residual_l1_magnitude is not None and residual_l2_magnitude is not None:
                print_msg += f" residual_l1={residual_l1_magnitude:.4f} residual_l2={residual_l2_magnitude:.4f}"

            # Add gradient norms to print statement
            if "train/actor_grad_norm" in metrics:
                print_msg += f" actor_grad_norm={metrics['train/actor_grad_norm']:.4f}"

            # Add L2 penalty if active
            if "train/actor_l2_penalty" in metrics:
                print_msg += f" l2_penalty={metrics['train/actor_l2_penalty']:.4f}"

            # Add timing percentages to print statement
            if timing_stats:
                env_pct = timing_stats.get("timing/env_step_percentage", 0)
                grad_pct = timing_stats.get("timing/gradient_update_percentage", 0)
                batch_pct = timing_stats.get("timing/batch_sampling_percentage", 0)
                eval_pct = timing_stats.get("timing/evaluation_percentage", 0)
                print_msg += (
                    f" | Time%: env={env_pct:.1f} grad={grad_pct:.1f} batch={batch_pct:.1f} eval={eval_pct:.1f}"
                )

            print(print_msg)

    if cfg.environment_backend == "marvin":
        _save_marvin_state(global_step)

    print(f"Training finished in {time.time() - train_start_time:.2f} seconds.")

    # Persist an actor+vision-encoder bundle for the standalone Marvin online
    # collector.  This is opt-in so published benchmark runs are unchanged.
    if cfg.environment_backend == "marvin" and cfg.marvin.get("residual_checkpoint") and checkpoint_dir is None:
        checkpoint_path = Path(str(cfg.marvin["residual_checkpoint"])).expanduser()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": agent.actor.state_dict(),
                "encoders": agent.encoders.state_dict(),
                "obs_shape": (img_c, img_h, img_w),
                "prop_shape": (lowdim_dim,),
                "action_dim": action_dim,
                "rl_cameras": list(image_keys),
                "actor_action_scale": float(cfg.agent.actor.action_scale),
            },
            checkpoint_path,
        )
        print(f"Saved Marvin residual actor checkpoint to {checkpoint_path}")

    # Clean up entire run directory after successful completion (videos/logs are saved to wandb)
    if run_cache_dir.exists() and cfg.environment_backend != "marvin":
        print(f"Cleaning up run directory: {run_cache_dir}")
        shutil.rmtree(run_cache_dir)
        print("Run directory cleaned up successfully.")


# -----------------------------------------------------------------------------
# Hydra entry point -----------------------------------------------------------
# -----------------------------------------------------------------------------
@hydra.main(version_base=None, config_name="residual_td3_dexmg_config")
def hydra_entry(cfg: ResidualTD3DexmgConfig):
    cfg_conf = OmegaConf.structured(cfg)
    main(cfg_conf)


if __name__ == "__main__":
    hydra_entry()
