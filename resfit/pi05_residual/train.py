"""Learner-only Pi0.5 residual TD3 process; never connects to the robot."""

from __future__ import annotations

import argparse
import io
import json
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer

from resfit.rl_finetuning.marvin.grpc_transport import start_learner_server
from resfit.rl_finetuning.marvin.replay import atomic_write_episode, inspect_replay_directory, validate_manifest
from resfit.rl_finetuning.utils.checkpoint import save_checkpoint
from resfit.rl_finetuning.utils.rb_transforms import MultiStepTransform

from .config import Pi05ResidualConfig
from .replay import PythonPrioritizedSampler
from .runtime import ACTOR_CONTRACT, actor_bundle, atomic_save_bundle, make_agent, replay_identity


TRAINING_CONTRACT = "pi05_residual_10hz_5deg_frozen_resnet_v8"


def _learner_step_budget(online_transitions: int, steps_per_transition: int) -> int:
    return online_transitions * steps_per_transition


def _transition(data, row: int, cameras: tuple[str, ...], *, force_done: bool = False) -> TensorDict:
    obs = {
        "observation.state": torch.as_tensor(data["state"][row], dtype=torch.float32),
        "observation.base_action": torch.as_tensor(data["base_action"][row], dtype=torch.float32),
    }
    nxt = {
        "observation.state": torch.as_tensor(data["next_state"][row], dtype=torch.float32),
        "observation.base_action": torch.as_tensor(data["next_base_action"][row], dtype=torch.float32),
    }
    for camera in cameras:
        short = camera.replace("observation.images.", "")
        obs[camera] = torch.as_tensor(data[f"image_{short}"][row])
        nxt[camera] = torch.as_tensor(data[f"next_image_{short}"][row])
    done = bool(force_done or data["terminated"][row] or data["truncated"][row])
    transition = TensorDict({
        "obs": TensorDict(obs, batch_size=[]),
        "action": torch.as_tensor(data["action"][row], dtype=torch.float32),
        "next": TensorDict({
            "obs": TensorDict(nxt, batch_size=[]),
            "reward": torch.tensor(float(data["reward"][row])),
            "done": torch.tensor(done),
        }, batch_size=[]),
        "_priority": torch.tensor(10.0),
    }, batch_size=[])
    if "executed_residual_action" in data:
        transition["executed_residual_action"] = torch.as_tensor(
            data["executed_residual_action"][row], dtype=torch.float32
        )
    if "executed_absolute_action" in data:
        transition["executed_absolute_action"] = torch.as_tensor(
            data["executed_absolute_action"][row], dtype=torch.float32
        )
    return transition.unsqueeze(0)


def _add_episode(data, replay, cameras: tuple[str, ...], expert_replay=None) -> int:
    for row in range(len(data["reward"])):
        replay.add(_transition(data, row, cameras))
        if expert_replay is not None and "intervened" in data and bool(data["intervened"][row]):
            # The expert stream may contain gaps between takeover runs. Do not
            # let an n-step target cross a policy-controlled interval.
            run_ends = row + 1 == len(data["reward"]) or not bool(data["intervened"][row + 1])
            expert_replay.add(_transition(data, row, cameras, force_done=run_ends))
    return len(data["reward"])


def _ingest_new_replay_files(
    replay_dir: str, identity: dict, known_episode_ids: set[int], replay,
    cameras: tuple[str, ...], expert_replay=None,
) -> int:
    """Read committed collector episodes from the shared replay directory."""
    added = 0
    for manifest_path in sorted(Path(replay_dir).glob("episode_*.json")):
        episode_id = int(manifest_path.stem.removeprefix("episode_"))
        if episode_id in known_episode_ids:
            continue
        episode_path = manifest_path.with_suffix(".npz")
        if not episode_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        validate_manifest(manifest, identity, manifest_path)
        if int(manifest["episode_id"]) != episode_id:
            raise ValueError(f"Replay filename/id mismatch: {manifest_path}")
        if manifest.get("source") not in {"pi05_residual_rl_episode", "pi05_residual_grpc"}:
            known_episode_ids.add(episode_id)
            continue
        with np.load(episode_path, allow_pickle=False) as data:
            if len(data["reward"]) != int(manifest["transitions"]):
                raise ValueError(f"Replay length mismatch: {episode_path}")
            if expert_replay is None:
                added += _add_episode(data, replay, cameras)
            else:
                added += _add_episode(data, replay, cameras, expert_replay)
        known_episode_ids.add(episode_id)
        print(
            f"loaded replay episode={episode_id}; online_buffer={len(replay)} "
            f"expert_buffer={len(expert_replay) if expert_replay is not None else 0}"
        )
    return added


def _make_replay(args, batch_size: int):
    common = dict(
        storage=LazyTensorStorage(args.replay_capacity, device="cpu"),
        transform=MultiStepTransform(n_steps=args.n_step, gamma=args.gamma),
        batch_size=batch_size,
        pin_memory=True,
    )
    if args.sampling_strategy == "prioritized_replay":
        try:
            return TensorDictPrioritizedReplayBuffer(
                alpha=args.priority_alpha,
                beta=args.priority_beta,
                eps=1e-6,
                priority_key="_priority",
                **common,
            )
        except NameError:
            return TensorDictReplayBuffer(
                sampler=PythonPrioritizedSampler(
                    args.replay_capacity,
                    args.priority_alpha,
                    args.priority_beta,
                    1e-6,
                ),
                priority_key="_priority",
                **common,
            )
    return TensorDictReplayBuffer(**common)


def _load_checkpoint(agent, path: Path) -> tuple[int, bool, int]:
    if not path.is_file():
        return 0, False, 0
    checkpoint = torch.load(path, map_location=agent.cfg.device, weights_only=False)
    if checkpoint.get("training_contract") != TRAINING_CONTRACT:
        raise ValueError(f"incompatible learner checkpoint: {path}; use a new --checkpoint path")
    agent.load_state_dict(checkpoint["agent_state_dict"])
    optimizers = checkpoint.get("optimizer_state_dict", {})
    for name in ("actor_opt", "critic_opt", "encoder_opt"):
        if name in optimizers:
            getattr(agent, name).load_state_dict(optimizers[name])
    for group in agent.actor_opt.param_groups:
        group["lr"] = agent.cfg.actor_lr
    for optimizer in (agent.critic_opt, agent.encoder_opt):
        for group in optimizer.param_groups:
            group["lr"] = agent.cfg.critic_lr
    completed = bool(checkpoint.get("critic_warmup_completed", False))
    if checkpoint.get("schedule_version") == 2:
        return (
            int(checkpoint.get("global_step", 0)),
            completed,
            int(checkpoint.get("critic_warmup_updates", 0)),
        )
    # Legacy Pi0.5 checkpoints counted each gradient update as global_step.
    # Preserve unfinished critic warmup, then start the corrected UTD counter.
    legacy_step = int(checkpoint.get("global_step", 0))
    return 0, completed, 0 if completed else legacy_step


def run(args) -> None:
    if not 0 < args.actor_updates_per_round <= args.updates_per_round:
        raise ValueError("actor_updates_per_round must be in [1, updates_per_round]")
    if args.updates_per_round % args.actor_updates_per_round:
        raise ValueError("updates_per_round must be divisible by actor_updates_per_round")
    if not 0.0 <= args.offline_fraction < 1.0:
        raise ValueError("--offline-fraction must be in [0, 1)")
    if args.max_learner_steps_per_online_transition < 1:
        raise ValueError("--max-learner-steps-per-online-transition must be >= 1")
    if args.offline_npz and args.offline_replay_dir:
        raise ValueError("use only one of --offline-npz and --offline-replay-dir")
    if args.offline_fraction == 0.0 and (args.offline_npz or args.offline_replay_dir):
        raise ValueError("an offline replay source requires positive --offline-fraction")
    if args.offline_npz and not Path(args.offline_npz).is_file():
        raise FileNotFoundError(args.offline_npz)
    cameras = tuple(args.cameras.split(","))
    cfg = Pi05ResidualConfig(
        pi05_checkpoint=args.pi05_checkpoint,
        pi05_config=args.pi05_config,
        task=args.task,
        cameras=cameras,
        chunk_size=args.chunk_size,
        rl_chunk_size=args.rl_chunk_size,
        residual_joint_limit_rad=float(np.deg2rad(args.residual_joint_deg)),
        residual_ema_alpha=args.residual_ema_alpha,
        residual_slew_limit_rad=float(np.deg2rad(args.residual_slew_limit_deg)),
        action_lowpass_alpha=args.action_lowpass_alpha,
        bridge_joint_step_limit_rad=args.bridge_joint_step_limit_rad,
        device=args.device,
    )
    identity = replay_identity(cfg)
    bundle_path = Path(args.residual_checkpoint).expanduser().resolve()
    if bundle_path.is_file():
        existing_bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        if existing_bundle.get("identity") != identity or existing_bundle.get("actor_contract") != ACTOR_CONTRACT:
            raise ValueError(f"incompatible actor bundle: {bundle_path}; use a new --residual-checkpoint path")
    device = args.device if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and args.gpu_memory_fraction < 1.0:
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    agent = make_agent(cfg, device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    updates, critic_warmup_completed, critic_warmup_updates = _load_checkpoint(
        agent, checkpoint_path
    )

    online_n = max(1, int(args.batch_size * (1.0 - args.offline_fraction)))
    offline_n = args.batch_size - online_n
    online_replay = _make_replay(args, online_n)
    offline_replay = _make_replay(args, max(1, offline_n))
    known_episode_ids: set[int] = set()
    online_transitions_seen = 0
    episodes, _, _ = inspect_replay_directory(args.replay_dir, identity)
    for path, manifest in episodes:
        if manifest.get("source") not in {
            "pi05_residual_rl_episode",
            "pi05_residual_grpc",
        }:
            print(f"skipping pre-B/A/C replay episode: {path.name}")
            continue
        with np.load(path, allow_pickle=False) as data:
            _add_episode(data, online_replay, cameras, offline_replay)
            online_transitions_seen += len(data["reward"])
        known_episode_ids.add(int(manifest["episode_id"]))
    if args.offline_npz:
        with np.load(args.offline_npz, allow_pickle=False) as data:
            _add_episode(data, offline_replay, cameras)
    elif args.offline_replay_dir:
        offline_episodes, _, _ = inspect_replay_directory(
            args.offline_replay_dir, identity
        )
        for path, manifest in offline_episodes:
            if manifest.get("source") not in {
                "pi05_residual_rl_episode",
                "pi05_residual_grpc",
                "pi05_residual_offline_expert",
            }:
                print(f"skipping incompatible offline episode: {path.name}")
                continue
            with np.load(path, allow_pickle=False) as data:
                _add_episode(data, offline_replay, cameras)

    shutdown = threading.Event()
    parameter_queue = queue.Queue(maxsize=1)
    transition_queue = queue.Queue(maxsize=8)
    server = None
    if args.async_grpc:
        server = start_learner_server(
            host=args.grpc_host,
            port=args.grpc_port,
            parameters_queue=parameter_queue,
            transition_queue=transition_queue,
            shutdown_event=shutdown,
            push_frequency_s=args.grpc_parameter_push_s,
        )

    def publish() -> None:
        bundle = actor_bundle(agent, cfg, updates)
        atomic_save_bundle(bundle, args.residual_checkpoint)
        if args.async_grpc:
            buffer = io.BytesIO()
            torch.save(bundle, buffer)
            try:
                while True:
                    parameter_queue.get_nowait()
            except queue.Empty:
                pass
            parameter_queue.put_nowait(buffer.getvalue())

    def save() -> None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
        save_checkpoint(
            agent,
            temporary,
            updates,
            config=vars(args),
            critic_warmup_completed=critic_warmup_completed,
            critic_warmup_updates=critic_warmup_updates,
            schedule_version=2,
            training_contract=TRAINING_CONTRACT,
        )
        os.replace(temporary, checkpoint_path)

    def gradient_update(update_actor: bool):
        online_batch = online_replay.sample(online_n)
        parts = [online_batch]
        if offline_n:
            parts.append(offline_replay.sample(offline_n))
        batch = torch.cat(parts, dim=0).to(device, non_blocking=True)
        metrics = agent.update(
            batch,
            stddev=0.0,
            update_actor=update_actor,
        )
        if args.sampling_strategy == "prioritized_replay":
            batch["_priority"] = metrics["_td_errors"].to(batch.device)
            online_replay.update_tensordict_priority(batch[:online_n])
            if offline_n:
                offline_replay.update_tensordict_priority(batch[online_n:])
        return metrics

    publish()
    last_wait_message = 0.0
    last_replay_scan = 0.0
    if critic_warmup_updates >= args.critic_warmup_steps:
        critic_warmup_completed = True
    try:
        while args.total_updates <= 0 or updates < args.total_updates:
            if time.monotonic() - last_replay_scan >= 1.0:
                online_transitions_seen += _ingest_new_replay_files(
                    args.replay_dir, identity, known_episode_ids, online_replay,
                    cameras, offline_replay,
                )
                last_replay_scan = time.monotonic()
            while True:
                try:
                    payload = transition_queue.get_nowait()
                except queue.Empty:
                    break
                episode = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
                episode_id = int(episode.pop("_episode_id").item())
                payload_identity = episode.pop("_identity")
                if payload_identity != identity:
                    raise ValueError("collector replay contract does not match learner")
                if episode_id in known_episode_ids:
                    continue
                arrays = {key: value.numpy() for key, value in episode.items()}
                manifest = {
                    **identity,
                    "episode_id": episode_id,
                    "transitions": len(arrays["reward"]),
                    "source": "pi05_residual_grpc",
                    "success": bool(np.any(arrays["reward"] > 0.5)),
                }
                path = Path(args.replay_dir) / f"episode_{episode_id:06d}.npz"
                if not path.exists():
                    atomic_write_episode(args.replay_dir, episode_id, arrays, manifest)
                _add_episode(arrays, online_replay, cameras, offline_replay)
                online_transitions_seen += len(arrays["reward"])
                known_episode_ids.add(episode_id)
                print(f"received episode={episode_id}; online_buffer={len(online_replay)}")

            required = max(args.learning_starts, online_n)
            if len(online_replay) < required or (offline_n and len(offline_replay) < offline_n):
                if time.monotonic() - last_wait_message > 5:
                    print(
                        f"waiting for replay: online={len(online_replay)}/{required} "
                        f"expert={len(offline_replay)}/{offline_n}"
                    )
                    last_wait_message = time.monotonic()
                time.sleep(0.1)
                continue

            if not critic_warmup_completed:
                started = time.perf_counter()
                metrics = gradient_update(update_actor=False)
                critic_warmup_updates += 1
                if critic_warmup_updates % args.log_every == 0:
                    print(
                        f"critic_warmup={critic_warmup_updates}/{args.critic_warmup_steps} "
                        f"critic_loss={metrics['train/critic_loss']:.5f} "
                        f"update_s={time.perf_counter() - started:.3f}"
                    )
                if critic_warmup_updates >= args.critic_warmup_steps:
                    critic_warmup_completed = True
                    print(f"critic warmup completed after {critic_warmup_updates} updates")
                    publish()
                    save()
                continue

            learner_step_budget = _learner_step_budget(
                online_transitions_seen,
                args.max_learner_steps_per_online_transition,
            )
            if updates >= learner_step_budget:
                if time.monotonic() - last_wait_message > 5:
                    print(
                        "waiting for new online replay: "
                        f"learner_step={updates} budget={learner_step_budget} "
                        f"online_transitions={online_transitions_seen}"
                    )
                    last_wait_message = time.monotonic()
                time.sleep(0.1)
                continue

            round_started = time.perf_counter()
            actor_cadence = args.updates_per_round // args.actor_updates_per_round
            for round_index in range(args.updates_per_round):
                update_actor = (round_index + 1) % actor_cadence == 0
                metrics = gradient_update(update_actor=update_actor)
            updates += 1
            round_seconds = time.perf_counter() - round_started
            if updates % args.log_every == 0:
                print(
                    f"learner_step={updates} utd={args.updates_per_round} "
                    f"online={len(online_replay)} offline={len(offline_replay)} "
                    f"critic_loss={metrics['train/critic_loss']:.5f} "
                    f"round_s={round_seconds:.3f}"
                )
            if updates % args.publish_every == 0:
                publish()
            if updates % args.checkpoint_every == 0:
                save()
            if args.learner_sleep > 0:
                time.sleep(max(0.0, args.learner_sleep - (time.perf_counter() - round_started)))
    except KeyboardInterrupt:
        print("learner interrupted; saving latest state")
    finally:
        publish()
        save()
        shutdown.set()
        if server is not None:
            server.stop(10)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pi05-checkpoint", required=True)
    p.add_argument("--pi05-config", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--replay-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--residual-checkpoint", required=True)
    p.add_argument("--cameras", default="observation.images.front,observation.images.left_wrist,observation.images.right_wrist")
    p.add_argument("--chunk-size", type=int, default=30)
    p.add_argument("--rl-chunk-size", type=int, default=10)
    p.add_argument("--residual-joint-deg", type=float, default=5.0)
    p.add_argument("--residual-ema-alpha", type=float, default=0.35)
    p.add_argument("--residual-slew-limit-deg", type=float, default=1.0)
    p.add_argument("--action-lowpass-alpha", type=float, default=1.0)
    p.add_argument("--bridge-joint-step-limit-rad", type=float, default=0.08)
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpu-memory-fraction", type=float, default=0.80)
    p.add_argument("--total-updates", type=int, default=0, help="0 runs continuously")
    p.add_argument("--learning-starts", type=int, default=1000)
    p.add_argument("--critic-warmup-steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--replay-capacity", type=int, default=200000)
    p.add_argument("--offline-npz", default="")
    p.add_argument("--offline-replay-dir", default="")
    p.add_argument("--offline-fraction", type=float, default=0.0)
    p.add_argument("--n-step", type=int, default=3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--updates-per-round", type=int, default=4)
    p.add_argument("--actor-updates-per-round", type=int, default=1)
    p.add_argument("--learner-sleep", type=float, default=0.01)
    p.add_argument(
        "--max-learner-steps-per-online-transition",
        type=int,
        default=1,
        help="cap formal UTD rounds per collected online transition",
    )
    p.add_argument("--sampling-strategy", choices=("uniform", "prioritized_replay"), default="prioritized_replay")
    p.add_argument("--priority-alpha", type=float, default=0.6)
    p.add_argument("--priority-beta", type=float, default=0.4)
    p.add_argument("--publish-every", type=int, default=50)
    p.add_argument("--checkpoint-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--async-grpc", action="store_true")
    p.add_argument("--grpc-host", default="127.0.0.1")
    p.add_argument("--grpc-port", type=int, default=50051)
    p.add_argument("--grpc-parameter-push-s", type=float, default=1.0)
    run(p.parse_args())


if __name__ == "__main__":
    main()
