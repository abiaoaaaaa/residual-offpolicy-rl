"""Collect a safe Marvin replay with a frozen ACT base and zero/small residual.

The collector deliberately does not perform learner updates or random warm-up.
The resulting NPZ episodes are an auditable interchange format; the learner
must convert them into its TensorDict replay after checking the manifest.
"""

from __future__ import annotations

import argparse
import io
import queue
import threading
from pathlib import Path

import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from resfit.rl_finetuning.config.rlpd import QAgentConfig
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.marvin.env import MarvinResidualConfig, MarvinResidualVecEnv
from resfit.rl_finetuning.marvin.replay import (
    atomic_write_episode,
    inspect_replay_directory,
    replay_identity,
    usable_nstep_transitions,
)
from resfit.rl_finetuning.utils.normalization import ActionScaler, StateStandardizer
from resfit.rl_finetuning.marvin.grpc_transport import (
    actor_stub,
    receive_parameters,
    send_transition_batches,
)


def _agent_from_bundle(bundle: dict, device: str) -> tuple[QAgent, float]:
    cameras = list(bundle["rl_cameras"])
    cfg = QAgentConfig(device=device)
    cfg.actor.action_scale = float(bundle.get("actor_action_scale", cfg.actor.action_scale))
    agent = QAgent(
        obs_shape=tuple(bundle["obs_shape"]),
        prop_shape=tuple(bundle["prop_shape"]),
        action_dim=int(bundle["action_dim"]),
        rl_cameras=cameras,
        cfg=cfg,
        residual_actor=True,
    ).to(device)
    agent.actor.load_state_dict(bundle["actor"])
    agent.encoders.load_state_dict(bundle["encoders"])
    agent.eval()
    agent.actor.eval()
    agent.critic.eval()
    return agent, cfg.actor.action_scale


def load_residual_agent(checkpoint: str, device: str) -> tuple[QAgent, float]:
    """Load the actor bundle emitted by train_residual_td3.py."""
    return _agent_from_bundle(torch.load(checkpoint, map_location=device, weights_only=False), device)


def apply_residual_bundle(agent: QAgent | None, bundle: dict, device: str) -> tuple[QAgent, float, int]:
    if agent is None:
        agent, scale = _agent_from_bundle(bundle, device)
    else:
        if list(bundle["rl_cameras"]) != list(agent.rl_cameras):
            raise ValueError("Received residual checkpoint has different camera keys")
        agent.actor.load_state_dict(bundle["actor"])
        agent.encoders.load_state_dict(bundle["encoders"])
        scale = float(bundle["actor_action_scale"])
    return agent, scale, int(bundle.get("global_step", -1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-policy-checkpoint", required=True)
    p.add_argument("--base-policy-chunk-size", type=int, default=20)
    p.add_argument("--base-policy-n-action-steps", type=int, default=5)
    p.add_argument("--offline-dataset-root", required=True)
    p.add_argument("--offline-dataset-name", default="marvin_usb_rl")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--marvin-root", default="/home/xlb/code/强化微调ACT")
    p.add_argument("--device", default="cuda")
    p.add_argument("--task", default="Insert the USB plug")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument(
        "--min-transitions",
        type=int,
        default=0,
        help="Continue until the directory has this many usable transitions after n-step conversion",
    )
    p.add_argument("--n-step", type=int, default=3)
    p.add_argument("--resume", action="store_true", help="Validate and append to an existing replay directory")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument(
        "--residual-scale",
        type=float,
        default=None,
        help="Optional hardware cap in normalized residual units; omit for the original ResFiT path",
    )
    p.add_argument("--actor-action-scale", type=float, default=0.1)
    p.add_argument("--min-action-range", type=float, default=0.1)
    p.add_argument("--min-state-std", type=float, default=0.1)
    p.add_argument("--residual-checkpoint", default="", help="Actor bundle saved by the residual learner")
    p.add_argument("--contact-start-step", type=int, default=0)
    p.add_argument("--enable-gripper-residual", action="store_true")
    p.add_argument("--allow-real-robot", action="store_true")
    p.add_argument("--async-grpc", action="store_true")
    p.add_argument("--learner-host", default="127.0.0.1")
    p.add_argument("--learner-port", type=int, default=50051)
    p.add_argument("--gpu-memory-fraction", type=float, default=0.18)
    p.add_argument("--continuous", action="store_true")
    args = p.parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.min_transitions < 0:
        raise ValueError("--min-transitions must be non-negative")
    if args.n_step <= 0:
        raise ValueError("--n-step must be positive")
    if not 1 <= args.base_policy_n_action_steps <= args.base_policy_chunk_size:
        raise ValueError("--base-policy-n-action-steps must be in [1, --base-policy-chunk-size]")
    if not 0.0 < args.gpu_memory_fraction < 1.0:
        raise ValueError("--gpu-memory-fraction must be in (0, 1)")
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
        print(f"Marvin actor CUDA memory fraction: {args.gpu_memory_fraction:.0%}")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    residual_agent = None
    actor_action_scale = args.actor_action_scale
    if args.residual_checkpoint:
        residual_agent, actor_action_scale = load_residual_agent(args.residual_checkpoint, args.device)

    grpc_shutdown = threading.Event()
    grpc_parameters = queue.Queue(maxsize=1)
    grpc_transitions = queue.Queue(maxsize=2)
    grpc_channel = None
    if args.async_grpc:
        stub, grpc_channel = actor_stub(args.learner_host, args.learner_port)
        threading.Thread(
            target=receive_parameters,
            args=(stub, grpc_parameters, grpc_shutdown),
            daemon=True,
        ).start()
        threading.Thread(
            target=send_transition_batches,
            args=(stub, grpc_transitions, grpc_shutdown),
            daemon=True,
        ).start()

    dataset = LeRobotDataset(args.offline_dataset_name, root=args.offline_dataset_root)
    action_scaler = ActionScaler.from_dataset_stats(
        dataset.meta.stats["action"],
        action_scale=actor_action_scale,
        min_range_per_dim=args.min_action_range,
        absolute_min=-1.0,
        absolute_max=1.0,
        device="cpu",
    )
    state_standardizer = StateStandardizer.from_dataset_stats(
        dataset.meta.stats["observation.state"],
        min_std=args.min_state_std,
        device="cpu",
    )
    cfg = MarvinResidualConfig(
        base_policy_checkpoint=args.base_policy_checkpoint,
        base_policy_chunk_size=args.base_policy_chunk_size,
        base_policy_n_action_steps=args.base_policy_n_action_steps,
        marvin_root=args.marvin_root,
        device=args.device,
        task=args.task,
        host=args.host,
        port=args.port,
        max_steps=args.max_steps,
        residual_scale=args.residual_scale,
        contact_start_step=args.contact_start_step,
        enable_gripper_residual=args.enable_gripper_residual,
        allow_real_robot=args.allow_real_robot,
    )
    env = MarvinResidualVecEnv(cfg, action_scaler=action_scaler, state_standardizer=state_standardizer)
    identity = replay_identity(
        base_policy_checkpoint=args.base_policy_checkpoint,
        base_policy_chunk_size=args.base_policy_chunk_size,
        base_policy_n_action_steps=args.base_policy_n_action_steps,
        offline_dataset_root=args.offline_dataset_root,
        offline_dataset_name=args.offline_dataset_name,
        actor_action_scale=actor_action_scale,
        min_action_range=args.min_action_range,
        min_state_std=args.min_state_std,
    )
    existing_episodes, total_transitions, next_episode_id = inspect_replay_directory(output, identity)
    usable_transitions = usable_nstep_transitions(existing_episodes, args.n_step)
    if existing_episodes and not args.resume:
        raise FileExistsError(f"Replay directory already contains episodes; pass --resume to append safely: {output}")
    print(
        f"Replay collection start: existing_episodes={len(existing_episodes)}, "
        f"existing_transitions={total_transitions}, usable_nstep_transitions={usable_transitions}, "
        f"next_episode_id={next_episode_id}"
    )
    try:
        collected_episodes = 0
        prefetched_obs = None
        while args.continuous or collected_episodes < args.episodes or usable_transitions < args.min_transitions:
            if args.async_grpc:
                latest = None
                try:
                    while True:
                        latest = grpc_parameters.get_nowait()
                except queue.Empty:
                    pass
                if latest is not None:
                    bundle = torch.load(io.BytesIO(latest), map_location=args.device, weights_only=True)
                    if bundle.get("base_policy_sha256") != identity["base_policy_sha256"]:
                        raise ValueError("Learner parameters use a different ACT base policy")
                    if bundle.get("action_contract") != identity["action_contract"]:
                        raise ValueError("Learner parameters use a different Marvin action contract")
                    residual_agent, received_scale, received_step = apply_residual_bundle(
                        residual_agent, bundle, args.device
                    )
                    if not np.isclose(received_scale, actor_action_scale):
                        raise ValueError(
                            f"Learner actor_action_scale changed: {actor_action_scale} -> {received_scale}"
                        )
                    print(f"Applied learner parameters at episode boundary: global_step={received_step}")
            episode = next_episode_id
            if prefetched_obs is None:
                obs, _ = env.reset()
            else:
                obs = prefetched_obs
                prefetched_obs = None
            rows = []
            episode_success = False
            for _step in range(args.max_steps):
                command_obs = {k: v.clone() for k, v in obs.items()}
                if residual_agent is None:
                    residual_action = torch.zeros((1, 5), dtype=torch.float32)
                else:
                    residual_obs = {k: v.to(args.device) for k, v in obs.items()}
                    with torch.inference_mode():
                        residual_action = residual_agent.act(residual_obs, eval_mode=True, stddev=0.0, cpu=False)
                next_obs, reward, terminated, truncated, info = env.step(residual_action)
                done = bool(terminated[0] or truncated[0])
                # The env auto-resets after a terminal step.  Persist the
                # terminal observation in the replay, not the reset state's
                # observation, otherwise the Bellman transition crosses
                # episode boundaries.
                transition_next_obs = info["final_obs"][0] if done and info.get("final_obs", [None])[0] is not None else next_obs
                episode_success = episode_success or bool(float(reward[0]) > 0.8)
                rows.append({
                    "state": obs["observation.state"].numpy()[0],
                    "next_state": transition_next_obs["observation.state"].numpy()[0],
                    "base_action": obs["observation.base_action"].numpy()[0],
                    "residual_action": info["residual_action"].numpy()[0],
                    "command_action": info["scaled_action"].numpy()[0],
                    "next_base_action": transition_next_obs["observation.base_action"].numpy()[0],
                    "actual_action": np.asarray(info["actual_action"], dtype=np.float32),
                    "raw_reward": float(info.get("raw_reward", reward[0])),
                    "reward": float(reward[0]), "success": bool(float(reward[0]) > 0.8),
                    "terminated": bool(terminated[0]), "truncated": bool(truncated[0]),
                    **{f"image_{key}": command_obs[f"observation.images.{key}"].numpy()[0] for key in ("side_policy", "wrist_1buchong", "wrist_1", "wrist_2")},
                    **{f"next_image_{key}": transition_next_obs[f"observation.images.{key}"].numpy()[0] for key in ("side_policy", "wrist_1buchong", "wrist_1", "wrist_2")},
                })
                obs = next_obs
                if terminated[0] or truncated[0]:
                    # MarvinResidualVecEnv already auto-resets on a terminal
                    # transition. Reuse that observation for the next episode
                    # instead of issuing a second hardware reset.
                    prefetched_obs = next_obs
                    break
            arrays = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
            manifest = {
                **identity,
                "episode_id": episode,
                "transitions": len(rows),
                "source": "zero_residual_collector" if residual_agent is None else "residual_collector",
                "residual_checkpoint": str(Path(args.residual_checkpoint).resolve()) if args.residual_checkpoint else "zero_residual",
                "residual_scale": args.residual_scale,
                "success": episode_success,
            }
            atomic_write_episode(output, episode, arrays, manifest)
            if args.async_grpc:
                buffer = io.BytesIO()
                grpc_episode = {key: torch.from_numpy(value) for key, value in arrays.items()}
                grpc_episode["_episode_id"] = torch.tensor(episode, dtype=torch.int64)
                torch.save(grpc_episode, buffer)
                grpc_transitions.put(buffer.getvalue())
            collected_episodes += 1
            next_episode_id += 1
            total_transitions += len(rows)
            usable_transitions += max(0, len(rows) - args.n_step + 1)
            print(
                f"Saved replay episode={episode} transitions={len(rows)}; "
                f"directory_total={total_transitions}, usable_nstep_total="
                f"{usable_transitions}/{args.min_transitions or usable_transitions}"
            )
    finally:
        grpc_shutdown.set()
        if grpc_channel is not None:
            grpc_channel.close()
        env.close()


if __name__ == "__main__":
    main()
