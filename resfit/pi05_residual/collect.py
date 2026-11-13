"""Real-time Pi0.5 rollout collector; never performs learner updates."""

from __future__ import annotations

import argparse
import io
import queue
import signal
import threading
import time

import numpy as np
import torch

from resfit.rl_finetuning.marvin.replay import atomic_write_episode, inspect_replay_directory
from resfit.rl_finetuning.off_policy.common_utils import utils

from .config import Pi05ResidualConfig
from .keyboard import KeyboardListener
from .policy import Pi05BasePolicy
from .runtime import ACTOR_CONTRACT, apply_actor_bundle, replay_identity
from .vlahost import VLAHostClient, VLAHostConfig, VLAHostResidualEnv


def _episode_arrays(rows: list[dict]) -> dict[str, np.ndarray]:
    return {key: np.asarray([row[key] for row in rows]) for key in rows[0]}


def _residual_probe(cfg: Pi05ResidualConfig, joint: int, degrees: float) -> torch.Tensor:
    residual = torch.zeros(cfg.action_dim)
    residual[joint] = float(np.deg2rad(degrees)) / cfg.rl_joint_delta_limit[joint]
    return residual


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bridge-url", required=True)
    p.add_argument("--pi05-checkpoint", required=True)
    p.add_argument("--pi05-config", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--episodes", type=int, default=0, help="0 runs continuously")
    p.add_argument("--pi-session-minutes", type=float, default=30.0)
    p.add_argument("--reset-sleep-seconds", type=float, default=3.0)
    p.add_argument("--state-dim", type=int, default=64)
    p.add_argument("--action-dim", type=int, default=16)
    p.add_argument("--chunk-size", type=int, default=30)
    p.add_argument("--rl-chunk-size", type=int, default=10)
    p.add_argument("--residual-joint-deg", type=float, default=5.0)
    p.add_argument("--residual-ema-alpha", type=float, default=0.35)
    p.add_argument("--residual-slew-limit-deg", type=float, default=1.0)
    p.add_argument("--action-lowpass-alpha", type=float, default=1.0)
    p.add_argument("--bridge-joint-step-limit-rad", type=float, default=0.08)
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpu-memory-fraction", type=float, default=0.20)
    p.add_argument("--control-hz", type=float, default=10.0)
    p.add_argument("--keyboard-device", default=None)
    p.add_argument("--residual-checkpoint", default="")
    p.add_argument("--async-grpc", action="store_true")
    p.add_argument(
        "--pi-only",
        action="store_true",
        help="force zero residual, disable learner/actor, and log every Pi0.5 chunk refresh",
    )
    p.add_argument(
        "--residual-shadow",
        action="store_true",
        help="run a fixed residual actor for diagnostics but always send zero residual",
    )
    p.add_argument("--residual-probe-joint", type=int, default=None)
    p.add_argument("--residual-probe-deg", type=float, default=0.5)
    p.add_argument("--learner-host", default="127.0.0.1")
    p.add_argument("--learner-port", type=int, default=50051)
    args = p.parse_args()
    if args.episodes < 0 or args.pi_session_minutes <= 0 or args.control_hz <= 0:
        raise ValueError("episodes must be non-negative; pi-session-minutes/control-hz must be positive")
    if args.control_hz != 10.0:
        raise ValueError("30Hz Pi0.5 chunks downsampled by 3 require --control-hz 10")
    if args.pi_only and (args.residual_checkpoint or args.async_grpc):
        raise ValueError("--pi-only cannot be combined with a residual checkpoint or --async-grpc")
    if args.residual_shadow and (
        args.pi_only or not args.residual_checkpoint or args.async_grpc
    ):
        raise ValueError(
            "--residual-shadow requires --residual-checkpoint and cannot use --pi-only or --async-grpc"
        )
    probe_enabled = args.residual_probe_joint is not None
    if probe_enabled and (
        not 0 <= args.residual_probe_joint < 14
        or not 0 < abs(args.residual_probe_deg) <= args.residual_joint_deg
        or args.pi_only
        or args.residual_shadow
        or args.residual_checkpoint
        or args.async_grpc
    ):
        raise ValueError(
            "residual probe requires arm joint 0..13, nonzero degrees within the residual limit, "
            "and cannot use Pi-only, shadow, actor checkpoint, or gRPC"
        )

    if args.device.startswith("cuda") and torch.cuda.is_available() and args.gpu_memory_fraction < 1.0:
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    cfg = Pi05ResidualConfig(
        pi05_checkpoint=args.pi05_checkpoint,
        pi05_config=args.pi05_config,
        task=args.task,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        chunk_size=args.chunk_size,
        rl_chunk_size=args.rl_chunk_size,
        residual_joint_limit_rad=float(np.deg2rad(args.residual_joint_deg)),
        residual_ema_alpha=args.residual_ema_alpha,
        residual_slew_limit_rad=float(np.deg2rad(args.residual_slew_limit_deg)),
        action_lowpass_alpha=args.action_lowpass_alpha,
        bridge_joint_step_limit_rad=args.bridge_joint_step_limit_rad,
        device=args.device,
        max_episode_steps=max(1, int(args.pi_session_minutes * 60 * args.control_hz) + 1),
    )
    identity = replay_identity(cfg)
    existing_episodes, _, next_episode_id = inspect_replay_directory(
        args.output_dir, identity
    )
    total_transitions = sum(
        int(manifest["transitions"])
        for _, manifest in existing_episodes
        if manifest.get("source") != "pi05_zero_residual_collector"
    )
    bridge = VLAHostClient(VLAHostConfig(
        server_url=args.bridge_url,
        step_frequency=args.control_hz,
        max_steps=cfg.max_episode_steps,
    ))
    env = VLAHostResidualEnv(bridge, Pi05BasePolicy(cfg), cfg)
    keyboard = KeyboardListener(args.keyboard_device)
    residual_agent = None
    if args.residual_checkpoint:
        bundle = torch.load(args.residual_checkpoint, map_location=args.device, weights_only=False)
        residual_agent, step = apply_actor_bundle(residual_agent, bundle, cfg, args.device)
        print(f"loaded residual actor step={step}")

    shutdown = threading.Event()
    parameter_queue = queue.Queue(maxsize=1)
    channel = None
    if args.async_grpc:
        from resfit.rl_finetuning.marvin.grpc_transport import (
            actor_stub,
            receive_parameters,
        )
        stub, channel = actor_stub(args.learner_host, args.learner_port)
        threading.Thread(
            target=receive_parameters,
            args=(stub, parameter_queue, shutdown),
            daemon=True,
        ).start()

    rl_armed = False
    rl_started_at = 0.0
    collected = 0
    pi_chunk = 1
    shadow_actor_ms: list[float] = []
    shadow_output_max: list[float] = []
    shadow_physical_deg_max: list[float] = []
    probe_applied_deg: list[float] = []
    rows: list[dict] = []
    success = False
    loop_seconds: list[float] = []
    base_inference_seconds: list[float] = []
    residual_abs_max: list[float] = []
    correction_abs_max: list[float] = []

    def append_row(
        current_obs, next_obs, *, action, absolute_action, residual_action,
        bridge_action_chunk, intervened, reward, terminated, truncated, tick,
        inference_seconds=0.0, executed_residual_action=None,
    ) -> None:
        base = current_obs["observation.base_action"].numpy()[0]
        normalized = np.asarray(action, dtype=np.float32).reshape(-1)
        residual = np.asarray(residual_action, dtype=np.float32).reshape(-1)
        row = {
            "state": current_obs["observation.state"].numpy()[0],
            "next_state": next_obs["observation.state"].numpy()[0],
            "base_action": base,
            "next_base_action": next_obs["observation.base_action"].numpy()[0],
            "residual_action": residual,
            "action": normalized,
            "raw_absolute_action": np.asarray(absolute_action, dtype=np.float32),
            "absolute_action": np.asarray(absolute_action, dtype=np.float32),
            "executed_absolute_action": np.asarray(absolute_action, dtype=np.float32),
            "executed_residual_action": np.asarray(
                residual if executed_residual_action is None else executed_residual_action,
                dtype=np.float32,
            ),
            "bridge_action_chunk": np.asarray(bridge_action_chunk, dtype=np.float32),
            "intervened": bool(intervened),
            "reward": reward,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        for camera in cfg.cameras:
            key = camera.replace("observation.images.", "")
            row[f"image_{key}"] = current_obs[camera].numpy()[0]
            row[f"next_image_{key}"] = next_obs[camera].numpy()[0]
        rows.append(row)
        loop_seconds.append(time.perf_counter() - tick)
        base_inference_seconds.append(float(inference_seconds))
        residual_abs_max.append(float(np.max(np.abs(residual))))
        correction_abs_max.append(float(np.max(np.abs(normalized - base))))

    def label(events: list[str]) -> tuple[float, bool, bool, bool]:
        nonlocal success
        reset_requested = "reset" in events
        if not rl_armed or time.monotonic() - rl_started_at < 0.25:
            return 0.0, False, reset_requested, reset_requested
        if "c" in events:
            success = True
            print(f"RL episode={next_episode_id} success at step={len(rows)}")
            return 1.0, True, False, reset_requested
        if "a" in events or "q" in events:
            print(f"RL episode={next_episode_id} failure at step={len(rows)}")
            return 0.0, True, False, reset_requested
        return 0.0, False, reset_requested, reset_requested

    def finish_without_row(events: list[str]) -> None:
        nonlocal rl_armed
        reward, terminated, truncated, _ = label(events)
        if not rl_armed or not (terminated or truncated):
            return
        if rows:
            rows[-1]["reward"] = reward
            rows[-1]["terminated"] = terminated
            rows[-1]["truncated"] = truncated
            save_rl_episode(truncated=bool(truncated and not terminated))
        rl_armed = False

    def save_rl_episode(*, truncated: bool = False) -> None:
        nonlocal rows, success, collected, next_episode_id, total_transitions
        nonlocal loop_seconds, base_inference_seconds, residual_abs_max, correction_abs_max
        if not rows:
            return
        if truncated:
            rows[-1]["truncated"] = True
        arrays = _episode_arrays(rows)
        manifest = {
            **identity,
            "episode_id": next_episode_id,
            "transitions": len(rows),
            "source": "pi05_residual_rl_episode",
            "success": success,
            "rl_armed": True,
            "actor_contract": ACTOR_CONTRACT,
            "control_hz": args.control_hz,
            "mean_loop_ms": float(np.mean(loop_seconds) * 1000.0),
            "p95_loop_ms": float(np.percentile(loop_seconds, 95) * 1000.0),
            "base_inference_refreshes": int(sum(x > 0.0 for x in base_inference_seconds)),
            "max_base_inference_ms": float(max(base_inference_seconds, default=0.0) * 1000.0),
            "max_residual_normalized": float(max(residual_abs_max, default=0.0)),
            "max_correction_normalized": float(max(correction_abs_max, default=0.0)),
            "intervention_transitions": int(sum(row["intervened"] for row in rows)),
        }
        previous_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            atomic_write_episode(args.output_dir, next_episode_id, arrays, manifest)
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
        total_transitions += len(rows)
        print(
            f"saved RL episode={next_episode_id} transitions={len(rows)} "
            f"success={success} intervention={manifest['intervention_transitions']} "
            f"replay_total={total_transitions}"
        )
        next_episode_id += 1
        collected += 1
        rows, loop_seconds, base_inference_seconds = [], [], []
        residual_abs_max, correction_abs_max = [], []
        success = False

    def apply_latest_actor() -> None:
        nonlocal residual_agent
        latest = None
        try:
            while True:
                latest = parameter_queue.get_nowait()
        except queue.Empty:
            pass
        if latest is not None:
            bundle = torch.load(io.BytesIO(latest), map_location=args.device, weights_only=True)
            residual_agent, step = apply_actor_bundle(residual_agent, bundle, cfg, args.device)
            print(f"applied learner actor at RL episode boundary: step={step}")

    try:
        reset_started = time.perf_counter()
        obs, _ = env.reset(home_sleep_s=args.reset_sleep_seconds)
        bridge.start_intervention_session()
        if bridge.wait_for_policy_control():
            obs, _ = env.reset(go_home=False)
        if not (args.pi_only or args.residual_shadow or probe_enabled) and (
            not isinstance(bridge.intervention, dict)
            or "executed_action_fresh" not in bridge.intervention
        ):
            raise RuntimeError(
                "VLAHost must be rebuilt/restarted with executed_action_fresh before intervention training"
            )
        if args.pi_only:
            print(
                f"pi_only chunk=1 ready startup_s={time.perf_counter() - reset_started:.3f} "
                "residual_max=0"
            )
        elif args.residual_shadow:
            print(
                f"residual_shadow chunk=1 ready startup_s={time.perf_counter() - reset_started:.3f} "
                "applied_residual_max=0"
            )
        elif probe_enabled:
            print(
                f"residual_probe chunk=1 ready joint={args.residual_probe_joint} "
                f"requested_deg={args.residual_probe_deg:.3f}"
            )
        session_deadline = time.monotonic() + args.pi_session_minutes * 60.0
        while args.episodes == 0 or collected < args.episodes:
            bridge.assert_intervention_healthy()
            snapshot = bridge.intervention
            if not isinstance(snapshot, dict):
                raise RuntimeError("VLAHost /state lacks intervention arbitration data")
            if snapshot.get("phase") != "policy":
                tick = time.perf_counter()
                before = keyboard.pop()
                if "reset" in before:
                    if rl_armed:
                        save_rl_episode(truncated=True)
                        rl_armed = False
                    bridge.stop_intervention_session()
                    obs, _ = env.reset(home_sleep_s=args.reset_sleep_seconds)
                    bridge.start_intervention_session()
                    if bridge.wait_for_policy_control():
                        obs, _ = env.reset(go_home=False)
                    session_deadline = time.monotonic() + args.pi_session_minutes * 60.0
                    continue
                if not (args.pi_only or args.residual_shadow or probe_enabled) and "b" in before and not rl_armed:
                    apply_latest_actor()
                    rl_armed = True
                    rl_started_at = time.monotonic()
                    print(f"keyboard: RL episode={next_episode_id} started during VR")
                if snapshot.get("footkey") != 1 and bridge.wait_for_policy_control(timeout=0.5):
                    if rows:
                        rows[-1]["truncated"] = True  # break n-step continuity across handover
                    obs, _ = env.reset(go_home=False)
                    print("VR released: fresh Pi0.5 observation and policy control restored; RL episode retained")
                else:
                    raw_next, state = bridge.observation()
                    human_snapshot = state.get("intervention", {})
                    executed = None
                    if human_snapshot.get("phase") == "vr":
                        try:
                            executed = bridge.executed_intervention_action(human_snapshot)
                            next_obs, human_action, _ = env.observe_intervention(raw_next, executed)
                        except ValueError as exc:
                            next_obs, human_action, _ = env.observe_intervention(raw_next, None)
                            if rl_armed:
                                if rows:
                                    rows[-1]["truncated"] = True
                                print(f"intervention frame omitted (no truthful critic action): {exc}")
                        if rl_armed and human_action is not None:
                            events = before + keyboard.pop()
                            reward, terminated, truncated, reset_requested = label(events)
                            append_row(
                                obs, next_obs, action=human_action, absolute_action=executed,
                                residual_action=human_action - obs["observation.base_action"].numpy()[0],
                                bridge_action_chunk=np.full((3, cfg.action_dim), np.nan, dtype=np.float32),
                                intervened=True, reward=reward, terminated=terminated,
                                truncated=truncated, tick=tick,
                            )
                            if terminated or truncated:
                                save_rl_episode(truncated=bool(truncated and not terminated))
                                rl_armed = False
                            if reset_requested:
                                bridge.stop_intervention_session()
                                obs, _ = env.reset(home_sleep_s=args.reset_sleep_seconds)
                                bridge.start_intervention_session()
                                if bridge.wait_for_policy_control():
                                    obs, _ = env.reset(go_home=False)
                                session_deadline = time.monotonic() + args.pi_session_minutes * 60.0
                                continue
                        elif rl_armed:
                            finish_without_row(before + keyboard.pop())
                        obs = next_obs
                    else:
                        events = before + keyboard.pop()
                        finish_without_row(events)
                    remaining = 1.0 / args.control_hz - (time.perf_counter() - tick)
                    if remaining > 0:
                        time.sleep(remaining)
                continue
            if not rl_armed:
                apply_latest_actor()
            if time.monotonic() >= session_deadline:
                if rl_armed:
                    save_rl_episode(truncated=True)
                    rl_armed = False
                print("Pi0.5 30-minute session ended; going home")
                bridge.stop_intervention_session()
                obs, _ = env.reset(home_sleep_s=args.reset_sleep_seconds)
                bridge.start_intervention_session()
                if bridge.wait_for_policy_control():
                    obs, _ = env.reset(go_home=False)
                session_deadline = time.monotonic() + args.pi_session_minutes * 60.0
                continue

            while True:
                bridge.assert_intervention_healthy()
                if bridge.intervention.get("phase") != "policy":
                    break
                if not rl_armed:
                    apply_latest_actor()
                tick = time.perf_counter()
                before = keyboard.pop()
                if "reset" in before:
                    if rl_armed:
                        save_rl_episode(truncated=True)
                        rl_armed = False
                    print(
                        "keyboard: Ctrl+1 reset; going home, then opening grippers, "
                        f"sleeping {args.reset_sleep_seconds:g}s"
                    )
                    bridge.stop_intervention_session()
                    obs, _ = env.reset(home_sleep_s=args.reset_sleep_seconds)
                    bridge.start_intervention_session()
                    if bridge.wait_for_policy_control():
                        obs, _ = env.reset(go_home=False)
                    session_deadline = time.monotonic() + args.pi_session_minutes * 60.0
                    break
                diagnostic_mode = args.pi_only or args.residual_shadow or probe_enabled
                if not diagnostic_mode and "b" in before and not rl_armed:
                    rl_armed = True
                    rl_started_at = time.monotonic()
                    print(f"keyboard: RL episode={next_episode_id} started")
                if probe_enabled:
                    residual = _residual_probe(
                        cfg, args.residual_probe_joint, args.residual_probe_deg
                    )
                elif args.residual_shadow:
                    actor_started = time.perf_counter()
                    with torch.inference_mode(), utils.eval_mode(residual_agent):
                        proposed_residual = residual_agent.act(
                            obs, eval_mode=True, stddev=0.0, cpu=True
                        )[0]
                    actor_ms = (time.perf_counter() - actor_started) * 1000.0
                    physical = proposed_residual[:14] * torch.as_tensor(
                        cfg.rl_joint_delta_limit[:14]
                    )
                    physical_deg = float(torch.rad2deg(physical.abs()).max())
                    if not torch.isfinite(proposed_residual).all() or physical_deg > args.residual_joint_deg + 1e-4:
                        raise RuntimeError("shadow actor violated the finite five-degree residual contract")
                    shadow_actor_ms.append(actor_ms)
                    shadow_output_max.append(float(proposed_residual.abs().max()))
                    shadow_physical_deg_max.append(physical_deg)
                    residual = torch.zeros(cfg.action_dim)
                elif args.pi_only or not rl_armed:
                    residual = torch.zeros(cfg.action_dim)
                elif residual_agent is None:
                    residual = torch.zeros(cfg.action_dim)
                else:
                    with torch.inference_mode(), utils.eval_mode(residual_agent):
                        residual = residual_agent.act(
                            obs, eval_mode=True, stddev=0.0, cpu=True
                        )[0]

                next_obs, _, terminated, truncated, info = env.step(residual)
                if info.get("execution") == "active" and info.get("intervention", {}).get("phase") != "policy":
                    if rl_armed and rows:
                        rows[-1]["truncated"] = True
                    print("policy/VR boundary omitted: ownership changed during action execution")
                    obs = next_obs
                    continue
                if info.get("execution") != "active":
                    human_snapshot = info.get("intervention", {})
                    try:
                        executed = bridge.executed_intervention_action(human_snapshot)
                        human_action = env.policy_env.normalize_executed_action(
                            executed, info["base_reference_action"]
                        )
                    except ValueError as exc:
                        if rl_armed:
                            if rows:
                                rows[-1]["truncated"] = True
                            print(f"takeover boundary omitted (no truthful critic action): {exc}")
                            finish_without_row(before + keyboard.pop())
                    else:
                        if rl_armed:
                            events = before + keyboard.pop()
                            reward, human_terminated, human_truncated, reset_requested = label(events)
                            append_row(
                                obs, next_obs, action=human_action, absolute_action=executed,
                                residual_action=human_action - obs["observation.base_action"].numpy()[0],
                                bridge_action_chunk=np.full((3, cfg.action_dim), np.nan, dtype=np.float32),
                                intervened=True, reward=reward,
                                terminated=human_terminated, truncated=human_truncated,
                                tick=tick, inference_seconds=info.get("base_inference_seconds", 0.0),
                            )
                            if human_terminated or human_truncated:
                                save_rl_episode(truncated=bool(human_truncated and not human_terminated))
                                rl_armed = False
                            if reset_requested:
                                bridge.stop_intervention_session()
                                obs, _ = env.reset(home_sleep_s=args.reset_sleep_seconds)
                                bridge.start_intervention_session()
                                if bridge.wait_for_policy_control():
                                    obs, _ = env.reset(go_home=False)
                                session_deadline = time.monotonic() + args.pi_session_minutes * 60.0
                                break
                    obs = next_obs
                    continue
                if probe_enabled:
                    probe_applied_deg.append(
                        float(
                            torch.rad2deg(
                                info["residual_physical_rad"][0, args.residual_probe_joint]
                            )
                        )
                    )
                if args.pi_only and info.get("chunk_refreshed", False):
                    pi_chunk += 1
                    print(
                        f"pi_only chunk={pi_chunk} "
                        f"inference_s={float(info['base_inference_seconds']):.3f} "
                        f"prefetch_wait_s={float(info['base_prefetch_wait_seconds']):.3f} "
                        f"loop_s={time.perf_counter() - tick:.3f} residual_max=0"
                    )
                elif args.residual_shadow and info.get("chunk_refreshed", False):
                    pi_chunk += 1
                    print(
                        f"residual_shadow chunk={pi_chunk} "
                        f"actor_ms_mean={float(np.mean(shadow_actor_ms)):.1f} "
                        f"actor_ms_max={float(max(shadow_actor_ms)):.1f} "
                        f"actor_output_max={float(max(shadow_output_max)):.5f} "
                        f"physical_deg_max={float(max(shadow_physical_deg_max)):.3f} "
                        f"pi_inference_s={float(info['base_inference_seconds']):.3f} "
                        f"prefetch_wait_s={float(info['base_prefetch_wait_seconds']):.3f} "
                        f"loop_s={time.perf_counter() - tick:.3f} applied_residual_max=0"
                    )
                    shadow_actor_ms.clear()
                    shadow_output_max.clear()
                    shadow_physical_deg_max.clear()
                elif probe_enabled and info.get("chunk_refreshed", False):
                    pi_chunk += 1
                    print(
                        f"residual_probe chunk={pi_chunk} joint={args.residual_probe_joint} "
                        f"requested_deg={args.residual_probe_deg:.3f} "
                        f"applied_deg_min={float(min(probe_applied_deg)):.3f} "
                        f"applied_deg_max={float(max(probe_applied_deg)):.3f} "
                        f"pi_inference_s={float(info['base_inference_seconds']):.3f} "
                        f"prefetch_wait_s={float(info['base_prefetch_wait_seconds']):.3f} "
                        f"loop_s={time.perf_counter() - tick:.3f}"
                    )
                    probe_applied_deg.clear()
                events = before + keyboard.pop()
                reward, labelled_terminated, labelled_truncated, reset_requested = label(events)
                terminated = bool(terminated or labelled_terminated)
                truncated = bool(truncated or labelled_truncated)

                if rl_armed:
                    append_row(
                        obs, next_obs,
                        action=info["scaled_action"].numpy()[0],
                        absolute_action=info["absolute_action"],
                        residual_action=info["residual_action"].numpy()[0],
                        executed_residual_action=info["executed_residual_action"].numpy()[0],
                        bridge_action_chunk=info["bridge_action_chunk"],
                        intervened=False, reward=reward, terminated=terminated,
                        truncated=truncated, tick=tick,
                        inference_seconds=info.get("base_inference_seconds", 0.0),
                    )
                obs = next_obs
                remaining = 1.0 / args.control_hz - (time.perf_counter() - tick)
                if remaining > 0:
                    time.sleep(remaining)
                if rl_armed and (terminated or truncated):
                    save_rl_episode(truncated=bool(truncated and not terminated))
                    rl_armed = False
                elif not diagnostic_mode and "b" in events and not rl_armed:
                    rl_armed = True
                    rl_started_at = time.monotonic()
                    print(f"keyboard: RL episode={next_episode_id} started")
                if reset_requested:
                    print(
                        "keyboard: Ctrl+1 reset; going home, then opening grippers, "
                        f"sleeping {args.reset_sleep_seconds:g}s"
                    )
                    bridge.stop_intervention_session()
                    obs, _ = env.reset(home_sleep_s=args.reset_sleep_seconds)
                    bridge.start_intervention_session()
                    if bridge.wait_for_policy_control():
                        obs, _ = env.reset(go_home=False)
                    session_deadline = time.monotonic() + args.pi_session_minutes * 60.0
                    break
                if args.episodes > 0 and collected >= args.episodes:
                    break
                if time.monotonic() >= session_deadline:
                    break
    except KeyboardInterrupt:
        if rl_armed:
            save_rl_episode(truncated=True)
            print("collector interrupted; active RL episode saved")
        else:
            print("collector interrupted; no active RL episode")
    finally:
        shutdown.set()
        bridge.stop_intervention_session()
        if channel is not None:
            channel.close()


if __name__ == "__main__":
    main()
