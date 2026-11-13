"""Replay one LeRobot joint-action episode through the VLAHost bridge."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .config import Pi05ResidualConfig
from .vlahost import VLAHostClient, VLAHostConfig, _BridgeBackend


def load_episode(dataset: Path, episode_id: int) -> tuple[np.ndarray, np.ndarray]:
    info = json.loads((dataset / "meta" / "info.json").read_text())
    table = pq.read_table(
        dataset / "data" / "chunk-000" / "file-000.parquet",
        columns=["episode_index", "observation.state", "action"],
    )
    episode = np.asarray(table["episode_index"], dtype=np.int64)
    mask = episode == episode_id
    if not mask.any():
        raise ValueError(f"episode {episode_id} not found")
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)[mask]
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)[mask]
    if info.get("fps") != 30 or state.shape[1:] != (16,) or action.shape[1:] != (16,):
        raise ValueError(f"expected 30Hz [N,16] dataset, got fps={info.get('fps')} state={state.shape} action={action.shape}")
    return state, action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--episode", type=int, default=135)
    parser.add_argument("--bridge-url", default="http://127.0.0.1:8010")
    parser.add_argument("--bridge-joint-step-limit-rad", type=float, default=0.08)
    parser.add_argument(
        "--mode",
        choices=(
            "raw30",
            "decimate10_interp",
            "decimate10_feedback_interp",
            "decimate10_backend",
        ),
        default="raw30",
        help=(
            "raw dataset frames; 10Hz target + command-anchored interpolation; "
            "or online-style immediate-feedback-anchored interpolation"
        ),
    )
    parser.add_argument("--max-start-error-rad", type=float, default=0.15)
    parser.add_argument("--report-json", default="")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    dataset = Path(args.dataset).expanduser().resolve()
    states, actions = load_episode(dataset, args.episode)
    bridge = VLAHostClient(VLAHostConfig(server_url=args.bridge_url, step_frequency=10.0))
    current, _ = bridge.observation()
    current_state = np.asarray(current["observation.state"], dtype=np.float32)
    start_error = float(np.max(np.abs(current_state[:14] - states[0, :14])))
    max_frame_jump = float(np.max(np.abs(np.diff(actions[:, :14], axis=0)))) if len(actions) > 1 else 0.0
    print(
        f"episode={args.episode} frames={len(actions)} duration_s={len(actions)/30:.2f} "
        f"start_error_rad={start_error:.5f} max_dataset_frame_jump_rad={max_frame_jump:.5f}"
    )
    if start_error > args.max_start_error_rad:
        raise RuntimeError(f"start pose error {start_error:.3f} exceeds {args.max_start_error_rad:.3f} rad")
    if not args.execute:
        print("validation passed; no action sent (add --execute for real playback)")
        return

    # Preserve complete 30Hz dataset actions, or reproduce the online
    # residual path: select one target every 3 frames and interpolate it into
    # three 30Hz waypoints before sending one 100ms HTTP chunk.
    sent_rows = []
    previous = current_state.copy()
    backend = None
    if args.mode == "decimate10_backend":
        backend = _BridgeBackend(
            bridge,
            Pi05ResidualConfig(pi05_checkpoint="unused", task="unused"),
        )
        backend.reset(go_home=False)
    deadline = time.monotonic()
    for chunk_index, start in enumerate(range(0, len(actions), 3)):
        source = actions[start : start + 3]
        if len(source) < 3:
            source = np.concatenate((source, np.repeat(source[-1:], 3 - len(source), axis=0)))
        if args.mode == "raw30":
            requested = source.copy()
        else:
            target = source[-1].copy()
            fractions = np.arange(1, 4, dtype=np.float32)[:, None] / 3.0
            requested = previous[None] + fractions * (target[None] - previous[None])
        if backend is not None:
            started = time.monotonic()
            _, _, _, _, result = backend.step(target)
            elapsed = time.monotonic() - started
            dispatched = np.asarray(result["bridge_action_chunk"], dtype=np.float32)
            previous = dispatched[-1].copy()
            sent_rows.append({
                "chunk": chunk_index,
                "frame_start": start,
                "submit_elapsed_ms": elapsed * 1000.0,
                "requested": requested.tolist(),
                "dispatched": dispatched.tolist(),
                "seq": result.get("seq"),
                "mode": args.mode,
                "observation_started_after_s": result["observation_started_after_s"],
                "observation_completed_after_s": result["observation_completed_after_s"],
            })
            continue
        dispatched = requested.copy()
        for row in dispatched:
            row[:14] = previous[:14] + np.clip(row[:14] - previous[:14], -args.bridge_joint_step_limit_rad, args.bridge_joint_step_limit_rad)
            previous = row.copy()
        started = time.monotonic()
        result = bridge.action_chunk(dispatched, source_hz=30.0)
        elapsed = time.monotonic() - started
        feedback = None
        if args.mode == "decimate10_feedback_interp":
            feedback_raw, _ = bridge.observation()
            feedback = np.asarray(
                feedback_raw["observation.state"], dtype=np.float32
            )
            # Match _BridgeBackend.step(): the next 10Hz segment starts from
            # state sampled immediately after /action_chunk acknowledges it.
            previous = feedback.copy()
        sent_rows.append({
            "chunk": chunk_index,
            "frame_start": start,
            "submit_elapsed_ms": elapsed * 1000.0,
            "requested": requested.tolist(),
            "dispatched": dispatched.tolist(),
            "seq": result.get("seq"),
            "mode": args.mode,
            "feedback_after_submit": None if feedback is None else feedback.tolist(),
        })
        deadline += 0.1
        time.sleep(max(0.0, deadline - time.monotonic()))

    report = {
        "schema_version": 1,
        "dataset": str(dataset),
        "episode": args.episode,
        "frames": len(actions),
        "duration_s": len(actions) / 30.0,
        "start_error_rad": start_error,
        "max_dataset_frame_jump_rad": max_frame_jump,
        "source_hz": 30.0,
        "mode": args.mode,
        "chunks": sent_rows,
    }
    report_path = Path(args.report_json).expanduser() if args.report_json else dataset / "meta" / f"replay_episode_{args.episode:06d}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"replay complete; report={report_path.resolve()}")


if __name__ == "__main__":
    main()
