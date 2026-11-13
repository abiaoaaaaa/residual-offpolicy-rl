"""Replay a captured Pi0.5 trajectory through the residual bridge, without a model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .config import Pi05ResidualConfig
from .vlahost import VLAHostClient, VLAHostConfig, _BridgeBackend


def load_trajectory(path: str | Path) -> tuple[dict, np.ndarray]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported trajectory schema")
    actions = np.asarray(payload["actions_10hz"], dtype=np.float32)
    state = np.asarray(payload["state"], dtype=np.float32)
    if actions.shape != (10, 16) or state.shape != (16,):
        raise ValueError(f"expected state [16] and actions [10,16], got {state.shape}, {actions.shape}")
    if not np.isfinite(actions).all() or not np.isfinite(state).all():
        raise ValueError("trajectory contains non-finite values")
    return payload, actions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--bridge-url", default="http://127.0.0.1:8010")
    parser.add_argument("--max-start-error-rad", type=float, default=0.15)
    parser.add_argument("--bridge-joint-step-limit-rad", type=float, default=0.08)
    parser.add_argument("--report-json", default="")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    payload, actions = load_trajectory(args.trajectory)
    cfg = Pi05ResidualConfig(
        pi05_checkpoint=payload["pi05_checkpoint"],
        pi05_config=payload["pi05_config"],
        task=payload["task"],
        bridge_joint_step_limit_rad=args.bridge_joint_step_limit_rad,
    )
    bridge = VLAHostClient(VLAHostConfig(server_url=args.bridge_url, step_frequency=10.0))
    current, _ = bridge.observation()
    current_state = np.asarray(current["observation.state"], dtype=np.float32)
    captured_state = np.asarray(payload["state"], dtype=np.float32)
    start_error = float(np.max(np.abs(current_state[:14] - captured_state[:14])))
    print(f"start_error_rad={start_error:.5f}; actions={len(actions)}; duration=1.0s")
    if start_error > args.max_start_error_rad:
        raise RuntimeError(
            f"start pose differs from capture by {start_error:.3f} rad; "
            f"limit is {args.max_start_error_rad:.3f} rad"
        )
    if not args.execute:
        print("validation passed; no action sent (add --execute for real playback)")
        return

    backend = _BridgeBackend(bridge, cfg)
    backend.reset(go_home=False)
    deadline = time.monotonic()
    rows = []
    for index, target in enumerate(actions):
        started = time.monotonic()
        before, _ = bridge.observation()
        _, _, _, _, info = backend.step(target)
        after, _ = bridge.observation()
        elapsed = time.monotonic() - started
        waypoints = np.asarray(info["bridge_action_chunk"], dtype=np.float32)
        rows.append({
            "step": index,
            "submit_elapsed_ms": elapsed * 1000.0,
            "target": target.tolist(),
            "bridge_waypoints": waypoints.tolist(),
            "measured_before": np.asarray(before["observation.state"], dtype=np.float32).tolist(),
            "measured_after_submit": np.asarray(after["observation.state"], dtype=np.float32).tolist(),
        })
        deadline += 0.1
        time.sleep(max(0.0, deadline - time.monotonic()))

    report_path = Path(args.report_json).expanduser() if args.report_json else Path(args.trajectory).with_suffix(".replay_report.json")
    report = {
        "schema_version": 1,
        "trajectory": str(Path(args.trajectory).expanduser().resolve()),
        "replayed_unix_s": time.time(),
        "start_error_rad": start_error,
        "control_hz": 10.0,
        "source_hz": 30.0,
        "rows": rows,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"replay complete; report={report_path.resolve()}")


if __name__ == "__main__":
    main()
