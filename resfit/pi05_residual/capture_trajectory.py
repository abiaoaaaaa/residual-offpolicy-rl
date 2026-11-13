"""Capture one real observation and one frozen Pi0.5 action chunk as JSON."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .config import Pi05ResidualConfig
from .env import Pi05ResidualEnv
from .policy import Pi05BasePolicy
from .vlahost import VLAHostClient, VLAHostConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge-url", default="http://127.0.0.1:8010")
    parser.add_argument("--pi05-checkpoint", required=True)
    parser.add_argument("--pi05-config", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = Pi05ResidualConfig(
        pi05_checkpoint=args.pi05_checkpoint,
        pi05_config=args.pi05_config,
        task=args.task,
        device=args.device,
    )
    bridge = VLAHostClient(VLAHostConfig(server_url=args.bridge_url))
    raw, state_metadata = bridge.observation()
    policy = Pi05BasePolicy(cfg)
    adapter = Pi05ResidualEnv(None, policy, cfg)
    observation = adapter._tensor_obs(raw, cfg.image_size)

    started = time.perf_counter()
    chunk = policy.action_chunk(observation)[0].detach().cpu()
    inference_seconds = time.perf_counter() - started
    if tuple(chunk.shape) != (cfg.chunk_size, cfg.action_dim):
        raise ValueError(f"expected {(cfg.chunk_size, cfg.action_dim)}, got {tuple(chunk.shape)}")

    stride = cfg.chunk_size // cfg.rl_chunk_size
    selected_indices = list(range(stride - 1, cfg.chunk_size, stride))
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "captured_unix_s": time.time(),
        "bridge_url": args.bridge_url,
        "pi05_checkpoint": str(Path(args.pi05_checkpoint).expanduser().resolve()),
        "pi05_config": args.pi05_config,
        "task": args.task,
        "state": raw["observation.state"].tolist(),
        "camera_shapes": {
            key: list(raw[key].shape) for key in cfg.cameras
        },
        "pi_inference_seconds": inference_seconds,
        "source_hz": 30.0,
        "control_hz": 10.0,
        "actions_30hz": chunk.tolist(),
        "selected_indices": selected_indices,
        "actions_10hz": chunk[selected_indices].tolist(),
        "state_observation_generation": state_metadata.get("observation_generation"),
    }
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(
        f"saved {output}: state=16 actions_30hz={tuple(chunk.shape)} "
        f"actions_10hz=({len(selected_indices)}, 16) inference_ms={inference_seconds * 1000:.1f}"
    )


if __name__ == "__main__":
    main()
