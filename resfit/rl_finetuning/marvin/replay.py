from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 5
COORDINATE_FRAME = "resfit_marvin_bounded_v2"
ACTION_CONTRACT = "marvin_normalized_command_box_v1"


def usable_nstep_transitions(episodes: list[tuple[Path, dict[str, Any]]], n_step: int) -> int:
    if n_step <= 0:
        raise ValueError("n_step must be positive")
    return sum(max(0, int(manifest["transitions"]) - n_step + 1) for _path, manifest in episodes)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_policy_dir(checkpoint: str | Path) -> Path:
    checkpoint = Path(checkpoint).expanduser().resolve()
    if (checkpoint / "policy" / "config.json").is_file():
        checkpoint = checkpoint / "policy"
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"ACT checkpoint does not contain config.json: {checkpoint}")
    return checkpoint


def policy_weight_path(checkpoint: str | Path) -> Path:
    checkpoint = resolve_policy_dir(checkpoint)
    for name in ("model.safetensors", "pytorch_model.bin"):
        candidate = checkpoint / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"ACT checkpoint does not contain a supported weight file: {checkpoint}")


def dataset_stats_sha256(dataset_root: str | Path) -> str:
    stats_path = Path(dataset_root).expanduser().resolve() / "meta" / "episodes_stats.jsonl"
    if not stats_path.is_file():
        raise FileNotFoundError(f"Dataset episode statistics are missing: {stats_path}")
    return sha256_file(stats_path)


def replay_identity(
    *,
    base_policy_checkpoint: str | Path,
    base_policy_chunk_size: int,
    base_policy_n_action_steps: int,
    offline_dataset_root: str | Path,
    offline_dataset_name: str,
    actor_action_scale: float,
    min_action_range: float,
    min_state_std: float,
) -> dict[str, Any]:
    policy_dir = resolve_policy_dir(base_policy_checkpoint)
    dataset_root = Path(offline_dataset_root).expanduser().resolve()
    return {
        "schema_version": SCHEMA_VERSION,
        "coordinate_frame": COORDINATE_FRAME,
        "action_contract": ACTION_CONTRACT,
        "command_bounds": [-1.0, 1.0],
        "base_policy_checkpoint": str(policy_dir),
        "base_policy_sha256": sha256_file(policy_weight_path(policy_dir)),
        "base_policy_chunk_size": int(base_policy_chunk_size),
        "base_policy_n_action_steps": int(base_policy_n_action_steps),
        "offline_dataset_root": str(dataset_root),
        "offline_dataset_name": offline_dataset_name,
        "offline_dataset_stats_sha256": dataset_stats_sha256(dataset_root),
        "actor_action_scale": float(actor_action_scale),
        "min_action_range": float(min_action_range),
        "min_state_std": float(min_state_std),
    }


def validate_manifest(manifest: dict[str, Any], expected_identity: dict[str, Any], path: Path) -> None:
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected_identity.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Replay manifest identity mismatch at {path}: {mismatches}")


def inspect_replay_directory(
    replay_root: str | Path,
    expected_identity: dict[str, Any],
) -> tuple[list[tuple[Path, dict[str, Any]]], int, int]:
    replay_root = Path(replay_root).expanduser().resolve()
    replay_root.mkdir(parents=True, exist_ok=True)
    episodes: list[tuple[Path, dict[str, Any]]] = []
    total_transitions = 0
    max_episode_id = -1
    for episode_path in sorted(replay_root.glob("episode_*.npz")):
        manifest_path = episode_path.with_suffix(".json")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing Marvin replay manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        validate_manifest(manifest, expected_identity, manifest_path)
        episode_id = int(manifest["episode_id"])
        expected_stem = f"episode_{episode_id:06d}"
        if episode_path.stem != expected_stem:
            raise ValueError(f"Replay filename/id mismatch: {episode_path} versus episode_id={episode_id}")
        transitions = int(manifest["transitions"])
        if transitions <= 0:
            raise ValueError(f"Replay episode has no transitions: {manifest_path}")
        with np.load(episode_path, allow_pickle=False) as data:
            if "reward" not in data or len(data["reward"]) != transitions:
                raise ValueError(f"Replay NPZ/manifest transition mismatch: {episode_path}")
        episodes.append((episode_path, manifest))
        total_transitions += transitions
        max_episode_id = max(max_episode_id, episode_id)
    return episodes, total_transitions, max_episode_id + 1


def atomic_write_episode(
    replay_root: str | Path,
    episode_id: int,
    arrays: dict[str, np.ndarray],
    manifest: dict[str, Any],
) -> tuple[Path, Path]:
    replay_root = Path(replay_root).expanduser().resolve()
    replay_root.mkdir(parents=True, exist_ok=True)
    stem = f"episode_{episode_id:06d}"
    npz_path = replay_root / f"{stem}.npz"
    manifest_path = replay_root / f"{stem}.json"
    if npz_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing replay episode: {stem}")

    tmp_npz = replay_root / f".{stem}.npz.tmp"
    tmp_manifest = replay_root / f".{stem}.json.tmp"
    try:
        with tmp_npz.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        tmp_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(tmp_npz, npz_path)
        os.replace(tmp_manifest, manifest_path)
    finally:
        tmp_npz.unlink(missing_ok=True)
        tmp_manifest.unlink(missing_ok=True)
    return npz_path, manifest_path
