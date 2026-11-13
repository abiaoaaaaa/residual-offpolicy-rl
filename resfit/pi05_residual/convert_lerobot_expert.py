"""Convert labeled Joint LeRobot demonstrations into Pi0.5 residual replay."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch

from resfit.rl_finetuning.marvin.replay import atomic_write_episode

from .action_codec import JointDeltaCodec
from .config import Pi05ResidualConfig
from .observation import CONTEXT_DIM, residual_context
from .policy import Pi05BasePolicy
from .runtime import replay_identity


SOURCE_VIEWS = (
    "observation.images.top",
    "observation.images.wrist_L",
    "observation.images.wrist_R",
)


@dataclass(frozen=True)
class EpisodeSource:
    data_path: Path
    start: int
    stop: int
    video_paths: tuple[Path, Path, Path]


def _episode_sources(dataset: Path) -> dict[int, EpisodeSource]:
    rows = pq.read_table(sorted((dataset / "meta/episodes").rglob("*.parquet"))).to_pylist()
    file_bases: dict[tuple[int, int], int] = {}
    for row in rows:
        location = (int(row["data/chunk_index"]), int(row["data/file_index"]))
        file_bases[location] = min(
            file_bases.get(location, int(row["dataset_from_index"])),
            int(row["dataset_from_index"]),
        )
    result = {}
    for row in rows:
        episode = int(row["episode_index"])
        location = (int(row["data/chunk_index"]), int(row["data/file_index"]))
        videos = []
        for view in SOURCE_VIEWS:
            chunk = int(row[f"videos/{view}/chunk_index"])
            file_index = int(row[f"videos/{view}/file_index"])
            videos.append(dataset / "videos" / view / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4")
        result[episode] = EpisodeSource(
            data_path=dataset / "data" / f"chunk-{location[0]:03d}" / f"file-{location[1]:03d}.parquet",
            start=int(row["dataset_from_index"]) - file_bases[location],
            stop=int(row["dataset_to_index"]) - file_bases[location],
            video_paths=tuple(videos),
        )
    return result


def _matrix(table, key: str) -> np.ndarray:
    value = np.asarray(table[key].combine_chunks().to_pylist(), dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 16:
        raise ValueError(f"{key}: expected [N,16], got {value.shape}")
    return value


def _resize_pad(image: np.ndarray, size: int = 128) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(size / height, size / width)
    resized = cv2.resize(image, (round(width * scale), round(height * scale)))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - resized.shape[0]) // 2
    left = (size - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return np.moveaxis(canvas, -1, 0)


def _read_interval_images(
    source: EpisodeSource,
    start: int,
    end: int,
    needed: set[int],
    block_starts: set[int],
    on_block,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    captures = [cv2.VideoCapture(str(path)) for path in source.video_paths]
    if any(not capture.isOpened() for capture in captures):
        raise RuntimeError(f"failed to open episode videos: {source.video_paths}")
    replay_images = {}
    try:
        for capture in captures:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        for frame in range(start, end + 1):
            images = []
            for capture in captures:
                ok, bgr = capture.read()
                if not ok:
                    raise RuntimeError(f"failed to decode source frame {frame}")
                images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if frame in needed:
                replay_images[frame] = tuple(_resize_pad(image) for image in images)
            if frame in block_starts:
                on_block(frame, images)
    finally:
        for capture in captures:
            capture.release()
    return replay_images


def _convert_episode(
    episode: int,
    label: dict,
    source: EpisodeSource,
    policy: Pi05BasePolicy,
    cfg: Pi05ResidualConfig,
) -> tuple[dict[str, np.ndarray], dict]:
    table = pq.read_table(source.data_path, columns=["observation.state", "action"]).slice(
        source.start, source.stop - source.start
    )
    states = _matrix(table, "observation.state")
    expert_actions = _matrix(table, "action")
    intervals = label["rl_intervals"]
    if len(intervals) != 1:
        raise ValueError(f"episode {episode}: expected one expert interval, got {intervals}")
    start, end = map(int, intervals[0])
    end = min(end, len(states) - 1)
    if label.get("outcome") != "success" or int(label.get("terminal_frame", -1)) != end:
        raise ValueError(f"episode {episode}: only labeled terminal-success intervals are accepted")

    steps = []
    for block_start in range(start, end, cfg.chunk_size):
        for offset in range(0, cfg.chunk_size, 3):
            obs_frame = block_start + offset
            if obs_frame >= end:
                break
            steps.append(
                {
                    "block": block_start,
                    "slot": offset // 3,
                    "obs": obs_frame,
                    "target": min(obs_frame + 2, end),
                    "next": min(obs_frame + 3, end),
                }
            )
    needed = {step[key] for step in steps for key in ("obs", "next")}
    block_starts = {step["block"] for step in steps}
    codec = JointDeltaCodec(cfg.rl_joint_delta_limit)
    base_chunks: dict[int, np.ndarray] = {}
    pi_absolute_chunks: dict[int, np.ndarray] = {}

    def infer_block(frame: int, images: list[np.ndarray]) -> None:
        observation = {
            "observation.state": torch.from_numpy(states[frame][None]),
            **{
                camera: torch.from_numpy(image[None])
                for camera, image in zip(cfg.cameras, images)
            },
        }
        absolute = policy.action_chunk(observation).detach().cpu()
        selected = absolute[:, 2::3]
        base = codec.encode_chunk(observation["observation.state"], selected)
        base_chunks[frame] = base[0].numpy()
        pi_absolute_chunks[frame] = selected[0].numpy()

    replay_images = _read_interval_images(
        source, start, end, needed, block_starts, infer_block
    )
    state_std = policy.state_std.clamp_min(cfg.min_state_std).numpy()
    state_mean = policy.state_mean.numpy()
    limits = np.asarray(cfg.rl_joint_delta_limit, dtype=np.float32)
    residual_limit = cfg.residual_joint_limit_rad / limits[:14]
    rows = []
    context = np.zeros(CONTEXT_DIM, dtype=np.float32)
    previous_executed = None
    for step in steps:
        base = base_chunks[step["block"]][step["slot"]]
        pi_absolute = pi_absolute_chunks[step["block"]][step["slot"]]
        residual_reference = (expert_actions[step["target"]] - pi_absolute) / limits
        residual_reference[14:] = 0.0
        critic_action = np.clip(base + residual_reference, -1.0, 1.0)
        bounded_residual = residual_reference.copy()
        bounded_residual[:14] = np.clip(bounded_residual[:14], -residual_limit, residual_limit)
        bounded_residual[14:] = 0.0
        # Offline takeover rows must expose exactly the same context contract
        # as the live collector.  The expert absolute target is the action
        # that was actually executed during the takeover; the actor proposal
        # remains the replay ``action`` and disagreement is retained explicitly.
        proposed_residual = np.clip(critic_action - base, -1.0, 1.0)
        executed_residual = np.clip(
            (expert_actions[step["target"]] - pi_absolute) / limits, -1.0, 1.0
        )
        if previous_executed is None:
            previous_executed = states[step["obs"]].copy()
        next_context = residual_context(
            executed_residual,
            expert_actions[step["target"]] - previous_executed,
            proposed_residual - executed_residual,
            limits,
        )
        rows.append(
            {
                "state": np.concatenate(
                    [(states[step["obs"]] - state_mean) / state_std, context]
                ),
                "next_state": np.concatenate(
                    [(states[step["next"]] - state_mean) / state_std, next_context]
                ),
                "base_action": base,
                "reference_residual_action": residual_reference,
                "residual_action": bounded_residual,
                "proposed_scaled_action": proposed_residual + base,
                "action": proposed_residual + base,
                "executed_residual_action": executed_residual,
                "pi_absolute_action": pi_absolute,
                "expert_absolute_action": expert_actions[step["target"]],
                "executed_absolute_action": expert_actions[step["target"]],
                "reward": float(step["next"] >= end),
                "terminated": bool(step["next"] >= end),
                "truncated": False,
                **{
                    f"image_{camera.replace('observation.images.', '')}": replay_images[step["obs"]][index]
                    for index, camera in enumerate(cfg.cameras)
                },
                **{
                    f"next_image_{camera.replace('observation.images.', '')}": replay_images[step["next"]][index]
                    for index, camera in enumerate(cfg.cameras)
                },
            }
        )
        context = next_context
        previous_executed = expert_actions[step["target"]].copy()
    for index, row in enumerate(rows):
        row["next_base_action"] = rows[min(index + 1, len(rows) - 1)]["base_action"]
    arrays = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    joint_reference = np.abs(arrays["reference_residual_action"][:, :14] * limits[:14])
    stats = {
        "source_episode": episode,
        "source_interval": [start, end],
        "pi_inferences": len(block_starts),
        "max_reference_residual_rad": float(joint_reference.max()),
        "reference_over_5deg_fraction": float(
            np.mean(joint_reference > cfg.residual_joint_limit_rad)
        ),
    }
    return arrays, stats


def run(args) -> None:
    dataset = Path(args.dataset).expanduser().resolve()
    annotations = json.loads(Path(args.annotations).expanduser().read_text())
    if Path(annotations["dataset"]).resolve() != dataset:
        raise ValueError("annotation sidecar does not belong to the requested dataset")
    selected = [
        (int(episode), label)
        for episode, label in annotations["episodes"].items()
        if label.get("outcome") == "success" and label.get("rl_intervals")
    ]
    selected.sort()
    if args.max_episodes:
        selected = selected[: args.max_episodes]
    if not selected:
        raise ValueError("no successful expert intervals found")
    output = Path(args.output).expanduser().resolve()
    if any(output.glob("episode_*.npz")):
        raise FileExistsError(f"output already contains replay episodes: {output}")
    cfg = Pi05ResidualConfig(
        pi05_checkpoint=args.pi05_checkpoint,
        pi05_config=args.pi05_config,
        task=args.task,
        cameras=tuple(args.cameras.split(",")),
        device=args.device,
    )
    identity = replay_identity(cfg)
    sources = _episode_sources(dataset)
    policy = Pi05BasePolicy(cfg)
    summary = []
    for output_episode, (source_episode, label) in enumerate(selected):
        arrays, stats = _convert_episode(
            source_episode, label, sources[source_episode], policy, cfg
        )
        manifest = {
            **identity,
            "episode_id": output_episode,
            "transitions": len(arrays["reward"]),
            "source": "pi05_residual_offline_expert",
            "success": True,
            "source_dataset": str(dataset),
            "source_annotations": str(Path(args.annotations).expanduser().resolve()),
            **stats,
        }
        atomic_write_episode(output, output_episode, arrays, manifest)
        summary.append(manifest)
        print(
            f"converted {output_episode + 1}/{len(selected)} source_episode={source_episode} "
            f"transitions={len(arrays['reward'])} max_residual_rad={stats['max_reference_residual_rad']:.4f}"
        )
    report = {
        "episodes": len(summary),
        "transitions": sum(item["transitions"] for item in summary),
        "max_reference_residual_rad": max(item["max_reference_residual_rad"] for item in summary),
        "mean_reference_over_5deg_fraction": float(
            np.mean([item["reference_over_5deg_fraction"] for item in summary])
        ),
        "items": summary,
    }
    (output / "conversion_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"offline expert replay ready: {output}; episodes={report['episodes']} transitions={report['transitions']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pi05-checkpoint", required=True)
    parser.add_argument("--pi05-config", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--cameras",
        default="observation.images.front,observation.images.left_wrist,observation.images.right_wrist",
    )
    parser.add_argument("--max-episodes", type=int, default=0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
