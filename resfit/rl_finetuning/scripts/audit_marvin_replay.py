from __future__ import annotations

import argparse
import json

from resfit.rl_finetuning.marvin.replay import inspect_replay_directory, replay_identity, usable_nstep_transitions


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a formal Marvin Residual replay directory")
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--base-policy-checkpoint", required=True)
    parser.add_argument("--base-policy-chunk-size", type=int, default=20)
    parser.add_argument("--base-policy-n-action-steps", type=int, default=5)
    parser.add_argument("--offline-dataset-root", required=True)
    parser.add_argument("--offline-dataset-name", default="marvin_usb_rl")
    parser.add_argument("--actor-action-scale", type=float, default=0.1)
    parser.add_argument("--min-action-range", type=float, default=0.1)
    parser.add_argument("--min-state-std", type=float, default=0.1)
    parser.add_argument("--minimum-transitions", type=int, default=10_000)
    parser.add_argument("--n-step", type=int, default=3)
    args = parser.parse_args()

    identity = replay_identity(
        base_policy_checkpoint=args.base_policy_checkpoint,
        base_policy_chunk_size=args.base_policy_chunk_size,
        base_policy_n_action_steps=args.base_policy_n_action_steps,
        offline_dataset_root=args.offline_dataset_root,
        offline_dataset_name=args.offline_dataset_name,
        actor_action_scale=args.actor_action_scale,
        min_action_range=args.min_action_range,
        min_state_std=args.min_state_std,
    )
    episodes, transitions, next_episode_id = inspect_replay_directory(args.replay_dir, identity)
    usable_transitions = usable_nstep_transitions(episodes, args.n_step)
    report = {
        **identity,
        "replay_dir": args.replay_dir,
        "episodes": len(episodes),
        "transitions": transitions,
        "n_step": args.n_step,
        "usable_nstep_transitions": usable_transitions,
        "minimum_transitions": args.minimum_transitions,
        "ready": usable_transitions >= args.minimum_transitions,
        "next_episode_id": next_episode_id,
    }
    print(json.dumps(report, indent=2))
    if not report["ready"]:
        raise SystemExit(
            f"Replay is not ready: {usable_transitions} < {args.minimum_transitions} usable n-step transitions"
        )


if __name__ == "__main__":
    main()
