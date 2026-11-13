# pi0.5 Residual RL

This package keeps the pi0.5 behavior policy in the native OpenPI/LeRobot
policy API and reuses the proven residual TD3 learner from this repository.

## Data flow

```text
3 camera images + state
        |
        +--> native OpenPI infer()
        |          absolute joint target chunk [B, 30, 16]
        |          select absolute targets 2,5,...,29 (10Hz)
        |          sequential delta + 10Hz per-joint normalization
        |          normalized base delta chunk [B, 10, 16] in [-1, 1]
        |
        +--> residual actor (QAgent, residual_actor=True)
                   16D normalized state + 48D execution context
                   + base_action + 3 image encoders
                   residual action [B, A]
        |
        +--> clip(base_delta[t] + actor residual, -1, 1)
        |          add physical residual to Pi0.5's original absolute target
        |          raw absolute joint target (replay)
        |          low-pass first 14 joints at 10Hz
        |          interpolate 3 waypoints from measured state to target
        |          limit each arm waypoint to 0.08 rad from the previous one
        |          one POST /action_chunk with 3 waypoints at source_hz=30
        |          VLAHost interpolates again for its 500Hz chunk player
        |
        +--> replay: obs, base_action, actor-proposed normalized action,
                    separately logged filtered/executed action, reward, next obs
```

## Files

- `policy.py`: native OpenPI `create_trained_policy` loader and frozen
  base-action adapter. A LeRobot PI05 fallback is kept for converted local
  checkpoints.
- `action_codec.py`: normalizes the selected absolute targets as consecutive
  10Hz joint deltas for the residual actor and replay.
- `env.py`: Gym-style adapter that adds `observation.base_action` and applies
  the residual only after the base delta is normalized to `[-1, 1]`.
- `config.py`: residual RL configuration for three cameras and state.
- `train.py`: entry point that reuses `resfit.rl_finetuning.off_policy.rl.QAgent`
  and the existing replay/update implementation.
- `collect.py`: collect offline transitions with the frozen pi0.5 base policy.

The base policy owns image preprocessing/tokenization and always receives the
raw 16D state. The residual learner receives
`observation.images.<camera>`, a 64D `observation.state` (normalized raw 16D
state followed by 48D context), and `observation.base_action`. The context is,
in order, previous actual residual, actual absolute-action delta normalized by
the 10Hz joint limits, and actor-proposal versus actual-residual disagreement.
At reset it is zero; after every step it is built from the actual command.

## Runtime cadence

OpenPI predicts 30 absolute joint targets at 30Hz. We use targets at indices
`2,5,...,29`, giving ten 100ms actions. Each 10Hz step reads fresh images and
joint state. The actor's final `tanh` is in `[-1,1]`; its first 14 outputs are
scaled to at most 5 degrees per joint per step. Gripper residuals are zero.
Before execution, the physical residual passes through EMA (`alpha=0.35`) and
a 1 degree/100ms per-joint slew limit. The critic/replay `action` is the
actor-proposed normalized combined action. The filtered/executed normalized
residual is stored separately as `executed_residual_action`; `proposed_scaled_action`
is retained as an explicit diagnostic alias.
Replay also stores the Pi-target-plus-residual raw
absolute joint target, the filtered `absolute_action` used as the final
chunk target, and all three rate-limited `bridge_action_chunk` waypoints;
`next_obs` is measured after the chunk playback period. The
first 14 joint
targets use `filtered_t = (1 - alpha) * filtered_(t-1) + alpha * raw_t`, with
`alpha=1.0` by default, which disables smoothing. The filter starts from the
observed joint state after reset. Gripper targets are not filtered. Set
`--action-lowpass-alpha` below 1 to enable smoothing. Collector and learner
must use the same alpha.

The bridge transport is asynchronous: a successful `/action_chunk` response
only confirms scheduling. Each 10Hz collector step sends one chunk containing
three linearly interpolated 30Hz waypoints. The collector, not the bridge,
waits until the 100ms RL deadline before starting the next step. The server normally blends from its current
setpoint and plays a dense trajectory at 500Hz. The 14 arm joints are limited
to at most 0.08 rad per 30Hz waypoint, matching the LeRobot driver default;
this may leave the final bridge waypoint short of the requested Pi target.
Network, image decoding and
Pi0.5 inference can still make the *actual* RL loop slower than 10Hz; the
per-waypoint limit is not an acceleration or full hardware safety guarantee.
Repeated corrections in the same direction can accumulate, so 5 degrees per
100ms is an amplitude limit, not a guarantee of slow joint motion.

OpenPI's native `Policy.infer()` is not normalized: its output pipeline applies
`Unnormalize(norm_stats)`. This checkpoint has `use_delta_joint_actions=False`,
so its output is already an absolute joint target. The selected absolute Pi
targets are preserved for control; sequential deltas are only actor/replay
features. The residual is converted to radians and added to each absolute
target without re-anchoring it to delayed joint feedback.

## Replay and reward

Training uses the original residual path: separate online/offline replay
buffers, `offline_fraction` batch mixing, `MultiStepTransform`, uniform or
prioritized sampling with TD-error priority updates, critic-only warmup, actor
updates, and target-network soft updates. Keyboard labels are authoritative:
`b` starts RL control, `a` ends a failed episode, and `c` ends a successful
episode. Only the transitions between B and A/C form an RL episode and enter
online replay. `Ctrl+1` truncates an active RL episode, calls `go_home`, opens
both grippers after the home request returns, and waits three seconds. Pi0.5
keeps providing reference actions between RL episodes; its outer session resets
after 30 minutes. HTTP responses only
acknowledge transport and never supply reward.

The collector keeps a VLAHost `/intervention/session` heartbeat. The VR foot
switch gives the human control through VLAHost; the collector stops submitting
policy chunks but keeps the same B-to-A/C RL episode. A VR frame is added to
online replay only when `/state.intervention` confirms active human ownership,
both mux-output arm commands are fresh, and the actual absolute 16-joint target
can be converted without clipping into the critic's normalized Pi-reference
delta space. These frames carry `intervened=true`; the learner additionally
copies contiguous intervention runs into its expert buffer. Ambiguous takeover
boundaries or unrepresentable actions are omitted and terminate the preceding
n-step segment, never substituted with the unexecuted Pi/actor proposal. On
release, the collector submits a hold chunk with a fresh observation generation,
waits for policy ownership, and replans Pi0.5 without ending the RL episode.
This requires restarting VLAHost from the updated source exposing
`executed_action_fresh` in `/state.intervention`.

The collector commits each RL episode to the shared `online_replay` directory.
The learner scans that directory for new committed episodes once per second;
gRPC carries actor parameters to the collector. Large image transitions are
not sent through a blocking collector-side gRPC queue.

The learner matches the original Marvin residual schedule: wait for 1,000
online transitions, run 1,000 critic-only warmup updates, then use UTD=4 with
four critic updates and one actor update per learner step. The 10Hz actor and
target actions are deterministic. The real-time collector does not inject
random warmup actions.

For a real robot, `vlahost.py` follows the LeRobot VLAHost bridge contract:
`GET /state`, `quad_image` retrieval, `POST /action_chunk`, and `POST /go_home`.
The bridge-backed trainer uses `--bridge-url`; learning stays local while
physical execution remains on VLAHost.

## Run

Use the OpenPI environment, with this repository, OpenPI source, and the
Python-3.11-compatible LeRobot transport copy on `PYTHONPATH`:

```bash
export PYTHONPATH=/home/xlb/code/residual-offpolicy-rl:/home/xlb/data/code/code_new/lerobot_vlahost/src:/home/xlb/data/code/code_new/code_xlb/openpi/src
export RESIDUAL_KEYBOARD_DEVICE=/dev/input/by-id/<a4tech-keyboard>

# Terminal 1: learner (no VLAHost or Pi0.5 GPU model is loaded here)
/home/xlb/data/code/code_new/code_xlb/openpi/.venv/bin/python \
  -m resfit.pi05_residual.train \
  --pi05-checkpoint /home/xlb/data/models/pi05_zadai_ybs_joint_clean_951/149999 \
  --pi05-config pi05_zadai_ybs_joint_clean_951 \
  --task "Insert the tail of the yellow cable tie into its head to fasten it." \
  --replay-dir /home/xlb/data/pi05_residual_v8_clean951/online_replay \
  --checkpoint /home/xlb/data/pi05_residual_v8_clean951/learner.pt \
  --residual-checkpoint /home/xlb/data/pi05_residual_v8_clean951/actor.pt \
  --chunk-size 30 --rl-chunk-size 10 --residual-joint-deg 5 --residual-ema-alpha 0.35 --residual-slew-limit-deg 1.0 --action-lowpass-alpha 1 --bridge-joint-step-limit-rad 0.08 \
  --learning-starts 1000 --critic-warmup-steps 1000 \
  --batch-size 256 --updates-per-round 4 --actor-updates-per-round 1 \
  --max-learner-steps-per-online-transition 1 \
  --n-step 3 --sampling-strategy prioritized_replay \
  --offline-replay-dir /home/xlb/data/pi05_residual_v8_clean951/offline_expert_38 \
  --offline-fraction 0.5 \
  --async-grpc

# Terminal 2: real-time collector (B arms RL; A fails; C succeeds)
/home/xlb/data/code/code_new/code_xlb/openpi/.venv/bin/python \
  -m resfit.pi05_residual.collect \
  --bridge-url http://127.0.0.1:8010 \
  --pi05-checkpoint /home/xlb/data/models/pi05_zadai_ybs_joint_clean_951/149999 \
  --pi05-config pi05_zadai_ybs_joint_clean_951 \
  --task "Insert the tail of the yellow cable tie into its head to fasten it." \
  --output-dir /home/xlb/data/pi05_residual_v8_clean951/online_replay \
  --chunk-size 30 --rl-chunk-size 10 --residual-joint-deg 5 --residual-ema-alpha 0.35 --residual-slew-limit-deg 1.0 --action-lowpass-alpha 1 --bridge-joint-step-limit-rad 0.08 \
  --pi-session-minutes 30 --reset-sleep-seconds 3 \
  --async-grpc --control-hz 10
```

The current checkpoint's measured sequential-delta maxima are stored in
`Pi05ResidualConfig.joint_delta_limit`. The first 14 dimensions are radians;
dimensions 14 and 15 are gripper units. Three times these 30Hz limits define
the 10Hz normalization range. The old 30Hz replay and checkpoints are not
compatible; the commands above use a separate directory. Existing single-frame
`/action` and pre-fix `/action_chunk` replay/checkpoints use different action
contracts and must not be mixed with this absolute-target, rate-limited path.
