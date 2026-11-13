# Pi0.5 EEF residual v9 design

## Status

The offline core is implemented in `eef_v9.py` and `eef_runtime.py`: base-frame
SE(3) delta encoding, bounded residual composition, EMA/slew filtering and a
stateful numerical-IK planner are covered by `tests/test_pi05_eef_v9.py`.
The Pinocchio/MJCF robot adapter and VLAHost collector integration are not
enabled yet because the OpenPI environment does not provide working Pinocchio
bindings; no real-robot action is sent by these modules.

## Recommendation

先以当前 joint residual v8 完成 replay/观测契约验证，再单独建立
`pi05_eef_residual_v9`。不要把 16D joint v8 replay 与 EEF replay 混训。
EEF residual 更适合限制探索幅度和获得连续运动，但它依赖正确的 FK、姿态
表示和有状态 IK，因此必须作为新的 action contract 实现。

## Control chain

```text
Pi0.5 absolute joint target q_pi[t]
  -> FK(q_pi[t]) and FK(q_pi[t-1])
  -> reference EEF delta in SE(3)
  -> actor bounded EEF residual [dx,dy,dz,dRx,dRy,dRz,(gripper)]
  -> EMA + velocity/slew + workspace safety box
  -> stateful streaming IK (seed = previous solved q)
  -> joint target / 3-waypoint VLAHost action_chunk
```

For two arms the residual is 12D (6D per arm); grippers remain separate and are
zeroed or handled by a dedicated bounded scalar head. Use rotation-vector
increments in the actor contract, not Euler angles. Compose the increment on the
Pi reference pose, then solve IK; do not independently IK-solve each waypoint
from the current measured joint state.

Initial limits at 10 Hz should be deliberately small: translation 2--5 mm per
step and rotation 1--2 degrees per step, followed by the existing 30 Hz waypoint
limit and workspace/collision checks. The residual amplitude, velocity and
acceleration limits are separate controls.

## Observation and replay

Keep the v8 context idea, changing its dimensionality to the EEF action space:
previous actual EEF residual, actual EEF delta, and actor-proposal versus actual
EEF residual disagreement. Store `action` as actor-proposed EEF residual. Store
filtered EEF residual, solved joint command, actual joint/EEF delta, IK success,
IK error and bridge waypoints as separate fields. Any IK failure or safety clip
must be represented in `info`/replay and must not silently replace the actor
action.

## Acceptance tests before real robot

1. FK round-trip: Pi joint target -> EEF reference -> stateful IK has bounded
   position/orientation error on recorded joint trajectories.
2. Zero residual is numerically identical to the existing Pi joint target.
3. Constant small residual produces monotonic, bounded EEF motion without joint
   jumps or IK branch flips.
4. Replay `action` remains the proposal when EMA/slew/IK changes execution.
5. Run fixed-trajectory, Pi-only and shadow tests before enabling RL ownership.
