import torch

from resfit.pi05_residual.action_codec import JointDeltaCodec, ResidualActionFilter


def test_residual_filter_limits_step_and_converges_smoothly():
    filt = ResidualActionFilter(alpha=0.5, slew_limit_rad=0.1, dimensions=2)
    first = filt(torch.tensor([1.0, -1.0]))
    second = filt(torch.tensor([1.0, -1.0]))
    assert torch.allclose(first, torch.tensor([0.1, -0.1]))
    assert torch.allclose(second, torch.tensor([0.2, -0.2]))
from resfit.pi05_residual.config import Pi05ResidualConfig
from resfit.pi05_residual.env import Pi05ResidualEnv


def test_absolute_chunk_round_trip_and_full_queue():
    codec = JointDeltaCodec((2.0, 4.0))
    state = torch.tensor([[10.0, 20.0]])
    targets = torch.tensor([[[11.0, 22.0], [12.0, 18.0], [10.0, 20.0]]])
    delta = codec.encode_chunk(state, targets)
    assert torch.allclose(
        delta,
        torch.tensor([[[0.5, 0.5], [0.5, -1.0], [-1.0, 0.5]]]),
    )
    rebuilt = []
    current = state
    for step in delta.unbind(1):
        current = codec.decode_step(current, step)
        rebuilt.append(current)
    assert torch.allclose(torch.stack(rebuilt, 1), targets)


class _Policy:
    def __init__(self):
        self.calls = 0

    def action_chunk(self, observation):
        self.calls += 1
        state = observation["observation.state"]
        offsets = torch.tensor([[[float(i + 1), float((i % 3) - 1)] for i in range(30)]])
        return state[:, None] + offsets


class _Env:
    def __init__(self):
        self.state = torch.tensor([10.0, 20.0])

    def _obs(self):
        image = torch.zeros(4, 4, 3, dtype=torch.uint8)
        return {
            "observation.state": self.state.clone(),
            "observation.images.front": image,
            "observation.images.left_wrist": image,
            "observation.images.right_wrist": image,
        }

    def reset(self):
        return self._obs(), {}

    def step(self, action):
        self.state = torch.as_tensor(action)
        return self._obs(), 0.0, False, False, {}


def test_thirty_pi_targets_become_ten_100ms_residual_steps():
    cfg = Pi05ResidualConfig(
        pi05_checkpoint="unused",
        task="unused",
        state_dim=2,
        action_dim=2,
        chunk_size=30,
        rl_chunk_size=10,
        joint_delta_limit=(2.0, 4.0),
        action_lowpass_alpha=0.5,
    )
    policy = _Policy()
    env = Pi05ResidualEnv(_Env(), policy, cfg)
    env.reset()
    assert torch.allclose(env._base_chunk[0, :, 0], torch.full((10,), 0.5))
    residual = torch.tensor([5.0 / 180.0 * torch.pi / 6.0, -5.0 / 180.0 * torch.pi / 12.0])
    _, _, _, _, info = env.step(residual)
    assert torch.allclose(info["residual_action"], residual[None])
    assert torch.allclose(torch.as_tensor(info["residual_physical_rad"]), torch.tensor([[5.0 / 180.0 * torch.pi, -5.0 / 180.0 * torch.pi]]), atol=1e-6)
    expected_raw = torch.tensor([13.0 + torch.pi / 36.0, 21.0 - torch.pi / 36.0])
    assert torch.allclose(torch.as_tensor(info["raw_absolute_action"]), expected_raw, atol=1e-6)
    expected_filtered = 0.5 * (torch.tensor([10.0, 20.0]) + expected_raw)
    assert torch.allclose(torch.as_tensor(info["absolute_action"]), expected_filtered, atol=1e-6)
    assert torch.allclose(env.env.state, expected_filtered, atol=1e-6)
    _, _, _, _, second = env.step(torch.zeros(2))
    assert torch.allclose(
        torch.as_tensor(second["absolute_action"]),
        0.5 * (expected_filtered + torch.as_tensor(second["raw_absolute_action"])),
        atol=1e-6,
    )
    assert torch.allclose(torch.as_tensor(second["raw_absolute_action"]), torch.tensor([16.0, 21.0]))
    assert policy.calls == 1
    for _ in range(8):
        env.step(torch.zeros(2))
    assert policy.calls == 2


def test_lowpass_reset_starts_from_observed_state_and_alpha_one_is_passthrough():
    cfg = Pi05ResidualConfig(
        pi05_checkpoint="unused", task="unused", state_dim=2, action_dim=2,
        joint_delta_limit=(2.0, 4.0),
    )
    assert cfg.action_lowpass_alpha == 1.0
    env = Pi05ResidualEnv(_Env(), _Policy(), cfg)
    env.reset()
    _, _, _, _, info = env.step(torch.zeros(2))
    assert torch.allclose(
        torch.as_tensor(info["absolute_action"]),
        torch.as_tensor(info["raw_absolute_action"]),
    )
    env.env.state = torch.tensor([5.0, 7.0])
    env.reset()
    assert torch.allclose(torch.as_tensor(env._last_filtered_action), torch.tensor([5.0, 7.0]))

    filtered_cfg = Pi05ResidualConfig(
        pi05_checkpoint="unused", task="unused", state_dim=2, action_dim=2,
        joint_delta_limit=(2.0, 4.0), action_lowpass_alpha=0.5,
    )
    filtered_env = Pi05ResidualEnv(_Env(), _Policy(), filtered_cfg)
    filtered_env.env.state = torch.tensor([5.0, 7.0])
    filtered_env.reset()
    _, _, _, _, after_reset = filtered_env.step(torch.zeros(2))
    assert torch.allclose(
        torch.as_tensor(after_reset["absolute_action"]),
        0.5 * (torch.tensor([5.0, 7.0]) + torch.as_tensor(after_reset["raw_absolute_action"])),
    )


def test_zero_residual_preserves_pi_absolute_targets_despite_feedback_lag():
    class LaggingEnv(_Env):
        def step(self, action):
            self.state = self.state + 0.5 * (torch.as_tensor(action) - self.state)
            return self._obs(), 0.0, False, False, {}

    cfg = Pi05ResidualConfig(
        pi05_checkpoint="unused", task="unused", state_dim=2, action_dim=2,
        joint_delta_limit=(2.0, 4.0),
    )
    env = Pi05ResidualEnv(LaggingEnv(), _Policy(), cfg)
    env.reset()
    _, _, _, _, first = env.step(torch.zeros(2))
    _, _, _, _, second = env.step(torch.zeros(2))
    assert torch.allclose(torch.as_tensor(first["raw_absolute_action"]), torch.tensor([13.0, 21.0]))
    assert torch.allclose(torch.as_tensor(second["raw_absolute_action"]), torch.tensor([16.0, 21.0]))
    assert not torch.allclose(env.env.state, torch.tensor([16.0, 21.0]))


def test_base_normalization_clip_does_not_change_pi_absolute_target():
    cfg = Pi05ResidualConfig(
        pi05_checkpoint="unused", task="unused", state_dim=2, action_dim=2,
        joint_delta_limit=(0.1, 0.1),
    )
    env = Pi05ResidualEnv(_Env(), _Policy(), cfg)
    obs, _ = env.reset()
    assert torch.allclose(obs["observation.base_action"], torch.ones(1, 2))
    _, _, _, _, info = env.step(torch.zeros(2))
    assert torch.allclose(torch.as_tensor(info["raw_absolute_action"]), torch.tensor([13.0, 21.0]))
