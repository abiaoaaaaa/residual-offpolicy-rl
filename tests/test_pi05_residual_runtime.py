import time
import threading
import json
from collections import deque
from types import SimpleNamespace

import pytest
import numpy as np
import torch

from resfit.rl_finetuning.config.rlpd import ActorConfig
from resfit.rl_finetuning.off_policy.rl.actor import Actor
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.pi05_residual.config import Pi05ResidualConfig
from resfit.pi05_residual.collect import _residual_probe
from resfit.pi05_residual.env import Pi05ResidualEnv
from resfit.pi05_residual.keyboard import KeyboardListener
from resfit.pi05_residual.train import _add_episode, _ingest_new_replay_files, _learner_step_budget


def test_learner_budget_only_grows_with_online_transitions():
    assert _learner_step_budget(1000, 1) == 1000
    assert _learner_step_budget(1001, 1) == 1001
    assert _learner_step_budget(1000, 2) == 2000


def test_human_frames_remain_online_and_only_executed_frames_enter_expert():
    n = 4
    data = {
        "state": np.zeros((n, 2), dtype=np.float32),
        "next_state": np.zeros((n, 2), dtype=np.float32),
        "base_action": np.zeros((n, 2), dtype=np.float32),
        "next_base_action": np.zeros((n, 2), dtype=np.float32),
        "action": np.array([[0.1, 0], [0.2, 0], [0.3, 0], [0.4, 0]], dtype=np.float32),
        "intervened": np.array([False, True, True, False]),
        "reward": np.zeros(n, dtype=np.float32),
        "terminated": np.zeros(n, dtype=bool),
        "truncated": np.zeros(n, dtype=bool),
    }
    class Replay(list):
        add = list.append

    online, expert = Replay(), Replay()
    assert _add_episode(data, online, (), expert) == n
    assert len(online) == 4
    assert len(expert) == 2
    assert expert[0]["action"].squeeze(0)[0].item() == pytest.approx(0.2)
    assert not expert[0]["next", "done"].item()
    assert expert[1]["next", "done"].item()


def test_human_absolute_action_uses_pi_reference_without_clipping():
    cfg = Pi05ResidualConfig(
        pi05_checkpoint="unused", task="unused", state_dim=2, action_dim=2,
        chunk_size=30, rl_chunk_size=10, rl_image_size=4, image_size=8,
        joint_delta_limit=(2.0, 4.0),
    )
    env = Pi05ResidualEnv(_ToyEnv(), _SlowPolicy(), cfg)
    obs, _ = env.reset()
    reference = env._base_reference_action.numpy()[0]
    target = reference + np.array([3.0, -6.0], dtype=np.float32)
    next_obs, action, used_reference = env.observe_intervention(env.env._obs(), target)
    np.testing.assert_allclose(used_reference, reference)
    np.testing.assert_allclose(action, [0.5, -0.5])
    assert "observation.base_action" in next_obs
    with pytest.raises(ValueError, match="normalized action range"):
        env.normalize_executed_action(reference + np.array([7.0, 0.0], dtype=np.float32), reference)


def test_learner_loads_new_disk_episode_once_when_grpc_is_stalled(tmp_path, monkeypatch):
    import resfit.pi05_residual.train as train_module

    np.savez(tmp_path / "episode_000021.npz", reward=np.zeros(3, dtype=np.float32))
    (tmp_path / "episode_000021.json").write_text(json.dumps({
        "episode_id": 21, "source": "pi05_residual_rl_episode",
        "transitions": 3, "schema_version": 7,
    }))
    replay = []
    known = set()
    monkeypatch.setattr(train_module, "_add_episode", lambda data, buffer, cameras: buffer.append(3) or 3)

    assert _ingest_new_replay_files(str(tmp_path), {"schema_version": 7}, known, replay, ()) == 3
    assert _ingest_new_replay_files(str(tmp_path), {"schema_version": 7}, known, replay, ()) == 0
    assert known == {21}
    assert replay == [3]


def test_residual_probe_converts_physical_degrees_to_normalized_delta():
    cfg = Pi05ResidualConfig(pi05_checkpoint="unused", task="unused")
    residual = _residual_probe(cfg, joint=0, degrees=0.5)
    physical = residual * torch.as_tensor(cfg.rl_joint_delta_limit)
    assert torch.rad2deg(physical[0]) == pytest.approx(0.5)
    assert torch.count_nonzero(residual).item() == 1


def test_actor_tanh_scales_each_joint_and_starts_at_zero():
    cfg = ActorConfig(
        feature_dim=8, hidden_dim=8, num_layers=1, orth=0,
        actor_last_layer_init_scale=0.0,
        action_scale=(0.25, 0.5),
    )
    actor = Actor(4, 4, 2, 2, cfg, residual_actor=True)
    obs = {
        "feat": torch.zeros(1, 4),
        "observation.state": torch.zeros(1, 2),
        "observation.base_action": torch.zeros(1, 2),
    }
    assert torch.allclose(actor(obs, 0.0).mean, torch.zeros(1, 2))
    final_linear = next(m for m in reversed(list(actor.policy)) if isinstance(m, torch.nn.Linear))
    with torch.no_grad():
        final_linear.bias.copy_(torch.tensor([10.0, -10.0]))
    assert torch.allclose(actor(obs, 0.0).mean, torch.tensor([[0.25, -0.5]]), atol=1e-6)


def test_frozen_encoder_stays_in_eval_mode_during_training():
    agent = QAgent.__new__(QAgent)
    torch.nn.Module.__init__(agent)
    agent.cfg = SimpleNamespace(freeze_encoder=True)
    agent.encoders = torch.nn.ModuleList([torch.nn.BatchNorm1d(2)])
    agent.actor = torch.nn.Linear(2, 2)
    agent.critic = torch.nn.Linear(2, 1)
    agent.critic_target = torch.nn.Linear(2, 1).eval()
    agent.bc_policies = []
    agent.train(True)
    assert not agent.encoders.training
    assert agent.actor.training
    assert agent.critic.training


class _SlowPolicy:
    def __init__(self):
        self.calls = 0
        self.thread_ids = []
        self.state_mean = torch.tensor([10.0, 20.0])
        self.state_std = torch.tensor([2.0, 4.0])

    def action_chunk(self, obs):
        self.calls += 1
        self.thread_ids.append(threading.get_ident())
        time.sleep(0.001)
        state = obs["observation.state"]
        return state[:, None] + torch.tensor([[[1.0, 2.0]]]).expand(1, 30, 2)


class _ToyEnv:
    def __init__(self):
        self.state = torch.tensor([10.0, 20.0])

    def _obs(self):
        image = torch.zeros(8, 8, 3, dtype=torch.uint8)
        return {
            "observation.state": self.state.clone(),
            "observation.images.front": image,
            "observation.images.left_wrist": image,
            "observation.images.right_wrist": image,
        }

    def reset(self):
        return self._obs(), {}

    def step(self, action):
        self.state = torch.as_tensor(action, dtype=torch.float32)
        return self._obs(), 0.0, False, False, {}


def test_rl_observation_is_standardized_but_pi_input_stays_raw():
    cfg = Pi05ResidualConfig(
        pi05_checkpoint="unused", task="unused", state_dim=2, action_dim=2,
        chunk_size=30, rl_chunk_size=10, rl_image_size=4, image_size=8,
        joint_delta_limit=(2.0, 4.0),
    )
    policy = _SlowPolicy()
    env = Pi05ResidualEnv(_ToyEnv(), policy, cfg)
    obs, _ = env.reset()
    assert torch.allclose(obs["observation.state"], torch.zeros(1, 2))
    residual = torch.tensor([0.005, -0.005])
    refreshes = []
    for _ in range(5):
        _, _, _, _, info = env.step(residual)
        refreshes.append(info["chunk_refreshed"])
        assert float(info["residual_action"].abs().max()) <= 0.005 + 1e-6
        assert float(info["scaled_action"].abs().max()) <= 1.0 + 1e-6
    assert policy.calls == 1
    assert sum(refreshes) == 0
    assert any(info_value > 0.0 for info_value in [0.001])


def test_chunk_refresh_timing_is_reported_on_the_tenth_step():
    cfg = Pi05ResidualConfig(
        pi05_checkpoint="unused", task="unused", state_dim=2, action_dim=2,
        chunk_size=30, rl_chunk_size=10, rl_image_size=4, image_size=8,
        joint_delta_limit=(2.0, 4.0),
    )
    policy = _SlowPolicy()
    env = Pi05ResidualEnv(_ToyEnv(), policy, cfg)
    env.reset()
    infos = [env.step(torch.zeros(2))[-1] for _ in range(10)]
    assert [info["chunk_refreshed"] for info in infos] == [False] * 9 + [True]
    assert infos[-1]["base_inference_seconds"] > 0.0
    assert infos[-1]["base_prefetch_wait_seconds"] >= 0.0
    assert policy.calls == 2
    assert policy.thread_ids[0] == threading.get_ident()
    assert policy.thread_ids[1] != threading.get_ident()


def test_ctrl_1_emits_one_reset_event():
    listener = KeyboardListener.__new__(KeyboardListener)
    listener._events = deque()
    listener._lock = threading.Lock()
    listener._ctrl_down = False
    listener.ecodes = SimpleNamespace(
        EV_KEY=1,
        bytype={1: {29: "KEY_LEFTCTRL", 2: "KEY_1"}},
    )
    events = [
        SimpleNamespace(type=1, code=29, value=1),
        SimpleNamespace(type=1, code=2, value=1),
        SimpleNamespace(type=1, code=29, value=0),
    ]
    listener.device = SimpleNamespace(read_loop=lambda: iter(events))
    listener._loop()
    assert listener.pop() == ["reset"]


def test_environment_rejects_unscaled_actor_output_instead_of_clipping_it():
    cfg = Pi05ResidualConfig(
        pi05_checkpoint="unused", task="unused", state_dim=2, action_dim=2,
        chunk_size=30, rl_chunk_size=10, joint_delta_limit=(2.0, 4.0),
    )
    env = Pi05ResidualEnv(_ToyEnv(), _SlowPolicy(), cfg)
    env.reset()
    with pytest.raises(ValueError, match="five-degree"):
        env.step(torch.tensor([1.0, -1.0]))
