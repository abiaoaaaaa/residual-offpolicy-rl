import numpy as np
import pytest

from resfit.pi05_residual.config import Pi05ResidualConfig
from resfit.pi05_residual.vlahost import VLAHostClient, VLAHostConfig, _BridgeBackend


class _Bridge:
    def __init__(self):
        self.cfg = VLAHostConfig(step_frequency=10.0)
        self.state = np.arange(16, dtype=np.float32)
        self.sent = []

    def reset(self):
        pass

    def observation(self):
        return {"observation.state": self.state.copy()}, {}

    def action_chunk(self, actions, *, source_hz):
        self.sent.append((actions.copy(), source_hz))
        self.state = actions[-1].copy()
        return {"success": True, "execution": "active"}


def test_bridge_interpolates_three_30hz_waypoints_per_10hz_step():
    cfg = Pi05ResidualConfig(pi05_checkpoint="unused", task="unused")
    bridge = _Bridge()
    backend = _BridgeBackend(bridge, cfg)
    backend.reset(go_home=False)
    target = bridge.state + 0.21
    next_obs, _, _, _, info = backend.step(target)
    waypoints, source_hz = bridge.sent[0]
    np.testing.assert_allclose(waypoints, np.stack([
        np.arange(16) + 0.07, np.arange(16) + 0.14, target,
    ]), atol=1e-6)
    assert source_hz == 30.0
    np.testing.assert_allclose(info["bridge_action_chunk"], waypoints)
    np.testing.assert_allclose(next_obs["observation.state"], target)


def test_bridge_limits_each_arm_waypoint_like_lerobot_driver():
    cfg = Pi05ResidualConfig(pi05_checkpoint="unused", task="unused")
    bridge = _Bridge()
    backend = _BridgeBackend(bridge, cfg)
    backend.reset(go_home=False)
    backend.step(bridge.state + 1.0)
    waypoints, _ = bridge.sent[0]
    arm_motion = np.diff(np.vstack([np.arange(14, dtype=np.float32), waypoints[:, :14]]), axis=0)
    assert np.max(np.abs(arm_motion)) <= 0.08 + 1e-6
    np.testing.assert_allclose(waypoints[-1, :14], np.arange(14) + 0.24, atol=1e-6)
    np.testing.assert_allclose(waypoints[-1, 14:], np.arange(14, 16) + 1.0)


def test_bridge_anchors_next_chunk_to_previous_command_not_lagging_feedback():
    class _LagBridge(_Bridge):
        def action_chunk(self, actions, *, source_hz):
            self.sent.append((actions.copy(), source_hz))
            return {"success": True, "execution": "active"}

    cfg = Pi05ResidualConfig(pi05_checkpoint="unused", task="unused")
    bridge = _LagBridge()
    initial = bridge.state.copy()
    backend = _BridgeBackend(bridge, cfg)
    backend.reset(go_home=False)
    backend.step(initial + 0.12)
    backend.step(initial + 0.24)
    second, _ = bridge.sent[1]
    np.testing.assert_allclose(second[0], initial + 0.16, atol=1e-6)


def test_shadow_chunk_does_not_advance_policy_command_anchor():
    class _ShadowBridge(_Bridge):
        def action_chunk(self, actions, *, source_hz):
            self.sent.append((actions.copy(), source_hz))
            return {"success": True, "execution": "shadow"}

    bridge = _ShadowBridge()
    backend = _BridgeBackend(bridge, Pi05ResidualConfig(pi05_checkpoint="unused", task="unused"))
    backend.reset(go_home=False)
    initial = backend._command_anchor.copy()
    _, _, _, _, info = backend.step(initial + 0.12)
    assert info["execution"] == "shadow"
    np.testing.assert_allclose(backend._command_anchor, initial)


def test_vlahost_chunk_posts_waypoints_and_rejects_shadow_execution():
    client = VLAHostClient(VLAHostConfig())
    captured = {}

    class _Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"success": True, "execution": "shadow"}

    def post(url, *, json, timeout):
        captured.update(url=url, body=json, timeout=timeout)
        return _Response()

    client.session.post = post
    with pytest.raises(RuntimeError, match="did not execute"):
        client.action_chunk(np.zeros((3, 16), dtype=np.float32), source_hz=30.0)
    assert captured["url"] == "http://127.0.0.1:8010/action_chunk"
    assert captured["body"]["source_hz"] == 30.0
    assert len(captured["body"]["actions"]) == 3
    assert set(captured["body"]["actions"][0]) == {
        "jointcmd_left", "jointcmd_right", "gripper_left", "gripper_right",
    }


def test_vlahost_reset_opens_grippers_after_go_home():
    client = VLAHostClient(VLAHostConfig())
    calls = []

    class _Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"success": True}

    def post(url, **kwargs):
        calls.append((url, kwargs.get("json")))
        return _Response()

    client.session.post = post
    client.reset()
    assert calls == [
        ("http://127.0.0.1:8010/go_home", None),
        ("http://127.0.0.1:8010/action", {"gripper_left": 0.0, "gripper_right": 0.0}),
    ]


def test_vlahost_intervention_requires_generation_and_never_treats_shadow_as_executed():
    client = VLAHostClient(VLAHostConfig())
    client._intervention_enabled = True
    client._observation_generation = 9
    captured = {}

    class _Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"success": True, "execution": "shadow"}

    def post(url, *, json, timeout):
        captured.update(json)
        return _Response()

    client.session.post = post
    result = client.action_chunk(np.zeros((3, 16), dtype=np.float32), source_hz=30.0)
    assert captured["observation_generation"] == 9
    assert result["execution"] == "shadow"


def test_executed_human_action_requires_active_fresh_mux_output():
    snapshot = {
        "active": True, "intervene_flag": True, "source": "human",
        "phase": "vr", "executed_action_fresh": True,
        "executed_action": [0.05] * 16,
    }
    np.testing.assert_allclose(VLAHostClient.executed_intervention_action(snapshot), [0.05] * 16)
    for key, value in (("active", False), ("executed_action_fresh", False), ("source", "policy")):
        invalid = {**snapshot, key: value}
        with pytest.raises(ValueError, match="no fresh executed human action"):
            VLAHostClient.executed_intervention_action(invalid)


def test_vlahost_release_submits_hold_from_current_state_and_waits_for_policy():
    client = VLAHostClient(VLAHostConfig())
    client._intervention_enabled = True
    sent = []
    states = iter([
        {"joint_states": {"positions": [0.1] * 14},
         "gripper_left": {"position": 0.0}, "gripper_right": {"position": 0.0},
         "intervention": {"phase": "waiting_policy", "footkey": 0, "refresh_generation": 4}},
        {"intervention": {"phase": "policy", "footkey": 0, "refresh_generation": 4}},
    ])

    class _Response:
        def __init__(self, value):
            self.value = value

        def raise_for_status(self):
            pass

        def json(self):
            return self.value

    client.session.get = lambda *_args, **_kwargs: _Response(next(states))
    def post(_url, *, json, **_kwargs):
        sent.append(json)
        return _Response({"success": True, "execution": "resume_pending"})
    client.session.post = post

    assert client.wait_for_policy_control(timeout=1.0)
    assert sent[0]["observation_generation"] == 4
    assert sent[0]["actions"][0]["jointcmd_left"] == pytest.approx([0.1] * 7)
