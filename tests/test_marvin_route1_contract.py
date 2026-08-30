from __future__ import annotations

import json
import queue

import numpy as np
import pytest
import torch

from resfit.rl_finetuning.marvin.replay import atomic_write_episode, inspect_replay_directory, usable_nstep_transitions
from resfit.rl_finetuning.config.rlpd import VitEncoderConfig
from resfit.rl_finetuning.off_policy.networks.encoder import VitEncoder
from resfit.rl_finetuning.utils.normalization import ActionScaler
from lerobot.scripts.rl.learner_service import LearnerService
from resfit.rl_finetuning.marvin.replay import critic_action


def test_marvin_scaler_never_expands_past_physical_box():
    scaler = ActionScaler(
        torch.full((5,), -1.0),
        torch.full((5,), 1.0),
        action_scale=0.1,
        absolute_min=-1.0,
        absolute_max=1.0,
    )
    raw_act = torch.tensor([[0.0, 0.5, -0.5, 1.0775393, -1.2]])
    scaled = scaler.scale(raw_act)
    command = scaler.unscale(scaled)
    assert torch.all(command <= 1.0)
    assert torch.all(command >= -1.0)
    torch.testing.assert_close(command[0, 3:], torch.tensor([1.0, -1.0]))


def test_vit_encoder_uses_marvin_image_size():
    encoder = VitEncoder((3, 128, 128), VitEncoderConfig())
    assert encoder.vit.num_patches == 225
    assert encoder(torch.zeros(2, 3, 128, 128)).shape == (2, 225 * 128)


def test_replay_manifest_identity_and_atomic_write(tmp_path):
    identity = {
        "schema_version": 4,
        "coordinate_frame": "resfit_marvin_bounded_v2",
        "action_contract": "marvin_normalized_command_box_v1",
    }
    arrays = {"reward": np.asarray([0.0, 1.0], dtype=np.float32)}
    manifest = {**identity, "episode_id": 0, "transitions": 2}
    atomic_write_episode(tmp_path, 0, arrays, manifest)
    episodes, transitions, next_episode_id = inspect_replay_directory(tmp_path, identity)
    assert len(episodes) == 1
    assert transitions == 2
    assert next_episode_id == 1
    assert usable_nstep_transitions(episodes, n_step=2) == 1

    manifest_path = tmp_path / "episode_000000.json"
    corrupted = json.loads(manifest_path.read_text())
    corrupted["coordinate_frame"] = "obsolete"
    manifest_path.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError, match="identity mismatch"):
        inspect_replay_directory(tmp_path, identity)


def test_empty_parameter_stream_waits_instead_of_busy_spinning():
    class StopAfterWait:
        def __init__(self):
            self.stopped = False
            self.waits = []

        def is_set(self):
            return self.stopped

        def wait(self, timeout):
            self.waits.append(timeout)
            self.stopped = True

    shutdown = StopAfterWait()
    service = LearnerService(
        shutdown_event=shutdown,
        parameters_queue=queue.Queue(),
        seconds_between_pushes=1.0,
        transition_queue=queue.Queue(),
        interaction_message_queue=queue.Queue(),
        queue_get_timeout=0.0,
    )

    assert list(service.StreamParameters(None, None)) == []
    assert shutdown.waits == [1.0]


def test_critic_action_prefers_applied_pre_smoothing_action():
    data = {
        "command_action": np.full((1, 5), 0.2, dtype=np.float32),
        "actual_action": np.full((1, 5), 0.9, dtype=np.float32),
        "applied_action": np.full((1, 5), 0.8, dtype=np.float32),
    }
    np.testing.assert_allclose(critic_action(data, 0), np.full(5, 0.8))
    data.pop("applied_action")
    np.testing.assert_allclose(critic_action(data, 0), np.full(5, 0.9))
    data.pop("actual_action")
    np.testing.assert_allclose(critic_action(data, 0), np.full(5, 0.2))
