from pathlib import Path

from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer

from resfit.rl_finetuning.marvin.replay import (
    cached_episode_ids,
    manifest_sha1,
    recover_pending_cache_metadata,
    replay_manifest_fingerprint,
)
from resfit.rl_finetuning.utils.rb_transforms import MultiStepTransform, get_multi_step_transform


def _episodes(count: int):
    return [
        (Path(f"episode_{episode_id:06d}.npz"), {"episode_id": episode_id, "transitions": 3})
        for episode_id in range(count)
    ]


def test_incremental_metadata_accepts_appended_episodes():
    episodes = _episodes(3)
    metadata = {
        "episode_manifest_sha1": {
            str(manifest["episode_id"]): manifest_sha1(manifest)
            for _path, manifest in episodes[:2]
        }
    }
    assert cached_episode_ids(metadata, episodes) == {0, 1}


def test_incremental_metadata_rejects_changed_episode():
    episodes = _episodes(2)
    metadata = {"episode_manifest_sha1": {"0": manifest_sha1(episodes[0][1])}}
    episodes[0][1]["transitions"] = 4
    assert cached_episode_ids(metadata, episodes) is None


def test_legacy_cache_is_accepted_as_replay_prefix():
    episodes = _episodes(3)
    prefix = [manifest for _path, manifest in episodes[:2]]
    metadata = {
        "replay_manifest_fingerprint": replay_manifest_fingerprint(prefix),
        "raw_replay_transitions": 6,
    }
    assert cached_episode_ids(metadata, episodes) == {0, 1}


def test_pending_commit_is_recovered(tmp_path):
    (tmp_path / "storage").mkdir()
    (tmp_path / "writer").mkdir()
    (tmp_path / "storage" / "storage_metadata.json").write_text('{"len": 9}')
    (tmp_path / "writer" / "metadata.json").write_text('{"cursor": 9}')
    (tmp_path / ".user_metadata.pending.json").write_text('{"buffer_size": 9}')
    assert recover_pending_cache_metadata(tmp_path) == "committed"
    assert (tmp_path / "user_metadata.json").is_file()


def test_pending_commit_requires_transform_context(tmp_path):
    (tmp_path / "storage").mkdir()
    (tmp_path / "writer").mkdir()
    (tmp_path / "storage" / "storage_metadata.json").write_text('{"len": 9}')
    (tmp_path / "writer" / "metadata.json").write_text('{"cursor": 9}')
    (tmp_path / ".user_metadata.pending.json").write_text(
        '{"buffer_size": 9, "transform_buffer": "transform_buffer_00000009.pt"}'
    )
    assert recover_pending_cache_metadata(tmp_path) == "invalid"


def test_get_multi_step_transform_unwraps_torchrl_compose():
    expected = MultiStepTransform(n_steps=3, gamma=0.99)
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(10),
        transform=expected,
    )
    assert get_multi_step_transform(replay_buffer) is replay_buffer._transform[0]
