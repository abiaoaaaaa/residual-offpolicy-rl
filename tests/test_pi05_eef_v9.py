import numpy as np
from scipy.spatial.transform import Rotation

from resfit.pi05_residual.eef_v9 import (
    EEFV9Codec,
    EEFV9Limits,
    EEFResidualFilter,
    StatefulNumericalIK,
    apply_base_delta,
    eef_pose_delta,
)
from resfit.pi05_residual.eef_runtime import EEFV9Planner


def fake_fk(q):
    q = np.asarray(q, dtype=np.float64)
    pose = np.zeros((2, 7), dtype=np.float32)
    pose[0, :3] = q[:3]
    pose[0, 3:7] = Rotation.from_rotvec(q[3:6]).as_quat()
    pose[1, :3] = q[7:10]
    pose[1, 3:7] = Rotation.from_rotvec(q[10:13]).as_quat()
    return pose


def make_pose(left_xyz, right_xyz):
    pose = np.zeros((2, 7), dtype=np.float32)
    pose[:, 3:7] = [0, 0, 0, 1]
    pose[0, :3], pose[1, :3] = left_xyz, right_xyz
    return pose


def test_chunk_delta_uses_measured_pose_then_previous_01s_reference():
    measured = make_pose([0, 0, 0], [1, 0, 0])
    chunk = np.stack([
        make_pose([0.01, 0, 0], [1.01, 0, 0]),
        make_pose([0.02, 0, 0], [1.02, 0, 0]),
        make_pose([0.05, 0, 0], [1.05, 0, 0]),
    ])
    delta = eef_pose_delta(
        np.concatenate([measured[None], chunk[:-1]], axis=0), chunk
    )
    np.testing.assert_allclose(delta[:, 0], [.01, .01, .03], atol=1e-7)
    codec = EEFV9Codec()
    normalized = codec.encode_chunk(measured, chunk)
    np.testing.assert_allclose(normalized[:, 0], [1.0, 1.0, 1.0], atol=1e-6)


def test_zero_residual_preserves_absolute_reference():
    reference = make_pose([.1, .2, .3], [1.1, .2, .3])
    codec = EEFV9Codec()
    actual = codec.apply_residual(reference, np.zeros(12, dtype=np.float32))
    np.testing.assert_allclose(actual[0], reference, atol=1e-7)


def test_filter_limits_physical_step():
    limits = EEFV9Limits(translation_m=.005, rotation_rad=.02)
    filt = EEFResidualFilter(limits, alpha=1.0)
    output = filt(np.ones(12, dtype=np.float32))
    np.testing.assert_allclose(output, np.ones(12), atol=1e-6)
    output2 = filt(-np.ones(12, dtype=np.float32))
    np.testing.assert_allclose(output2, np.zeros(12), atol=1e-6)


def test_stateful_ik_uses_previous_seed_and_round_trips():
    q = np.array([.1, .2, .3, .01, -.02, .03, 0, 1.1, 1.2, 1.3, -.01, .02, -.03, 0])
    ik = StatefulNumericalIK(fake_fk)
    ik.reset(q)
    solved, error, success = ik.solve(fake_fk(q))
    np.testing.assert_allclose(solved, q, atol=1e-7)
    assert success and error < 1e-8


def test_end_to_end_planner_zero_residual_tracks_selected_pi_targets():
    current = np.zeros(14, dtype=np.float64)
    chunk = np.stack([current + 0.001 * i for i in range(30)])
    planner = EEFV9Planner(fake_fk)
    base_delta, solved, steps = planner.plan(
        current, chunk, np.zeros((10, 12), dtype=np.float32)
    )
    np.testing.assert_equal(base_delta.shape, (10, 12))
    for got, target in zip(solved, chunk[2::3]):
        np.testing.assert_allclose(fake_fk(got), fake_fk(target), atol=1e-5)
    assert len(steps) == 10
    assert max(step.ik_error for step in steps) < 1e-6
    assert all(step.ik_success for step in steps)
