from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch


MARVIN_CAMERAS = ("side_policy", "wrist_1buchong", "wrist_1", "wrist_2")
MARVIN_STATE_KEYS = (
    "gripper_pose.pos", "last_action.dx.pos", "last_action.dy.pos", "last_action.dz.pos", "last_action.drz.pos", "last_action.gripper.pos",
    "tcp_force.x.pos", "tcp_force.y.pos", "tcp_force.z.pos", "tcp_pose.x.pos", "tcp_pose.y.pos", "tcp_pose.z.pos", "tcp_pose.rx.pos", "tcp_pose.ry.pos", "tcp_pose.rz.pos",
    "tcp_torque.x.pos", "tcp_torque.y.pos", "tcp_torque.z.pos", "tcp_vel.x.pos", "tcp_vel.y.pos", "tcp_vel.z.pos", "tcp_vel.rx.pos", "tcp_vel.ry.pos", "tcp_vel.rz.pos",
)


@dataclass
class MarvinResidualConfig:
    base_policy_checkpoint: str
    base_policy_chunk_size: int = 20
    base_policy_n_action_steps: int = 5
    marvin_root: str = "/home/xlb/code/强化微调ACT"
    device: str = "cuda"
    task: str = "Insert the USB plug"
    host: str = "127.0.0.1"
    port: int = 8765
    max_steps: int = 200
    # Optional real-robot safety cap in *normalized residual-action* units.
    # None follows the published ResFiT path: the actor controls the residual
    # magnitude and ActionScaler clips only the combined normalized action.
    residual_scale: float | None = None
    contact_start_step: int = 0
    enable_gripper_residual: bool = False
    allow_real_robot: bool = False
    replay_dir: str = ""
    residual_checkpoint: str = ""
    training_checkpoint_dir: str = ""
    resume_checkpoint: str = ""
    minimum_replay_transitions: int = 10_000
    learner_only: bool = False
    learner_updates_per_run: int = 500
    training_checkpoint_keep_last: int = 2
    async_grpc: bool = False
    grpc_host: str = "127.0.0.1"
    grpc_port: int = 50051
    grpc_parameter_push_s: float = 1.0
    learner_gpu_memory_fraction: float = 0.8
    actor_publish_interval: int = 50
    learner_update_sleep_s: float = 0.01


class MarvinACTBasePolicy:
    """Load a route-1 ACT checkpoint with ResFiT's bundled LeRobot 0.1.0."""

    def __init__(self, cfg: MarvinResidualConfig):
        from resfit.lerobot.policies.act.modeling_act import ACTPolicy

        checkpoint = Path(cfg.base_policy_checkpoint).expanduser().resolve()
        if (checkpoint / "policy" / "config.json").is_file():
            checkpoint = checkpoint / "policy"
        if not (checkpoint / "config.json").is_file():
            raise FileNotFoundError(
                f"ACT checkpoint must contain config.json (or policy/config.json): {checkpoint}"
            )
        self.device = torch.device(cfg.device)
        self.policy = ACTPolicy.from_pretrained(checkpoint).to(self.device)
        checkpoint_chunk_size = int(self.policy.config.chunk_size)
        if checkpoint_chunk_size != int(cfg.base_policy_chunk_size):
            raise ValueError(
                "ACT checkpoint chunk_size mismatch: "
                f"checkpoint={checkpoint_chunk_size}, expected={cfg.base_policy_chunk_size}"
            )
        if not 1 <= int(cfg.base_policy_n_action_steps) <= checkpoint_chunk_size:
            raise ValueError(
                "base_policy_n_action_steps must be in [1, chunk_size], got "
                f"{cfg.base_policy_n_action_steps} for chunk_size={checkpoint_chunk_size}"
            )
        self.policy.config.n_action_steps = int(cfg.base_policy_n_action_steps)
        self.policy.eval()
        self.config = self.policy.config
        print(
            "Marvin ACT receding horizon: "
            f"predict={checkpoint_chunk_size} actions, execute={self.config.n_action_steps} actions per inference"
        )

    def reset(self, env_ids=None):
        self.policy.reset(env_ids=env_ids)

    def action(self, observation: dict) -> np.ndarray:
        state = np.asarray(observation.get("state"), dtype=np.float32).reshape(24)
        batch = {"observation.state": torch.from_numpy(state).unsqueeze(0).to(self.device)}
        for key in MARVIN_CAMERAS:
            image = np.asarray(observation[key], dtype=np.uint8)
            if image.ndim != 3:
                raise ValueError(f"Expected a 3-D image for {key}, got {image.shape}")
            if image.shape[-1] == 3:
                image = image.transpose(2, 0, 1)
            batch[f"observation.images.{key}"] = (
                torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0).to(self.device).float() / 255.0
            )
        with torch.inference_mode():
            return self.policy.select_action(batch)[0].detach().cpu().numpy().astype(np.float32)

    def select_action(self, batch: dict) -> torch.Tensor:
        """Compatibility method used by the upstream offline-buffer builder."""
        policy_batch = {key: batch[key].to(self.device) for key in self.policy.config.input_features}
        with torch.inference_mode():
            return self.policy.select_action(policy_batch)


class MarvinResidualVecEnv:
    """Single-instance vector API expected by the Residual TD3 learner."""

    num_envs = 1

    def __init__(self, cfg: MarvinResidualConfig, base_policy=None, action_scaler=None, state_standardizer=None):
        from lerobot_robot_marvin.client import MarvinBridgeClient
        from lerobot_robot_marvin.heartbeat import BridgeHeartbeat

        self.cfg = cfg
        self.client = MarvinBridgeClient(cfg.host, cfg.port)
        self._heartbeat_client = MarvinBridgeClient(cfg.host, cfg.port)
        self._heartbeat = BridgeHeartbeat(self._heartbeat_client)
        self.base_policy = base_policy or MarvinACTBasePolicy(cfg)
        if action_scaler is None or state_standardizer is None:
            raise ValueError("MarvinResidualVecEnv requires the offline-data ActionScaler and StateStandardizer")
        self.action_scaler = action_scaler
        self.state_standardizer = state_standardizer
        self.step_count = 0
        self._base_action = np.zeros(5, dtype=np.float32)
        self._connected = False
        self.metadata = {"horizon": cfg.max_steps}
        self.vec_env = self
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1, 5), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "observation.state": gym.spaces.Box(-np.inf, np.inf, shape=(1, 24), dtype=np.float32),
            "observation.base_action": gym.spaces.Box(-1.0, 1.0, shape=(1, 5), dtype=np.float32),
            **{f"observation.images.{key}": gym.spaces.Box(0, 255, shape=(1, 3, 128, 128), dtype=np.uint8) for key in MARVIN_CAMERAS},
        })

    def _obs(self, raw: dict, base_action: np.ndarray) -> dict[str, torch.Tensor]:
        raw_state = torch.from_numpy(np.asarray(raw["state"], np.float32)).reshape(1, 24)
        raw_base_action = torch.from_numpy(np.asarray(base_action, np.float32)).reshape(1, 5)
        result = {
            "observation.state": self.state_standardizer.standardize(raw_state).cpu(),
            "observation.base_action": self.action_scaler.scale(raw_base_action).cpu(),
        }
        for key in MARVIN_CAMERAS:
            image = np.asarray(raw[key], dtype=np.uint8).transpose(2, 0, 1)
            result[f"observation.images.{key}"] = torch.from_numpy(image).unsqueeze(0)
        return result

    def reset(self, **_kwargs):
        if not self._connected:
            status = self.client.status()
            if status.get("backend") == "hardware" and not self.cfg.allow_real_robot:
                raise RuntimeError("Hardware bridge detected; set marvin.allow_real_robot=true only after mock validation")
            self.client.connect()
            self._connected = True
            self._heartbeat.start()
        raw, info = self.client.reset()
        self._heartbeat.note_primary_contact()
        self.base_policy.reset()
        self.step_count = 0
        self._base_action = self.base_policy.action(raw)
        self._heartbeat.check()
        return self._obs(raw, self._base_action), {"reset_info": info}

    def step(self, residual_action: torch.Tensor):
        if isinstance(residual_action, torch.Tensor):
            residual = residual_action.detach().cpu().numpy().reshape(5)
        else:
            residual = np.asarray(residual_action, dtype=np.float32).reshape(5)
        bounded = residual.copy()
        if self.cfg.residual_scale is not None:
            bounded = np.clip(bounded, -self.cfg.residual_scale, self.cfg.residual_scale)
        if not self.cfg.enable_gripper_residual:
            bounded[4] = 0.0
        if self.step_count < self.cfg.contact_start_step:
            bounded[:] = 0.0
        base = self._base_action.copy()
        base_normalized = self.action_scaler.scale(torch.from_numpy(base).reshape(1, 5)).cpu().numpy().reshape(5)
        combined_normalized = np.clip(base_normalized + bounded, -1.0, 1.0).astype(np.float32)
        command = (
            self.action_scaler.unscale(torch.from_numpy(combined_normalized).reshape(1, 5))
            .cpu().numpy().reshape(5).astype(np.float32)
        )
        if not np.all(np.isfinite(command)):
            raise ValueError(f"Marvin command contains NaN or Inf: {command.tolist()}")
        if np.any(command < -1.0) or np.any(command > 1.0):
            raise ValueError(
                "Marvin ActionScaler violated the hard command box [-1, 1]; "
                f"command={command.tolist()}, limits={self.action_scaler.limits}"
            )
        self._heartbeat.check()
        transition = self.client.step(command)
        self._heartbeat.note_primary_contact()
        self.step_count += 1
        episode_steps = self.step_count
        terminated = bool(transition.get("terminated", False))
        truncated = bool(transition.get("truncated", False) or (self.step_count >= self.cfg.max_steps and not terminated))
        done = terminated or truncated
        raw_next = transition["observation"]
        final_raw = raw_next
        # Compute the base action for the actual terminal observation before
        # resetting. Otherwise final_obs would contain the reset state's base
        # action and corrupt (s, a_base, a, s', a'_base, r, done).
        next_base_action = self.base_policy.action(raw_next)
        self._heartbeat.check()
        # Apply the task-level binary mapping at the residual layer so the
        # same rule is used for Bridge data and existing replay data.
        raw_reward = float(transition.get("reward", 0.0))
        reward = 1.0 if raw_reward > 0.8 else 0.0
        success = bool(reward == 1.0)
        if done:
            next_obs, _ = self.reset()
        else:
            self._base_action = next_base_action
            next_obs = self._obs(raw_next, self._base_action)
        info = {
            "scaled_action": torch.from_numpy(combined_normalized).reshape(1, 5),
            "residual_action": torch.from_numpy(bounded).reshape(1, 5),
            "actual_action": np.asarray(transition.get("actual_action", command), np.float32),
            "next_base_action": self._base_action.copy(),
            "final_obs": [self._obs(final_raw, next_base_action)] if done else [None],
            "final_info": {"episode_steps": np.asarray([episode_steps]), "_episode_steps": np.asarray([0]), "succeed": success},
            "succeed": success,
            "raw_reward": raw_reward,
        }
        return next_obs, torch.tensor([reward]), torch.tensor([terminated]), torch.tensor([truncated]), info

    def close(self):
        if self._connected:
            self._heartbeat.close()
            try:
                self.client.disconnect()
            finally:
                self._connected = False

    def seed(self, _seed=None):
        return None
