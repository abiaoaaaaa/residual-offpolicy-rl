"""LeRobot-VLAHost compatible HTTP rollout adapter for pi0.5 residual RL."""

from __future__ import annotations

import base64
import threading
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

import cv2
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import Pi05ResidualConfig
from .env import Pi05ResidualEnv
from .policy import Pi05BasePolicy


@dataclass
class VLAHostConfig:
    server_url: str = "http://127.0.0.1:8010"
    timeout: float = 2.0
    go_home_timeout: float = 45.0
    image_split_mode: str = "hero3"
    step_frequency: float = 10.0
    max_steps: int = 300
    task_description: str = ""


def _decode_quad(session, base_url: str, payload: dict, timeout: float) -> np.ndarray:
    if "data" in payload:
        raw = base64.b64decode(payload["data"])
    else:
        response = session.get(
            urljoin(base_url.rstrip("/") + "/", str(payload["stream_url"]).lstrip("/")),
            stream=True,
            timeout=timeout,
        )
        response.raise_for_status()
        raw_buffer = bytearray()
        found_start = False
        try:
            for chunk in response.iter_content(chunk_size=4096):
                if not chunk:
                    continue
                if not found_start:
                    start = chunk.find(b"\xff\xd8")
                    if start < 0:
                        continue
                    raw_buffer.extend(chunk[start:])
                    found_start = True
                else:
                    raw_buffer.extend(chunk)
                end = raw_buffer.find(b"\xff\xd9")
                if end >= 0:
                    raw_buffer = raw_buffer[: end + 2]
                    break
        finally:
            response.close()
        if not raw_buffer:
            raise ValueError("VLAHost MJPEG stream returned no complete JPEG")
        raw = bytes(raw_buffer)
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("VLAHost quad_image could not be decoded")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _split(image: np.ndarray, mode: str) -> dict[str, np.ndarray]:
    h, w = image.shape[:2]
    if mode == "hero3":
        top = int(round(h * 2 / 3))
        return {"top": image[:top], "wrist_L": image[top:, : w // 2], "wrist_R": image[top:, w // 2 :]}
    mid_h, mid_w = h // 2, w // 2
    return {"top": image[:mid_h, :mid_w], "wrist_L": image[:mid_h, mid_w:], "wrist_R": image[mid_h:, :mid_w]}


class VLAHostClient:
    """HTTP client matching the VLAHost state and action-chunk contract."""

    def __init__(self, cfg: VLAHostConfig):
        self.cfg = cfg
        self.session = self._new_session()
        self.intervention = None
        self._observation_generation = None
        self._intervention_stop = threading.Event()
        self._intervention_thread = None
        self._intervention_enabled = False
        self._intervention_error = None

    @staticmethod
    def _new_session() -> requests.Session:
        session = requests.Session()
        # State and camera reads are idempotent. Retry short bridge disconnects,
        # but never retry action POSTs because that could execute motion twice.
        retries = Retry(
            total=4,
            connect=4,
            read=4,
            backoff_factor=0.1,
            allowed_methods=frozenset({"GET"}),
        )
        session.mount("http://", HTTPAdapter(max_retries=retries))
        return session

    def _url(self, path: str) -> str:
        return urljoin(self.cfg.server_url.rstrip("/") + "/", path.lstrip("/"))

    def reset(self) -> None:
        response = self.session.post(
            self._url("go_home"), timeout=self.cfg.go_home_timeout
        )
        response.raise_for_status()
        result = response.json()
        if not result.get("success", False):
            raise RuntimeError(f"VLAHost go_home failed: {result}")
        # /go_home homes the end effector and can overwrite an earlier open.
        # Match the LeRobot HTTP driver: open after the home request returns.
        response = self.session.post(
            self._url("action"),
            json={"gripper_left": 0.0, "gripper_right": 0.0},
            timeout=self.cfg.timeout,
        )
        response.raise_for_status()
        result = response.json()
        if not result.get("success", False):
            raise RuntimeError(f"VLAHost gripper reset failed: {result}")
        # go_home plus the configured settling sleep can outlive the server's
        # keep-alive socket. Force the following /state onto a fresh connection.
        self.session.close()
        self.session = self._new_session()

    def observation(self) -> tuple[dict[str, Any], dict[str, Any]]:
        response = self.session.get(self._url("state"), timeout=self.cfg.timeout)
        response.raise_for_status()
        state = response.json()
        self.intervention = state.get("intervention")
        if isinstance(self.intervention, dict):
            self._observation_generation = self.intervention.get("refresh_generation")
        joints = state.get("joint_states", {})
        positions = np.asarray(joints.get("positions", []), dtype=np.float32)
        if positions.size != 14:
            raise ValueError(f"expected 14 joint positions, got {positions.size}")
        def gripper_position(value) -> float:
            if isinstance(value, dict):
                value = value.get("position", 0.0)
            elif isinstance(value, list):
                value = value[0] if value else 0.0
            # The Zadai Pi0.5 checkpoint defines grippers in [0, 1]. Some
            # VLAHost firmware revisions report a raw value slightly above 1.
            return float(np.clip(float(value or 0.0), 0.0, 1.0))

        grippers = np.asarray([
            gripper_position(state.get("gripper_left")),
            gripper_position(state.get("gripper_right")),
        ], dtype=np.float32)
        images = _split(
            _decode_quad(self.session, self.cfg.server_url, state["quad_image"], self.cfg.timeout),
            self.cfg.image_split_mode,
        )
        return {
            # OpenPI Zadai order: left arm 7, right arm 7, then two grippers.
            "observation.state": np.concatenate([positions, grippers]),
            "observation.images.front": images["top"],
            "observation.images.left_wrist": images["wrist_L"],
            "observation.images.right_wrist": images["wrist_R"],
        }, state

    def action(self, action: np.ndarray) -> dict[str, Any]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        payload = {"action": action.tolist()}
        if action.size == 16:
            payload = {
                "jointcmd_left": action[:7].tolist(),
                "jointcmd_right": action[7:14].tolist(),
                "gripper_left": float(np.clip(action[14], 0.0, 1.0)),
                "gripper_right": float(np.clip(action[15], 0.0, 1.0)),
            }
        response = self.session.post(self._url("action"), json=payload, timeout=self.cfg.timeout)
        response.raise_for_status()
        result = response.json()
        if not result.get("success", False):
            raise RuntimeError(f"VLAHost rejected action: {result}")
        return result

    def action_chunk(self, actions: np.ndarray, *, source_hz: float) -> dict[str, Any]:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 16 or not np.isfinite(actions).all():
            raise ValueError("VLAHost chunk requires finite [N, 16] joint targets")
        payload = {
            "source_hz": float(source_hz),
            "actions": [
                {
                    "jointcmd_left": action[:7].tolist(),
                    "jointcmd_right": action[7:14].tolist(),
                    "gripper_left": float(np.clip(action[14], 0.0, 1.0)),
                    "gripper_right": float(np.clip(action[15], 0.0, 1.0)),
                }
                for action in actions
            ],
        }
        if self._intervention_enabled:
            if self._observation_generation is None:
                raise RuntimeError("VLAHost intervention requires a fresh /state generation")
            payload["observation_generation"] = self._observation_generation
        response = self.session.post(self._url("action_chunk"), json=payload, timeout=self.cfg.timeout)
        response.raise_for_status()
        result = response.json()
        if not result.get("success", False):
            raise RuntimeError(f"VLAHost rejected action chunk: {result}")
        if result.get("execution") != "active" and not (
            self._intervention_enabled and result.get("execution") in {"shadow", "resume_pending"}
        ):
            raise RuntimeError(f"VLAHost did not execute action chunk: {result}")
        return result

    def set_intervention_session(self, enabled: bool) -> None:
        response = requests.post(
            self._url("intervention/session"),
            json={"enabled": enabled}, timeout=self.cfg.timeout,
        )
        response.raise_for_status()
        result = response.json()
        if not result.get("success", False):
            raise RuntimeError(f"VLAHost intervention session rejected: {result}")
        self._intervention_enabled = enabled

    def start_intervention_session(self) -> None:
        self.set_intervention_session(True)
        self._intervention_error = None
        self._intervention_stop.clear()

        def heartbeat():
            while not self._intervention_stop.wait(0.5):
                try:
                    self.set_intervention_session(True)
                except (requests.RequestException, RuntimeError) as exc:
                    self._intervention_error = exc
                    return

        self._intervention_thread = threading.Thread(target=heartbeat, daemon=True)
        self._intervention_thread.start()

    def assert_intervention_healthy(self) -> None:
        if not self._intervention_enabled or self._intervention_error is not None:
            raise RuntimeError(f"VLAHost intervention ownership lost: {self._intervention_error}")

    @staticmethod
    def executed_intervention_action(snapshot: dict[str, Any]) -> np.ndarray:
        """Use only fresh mux-output targets from an explicitly active VR step."""
        if not (
            isinstance(snapshot, dict)
            and snapshot.get("active") is True
            and snapshot.get("intervene_flag") is True
            and snapshot.get("source") == "human"
            and snapshot.get("phase") == "vr"
            and snapshot.get("executed_action_fresh") is True
        ):
            raise ValueError("VLAHost has no fresh executed human action for this frame")
        action = np.asarray(snapshot.get("executed_action"), dtype=np.float32).reshape(-1)
        if action.shape != (16,) or not np.isfinite(action).all():
            raise ValueError("VLAHost executed human action is not a finite 16-joint target")
        return action

    def stop_intervention_session(self) -> None:
        self._intervention_stop.set()
        if self._intervention_thread is not None:
            self._intervention_thread.join(timeout=self.cfg.timeout + 0.5)
            self._intervention_thread = None
        if self._intervention_enabled:
            self.set_intervention_session(False)

    def wait_for_policy_control(self, timeout: float = 5.0) -> bool:
        """Resume from a fresh observation generation; hold while VR owns control."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            response = self.session.get(self._url("state"), timeout=self.cfg.timeout)
            response.raise_for_status()
            state = response.json()
            snapshot = state.get("intervention")
            if not isinstance(snapshot, dict):
                raise RuntimeError("VLAHost /state lacks intervention arbitration data")
            self.intervention = snapshot
            phase = snapshot.get("phase")
            if phase == "policy":
                return True
            if phase == "waiting_policy" and snapshot.get("footkey") != 1:
                joints = state.get("joint_states", {}).get("positions", [])
                if len(joints) != 14:
                    raise RuntimeError("cannot resume policy without 14 measured joints")
                def grip(name):
                    value = state.get(name, 0.0)
                    if isinstance(value, dict):
                        value = value.get("position", 0.0)
                    elif isinstance(value, list):
                        value = value[0] if value else 0.0
                    return float(np.clip(value, 0.0, 1.0))
                hold = np.asarray([*joints, grip("gripper_left"), grip("gripper_right")], dtype=np.float32)
                if not np.isfinite(hold).all():
                    raise RuntimeError("cannot resume policy from nonfinite state")
                self._observation_generation = snapshot.get("refresh_generation")
                result = self.action_chunk(np.repeat(hold[None], 3, axis=0), source_hz=30.0)
                if result.get("execution") not in {"resume_pending", "active"}:
                    continue
            time.sleep(0.05)
        return False


class VLAHostResidualEnv:
    """Bridge-backed env; residual actor trains locally, robot stays remote."""

    def __init__(self, bridge: VLAHostClient, policy: Pi05BasePolicy, cfg: Pi05ResidualConfig):
        self.bridge = bridge
        self.policy_env = Pi05ResidualEnv(_BridgeBackend(bridge, cfg), policy, cfg)
        self.cfg = cfg
        self.steps = 0

    def reset(self, **kwargs):
        self.steps = 0
        return self.policy_env.reset(**kwargs)

    def step(self, action):
        self.steps += 1
        return self.policy_env.step(action)

    def observe_intervention(self, raw_next, executed_action):
        self.steps += 1
        return self.policy_env.observe_intervention(raw_next, executed_action)


class _BridgeBackend:
    def __init__(self, bridge: VLAHostClient, cfg: Pi05ResidualConfig):
        self.bridge, self.cfg = bridge, cfg
        self.steps = 0
        self._command_anchor = None
        self._observation_latency_s = 0.007

    def reset(self, **kwargs):
        go_home = bool(kwargs.pop("go_home", True))
        home_sleep_s = float(kwargs.pop("home_sleep_s", 0.0))
        if kwargs:
            raise TypeError(f"unsupported reset arguments: {sorted(kwargs)}")
        if go_home:
            self.bridge.reset()
            if home_sleep_s > 0:
                time.sleep(home_sleep_s)
        self.steps = 0
        raw, info = self.bridge.observation()
        if go_home and np.max(raw["observation.state"][-2:]) > 0.1:
            print(
                "warning: grippers remain above 0.1 after post-home open command: "
                f"{raw['observation.state'][-2:].tolist()}"
            )
        self._command_anchor = np.asarray(
            raw["observation.state"], dtype=np.float32
        ).copy()
        return raw, info

    def step(self, action):
        cycle_started = time.monotonic()
        target = np.asarray(action, dtype=np.float32).reshape(-1)
        if self._command_anchor is None or target.shape != (16,) or not np.isfinite(target).all():
            raise ValueError("reset and a finite 16-joint target are required before step")
        substeps = self.cfg.chunk_size // self.cfg.rl_chunk_size
        fractions = np.arange(1, substeps + 1, dtype=np.float32)[:, None] / substeps
        actions = self._command_anchor[None] + fractions * (
            target - self._command_anchor
        )[None]
        # Match LeRobot's per-waypoint max-relative-target protection.
        previous = self._command_anchor[:14].copy()
        limit = self.cfg.bridge_joint_step_limit_rad
        for waypoint in actions:
            waypoint[:14] = previous + np.clip(waypoint[:14] - previous, -limit, limit)
            previous = waypoint[:14].copy()
        # Keep the learner at 10Hz and let VLAHost play the complete 3-point
        # trajectory continuously at 30Hz. One HTTP chunk avoids boundary gaps
        # caused by three separate single-waypoint requests.
        result = self.bridge.action_chunk(
            actions, source_hz=self.bridge.cfg.step_frequency * substeps
        )
        result = {**result, "bridge_action_chunk": actions.copy()}
        # Never use immediate, lagging feedback as the next trajectory anchor.
        # The command endpoint keeps consecutive chunks continuous.
        if result.get("execution") == "active":
            self._command_anchor = actions[-1].copy()

        # Sample next_obs at the end of the 10Hz transition. Start the request
        # slightly early so its response, rather than its request, lands near
        # the cycle boundary. collect.py only sleeps any small remainder.
        period_s = 1.0 / self.bridge.cfg.step_frequency
        observe_at = cycle_started + period_s - self._observation_latency_s
        remaining = observe_at - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        observation_started = time.monotonic()
        raw, info = self.bridge.observation()
        observation_latency_s = time.monotonic() - observation_started
        self._observation_latency_s = (
            0.8 * self._observation_latency_s + 0.2 * observation_latency_s
        )
        result["observation_started_after_s"] = observation_started - cycle_started
        result["observation_completed_after_s"] = time.monotonic() - cycle_started
        self.steps += 1
        reward = float(result.get("reward", info.get("reward", 0.0)))
        terminated = bool(result.get("terminated", False))
        truncated = bool(
            result.get("truncated", False)
            or self.steps >= self.cfg.max_episode_steps
        )
        return raw, reward, terminated, truncated, {**info, **result}
