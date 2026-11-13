"""Shared actor/checkpoint contract for the Pi0.5 collector and learner."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch

from resfit.rl_finetuning.config.rlpd import QAgentConfig
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent

from .config import Pi05ResidualConfig

ACTOR_CONTRACT = "tanh_raw_times_5deg_per_joint_v1"


def make_agent(cfg: Pi05ResidualConfig, device: str) -> QAgent:
    agent_cfg = QAgentConfig(device=device, enc_type="resnet")
    agent_cfg.freeze_encoder = True
    agent_cfg.actor_lr = 1e-6
    agent_cfg.critic_lr = 1e-4
    agent_cfg.critic_target_tau = 0.005
    scale = tuple(
        cfg.residual_joint_limit_rad / limit if i < 14 else 0.0
        for i, limit in enumerate(cfg.rl_joint_delta_limit)
    )
    agent_cfg.actor.action_scale = scale
    agent_cfg.actor.actor_last_layer_init_scale = 0.0
    agent_cfg.target_action_noise = False
    agent_cfg.actor.residual_action_mask = tuple(
        0.0 if i in {14, 15} else 1.0 for i in range(cfg.action_dim)
    )
    return QAgent(
        obs_shape=(3, cfg.rl_image_size, cfg.rl_image_size),
        prop_shape=(cfg.state_dim,),
        action_dim=cfg.action_dim,
        rl_cameras=list(cfg.cameras),
        cfg=agent_cfg,
        residual_actor=True,
    )


def replay_identity(cfg: Pi05ResidualConfig) -> dict:
    checkpoint = Path(cfg.pi05_checkpoint).expanduser().resolve()
    files = [checkpoint / "params" / "manifest.ocdbt"]
    files += list((checkpoint / "assets").glob("*/norm_stats.json"))
    digest = hashlib.sha256()
    for path in sorted(files):
        if not path.is_file():
            raise FileNotFoundError(f"Pi0.5 identity file is missing: {path}")
        digest.update(path.read_bytes())
    return {
        "schema_version": 8,
        "coordinate_frame": "pi05_normalized_sequential_10hz_joint_delta_v2",
        "action_contract": "pi_absolute_plus_actor_proposal_filtered_execution_context_obs_v8",
        "visual_encoder_contract": "three_independent_imagenet_resnet18_frozen_eval_v1",
        "action_lowpass_alpha": cfg.action_lowpass_alpha,
        "bridge_joint_step_limit_rad": cfg.bridge_joint_step_limit_rad,
        "pi05_checkpoint": str(checkpoint),
        "pi05_checkpoint_sha256": digest.hexdigest(),
        "pi05_config": cfg.pi05_config,
        "chunk_size": cfg.chunk_size,
        "cameras": list(cfg.cameras),
        "state_dim": cfg.state_dim,
        "raw_state_dim": cfg.raw_state_dim,
        "context_dim": cfg.state_dim - cfg.raw_state_dim,
        "action_dim": cfg.action_dim,
        "rl_image_size": cfg.rl_image_size,
        "joint_delta_limit": list(cfg.joint_delta_limit),
        "rl_joint_delta_limit": list(cfg.rl_joint_delta_limit),
        "rl_chunk_size": cfg.rl_chunk_size,
        "residual_joint_limit_rad": cfg.residual_joint_limit_rad,
        "residual_ema_alpha": cfg.residual_ema_alpha,
        "residual_slew_limit_rad": cfg.residual_slew_limit_rad,
    }


def actor_bundle(agent: QAgent, cfg: Pi05ResidualConfig, step: int) -> dict:
    identity = replay_identity(cfg)
    return {
        "actor": agent.actor.state_dict(),
        "encoders": agent.encoders.state_dict(),
        "global_step": step,
        "identity": identity,
        "actor_contract": ACTOR_CONTRACT,
    }


def apply_actor_bundle(
    agent: QAgent | None, bundle: dict, cfg: Pi05ResidualConfig, device: str
) -> tuple[QAgent, int]:
    expected = replay_identity(cfg)
    if bundle.get("identity") != expected:
        raise ValueError("learner actor bundle does not match the Pi0.5 rollout contract")
    if bundle.get("actor_contract") != ACTOR_CONTRACT:
        raise ValueError("learner actor bundle uses an incompatible residual action contract")
    if agent is None:
        agent = make_agent(cfg, device)
    agent.actor.load_state_dict(bundle["actor"])
    agent.encoders.load_state_dict(bundle["encoders"])
    agent.eval()
    return agent, int(bundle.get("global_step", -1))


def atomic_save_bundle(bundle: dict, path: str | Path) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(bundle, temporary)
    os.replace(temporary, path)
