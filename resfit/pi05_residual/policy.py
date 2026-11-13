from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import torch

from .config import Pi05ResidualConfig


class Pi05BasePolicy:
    """Frozen pi0.5 using the native LeRobot/OpenPI policy API."""

    def __init__(self, cfg: Pi05ResidualConfig):
        self.cfg = cfg
        self._native = None
        self.state_mean = torch.zeros(cfg.raw_state_dim, dtype=torch.float32)
        self.state_std = torch.ones(cfg.raw_state_dim, dtype=torch.float32)
        try:
            # OpenPI's native policy loader imports its legacy LeRobot dataset
            # modules.  Put the pinned compatibility copy first when a
            # vlahost checkout is also on PYTHONPATH.
            compat_lerobot = Path(__file__).parents[2] / "deps" / "lerobot"
            if compat_lerobot.is_dir() and str(compat_lerobot) not in sys.path:
                sys.path.insert(0, str(compat_lerobot))
            from openpi.policies.policy_config import create_trained_policy
            from openpi.shared.normalize import load as load_norm_stats
            from openpi.training.config import get_config

            train_config = get_config(cfg.pi05_config or cfg.task)
            self._prompt = cfg.task or getattr(
                train_config.data, "default_prompt", ""
            )
            # OpenPI infer() applies its output transforms and returns a full
            # chunk of absolute, dataset-unit joint targets.
            train_config = dataclasses.replace(
                train_config,
                model=dataclasses.replace(
                    train_config.model, action_horizon=cfg.chunk_size
                ),
            )
            norm_stats = cfg.dataset_stats
            checkpoint = Path(cfg.pi05_checkpoint).expanduser()
            if checkpoint.exists():
                candidates = list((checkpoint / "assets").glob("*/norm_stats.json"))
                if len(candidates) == 1:
                    # Absolute LeRobot repo ids otherwise escape checkpoint/assets
                    # when OpenPI joins the asset path.
                    if norm_stats is None:
                        norm_stats = load_norm_stats(candidates[0].parent)
                    raw_stats = json.loads(candidates[0].read_text())["norm_stats"]["state"]
                    self.state_mean = torch.tensor(raw_stats["mean"], dtype=torch.float32)
                    self.state_std = torch.tensor(raw_stats["std"], dtype=torch.float32)
            self._native = create_trained_policy(
                train_config,
                cfg.pi05_checkpoint,
                default_prompt=self._prompt or None,
                norm_stats=norm_stats,
                pytorch_device=cfg.device,
            )
            return
        except (ImportError, ModuleNotFoundError):
            pass
        try:
            from lerobot.configs.types import FeatureType, PolicyFeature
            from lerobot.policies.pi05.configuration_pi05 import PI05Config
            from lerobot.policies.pi05.modeling_pi05 import PI05Policy
            from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
        except ImportError as exc:
            raise ImportError(
                "Install the OpenPI/LeRobot pi0.5 environment before using "
                "resfit.pi05_residual"
            ) from exc

        policy_cfg = PI05Config(
            device=cfg.device,
            dtype=cfg.dtype,
            chunk_size=cfg.chunk_size,
            n_action_steps=cfg.chunk_size,
            max_action_dim=max(32, cfg.action_dim),
            max_state_dim=max(32, cfg.raw_state_dim),
            image_resolution=(224, 224),
            num_inference_steps=cfg.num_inference_steps,
            gradient_checkpointing=False,
            compile_model=False,
        )
        policy_cfg.input_features = {
            key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
            for key in cfg.cameras
        }
        policy_cfg.input_features["observation.state"] = PolicyFeature(
            type=FeatureType.STATE, shape=(cfg.raw_state_dim,)
        )
        policy_cfg.output_features = {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(cfg.action_dim,))
        }
        self.policy = PI05Policy.from_pretrained(
            cfg.pi05_checkpoint, config=policy_cfg, strict=False
        )
        self.preprocessor, self.postprocessor = make_pi05_pre_post_processors(
            config=policy_cfg, dataset_stats=cfg.dataset_stats
        )
        self.policy.eval()
        for parameter in self.policy.parameters():
            parameter.requires_grad_(False)
        self._prompt = cfg.task

    @torch.no_grad()
    def action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        if self._native is not None:
            def image(key: str):
                value = observation[key].detach().cpu()
                if value.ndim == 4:
                    value = value[0]
                if value.shape[0] == 3:
                    value = value.permute(1, 2, 0)
                value = value.numpy()
                if value.dtype != "uint8":
                    value = (value.clip(0, 1) * 255).astype("uint8")
                return value

            raw = {
                "image": image(self.cfg.cameras[0]),
                "left_wrist_image": image(self.cfg.cameras[1]),
                "right_wrist_image": image(self.cfg.cameras[2]),
                "state": observation["observation.state"].detach().cpu().numpy()[0],
                "prompt": self._prompt,
            }
            output = self._native.infer(raw)
            chunk = torch.as_tensor(output["actions"], dtype=torch.float32).unsqueeze(0)
            return chunk[..., : self.cfg.action_dim]
        batch = {key: value for key, value in observation.items() if key != "observation.base_action"}
        batch["task"] = self._prompt
        processed = self.preprocessor(batch)
        chunk = self.policy.predict_action_chunk(processed)
        if chunk.ndim != 3:
            raise ValueError(f"pi0.5 must return [B,H,A], got {tuple(chunk.shape)}")
        return chunk[..., : self.cfg.action_dim].float()
