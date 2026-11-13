# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

import torch
from torch import nn

from resfit.rl_finetuning.config.rlpd import VitEncoderConfig
from resfit.rl_finetuning.off_policy.networks.min_vit import MinVit


class ResNetEncoder(nn.Module):
    """Small spatial ResNet encoder for the three-camera residual actor."""

    def __init__(self, obs_shape: tuple[int, int, int], cfg: VitEncoderConfig):
        super().__init__()
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except ImportError as exc:
            raise ImportError("enc_type='resnet' requires torchvision") from exc
        net = resnet18(weights=ResNet18_Weights.DEFAULT)
        net.fc = nn.Identity()
        self.backbone = nn.Sequential(*list(net.children())[:-2])
        self.patch_repr_dim = 512
        self.register_buffer(
            "image_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        )
        h, w = obs_shape[-2:]
        self.repr_dim = self.patch_repr_dim * max(1, (h + 31) // 32) * max(1, (w + 31) // 32)

    def forward(self, obs, flatten=True):
        feats = self.backbone((obs - self.image_mean) / self.image_std)
        feats = feats.flatten(2)
        feats = feats.transpose(1, 2)
        return feats.flatten(1) if flatten else feats


class VitEncoder(nn.Module):
    def __init__(self, obs_shape: tuple[int, int, int], cfg: VitEncoderConfig):
        super().__init__()
        self.obs_shape = obs_shape
        self.cfg = cfg
        self.vit = MinVit(
            embed_style=cfg.embed_style,
            embed_dim=cfg.embed_dim,
            embed_norm=cfg.embed_norm,
            num_head=cfg.num_heads,
            depth=cfg.depth,
            image_size=obs_shape[-2:],
        )

        self.num_patch = self.vit.num_patches
        self.patch_repr_dim = self.cfg.embed_dim
        self.repr_dim = self.cfg.embed_dim * self.vit.num_patches

    def forward(self, obs, flatten=True) -> torch.Tensor:
        if obs.max() > 5:
            obs = obs / 255.0
        obs = obs - 0.5
        feats: torch.Tensor = self.vit.forward(obs)
        if flatten:
            # [B, D, N] -> [B, D*N]
            feats = feats.flatten(1, 2)
        return feats


if __name__ == "__main__":
    vit_encoder = VitEncoder(obs_shape=(3, 84, 84), cfg=VitEncoderConfig())
    obs = torch.rand(10, 3, 84, 84)
    feats: torch.Tensor = vit_encoder(obs)
    print(feats.size())  # (10, 10368), i.e., 81 patches * 128 dimensions
