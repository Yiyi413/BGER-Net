from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .srm import SRM


def _group_norm(channels: int) -> nn.GroupNorm:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class HierarchicalBackbone(nn.Module):
    """Shared timm backbone returning 1/4, 1/8, 1/16 and 1/32 features."""

    def __init__(
        self,
        name: str = "convnext_tiny",
        pretrained: bool = True,
        freeze: bool = False,
        out_indices: Sequence[int] = (0, 1, 2, 3),
        checkpoint_path: Optional[str | Path] = None,
    ):
        super().__init__()
        self.freeze = freeze
        try:
            import timm
        except ImportError as exc:
            raise ImportError("BGER-Net requires timm. Install it with: pip install timm") from exc

        pretrained_options: Dict[str, Any] = {}
        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path).expanduser().resolve()
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"backbone checkpoint does not exist: {checkpoint_path}"
                )
            pretrained_options["pretrained_cfg_overlay"] = {
                "file": str(checkpoint_path),
                "hf_hub_id": None,
                "url": None,
            }

        try:
            self.model = timm.create_model(
                name,
                pretrained=pretrained or checkpoint_path is not None,
                features_only=True,
                out_indices=tuple(out_indices),
                **pretrained_options,
            )
        except Exception as exc:
            if pretrained:
                raise RuntimeError(
                    f"Could not load pretrained weights for {name}. "
                    "A frozen random backbone is unsuitable for training. "
                    "Make the pretrained weights available or use pretrained=False "
                    "only for smoke tests."
                ) from exc
            raise

        self.channels = tuple(self.model.feature_info.channels())
        self.reductions = tuple(self.model.feature_info.reduction())
        if self.reductions != (4, 8, 16, 32):
            raise ValueError(
                f"Expected backbone reductions (4, 8, 16, 32), got {self.reductions}."
            )
        if self.freeze:
            self.requires_grad_(False)
            self.eval()

    def train(self, mode: bool = True):
        if self.freeze:
            super().train(False)
            self.model.eval()
        else:
            super().train(mode)
        return self

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        if self.freeze:
            self.model.eval()
            with torch.no_grad():
                return tuple(self.model(x))
        return tuple(self.model(x))


class AlignBlock(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),
            _group_norm(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CrossAttentionBlock(nn.Module):
    """RGB-query, forensic-key/value cross attention with reduced K/V tokens."""

    def __init__(
        self,
        embed_dim: int = 96,
        num_heads: int = 4,
        kv_reduction: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.kv_reduction = kv_reduction
        self.q_norm = nn.LayerNorm(embed_dim)
        self.kv_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, rgb_feat: torch.Tensor, noise_feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = rgb_feat.shape
        q = rgb_feat.flatten(2).transpose(1, 2)

        kv_feat = noise_feat
        if self.kv_reduction > 1:
            kv_feat = F.avg_pool2d(
                kv_feat,
                kernel_size=self.kv_reduction,
                stride=self.kv_reduction,
                ceil_mode=True,
            )
        kv = kv_feat.flatten(2).transpose(1, 2)
        attn_out, _ = self.attn(
            self.q_norm(q),
            self.kv_norm(kv),
            self.kv_norm(kv),
            need_weights=False,
        )
        x = q + attn_out
        x = x + self.ffn(self.out_norm(x))
        return x.transpose(1, 2).reshape(b, c, h, w)


class SobelEdge(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor(
            [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 3:
            gray = (
                0.2989 * x[:, 0:1]
                + 0.5870 * x[:, 1:2]
                + 0.1140 * x[:, 2:3]
            )
        else:
            gray = x.mean(dim=1, keepdim=True)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        edge = torch.sqrt(gx.square() + gy.square() + 1e-6)
        scale = edge.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return edge / scale


class BoundaryConditionBlock(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        hidden = max(16, embed_dim // 2)
        self.block = nn.Sequential(
            nn.Conv2d(embed_dim * 2 + 1, hidden, 3, padding=1, bias=False),
            _group_norm(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(
        self,
        discrepancy: torch.Tensor,
        agreement: torch.Tensor,
        edge: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat((edge, discrepancy, agreement), dim=1)
        return torch.sigmoid(self.block(x))


class SharedExpertBlock(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size=3,
                padding=1,
                groups=embed_dim,
                bias=False,
            ),
            _group_norm(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False),
            _group_norm(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SharedModulatedExperts(nn.Module):
    evidence_names = ("R", "Q", "D", "P")

    def __init__(self, embed_dim: int):
        super().__init__()
        self.shared = SharedExpertBlock(embed_dim)
        self.gamma = nn.ParameterDict(
            {
                name: nn.Parameter(torch.ones(1, embed_dim, 1, 1))
                for name in self.evidence_names
            }
        )
        self.beta = nn.ParameterDict(
            {
                name: nn.Parameter(torch.zeros(1, embed_dim, 1, 1))
                for name in self.evidence_names
            }
        )

    def forward(self, evidences: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            name: self.gamma[name] * self.shared(evidences[name]) + self.beta[name]
            for name in self.evidence_names
        }


class Router(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 48, topk: int = 2):
        super().__init__()
        if not 1 <= topk <= 4:
            raise ValueError("topk must be in [1, 4]")
        self.topk = topk
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
        dense = torch.softmax(self.mlp(pooled), dim=1).view(-1, 4, 1, 1)
        if self.topk == 4:
            return dense, dense
        _, indices = torch.topk(dense, self.topk, dim=1)
        mask = torch.zeros_like(dense).scatter_(1, indices, 1.0)
        sparse = dense * mask
        sparse = sparse / sparse.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return sparse, dense


class BRSERBlock(nn.Module):
    evidence_names = ("R", "Q", "D", "P")

    def __init__(self, embed_dim: int = 96, topk: int = 2):
        super().__init__()
        self.boundary_condition = BoundaryConditionBlock(embed_dim)
        self.boundary_router = Router(embed_dim * 2 + 1, topk=topk)
        self.region_router = Router(embed_dim * 3, topk=topk)

    @staticmethod
    def _weighted_sum(
        experts: Dict[str, torch.Tensor],
        weights: torch.Tensor,
    ) -> torch.Tensor:
        stacked = torch.stack([experts[name] for name in BRSERBlock.evidence_names], dim=1)
        return (stacked * weights.unsqueeze(2)).sum(dim=1)

    def forward(
        self,
        evidences: Dict[str, torch.Tensor],
        experts: Dict[str, torch.Tensor],
        edge: torch.Tensor,
        route_mode: str = "dual",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        region_input = torch.cat((evidences["R"], evidences["Q"], evidences["P"]), dim=1)
        w_rg, w_rg_dense = self.region_router(region_input)
        z_region = self._weighted_sum(experts, w_rg)

        if route_mode == "single":
            zeros = torch.zeros_like(w_rg)
            boundary_map = torch.zeros_like(edge)
            return z_region, {
                "boundary_map": boundary_map,
                "boundary_weights": zeros,
                "region_weights": w_rg,
                "boundary_weights_dense": zeros,
                "region_weights_dense": w_rg_dense,
            }
        if route_mode != "dual":
            raise ValueError(f"route_mode must be 'single' or 'dual', got {route_mode!r}")

        boundary_map = self.boundary_condition(evidences["D"], evidences["P"], edge)
        boundary_input = torch.cat((evidences["D"], evidences["P"], edge), dim=1)
        w_bd, w_bd_dense = self.boundary_router(boundary_input)
        z_boundary = self._weighted_sum(experts, w_bd)
        fused = boundary_map * z_boundary + (1.0 - boundary_map) * z_region
        aux = {
            "boundary_map": boundary_map,
            "boundary_weights": w_bd,
            "region_weights": w_rg,
            "boundary_weights_dense": w_bd_dense,
            "region_weights_dense": w_rg_dense,
        }
        return fused, aux


class DecoderBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            _group_norm(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            _group_norm(channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ProgressiveDecoder(nn.Module):
    def __init__(self, decoder_dim: int = 96):
        super().__init__()
        self.level4 = DecoderBlock(decoder_dim)
        self.level3 = DecoderBlock(decoder_dim)
        self.level2 = DecoderBlock(decoder_dim)
        self.level1 = DecoderBlock(decoder_dim)
        self.mask_head = nn.Conv2d(decoder_dim, 1, kernel_size=1)
        self.boundary_head = nn.Conv2d(decoder_dim, 1, kernel_size=1)

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        z3: torch.Tensor,
        z4: torch.Tensor,
        out_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        u4 = self.level4(z4)
        u3 = self.level3(
            z3 + F.interpolate(u4, size=z3.shape[-2:], mode="bilinear", align_corners=False)
        )
        u2 = self.level2(
            z2 + F.interpolate(u3, size=z2.shape[-2:], mode="bilinear", align_corners=False)
        )
        u1 = self.level1(
            z1 + F.interpolate(u2, size=z1.shape[-2:], mode="bilinear", align_corners=False)
        )
        mask = F.interpolate(
            self.mask_head(u1), size=out_size, mode="bilinear", align_corners=False
        )
        boundary = F.interpolate(
            self.boundary_head(u1), size=out_size, mode="bilinear", align_corners=False
        )
        return mask, boundary


class BGERNet(nn.Module):
    """Boundary-Guided Evidence Routing Network.

    Tensor calls keep the native BGER-Net interface::

        model(image, noise=None) -> {"mask_logits": ..., "boundary_logits": ...}

    Owl calls use its batch and output protocols::

        model(batch, context) -> {"loss": ..., "eval": ..., "visual:*": ...}
    """

    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        out_indices: Sequence[int] = (0, 1, 2, 3),
        backbone_checkpoint: Optional[str | Path] = None,
        embed_dim: int = 96,
        num_heads: int = 4,
        topk: int = 2,
        decoder_dim: int = 96,
        kv_reductions: Sequence[int] = (16, 8, 4, 2),
        ablation: str = "F",
    ):
        super().__init__()
        if decoder_dim != embed_dim:
            raise ValueError("The first implementation requires decoder_dim == embed_dim")
        if len(kv_reductions) != 4:
            raise ValueError(
                "kv_reductions must contain values for 1/4, 1/8, 1/16 and 1/32"
            )
        ablation = ablation.upper()
        if ablation not in {"A", "B", "C", "D", "E", "F"}:
            raise ValueError(f"ablation must be one of A-F, got {ablation!r}")

        self.freeze_backbone = freeze_backbone
        self.ablation = ablation
        self.backbone = HierarchicalBackbone(
            name=backbone_name,
            pretrained=pretrained,
            freeze=freeze_backbone,
            out_indices=out_indices,
            checkpoint_path=backbone_checkpoint,
        )
        channels = self.backbone.channels
        self.rgb_align = nn.ModuleList([AlignBlock(c, embed_dim) for c in channels])
        self.noise_align = nn.ModuleList([AlignBlock(c, embed_dim) for c in channels])
        self.cross_attn = nn.ModuleList(
            [
                CrossAttentionBlock(embed_dim, num_heads, reduction)
                for reduction in kv_reductions
            ]
        )
        self.sobel = SobelEdge()
        self.default_noise = SRM()
        self.shared_experts = SharedModulatedExperts(embed_dim)
        self.brser = nn.ModuleList([BRSERBlock(embed_dim, topk) for _ in range(4)])
        self.decoder = ProgressiveDecoder(decoder_dim)

        self.register_buffer(
            "imagenet_mean",
            torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    @staticmethod
    def _to_unit_range(x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if x.detach().amax() > 1.5:
            x = x / 255.0
        return x

    def _normalize_rgb(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.imagenet_mean) / self.imagenet_std

    def _prepare_noise(self, noise: torch.Tensor) -> torch.Tensor:
        if noise.shape[1] == 1:
            noise = noise.repeat(1, 3, 1, 1)
        elif noise.shape[1] != 3:
            raise ValueError(f"noise must have 1 or 3 channels, got {noise.shape[1]}")
        mean = noise.mean(dim=(-2, -1), keepdim=True)
        std = noise.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        noise = 0.5 + 0.25 * torch.tanh((noise - mean) / std)
        return self._normalize_rgb(noise)

    @staticmethod
    def _build_evidences(
        rgb: torch.Tensor,
        noise: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return {
            "R": rgb,
            "Q": noise,
            "D": torch.abs(rgb - noise),
            "P": torch.abs(rgb * noise),
        }

    def forward_tensor(
        self,
        x: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> Dict[str, torch.Tensor]:
        out_size = x.shape[-2:]
        raw_x = self._to_unit_range(x)
        rgb_feats = self.backbone(self._normalize_rgb(raw_x))
        rgb_aligned = [align(feat) for align, feat in zip(self.rgb_align, rgb_feats)]
        router_weights = []
        boundary_maps = []

        if self.ablation == "A":
            fused_features = rgb_aligned
        else:
            if noise is None:
                noise = self.default_noise(raw_x)
            else:
                noise = noise.float()
            noise_feats = self.backbone(self._prepare_noise(noise))
            noise_aligned = [
                align(feat) for align, feat in zip(self.noise_align, noise_feats)
            ]

            if self.ablation in {"C", "D", "E", "F"}:
                rgb_aligned = [
                    attn(rgb_feat, noise_feat)
                    for attn, rgb_feat, noise_feat in zip(
                        self.cross_attn, rgb_aligned, noise_aligned
                    )
                ]

            if self.ablation in {"B", "C"}:
                fused_features = [
                    rgb_feat + noise_feat
                    for rgb_feat, noise_feat in zip(rgb_aligned, noise_aligned)
                ]
            else:
                full_edge = self.sobel(raw_x) if self.ablation == "F" else None
                fused_features = []
                for rgb_feat, noise_feat, brser in zip(
                    rgb_aligned, noise_aligned, self.brser
                ):
                    evidences = self._build_evidences(rgb_feat, noise_feat)
                    experts = self.shared_experts(evidences)
                    if self.ablation == "D":
                        fused = torch.stack(
                            [experts[name] for name in BRSERBlock.evidence_names],
                            dim=1,
                        ).mean(dim=1)
                        fused_features.append(fused)
                        continue

                    if full_edge is None:
                        edge = torch.zeros(
                            (rgb_feat.shape[0], 1, *rgb_feat.shape[-2:]),
                            device=rgb_feat.device,
                            dtype=rgb_feat.dtype,
                        )
                    else:
                        edge = F.interpolate(
                            full_edge,
                            size=rgb_feat.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    route_mode = "single" if self.ablation == "E" else "dual"
                    fused, aux = brser(
                        evidences, experts, edge, route_mode=route_mode
                    )
                    fused_features.append(fused)
                    router_weights.append(
                        {
                            key: aux[key]
                            for key in (
                                "boundary_weights",
                                "region_weights",
                                "boundary_weights_dense",
                                "region_weights_dense",
                            )
                        }
                    )
                    boundary_maps.append(aux["boundary_map"])

        mask_logits, boundary_logits = self.decoder(
            fused_features[0],
            fused_features[1],
            fused_features[2],
            fused_features[3],
            out_size=out_size,
        )
        outputs = {
            "mask_logits": mask_logits,
            "boundary_logits": boundary_logits,
        }
        if return_aux:
            outputs["router_weights"] = router_weights
            outputs["boundary_maps"] = boundary_maps
        return outputs

    def _forward_owl_batch(
        self,
        batch: Dict[str, Any],
        context: Any = None,
    ) -> Dict[str, Any]:
        """Adapt an Owl DatasetBatch to BGER-Net's native tensor interface."""
        del context
        if "tp" not in batch:
            raise KeyError("Owl batch must contain the input image field 'tp'")

        noise = batch.get("noise")
        outputs = self.forward_tensor(batch["tp"], noise=noise, return_aux=False)
        mask_logits = outputs["mask_logits"]
        boundary_logits = outputs["boundary_logits"]

        # Owl forwards the loss payload to the criterion without modification.
        return {
            "loss": (mask_logits, boundary_logits),
            "eval": mask_logits,
            "visual:mask": torch.sigmoid(mask_logits),
            "visual:boundary": torch.sigmoid(boundary_logits),
        }

    def forward(
        self,
        x: Any,
        noise: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> Dict[str, Any]:
        if isinstance(x, dict):
            # Owl calls model(batch, context), so its context arrives through
            # the second positional argument of the native-compatible API.
            return self._forward_owl_batch(x, context=noise)
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a Tensor or Owl batch dict, got {type(x).__name__}")
        return self.forward_tensor(x, noise=noise, return_aux=return_aux)


class OwlBGERCriterion(nn.Module):
    """Owl criterion adapter for BGER-Net mask and boundary supervision."""

    def __init__(self, lambda_boundary: float = 0.3):
        super().__init__()
        # Import lazily so the model remains usable when copied without the
        # project's standalone training utilities.
        from losses.bger_loss import BGERLoss

        self.loss_function = BGERLoss(lambda_boundary=lambda_boundary)

    def forward(
        self,
        loss_output: Tuple[torch.Tensor, torch.Tensor],
        batch: Dict[str, Any],
        context: Any = None,
    ) -> Dict[str, torch.Tensor]:
        del context
        if not isinstance(loss_output, (tuple, list)) or len(loss_output) != 2:
            raise ValueError(
                "BGER-Net Owl loss output must contain mask and boundary logits"
            )
        if "gt" not in batch:
            raise KeyError("Owl batch must contain the ground-truth field 'gt'")

        mask_logits, boundary_logits = loss_output
        boundary_gt = batch.get("edge")
        if boundary_gt is not None and boundary_gt.shape[-2:] == (1, 1):
            boundary_gt = None

        loss = self.loss_function(
            {
                "mask_logits": mask_logits,
                "boundary_logits": boundary_logits,
            },
            mask_gt=batch["gt"],
            boundary_gt=boundary_gt,
        )
        return {"loss": loss}


def count_parameters(model: nn.Module) -> Dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BGERNet(pretrained=False).to(device)
    image = torch.rand(2, 3, 256, 256, device=device)
    output = model(image, return_aux=True)
    print(output["mask_logits"].shape, output["boundary_logits"].shape)
    print(count_parameters(model))
