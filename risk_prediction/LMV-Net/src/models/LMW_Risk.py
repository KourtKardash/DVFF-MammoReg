import os
import torch
import torch.nn as nn
from pathlib import Path
from typing import Sequence

from asymmetry_model.mirai_localized_dif_head import extract_mirai_backbone
from models.model_utils import ContinuousPosEncoding, CumulativeProbabilityLayer, CrossAttentionBlock
from registration.backends import RegistrationBackend
from registration.flow_ops import resize_flow, warp_with_flow

# The Mirai image encoder is a third-party checkpoint and is not redistributed
# with this repository. Obtain it from the Mirai authors (see README, "Data" and
# "License and attribution") and point either --mirai_encoder_path or the
# MIRAI_ENCODER_PATH environment variable at the downloaded file.
MIRAI_ENCODER_PATH_ENV = "MIRAI_ENCODER_PATH"


class LongitudinalFeatureProcessor(nn.Module):
    """
    Longitudinal risk pipeline:
      1. Optional ANTs affine pre-warp on prior images (benchmark-style).
      2. Predict residual DL flows on (current, prior_ants).
      3. Apply residual at image or feature level; build |f_cur - f_pri_aligned|.
    """

    def __init__(
        self,
        registration_backend: RegistrationBackend,
        finetune_all: bool = False,
        mirai_encoder_path: str | Path | None = None,
        alignment_level: str = "images",
    ):
        super().__init__()

        mirai_encoder_path = mirai_encoder_path or os.environ.get(MIRAI_ENCODER_PATH_ENV)
        if not mirai_encoder_path:
            raise FileNotFoundError(
                "MIRAI encoder weights were not given. The checkpoint is not "
                "redistributed with this repository: obtain it from the Mirai "
                "authors, then pass --mirai_encoder_path <file> or set "
                f"{MIRAI_ENCODER_PATH_ENV}=<file>."
            )
        mirai_path = Path(mirai_encoder_path)
        if not mirai_path.is_file():
            raise FileNotFoundError(f"MIRAI encoder weights not found: {mirai_path}.")

        alignment_level = alignment_level.lower()
        if alignment_level not in ("features", "images"):
            raise ValueError(
                f"alignment_level must be 'features' or 'images', got {alignment_level!r}"
            )
        self.alignment_level = alignment_level

        self.encoder = extract_mirai_backbone(str(mirai_path))
        self.registration_backend = registration_backend
        self.positional_encoding = ContinuousPosEncoding(dim=512)

        self.encoder.requires_grad_(False)
        self.registration_backend.requires_grad_(False)
        self.encoder.eval()
        self.registration_backend.eval()

        if finetune_all:
            print("Finetuning all layers of the encoder")
            for param in self.encoder.parameters():
                param.requires_grad = True
            self.encoder.train()
        else:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

    def _process_view(
        self,
        img_cur: torch.Tensor,
        img_pri: torch.Tensor,
        residual_flow: torch.Tensor,
        time_gap,
        ants_flow: torch.Tensor | None = None,
    ):
        img_pri_work = img_pri
        if ants_flow is not None:
            img_pri_work = warp_with_flow(
                img_pri_work, ants_flow.detach(), padding_mode="border"
            )

        flow = residual_flow.detach()

        if self.alignment_level == "images":
            img_pri_aligned = warp_with_flow(
                img_pri_work, flow, padding_mode="border"
            )
            img_cur_3c = img_cur.repeat(1, 3, 1, 1)
            img_pri_3c = img_pri_aligned.repeat(1, 3, 1, 1)
            f_cur = self.encoder(img_cur_3c)
            f_pri = self.encoder(img_pri_3c)
        else:
            img_cur_3c = img_cur.repeat(1, 3, 1, 1)
            img_pri_3c = img_pri_work.repeat(1, 3, 1, 1)
            f_cur = self.encoder(img_cur_3c)
            f_pri = self.encoder(img_pri_3c)

            flow_downsampled = resize_flow(
                flow, (f_cur.shape[2], f_cur.shape[3])
            )
            f_pri = warp_with_flow(f_pri, flow_downsampled, padding_mode="border")

        f_diff = torch.abs(f_cur - f_pri)
        B, C, H, W = f_diff.shape
        flattened_feats = f_diff.flatten(start_dim=2).permute(2, 0, 1)
        fdif_with_time = self.positional_encoding(flattened_feats, time_gap)
        f_diff = fdif_with_time.permute(1, 2, 0).view(B, C, H, W)

        f_long = torch.cat([f_cur, f_pri, f_diff], dim=1)
        return f_long

    def forward(
        self,
        img_cur_cc,
        img_pri_cc,
        img_cur_mlo,
        img_pri_mlo,
        time_gap,
        affine_mat_cc: Sequence[str] | None = None,
        affine_mat_mlo: Sequence[str] | None = None,
    ):
        flows = self.registration_backend.get_flows(
            img_cur_cc,
            img_pri_cc,
            img_cur_mlo,
            img_pri_mlo,
            affine_mat_cc=affine_mat_cc,
            affine_mat_mlo=affine_mat_mlo,
        )
        f_cc_long = self._process_view(
            img_cur_cc,
            img_pri_cc,
            flows["cc"],
            time_gap,
            ants_flow=flows.get("ants_cc"),
        )
        f_mlo_long = self._process_view(
            img_cur_mlo,
            img_pri_mlo,
            flows["mlo"],
            time_gap,
            ants_flow=flows.get("ants_mlo"),
        )
        return {
            'f_cc_long': f_cc_long,
            'f_mlo_long': f_mlo_long,
        }


class LongitudinalMultiViewRiskModel(nn.Module):
    def __init__(
        self,
        registration_backend: RegistrationBackend,
        max_followup: int = 5,
        num_attn_blocks: int = 1,
        feature_dim: int = 1536,
        finetune_all: bool = False,
        mirai_encoder_path: str | Path | None = None,
        alignment_level: str = "images",
        mammo_reg_net=None,
    ):
        super().__init__()

        if mammo_reg_net is not None and registration_backend is None:
            from registration.backends import MammoRegNetBackend
            registration_backend = MammoRegNetBackend(mammo_reg_net)

        if registration_backend is None:
            raise ValueError("registration_backend is required")

        self.longitudinal_feat_processor = LongitudinalFeatureProcessor(
            registration_backend=registration_backend,
            finetune_all=finetune_all,
            mirai_encoder_path=mirai_encoder_path,
            alignment_level=alignment_level,
        )

        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(in_channels=feature_dim, reduced_channels=feature_dim,
                                heads=4, dropout=0.3, drop_path=0.2, ffn_expansion_factor=2)
            for _ in range(num_attn_blocks)
        ])

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.view_fc = nn.Linear(feature_dim * 2, feature_dim)
        self.cumulative_risk = CumulativeProbabilityLayer(num_features=feature_dim, max_followup=max_followup)

    def forward(
        self,
        img_cur_cc,
        img_pri_cc,
        img_cur_mlo,
        img_pri_mlo,
        time_gap,
        affine_mat_cc: Sequence[str] | None = None,
        affine_mat_mlo: Sequence[str] | None = None,
    ):
        longitudinal_output = self.longitudinal_feat_processor(
            img_cur_cc,
            img_pri_cc,
            img_cur_mlo,
            img_pri_mlo,
            time_gap,
            affine_mat_cc=affine_mat_cc,
            affine_mat_mlo=affine_mat_mlo,
        )

        f_cc = longitudinal_output['f_cc_long']
        f_mlo = longitudinal_output['f_mlo_long']

        for blk in self.cross_attn_blocks:
            f_cc, f_mlo = blk(f_cc, f_mlo)

        pooled_cc = self.global_avg_pool(f_cc).flatten(1)
        pooled_mlo = self.global_avg_pool(f_mlo).flatten(1)

        pooled_multi = torch.cat([pooled_cc, pooled_mlo], dim=1)
        pooled_multi = self.view_fc(pooled_multi)

        risk_multi = self.cumulative_risk(pooled_multi)
        risk_cc = self.cumulative_risk(pooled_cc)
        risk_mlo = self.cumulative_risk(pooled_mlo)

        return {'risk_multi': risk_multi, 'risk_cc': risk_cc, 'risk_mlo': risk_mlo}
