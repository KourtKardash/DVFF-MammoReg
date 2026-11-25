import torch
import torch.nn as nn

from Methods.MammoRegNet.models.MammoRegNet import (
    AffineTransformer_block,
    Conv_encoder,
    SpatialTransformer_block,
    Trans_decoder,
)


class CrossViewFusionBlock(nn.Module):
    """Exchange context between MLO and CC feature maps at one pyramid level."""

    def __init__(self, channels: int):
        super().__init__()
        mid = max(channels // 4, 4)
        self.mlo_from_cc = nn.Sequential(
            nn.Conv2d(channels * 2, mid, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1),
        )
        self.cc_from_mlo = nn.Sequential(
            nn.Conv2d(channels * 2, mid, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1),
        )

    def forward(self, mlo: torch.Tensor, cc: torch.Tensor):
        mlo_out = mlo + self.mlo_from_cc(torch.cat([mlo, cc], dim=1))
        cc_out = cc + self.cc_from_mlo(torch.cat([cc, mlo], dim=1))
        return mlo_out, cc_out


class DVFFNet(nn.Module):
    """
    Dual-projection extension of MammoRegNet.

    Accepts four images (fixed/moving x MLO/CC), fuses cross-view features at
    the selected pyramid levels (the paper uses levels 2, 3 and 4, with 32, 64
    and 128 channels), adds a learnable view embedding at the bottleneck, and
    decodes one displacement field per projection with a shared decoder. With
    coarse_affine=False every decoder level is deformable; with
    coarse_affine=True the coarsest level is an affine head.
    """

    def __init__(
        self,
        in_channels: int = 1,
        enc_channels: int = 8,
        dec_channels: int = 16,
        use_checkpoint: bool = True,
        fusion_levels: tuple[int, ...] = (2, 3, 4),
        use_fusion: bool = True,
        coarse_affine: bool = True,
    ):
        super().__init__()
        self.enc_channels = enc_channels
        self.fusion_levels = tuple(fusion_levels)
        self.use_fusion = use_fusion
        self.coarse_affine = coarse_affine
        # Runtime switch for curriculum: modules exist, but fusion is skipped
        # until set_fusion_enabled(True).
        self.apply_fusion = bool(use_fusion)

        self.encoder = Conv_encoder(
            in_channels=in_channels,
            channel_num=enc_channels,
            use_checkpoint=use_checkpoint,
        )
        self.decoder = Trans_decoder(
            in_channels=enc_channels,
            channel_num=dec_channels,
            use_checkpoint=use_checkpoint,
            coarse_affine=coarse_affine,
        )

        if use_fusion:
            channel_sizes = [
                enc_channels,
                enc_channels * 2,
                enc_channels * 4,
                enc_channels * 8,
                enc_channels * 16,
            ]
            self.fusion_blocks = nn.ModuleList(
                [
                    CrossViewFusionBlock(channel_sizes[level])
                    if level in fusion_levels
                    else nn.Identity()
                    for level in range(len(channel_sizes))
                ]
            )
            self.view_embed = nn.Embedding(2, enc_channels * 16)
        else:
            self.fusion_blocks = None
            self.view_embed = None

        self.spatial_transformer = SpatialTransformer_block(mode="bilinear")
        self.affine_transformer = AffineTransformer_block(mode="bilinear")

    def _fuse_pyramid(self, mlo_feats, cc_feats):
        if not self.use_fusion or not self.apply_fusion:
            return mlo_feats, cc_feats

        fused_mlo = []
        fused_cc = []
        for level, (mlo, cc, fusion) in enumerate(
            zip(mlo_feats, cc_feats, self.fusion_blocks)
        ):
            if isinstance(fusion, nn.Identity):
                fused_mlo.append(mlo)
                fused_cc.append(cc)
            else:
                mlo_out, cc_out = fusion(mlo, cc)
                fused_mlo.append(mlo_out)
                fused_cc.append(cc_out)
        return fused_mlo, fused_cc

    def _decode_view(self, x_fix, x_mov, view_id: int):
        if self.use_fusion and self.apply_fusion:
            x_fix = list(x_fix)
            x_mov = list(x_mov)
            view_emb = self.view_embed(
                torch.tensor([view_id], device=x_fix[4].device, dtype=torch.long)
            ).view(1, -1, 1, 1)
            x_fix[4] = x_fix[4] + view_emb
            x_mov[4] = x_mov[4] + view_emb
        return self.decoder(x_fix, x_mov)

    def forward(
        self,
        fixed_mlo,
        fixed_cc,
        moving_mlo,
        moving_cc,
        moving_mask_mlo=None,
        moving_mask_cc=None,
    ):
        fix_mlo = self.encoder(fixed_mlo)
        fix_cc = self.encoder(fixed_cc)
        mov_mlo = self.encoder(moving_mlo)
        mov_cc = self.encoder(moving_cc)

        fix_mlo, fix_cc = self._fuse_pyramid(fix_mlo, fix_cc)
        mov_mlo, mov_cc = self._fuse_pyramid(mov_mlo, mov_cc)

        flow_mlo, affine_mlo = self._decode_view(fix_mlo, mov_mlo, view_id=0)
        flow_cc, affine_cc = self._decode_view(fix_cc, mov_cc, view_id=1)

        warped_mlo = self.spatial_transformer(moving_mlo, flow_mlo)
        warped_cc = self.spatial_transformer(moving_cc, flow_cc)
        affined_mlo = self.affine_transformer(moving_mlo, affine_mlo)
        affined_cc = self.affine_transformer(moving_cc, affine_cc)

        warped_mask_mlo = None
        warped_mask_cc = None
        if moving_mask_mlo is not None:
            warped_mask_mlo = self.spatial_transformer(moving_mask_mlo, flow_mlo)
        if moving_mask_cc is not None:
            warped_mask_cc = self.spatial_transformer(moving_mask_cc, flow_cc)

        return {
            "warped_mlo": warped_mlo,
            "warped_cc": warped_cc,
            "flow_mlo": flow_mlo,
            "flow_cc": flow_cc,
            "affined_mlo": affined_mlo,
            "affined_cc": affined_cc,
            "affine_mlo": affine_mlo,
            "affine_cc": affine_cc,
            "warped_mask_mlo": warped_mask_mlo,
            "warped_mask_cc": warped_mask_cc,
        }

    def register_pair(self, fixed, moving, moving_mask=None):
        """
        Single-image-pair registration without cross-view fusion.

        Intended for inference when only one projection pair is available,
        while using encoder/decoder weights from fusion-trained checkpoints.
        """
        fix_feats = self.encoder(fixed)
        mov_feats = self.encoder(moving)
        flow, affine = self.decoder(fix_feats, mov_feats)

        warped = self.spatial_transformer(moving, flow)
        affined = self.affine_transformer(moving, affine)
        warped_mask = None
        if moving_mask is not None:
            warped_mask = self.spatial_transformer(moving_mask, flow)

        return {
            "warped": warped,
            "flow": flow,
            "affine": affine,
            "affined": affined,
            "warped_mask": warped_mask,
        }

    FUSION_PREFIXES = ("fusion_blocks.", "view_embed.")
    _MAMMOREGNET_KEY_MAP = {
        "Encoder.": "encoder.",
        "Decoder.": "decoder.",
        "SpatialTransformer.": "spatial_transformer.",
        "AffineTransformer.": "affine_transformer.",
    }

    @classmethod
    def remap_mammoregnet_state_dict(cls, state_dict):
        """Rename single-view MammoRegNet keys (Encoder., Decoder., ...) to DVFFNet keys."""
        mapped = {}
        for key, value in state_dict.items():
            for src_prefix, dst_prefix in cls._MAMMOREGNET_KEY_MAP.items():
                if key.startswith(src_prefix):
                    mapped[dst_prefix + key[len(src_prefix) :]] = value
                    break
        return mapped

    def load_pretrained_mammoregnet(self, checkpoint_path: str) -> tuple[list[str], list[str]]:
        """
        Load encoder/decoder weights from a single-view MammoRegNet checkpoint.

        Maps MammoRegNet.Encoder -> encoder, MammoRegNet.Decoder -> decoder.
        Fusion blocks and view_embed remain randomly initialized when use_fusion=True.
        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        mapped = self.remap_mammoregnet_state_dict(state_dict)

        missing, unexpected = self.load_state_dict(mapped, strict=False)
        trainable_new = [k for k in missing if k.startswith(self.FUSION_PREFIXES)]
        return trainable_new, unexpected

    def load_checkpoint_state(self, state_dict) -> dict:
        """
        Load evaluation weights and fail loudly if they do not fit this model.

        Accepts a DVFFNet checkpoint (plain state dict or a training checkpoint
        with a "model" entry) or a single-view MammoRegNet checkpoint, whose keys
        are renamed first. Fusion weights in the checkpoint are ignored when this
        model was built with use_fusion=False, which is how a fusion-trained
        network is evaluated with fusion switched off. Any other missing or
        unexpected key raises, so that a wrong checkpoint, or a wrong
        --coarse_deform / --no_fusion flag, cannot silently leave layers at
        their random initialization.
        """
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        if any(key.startswith("Encoder.") for key in state_dict):
            state_dict = self.remap_mammoregnet_state_dict(state_dict)

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        ignored = []
        if not self.use_fusion:
            ignored = [k for k in unexpected if k.startswith(self.FUSION_PREFIXES)]
            unexpected = [k for k in unexpected if k not in ignored]

        if missing or unexpected:
            hint = ""
            if any(k.startswith(self.FUSION_PREFIXES) for k in missing):
                hint = " The checkpoint has no fusion weights; evaluate it with --no_fusion."
            elif any(k.startswith("decoder.") for k in missing + unexpected):
                hint = (
                    " The decoder heads do not match; check --coarse_deform against "
                    "how the checkpoint was trained."
                )
            raise RuntimeError(
                f"Checkpoint does not match the model: {len(missing)} missing and "
                f"{len(unexpected)} unexpected keys (first missing: {missing[:3]}, "
                f"first unexpected: {unexpected[:3]}).{hint}"
            )
        return {"ignored_fusion_keys": ignored}

    def set_encoder_trainable(self, trainable: bool):
        for param in self.encoder.parameters():
            param.requires_grad = trainable

    def set_decoder_trainable(self, trainable: bool):
        for param in self.decoder.parameters():
            param.requires_grad = trainable

    def set_fusion_enabled(self, enabled: bool):
        """Turn cross-view fusion on/off without rebuilding the graph."""
        if not self.use_fusion:
            self.apply_fusion = False
            return
        self.apply_fusion = bool(enabled)
        for param in self.fusion_parameters():
            param.requires_grad = self.apply_fusion

    def fusion_parameters(self):
        if not self.use_fusion:
            return []
        return list(self.fusion_blocks.parameters()) + list(self.view_embed.parameters())

    @staticmethod
    def fusion_levels_from_state_dict(state_dict) -> tuple[int, ...]:
        """Detect which pyramid levels have CrossViewFusionBlock weights."""
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        levels = []
        for level in range(5):
            prefix = f"fusion_blocks.{level}."
            if any(key.startswith(prefix) and key.endswith("weight") for key in state_dict):
                levels.append(level)
        return tuple(levels) if levels else (2, 3, 4)

    def encoder_parameters(self):
        return list(self.encoder.parameters())

    def decoder_parameters(self):
        return list(self.decoder.parameters())
