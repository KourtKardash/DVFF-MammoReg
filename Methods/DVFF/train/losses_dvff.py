import torch
import torch.nn.functional as F

from Methods.MammoRegNet.train.losses_mammoregnet import NCC, Regu_loss


def masked_mean_flow(flow: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return flow.mean(dim=(2, 3))
    if mask.shape[1] == 1:
        weights = mask
    else:
        weights = mask[:, :1]
    weights = weights / (weights.sum(dim=(2, 3), keepdim=True) + 1e-8)
    return (flow * weights).sum(dim=(2, 3))


def cross_view_consistency_loss(
    flow_mlo: torch.Tensor,
    flow_cc: torch.Tensor,
    mask_mlo: torch.Tensor | None = None,
    mask_cc: torch.Tensor | None = None,
) -> torch.Tensor:
    mu_mlo = masked_mean_flow(flow_mlo, mask_mlo)
    mu_cc = masked_mean_flow(flow_cc, mask_cc)
    return F.mse_loss(mu_mlo, mu_cc)


def affine_consistency_loss(affine_mlo: torch.Tensor, affine_cc: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(affine_mlo, affine_cc)


def dvff_registration_loss(
    outputs: dict,
    fixed_mlo: torch.Tensor,
    fixed_cc: torch.Tensor,
    moving_mask_mlo: torch.Tensor | None = None,
    moving_mask_cc: torch.Tensor | None = None,
    lambda_regu: float = 1.0,
    lambda_cvc: float = 0.01,
    lambda_affine: float = 0.01,
    coarse_affine: bool = True,
):
    ncc = NCC()

    ncc_mlo = 1.0 - ncc.loss(fixed_mlo, outputs["warped_mlo"])
    ncc_cc = 1.0 - ncc.loss(fixed_cc, outputs["warped_cc"])
    if coarse_affine:
        ncc_mlo += 1.0 - ncc.loss(fixed_mlo, outputs["affined_mlo"])
        ncc_cc += 1.0 - ncc.loss(fixed_cc, outputs["affined_cc"])
    ncc_loss = ncc_mlo + ncc_cc

    regu_loss = Regu_loss(outputs["flow_mlo"]) + Regu_loss(outputs["flow_cc"])
    #cvc_loss = cross_view_consistency_loss(
    #    outputs["flow_mlo"],
    #    outputs["flow_cc"],
    #    moving_mask_mlo,
    #    moving_mask_cc,
    #)
    #aff_loss = affine_consistency_loss(outputs["affine_mlo"], outputs["affine_cc"])

    total = (
        ncc_loss
        + lambda_regu * regu_loss
        #+ lambda_cvc * cvc_loss
        #+ lambda_affine * aff_loss
    )

    metrics = {
        "total": total,
        "ncc": ncc_loss,
        "ncc_mlo": ncc_mlo,
        "ncc_cc": ncc_cc,
        "regu": regu_loss,
        "cvc": 0,
        "affine_consistency": 0,
    }
    return total, metrics
