import os
import re
import time

import torch
import wandb
from tqdm import tqdm

from Methods.DVFF.models.dvff_net import DVFFNet
from Methods.DVFF.train.losses_dvff import dvff_registration_loss
from Methods.MammoRegNet.utils.utils import checkpoint, create_logger


def is_full_training_checkpoint(ckpt) -> bool:
    return (
        isinstance(ckpt, dict)
        and "model" in ckpt
        and "epoch" in ckpt
        and "optimizer" in ckpt
    )


def infer_start_epoch(resume_path: str, ckpt) -> int:
    if is_full_training_checkpoint(ckpt):
        return int(ckpt["epoch"]) + 1

    match = re.search(r"model_(\d+)\.pth$", os.path.basename(resume_path))
    if match:
        return int(match.group(1)) + 1

    raise ValueError(
        f"Cannot infer resume epoch from checkpoint: {resume_path}. "
        "Use training_checkpoint.pth or a file named model_N.pth."
    )


def save_training_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    use_fusion,
    coarse_affine,
    config,
):
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "use_fusion": use_fusion,
        "apply_fusion": bool(getattr(model, "apply_fusion", use_fusion)),
        "coarse_affine": coarse_affine,
        "fusion_levels": list(getattr(model, "fusion_levels", ())),
        "config": config,
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)


def apply_resume_checkpoint(
    resume_path,
    ckpt,
    model,
    optimizer,
    scheduler,
    logger,
):
    start_epoch = infer_start_epoch(resume_path, ckpt)

    if is_full_training_checkpoint(ckpt):
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        logger.info(
            f"Resumed full training state from {resume_path} "
            f"(completed epoch {ckpt['epoch']}, continuing from epoch {start_epoch})"
        )
        return start_epoch, ckpt

    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=True)
        logger.warning(
            f"Checkpoint at {resume_path} has model weights only; "
            "optimizer and scheduler were re-initialized."
        )
        return start_epoch, ckpt

    model.load_state_dict(ckpt, strict=False)
    logger.warning(
        f"Loaded legacy model weights from {resume_path}; "
        f"continuing from epoch {start_epoch} with a fresh optimizer."
    )
    return start_epoch, {"epoch": start_epoch - 1, "config": {}}

def build_optimizer(
    model,
    lr_fusion,
    lr_decoder,
    lr_encoder,
    weight_decay,
    include_fusion=False,
):
    param_groups = [
        {"name": "encoder", "params": model.encoder_parameters(), "lr": lr_encoder},
        {"name": "decoder", "params": model.decoder_parameters(), "lr": lr_decoder},
    ]
    if include_fusion:
        param_groups.append(
            {"name": "fusion", "params": model.fusion_parameters(), "lr": lr_fusion}
        )
    return torch.optim.Adam(param_groups, weight_decay=weight_decay)


def optimizer_has_group(optimizer, name: str) -> bool:
    return any(group.get("name") == name for group in optimizer.param_groups)


def enable_fusion_training(model, optimizer, lr_fusion, logger):
    model.set_fusion_enabled(True)
    if not optimizer_has_group(optimizer, "fusion") and model.fusion_parameters():
        optimizer.add_param_group(
            {"name": "fusion", "params": model.fusion_parameters(), "lr": lr_fusion}
        )
    logger.info(
        f"Fusion enabled; trainable parameters: {trainable_param_count(model):,}"
    )


def trainable_param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_dvff(
    train_loader,
    device,
    learning_rate_fusion,
    learning_rate_decoder,
    weight_decay,
    num_epochs,
    path_logger,
    path_model,
    path_out,
    use_scheduler,
    max_iterations,
    lambda_regu=1.0,
    lambda_cvc=0.0,
    lambda_affine=0.0,
    pretrained_path=None,
    use_fusion=True,
    resume_path=None,
    coarse_affine=True,
    fusion_warmup_epochs=0,
    learning_rate_encoder=1e-4,
    freeze_encoder=False,
    fusion_levels=(2, 3, 4),
):
    print("[INFO] Starting DVFF training...")
    start_time = time.time()

    logger = create_logger(path_logger)
    logger.info(f"Training DVFF for {num_epochs} epochs")
    logger.info(f"use_fusion={use_fusion}")
    logger.info(f"fusion_levels={fusion_levels}")
    logger.info(f"fusion_warmup_epochs={fusion_warmup_epochs}")
    logger.info(f"coarse_affine={coarse_affine}")
    logger.info(f"freeze_encoder={freeze_encoder}")
    if resume_path:
        logger.info(f"resume_path={resume_path}")

    model = DVFFNet(
        use_fusion=use_fusion,
        coarse_affine=coarse_affine,
        fusion_levels=fusion_levels,
    ).to(device)

    if resume_path:
        if pretrained_path:
            logger.info(
                "Skipping pretrained MammoRegNet weights because --resume is set."
            )
    elif pretrained_path:
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")
        missing, unexpected = model.load_pretrained_mammoregnet(pretrained_path)
        logger.info(f"Loaded MammoRegNet weights from {pretrained_path}")
        logger.info(f"New modules (random init): {missing}")
        if unexpected:
            logger.info(f"Unexpected keys in checkpoint: {unexpected}")
    else:
        logger.info("Training from scratch (no pretrained MammoRegNet weights)")

    model.set_encoder_trainable(not freeze_encoder)
    model.set_decoder_trainable(True)

    start_epoch = 0
    resume_ckpt = None
    if resume_path:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        resume_ckpt = torch.load(resume_path, map_location=device)
        start_epoch = infer_start_epoch(resume_path, resume_ckpt)

    fusion_on_now = bool(use_fusion) and start_epoch >= fusion_warmup_epochs
    if use_fusion:
        model.set_fusion_enabled(fusion_on_now)
        if fusion_on_now:
            logger.info(
                "Fusion on "
                f"(lr_fusion={learning_rate_fusion}, lr_decoder={learning_rate_decoder}, "
                f"lr_encoder={learning_rate_encoder})"
            )
        else:
            logger.info(
                f"Warmup until epoch {fusion_warmup_epochs}: encoder + decoder only, fusion off "
                f"(lr_encoder={learning_rate_encoder}, lr_decoder={learning_rate_decoder})"
            )
    else:
        model.set_fusion_enabled(False)
        logger.info(
            f"Fusion disabled; training encoder + decoder "
            f"(lr_encoder={learning_rate_encoder}, lr_decoder={learning_rate_decoder})"
        )

    optimizer = build_optimizer(
        model,
        learning_rate_fusion,
        learning_rate_decoder,
        learning_rate_encoder,
        weight_decay=weight_decay,
        include_fusion=fusion_on_now,
    )

    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
        logger.info("Using ExponentialLR scheduler")

    training_config = {
        "learning_rate_fusion": learning_rate_fusion,
        "learning_rate_decoder": learning_rate_decoder,
        "learning_rate_encoder": learning_rate_encoder,
        "weight_decay": weight_decay,
        "use_scheduler": use_scheduler,
        "lambda_regu": lambda_regu,
        "lambda_cvc": lambda_cvc,
        "lambda_affine": lambda_affine,
        "pretrained_path": pretrained_path,
        "fusion_warmup_epochs": fusion_warmup_epochs,
        "fusion_levels": list(fusion_levels),
        "freeze_encoder": freeze_encoder,
    }
    training_config["coarse_affine"] = coarse_affine

    if resume_path:
        saved_use_fusion = resume_ckpt.get("use_fusion")
        if saved_use_fusion is not None and saved_use_fusion != use_fusion:
            raise ValueError(
                f"Checkpoint use_fusion={saved_use_fusion} does not match "
                f"current run use_fusion={use_fusion}."
            )
        saved_coarse_affine = resume_ckpt.get("coarse_affine")
        if (
            saved_coarse_affine is not None
            and saved_coarse_affine != coarse_affine
        ):
            raise ValueError(
                f"Checkpoint coarse_affine={saved_coarse_affine} does not match "
                f"current run coarse_affine={coarse_affine}."
            )
        saved_levels = resume_ckpt.get("fusion_levels")
        if saved_levels is None:
            saved_levels = (resume_ckpt.get("config") or {}).get("fusion_levels")
        if saved_levels is None and use_fusion:
            saved_levels = DVFFNet.fusion_levels_from_state_dict(
                resume_ckpt.get("model", resume_ckpt)
            )
        if saved_levels is not None and tuple(saved_levels) != tuple(fusion_levels):
            raise ValueError(
                f"Checkpoint fusion_levels={tuple(saved_levels)} do not match "
                f"current run fusion_levels={tuple(fusion_levels)}. "
                "Resume with the same --fusion_levels, or start a new run."
            )
        start_epoch, resume_ckpt = apply_resume_checkpoint(
            resume_path,
            resume_ckpt,
            model,
            optimizer,
            scheduler,
            logger,
        )
        if start_epoch >= num_epochs:
            logger.info(
                f"Checkpoint already reached epoch {start_epoch - 1}; "
                f"nothing left to train (num_epochs={num_epochs})."
            )
            return

    training_checkpoint_path = os.path.join(path_out, "training_checkpoint.pth")

    logger.info(f"Trainable parameters: {trainable_param_count(model):,}")
    logger.info(f"Training epochs: {start_epoch} -> {num_epochs - 1}")

    wandb.init(
        project="dvff",
        resume="allow" if resume_path else None,
        config={
            "optimizer": "Adam",
            "architecture": "DVFFNet",
            "use_fusion": use_fusion,
            "coarse_affine": coarse_affine,
            "transfer_learning": bool(pretrained_path),
            "pretrained_path": pretrained_path,
            "resume_path": resume_path,
            "epochs": num_epochs,
            "start_epoch": start_epoch,
            "fusion_warmup_epochs": fusion_warmup_epochs,
            "fusion_levels": list(fusion_levels),
            "learning_rate_fusion": learning_rate_fusion,
            "learning_rate_decoder": learning_rate_decoder,
            "learning_rate_encoder": learning_rate_encoder,
            "weight_decay": weight_decay,
            "lambda_regu": lambda_regu,
            "lambda_cvc": lambda_cvc,
            "lambda_affine": lambda_affine,
            "freeze_encoder": freeze_encoder,
        },
    )
    wandb.define_metric("epoch", hidden=True)
    for name in [
        "Training Total Loss",
        "Training Loss NCC",
        "Training Loss NCC MLO",
        "Training Loss NCC CC",
        "Training Loss Regularisation",
        "Trainable Parameters",
        "Fusion Enabled",
    ]:
        wandb.define_metric(name, step_metric="epoch")

    for epoch in tqdm(
        range(start_epoch, num_epochs),
        desc="Epoch",
        initial=start_epoch,
        total=num_epochs,
    ):
        torch.cuda.empty_cache()
        logger.info(f"Epoch {epoch}/{num_epochs}")
        if (
            use_fusion
            and fusion_warmup_epochs > 0
            and epoch == fusion_warmup_epochs
            and not model.apply_fusion
        ):
            logger.info(
                f"Epoch {epoch}: ending warmup, enabling fusion at levels {fusion_levels}"
            )
            enable_fusion_training(model, optimizer, learning_rate_fusion, logger)
        model.train()

        totals = {
            "total": 0.0,
            "ncc": 0.0,
            "ncc_mlo": 0.0,
            "ncc_cc": 0.0,
            "regu": 0.0,
        }
        counter = 0

        for batch in train_loader:
            if counter >= max_iterations:
                break
            counter += 1

            fixed_mlo = batch["fixed_mlo"].to(device)
            fixed_cc = batch["fixed_cc"].to(device)
            moving_mlo = batch["moving_mlo"].to(device)
            moving_cc = batch["moving_cc"].to(device)

            outputs = model(fixed_mlo, fixed_cc, moving_mlo, moving_cc)
            loss, metrics = dvff_registration_loss(
                outputs,
                fixed_mlo,
                fixed_cc,
                lambda_regu=lambda_regu,
                lambda_cvc=lambda_cvc,
                lambda_affine=lambda_affine,
                coarse_affine=coarse_affine,
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            for key in totals:
                totals[key] += metrics[key].item()

        if scheduler:
            scheduler.step()

        if counter == 0:
            raise RuntimeError("Training loader returned zero batches.")

        for key in totals:
            totals[key] /= counter

        wandb.log(
            {
                "epoch": epoch,
                "Training Total Loss": totals["total"],
                "Training Loss NCC": totals["ncc"],
                "Training Loss NCC MLO": totals["ncc_mlo"],
                "Training Loss NCC CC": totals["ncc_cc"],
                "Training Loss Regularisation": totals["regu"],
                "Trainable Parameters": trainable_param_count(model),
                "Fusion Enabled": int(bool(model.apply_fusion)),
            }
        )
        logger.info(
            f"Losses: Total={totals['total']:.4f}, "
            f"NCC={totals['ncc']:.4f}, Regu={totals['regu']:.4f}"
        )

        model_path = os.path.join(path_out, f"model_{epoch}.pth")
        checkpoint(model, model_path)
        save_training_checkpoint(
            training_checkpoint_path,
            model,
            optimizer,
            scheduler,
            epoch,
            use_fusion,
            coarse_affine,
            training_config,
        )
        model.eval()

    total_time = time.time() - start_time
    logger.info(f"Total training time: {total_time:.2f}s")
    print(f"[INFO] Total training time: {total_time:.2f}s")

    torch.save(model.state_dict(), path_model)
    artifact = wandb.Artifact("dvff_model", type="model")
    artifact.add_file(path_model)
    wandb.log_artifact(artifact)
    wandb.finish()
