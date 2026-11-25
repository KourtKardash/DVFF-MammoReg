from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator

from registration.factory import load_registration_backend
from utils import get_risk_loss_BCE, compute_auc_x_year_auc
from utils.c_index import concordance_index_ipcw, get_censoring_dist


def load_mammoregnet(path, map_location="cpu"):
    """Backward-compatible helper; prefer load_registration_backend()."""
    backend = load_registration_backend(
        method="mammoregnet",
        checkpoint_path=path,
        map_location=map_location,
        coarse_affine=True,
    )
    return backend.model


def load_risk_state_dict(checkpoint_path, map_location="cpu"):
    # Full training checkpoints contain numpy scalars (epoch, best_c_index, etc.).
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def load_risk_model_weights(model, checkpoint, map_location="cpu"):
    """
    Load risk-head / MIRAI encoder weights from a training checkpoint.

    Registration backend weights are loaded separately via --registration_checkpoint
    and are intentionally skipped here.

    `checkpoint` may be a path or an already-loaded state dict.
    """
    if isinstance(checkpoint, (str, Path)):
        state_dict = load_risk_state_dict(checkpoint, map_location=map_location)
    else:
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    state_dict = {
        k: v for k, v in state_dict.items()
        if "registration_backend" not in k
    }
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = [
        k for k in incompatible.missing_keys
        if "registration_backend" not in k
    ]
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(
            f"Checkpoint is missing {len(missing)} risk-model keys "
            f"(registration backend excluded), e.g. {preview}"
        )
    return incompatible


def _registration_batch_kwargs(batch: dict) -> dict:
    mats_cc = batch.get("affine_mat_cc")
    mats_mlo = batch.get("affine_mat_mlo")
    if mats_cc is None and mats_mlo is None:
        return {}
    return {
        "affine_mat_cc": list(mats_cc) if mats_cc is not None else None,
        "affine_mat_mlo": list(mats_mlo) if mats_mlo is not None else None,
    }


def train_one_epoch(args, model_risk, train_loader, optimizer, accelerator, warmup_scheduler, global_step, warmup_steps):
    """
    Handles the training logic for a single epoch.

    Returns:
        A tuple containing (average_risk_loss, c_index, auc_results, global_step) for the epoch.
    """
    model_risk.train()
    running_risk_loss = 0.0
    all_preds, all_times, all_events = [], [], []

    for batch in train_loader:

        outputs = model_risk(
            batch["current_image_cc"],
            batch["previous_image_cc"],
            batch["current_image_mlo"],
            batch["previous_image_mlo"],
            batch["time_gap"],
            **_registration_batch_kwargs(batch),
        )

        risk_multi = outputs["risk_multi"]
        risk_cc = outputs["risk_cc"]
        risk_mlo = outputs["risk_mlo"]

        pos_weight_cap = getattr(args, "risk_pos_weight", 1.0)
        risk_loss_multi = get_risk_loss_BCE(risk_multi, batch["target"], batch["y_mask"], pos_weight_cap=pos_weight_cap)
        risk_loss_cc = get_risk_loss_BCE(risk_cc, batch["target"], batch["y_mask"], pos_weight_cap=pos_weight_cap)
        risk_loss_mlo = get_risk_loss_BCE(risk_mlo, batch["target"], batch["y_mask"], pos_weight_cap=pos_weight_cap)

        risk_loss = risk_loss_multi + risk_loss_cc + risk_loss_mlo

        running_risk_loss += risk_loss.item()

        optimizer.zero_grad()
        accelerator.backward(risk_loss)
        optimizer.step()

        if global_step < warmup_steps:
            warmup_scheduler.step()
        global_step += 1

        all_preds.append(accelerator.gather(torch.sigmoid(risk_multi).detach()))
        all_times.append(accelerator.gather(batch["event_times"]))
        all_events.append(accelerator.gather(batch["event_observed"]))

    avg_risk_loss = running_risk_loss / len(train_loader)
    c_index, auc_results = 0, {}
    if accelerator.is_main_process:
        preds = torch.cat(all_preds).cpu().numpy()
        times = torch.cat(all_times).cpu().numpy().astype(int)
        events = torch.cat(all_events).cpu().numpy()
        censoring_dist = get_censoring_dist(times, events)
        c_index = concordance_index_ipcw(times, preds, events, censoring_dist)
        auc_results = compute_auc_x_year_auc(preds, times, events)

    return avg_risk_loss, c_index, auc_results, global_step


def evaluate(args, model_risk, valid_loader, accelerator):
    """
    Handles the evaluation logic for a single epoch.

    Returns:
        A tuple containing (average_risk_loss, c_index, auc_results).
    """
    model_risk.eval()
    running_risk_loss = 0.0
    val_preds, val_times, val_events = [], [], []

    with torch.no_grad():
        for batch_val in valid_loader:
            outputs_val = model_risk(
                batch_val["current_image_cc"],
                batch_val["previous_image_cc"],
                batch_val["current_image_mlo"],
                batch_val["previous_image_mlo"],
                batch_val["time_gap"],
                **_registration_batch_kwargs(batch_val),
            )

            risk_pred_val = outputs_val["risk_multi"]
            risk_cc = outputs_val["risk_cc"]
            risk_mlo = outputs_val["risk_mlo"]

            pos_weight_cap = getattr(args, "risk_pos_weight", 1.0)
            risk_loss_multi = get_risk_loss_BCE(risk_pred_val, batch_val["target"], batch_val["y_mask"], pos_weight_cap=pos_weight_cap)
            risk_loss_cc = get_risk_loss_BCE(risk_cc, batch_val["target"], batch_val["y_mask"], pos_weight_cap=pos_weight_cap)
            risk_loss_mlo = get_risk_loss_BCE(risk_mlo, batch_val["target"], batch_val["y_mask"], pos_weight_cap=pos_weight_cap)

            risk_loss_val = risk_loss_multi + risk_loss_cc + risk_loss_mlo
            running_risk_loss += risk_loss_val.item()

            val_preds.append(accelerator.gather(torch.sigmoid(risk_pred_val).detach()))
            val_times.append(accelerator.gather(batch_val["event_times"]))
            val_events.append(accelerator.gather(batch_val["event_observed"]))

    avg_risk_loss = running_risk_loss / len(valid_loader)
    c_index, auc_results, class_pred_summary = 0, {}, []
    if accelerator.is_main_process:
        predictions_val = torch.cat(val_preds).cpu().numpy()
        event_times_val = torch.cat(val_times).cpu().numpy().astype(int)
        event_observed_val = torch.cat(val_events).cpu().numpy()
        censoring_dist_val = get_censoring_dist(event_times_val, event_observed_val)
        c_index = concordance_index_ipcw(event_times_val, predictions_val, event_observed_val, censoring_dist_val)
        auc_results = compute_auc_x_year_auc(predictions_val, event_times_val, event_observed_val)
        class_pred_summary = summarize_class_predictions(
            predictions_val, event_times_val, event_observed_val, n_per_class=5,
        )

    return avg_risk_loss, c_index, auc_results, class_pred_summary


CLASS_NO_CANCER = "no_cancer"


def summarize_class_predictions(predictions, event_times, event_observed, n_per_class=5):
    """
    Pick up to ``n_per_class`` random validation samples from each horizon class
    and average their predicted cumulative-risk curves. Re-sampled every call
    (i.e. every epoch), so the examples differ across epochs.

    Classes:
      - no_cancer                     (event_observed == 0)
      - cancer_y{k} for k = 1..T      (event_observed == 1 and event year == k)
        (CSAW-CC has no "cancer in year 5" samples, so that row is usually empty)

    Args:
        predictions: [N, T] predicted cumulative probabilities (sigmoid of cum logits)
        event_times: [N] 0-based event/censor year index
        event_observed: [N] 1 for cancer case, 0 for control
        n_per_class: number of random examples to average per class

    Returns:
        List of dicts: {name, target, count, mean_pred} (mean_pred None if empty).
    """
    predictions = np.asarray(predictions)
    event_times = np.asarray(event_times).astype(int)
    event_observed = np.asarray(event_observed).astype(int)
    T = predictions.shape[1]

    groups = {CLASS_NO_CANCER: np.where(event_observed == 0)[0]}
    for k in range(1, T + 1):
        groups[f"cancer_y{k}"] = np.where((event_observed == 1) & (event_times == (k - 1)))[0]

    def target_for(name):
        if name == CLASS_NO_CANCER:
            return [0] * T
        k = int(name.split("_y")[1])
        return [1 if (y + 1) >= k else 0 for y in range(T)]

    summary = []
    for name, idx in groups.items():
        entry = {"name": name, "target": target_for(name), "count": int(len(idx)), "mean_pred": None}
        if len(idx) > 0:
            pick = np.random.choice(idx, size=min(n_per_class, len(idx)), replace=False)
            entry["mean_pred"] = predictions[pick].mean(axis=0).tolist()
        summary.append(entry)
    return summary


def format_class_predictions(summary):
    """Return a list of aligned log lines, one per horizon class."""
    if not summary:
        return ["Per-class val check: (no data)"]

    lines = ["Per-class val check (mean predicted cumulative risk over up to 5 random samples):"]
    name_w = max(len(e["name"]) for e in summary)
    for e in summary:
        target = "[" + ",".join(str(t) for t in e["target"]) + "]"
        if e["mean_pred"] is None:
            pred = "(no samples)"
        else:
            pred = "[" + ", ".join(f"{p:.3f}" for p in e["mean_pred"]) + "]"
        lines.append(f"  {e['name']:<{name_w}} (n={e['count']:>4}) target={target} pred={pred}")
    return lines


def get_model_size(model, accelerator):
    """Calculates and logs the model size, handling unwrapping from Accelerator."""
    unwrapped_model = accelerator.unwrap_model(model)
    param_size = sum(p.numel() * p.element_size() for p in unwrapped_model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in unwrapped_model.buffers())
    total_size_mb = (param_size + buffer_size) / (1024 ** 2)

    if accelerator.is_main_process:
        print(f"Model size: {total_size_mb:.2f} MB (Parameters: {param_size / 1e6:.1f}M elements)")
    return total_size_mb


def linear_warmup(step, warmup_steps):
    """
    Linear warm-up function.
    Gradually increases the learning rate from 0 to base_lr over warmup_steps.
    """
    if warmup_steps == 0:
        return 1.0
    if step < warmup_steps:
        return step / warmup_steps
    return 1.0


def get_param_groups(model, base_lr, finetune_lr_scale=0.1):
    encoder_params = []
    new_module_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "encoder" in name:
            encoder_params.append(param)
        else:
            new_module_params.append(param)

    param_groups = []
    if new_module_params:
        param_groups.append({"params": new_module_params, "lr": base_lr})
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": base_lr * finetune_lr_scale})

    return param_groups


def save_checkpoint(
    accelerator,
    model,
    optimizer,
    scheduler,
    warmup_scheduler,
    epoch,
    global_step,
    best_c_index,
    path
):
    accelerator.save({
        "epoch": epoch,
        "global_step": global_step,
        "best_c_index": best_c_index,
        "model": accelerator.unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "warmup_scheduler": warmup_scheduler.state_dict(),
    }, path)


def load_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    scheduler,
    warmup_scheduler,
    accelerator
):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    load_risk_model_weights(accelerator.unwrap_model(model), ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

    if scheduler and ckpt["scheduler"] is not None:
        scheduler.load_state_dict(ckpt["scheduler"])

    warmup_scheduler.load_state_dict(ckpt["warmup_scheduler"])

    return (
        ckpt["epoch"] + 1,
        ckpt["global_step"],
        ckpt["best_c_index"],
    )
