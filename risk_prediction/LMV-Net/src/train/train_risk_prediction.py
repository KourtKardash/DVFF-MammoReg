import os
import torch
import wandb
from accelerate import Accelerator

from models import LongitudinalMultiViewRiskModel
from utils import create_logger
from train.train_utils import (
    evaluate,
    format_class_predictions,
    get_model_size,
    get_param_groups,
    linear_warmup,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
)
from registration.factory import load_registration_backend


def train_val(args, train_loader, valid_loader, path_loggger, path_model, accelerator: Accelerator):
    logger = create_logger(path_loggger) if accelerator.is_main_process else None
    if accelerator.is_main_process:
        logger.info(f"Number Training Epochs: {args.num_epochs}")

    path_saved_reg_model = args.registration_checkpoint
    if accelerator.is_main_process:
        print(f"Registration method: {args.registration_method}")
        print(f"Registration checkpoint: {path_saved_reg_model}")
        print(f"Alignment level: {args.alignment_level}")
        print(f"ANTs pre-warp: {args.ants_prewarp}")
        print(f"Coarse affine head: {args.coarse_affine}")
        if args.registration_method == "dvff":
            print(f"DVFF fusion: {getattr(args, 'dvff_use_fusion', True)}")

    reg_backend = load_registration_backend(
        method=args.registration_method,
        checkpoint_path=path_saved_reg_model,
        coarse_affine=args.coarse_affine,
        use_fusion=getattr(args, "dvff_use_fusion", True),
        compose_ants_affine=args.ants_prewarp,
        prewarp_in_loader=getattr(args, "ants_prewarp_in_loader", False),
    )

    model_risk = LongitudinalMultiViewRiskModel(
        registration_backend=reg_backend,
        max_followup=5,
        finetune_all=args.finetune_all,
        mirai_encoder_path=args.mirai_encoder_path,
        alignment_level=args.alignment_level,
    )
    get_model_size(model_risk, accelerator)

    param_groups = get_param_groups(model_risk, base_lr=args.learning_rate, finetune_lr_scale=0.1)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    if accelerator.is_main_process:
        name_lookup = {p: n for n, p in model_risk.named_parameters()}
        for i, group in enumerate(optimizer.param_groups):
            lr = group["lr"]
            print(f"\nParam group {i}: learning rate = {lr}")
            for j, p in enumerate(group["params"]):
                print(f"  {j}: {name_lookup[p]} | shape={tuple(p.shape)}")
                if j >= 4:
                    print("  ...")
                    break

    warmup_steps = args.warmup_steps
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: linear_warmup(step, warmup_steps),
    )

    scheduler = None
    if args.use_scheduler == "True":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.lr_decay,
            patience=args.patience_lr_scheduler,
            min_lr=1e-7
        )
        if accelerator.is_main_process:
            logger.info(f"Scheduler configured: {type(scheduler).__name__}")
            print(f"Scheduler configured: {type(scheduler).__name__}")

    model_risk, optimizer, train_loader, valid_loader, scheduler, warmup_scheduler = accelerator.prepare(
        model_risk, optimizer, train_loader, valid_loader, scheduler, warmup_scheduler
    )

    if accelerator.is_main_process and args.wandb:
        wandb.init(
            project="LMV_Risk_Prediction",
            resume="allow" if args.wandb_id else None,
            id=args.wandb_id,
            config=vars(args),
        )
        wandb.define_metric("epoch", hidden=True)
        for metric in [
            "Training Risk Loss", "Training C-index", "Validation Risk Loss", "Validation C-index", "Learning Rate",
            "Train Year 1 AUC", "Train Year 2 AUC", "Train Year 3 AUC", "Train Year 4 AUC", "Train Year 5 AUC",
            "Val Year 1 AUC", "Val Year 2 AUC", "Val Year 3 AUC", "Val Year 4 AUC", "Val Year 5 AUC",
        ]:
            wandb.define_metric(metric, step_metric="epoch")
    elif accelerator.is_main_process:
        print("[INFO] WandB logging disabled (--no-wandb or WANDB_MODE=disabled)")

    start_epoch = 0
    global_step = 0
    best_c_index = 0.0
    patience_counter = 0

    if args.resume_from is not None:
        start_epoch, global_step, best_c_index = load_checkpoint(
            args.resume_from,
            model_risk,
            optimizer,
            scheduler,
            warmup_scheduler,
            accelerator,
        )
        patience_counter = 0
        if accelerator.is_main_process:
            print(f"[INFO] Resumed from {args.resume_from} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.num_epochs):
        avg_train_loss, train_c_index, auc_results_train, global_step = train_one_epoch(
            args, model_risk, train_loader, optimizer, accelerator, warmup_scheduler, global_step, warmup_steps,
        )

        val_risk_loss, val_c_index, auc_results, class_pred_summary = evaluate(
            args, model_risk, valid_loader, accelerator,
        )

        should_stop = False
        if accelerator.is_main_process:
            print_msg = (
                f"##### Epoch: {epoch} ##### | Training Loss: {avg_train_loss:.4f}, "
                f"Training C-index: {train_c_index:.4f} | Validation Loss: {val_risk_loss:.4f}, "
                f"Validation C-index: {val_c_index:.4f}"
            )
            logger.info(print_msg)
            print(print_msg)

            for line in format_class_predictions(class_pred_summary):
                logger.info(line)
                print(line)

            if args.wandb:
                for e in class_pred_summary:
                    if e["mean_pred"] is None:
                        continue
                    for i, p in enumerate(e["mean_pred"]):
                        wandb.log({f"valcls/{e['name']}/P_y{i + 1}": p, "epoch": epoch})

            if args.wandb:
                wandb.log({
                    "epoch": epoch,
                    "Training Risk Loss": avg_train_loss,
                    "Training C-index": train_c_index,
                    "Validation Risk Loss": val_risk_loss,
                    "Validation C-index": val_c_index,
                })
            for year, auc in auc_results_train.items():
                logger.info(f"Train Year {year + 1}: AUC = {auc:.6f}")
                if args.wandb:
                    wandb.log({f"Train Year {year + 1} AUC": auc, "epoch": epoch})
                print(f"Train Year {year + 1}: AUC = {auc:.6f}")

            for year, auc in auc_results.items():
                logger.info(f"Val Year {year + 1}: AUC = {auc:.6f}")
                if args.wandb:
                    wandb.log({f"Val Year {year + 1} AUC": auc, "epoch": epoch})
                print(f"Val Year {year + 1}: AUC = {auc:.6f}")

            if scheduler and global_step >= warmup_steps:
                scheduler.step(val_c_index)

            unwrapped_model = accelerator.unwrap_model(model_risk)
            if epoch % 10 == 0:
                save_checkpoint(
                    accelerator, model_risk, optimizer, scheduler,
                    warmup_scheduler, epoch, global_step, best_c_index,
                    os.path.join(args.results_dir, f"checkpoint_{epoch:04d}.pth"),
                )

            if val_c_index > best_c_index:
                best_c_index = val_c_index
                patience_counter = 0
                save_checkpoint(
                    accelerator, model_risk, optimizer, scheduler,
                    warmup_scheduler, epoch, global_step, best_c_index,
                    os.path.join(args.results_dir, f"best_model_risk_prediction_id-{args.id_training}.pth"),
                )
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logger.info("Early stopping triggered.")
                    accelerator.save(
                        unwrapped_model.state_dict(),
                        os.path.join(args.results_dir, f"early_stopping_risk_prediction_id-{args.id_training}.pth"),
                    )
                    should_stop = True

            save_checkpoint(
                accelerator, model_risk, optimizer, scheduler,
                warmup_scheduler, epoch, global_step, best_c_index,
                os.path.join(args.results_dir, "checkpoint_last.pth"),
            )

        stop_tensor = torch.tensor([int(should_stop)], device=accelerator.device)
        stop_tensor = accelerator.reduce(stop_tensor, reduction="max")
        if stop_tensor.item() > 0:
            break

        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print("[INFO] Saving final model ...")
        unwrapped_model = accelerator.unwrap_model(model_risk)
        accelerator.save(unwrapped_model.state_dict(), path_model)

        if args.wandb:
            artifact = wandb.Artifact("model", type="model")
            artifact.add_file(path_model)
            wandb.log_artifact(artifact)
            wandb.finish()
