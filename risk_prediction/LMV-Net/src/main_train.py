import argparse
import os
import random
import torch
from torch.utils.data import DataLoader
import logging
import time
import kornia.augmentation as K_A
from kornia.constants import Resample
from datetime import datetime
import kornia.augmentation.container as K_C
import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)

from accelerate import Accelerator

from dataloaders import BreastCancerRiskDatasetCSAWCC
from dataloaders.horizon_balanced_sampler import HorizonBalancedBatchSampler
from registration.cli import add_registration_arguments, dataset_registration_kwargs, resolve_registration_settings
from train import train_val


def setup_logging(path_logger, is_main_process):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if is_main_process:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        file_handler = logging.FileHandler(path_logger, mode="w")
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(console_handler)
    return logger


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--path_out_dir", type=str, required=True)
    parser.add_argument("--mirai_encoder_path", type=str, default=None, help="Path to MIRAI encoder weights (optional)")
    add_registration_arguments(parser)
    parser.add_argument("--resume_from", type=str)
    parser.add_argument("--wandb_id", type=str, default=None, help="WandB run ID to resume logging")
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log to Weights & Biases (use --no-wandb on cluster without API key)",
    )
    parser.add_argument("--id_training", type=int, required=True)
    parser.add_argument("--augmentations", type=str, required=True)
    parser.add_argument("--use_scheduler", type=str, required=True)
    parser.add_argument("--optimizer", type=str)
    parser.add_argument("--warmup_steps", default=5000, type=int)
    parser.add_argument("--finetune_all", action="store_true", help="Whether to finetune the whole encoder")
    parser.add_argument(
        "--risk_pos_weight",
        type=float,
        default=6.0,
        help=(
            "Upper bound on the dynamic positive-class weight in the risk BCE loss. "
            "Cancer cells are rare, so this up-weights them to keep the yearly hazard "
            "branch from collapsing to zero. Effective weight = clamp(#neg/#pos, 1, cap) "
            "per batch. Set to 1.0 to disable."
        ),
    )

    parser.add_argument("--patience_lr_scheduler", default=5, type=int)
    parser.add_argument("--patience", default=15, type=int)
    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--use_horizon_balance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Case/control balanced training batches (~50%% cancer / ~50%% controls, "
            "random within each half) with periodic natural batches "
            "(see --balance_random_every)."
        ),
    )
    parser.add_argument(
        "--balance_random_every",
        type=int,
        default=5,
        help=(
            "When horizon balance is enabled: every N-th batch is drawn uniformly from "
            "the full train set (natural ~7%% case prevalence). "
            "Set to 0 to disable natural batches (pure 50/50 balancing)."
        ),
    )
    parser.add_argument("--dataset", type=str, default="CSAW", choices=["CSAW"])

    parser.add_argument("--lr_decay", default=0.5, type=float)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--num_epochs", default=100, type=int)
    parser.add_argument("--seed", default=2023, type=int)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    args = parser.parse_args()

    reg_suffix = f"_{args.registration_method}"
    align_suffix = f"_{args.alignment_level}_align"
    args.results_dir = args.path_out_dir + reg_suffix + align_suffix + '_' \
                  + str(args.learning_rate) + '_lr_' \
                       + str(args.weight_decay) + '_wd_' \
                       + str(args.num_epochs) + '_epochs_' \
                  + str(args.batch_size) + '_bs_' \
                  + datetime.now().strftime("%Y-%m-%d-%H-%M") + '/'

    return args


def main():
    args = parse_arguments()
    resolve_registration_settings(args)
    accelerator = Accelerator()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

    if args.augmentations == "True":
        # same_on_batch=True is essential here: the CSAW dataloader stacks the four
        # views [current_cc, current_mlo, prior_cc, prior_mlo] into one batch and
        # runs this pipeline once, so every random transform (crop / affine, and the
        # intensity ops) is IDENTICAL across the four views. That keeps the
        # current<->prior geometry aligned. Independent per-image geometric augs
        # (the previous behaviour) desynced the pair and broke the longitudinal /
        # registration comparison, which is why the geometric augs had been disabled.
        train_transform = K_C.AugmentationSequential(
            K_A.RandomCrop(size=(1094, 851), p=0.2),
            K_A.Resize((1152, 896), resample=Resample.NEAREST.name),
            K_A.RandomAffine(translate=(0.0, 0.1), scale=(1.0, 1.1), degrees=0, shear=0, p=0.5),
            K_A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.0, p=0.5),
            K_A.RandomGamma(gamma=(0.8, 1.2), gain=(0.9, 1.05), p=0.5),
            same_on_batch=True,
        )
        if accelerator.is_main_process:
            print("Train augmentations :", train_transform)
    else:
        train_transform = None

    if args.dataset != "CSAW":
        raise ValueError(
            f"--dataset must be 'CSAW', got {args.dataset!r}. The downstream "
            "experiment reported in the paper runs on CSAW-CC only."
        )
    print("Use CSAW-CC dataset")
    ds_reg_kwargs = dataset_registration_kwargs(args)
    train_dataset = BreastCancerRiskDatasetCSAWCC(
        args.csv_file, args.data_root, "train", transforms=train_transform, **ds_reg_kwargs
    )
    validation_dataset = BreastCancerRiskDatasetCSAWCC(
        args.csv_file, args.data_root, "val", transforms=None, **ds_reg_kwargs
    )

    if args.use_horizon_balance:
        if not args.shuffle:
            if accelerator.is_main_process:
                print("Note: --use_horizon_balance overrides train shuffle (batch sampler controls order).")
        train_batch_sampler = HorizonBalancedBatchSampler(
            train_dataset,
            batch_size=args.batch_size,
            seed=args.seed,
            random_batch_every=args.balance_random_every,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
        if accelerator.is_main_process:
            if args.balance_random_every:
                print(
                    f"Case/control balanced training: batch_size={args.batch_size}, "
                    f"random 50/50 + 1 natural batch every {args.balance_random_every}"
                )
            else:
                print(
                    f"Case/control balanced training: batch_size={args.batch_size}, "
                    f"all batches random 50/50 cancer/controls"
                )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=args.shuffle,
            pin_memory=args.pin_memory,
        )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=args.pin_memory,
    )

    model_path = f"model_risk_prediction_training_id_{args.id_training}_last_epoch.pth"
    logg_path = f"train_risk_prediction_training_id_{args.id_training}.log"

    path_out_model = os.path.join(args.results_dir, model_path)
    path_logger = os.path.join(args.results_dir, logg_path)

    if accelerator.is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)

    logger = setup_logging(path_logger, accelerator.is_main_process)

    start_time = time.time()

    if accelerator.is_main_process:
        logger.info("Training started...")
        logger.info(f"registration_method={args.registration_method}")
        logger.info(f"alignment_level={args.alignment_level}")
        logger.info(f"ants_prewarp={args.ants_prewarp}")
        logger.info(f"ants_prewarp_in_loader={getattr(args, 'ants_prewarp_in_loader', False)}")
        logger.info(f"affine_cache_root={args.affine_cache_root}")
        logger.info(f"risk_pos_weight={args.risk_pos_weight}")
        if args.registration_method == "dvff":
            logger.info(f"dvff_use_fusion={getattr(args, 'dvff_use_fusion', True)}")
    train_val(
        args,
        train_loader,
        validation_loader,
        path_logger,
        path_out_model,
        accelerator,
    )

    end_time = time.time()
    if accelerator.is_main_process:
        logger.info(f"Training completed in {(end_time - start_time) / 60:.2f} minutes")
        logger.info(f"Saving model to: {path_out_model}")


if __name__ == '__main__':
    main()
