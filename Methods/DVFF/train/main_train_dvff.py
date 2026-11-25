import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from Methods.DVFF.dataloaders.dataset_dvff import (
    DVFFDataset,
)
from Methods.DVFF.train.train_dvff import train_dvff


#OUTPUT_FOLDER = "outputs/DVFFNet/"
#os.makedirs(OUTPUT_FOLDER, exist_ok=True)
#os.makedirs(OUTPUT_FOLDER + "models", exist_ok=True)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Training script for DVFF."
    )
    parser.add_argument(
        "--data_root",
        default=None,
        type=str,
        help=(
            "Image directory. Default follows --coarse_deform: "
            "Dataset/affine_train with the flag, Dataset/train without it. "
            "Masks are the sibling directory <name>_masks."
        ),
    )
    parser.add_argument(
        "--metadata_path",
        default=None,
        type=str,
        help="CSV with File Path and view metadata. Default: <data_root>/metadata_train_correct.csv",
    )
    #parser.add_argument(
    #    "--path_out_dir",
    #    default=OUTPUT_FOLDER + "models",
    #    type=str,
    #    help="Output directory for models and logs",
    #)
    parser.add_argument("--use_scheduler", action="store_true")
    parser.add_argument("--max_iterations", type=int, default=100)
    parser.add_argument("--learning_rate_fusion", type=float, default=1e-3,
                        help="LR for fusion blocks and view_embed")
    parser.add_argument("--learning_rate_decoder", type=float, default=1e-4,
                        help="LR for decoder")
    parser.add_argument("--learning_rate_encoder", type=float, default=1e-4,
                        help="LR for encoder (from-scratch training)")
    parser.add_argument(
        "--fusion_warmup_epochs",
        type=int,
        default=0,
        help=(
            "Epochs of this stage to run with fusion off before enabling it. "
            "The published run uses 0: the single-view stage is a separate "
            "MammoRegNet training, and fusion is on from the first epoch here."
        ),
    )
    parser.add_argument(
        "--fusion_levels",
        type=str,
        default="2,3,4",
        help=(
            "Comma-separated pyramid levels for cross-view fusion. "
            "0 is the finest encoder skip and 4 is the coarsest. "
            "2,3,4 are the three coarsest levels (32, 64 and 128 channels)."
        ),
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        help="Optional single-view MammoRegNet checkpoint. Default: train from scratch",
    )
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Freeze encoder weights (only useful with --pretrained_path)",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--num_epochs", type=int, default=250)
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--shuffle", default=True, type=bool)
    parser.add_argument("--pin_memory", default=True, type=bool)
    parser.add_argument("--dataset_length", type=int, default=1000)
    parser.add_argument("--lambda_regu", type=float, default=1.0)
    parser.add_argument("--lambda_cvc", type=float, default=0.0)
    parser.add_argument("--lambda_affine", type=float, default=0.0)
    #parser.add_argument(
    #    "--pretrained_path",
    #    default="outputs/MammoRegNet/models/model_249.pth",
    #    type=str,
    #    help="MammoRegNet checkpoint for transfer learning",
    #)
    parser.set_defaults(use_fusion=True)
    parser.add_argument(
        "--use_fusion",
        "--use-fusion",
        dest="use_fusion",
        action="store_true",
        help="Enable cross-view fusion blocks and view_embed (default)",
    )
    parser.add_argument(
        "--no_fusion",
        "--no-fusion",
        dest="use_fusion",
        action="store_false",
        help="Disable cross-view fusion blocks and view_embed",
    )
    parser.add_argument("--seed", default=11, type=int)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Resume training from training_checkpoint.pth or model_N.pth. "
            "Prefer training_checkpoint.pth to restore optimizer/scheduler."
        ),
    )
    parser.add_argument(
        "--coarse_deform",
        action="store_true",
        help="Use deformable head at the coarsest decoder level instead of affine",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.seed is not None:
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.data_root is None:
        args.data_root = "Dataset/affine_train" if args.coarse_deform else "Dataset/train"
    if args.metadata_path is None:
        args.metadata_path = os.path.join(args.data_root, "metadata_train_correct.csv")

    if args.coarse_deform and args.use_fusion:
        run_name = f"DVFF_{args.seed}"
    elif args.coarse_deform:
        run_name = f"DVFF_no_fusion_{args.seed}"
    elif args.use_fusion:
        run_name = f"DVFF_no_affine_{args.seed}"
    else:
        run_name = f"DVFF_no_affine_no_fusion_{args.seed}"
    OUTPUT_FOLDER = f"outputs/{run_name}/models"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    train_dataset = DVFFDataset(
        args.data_root,
        metadata_path=args.metadata_path,
        length=args.dataset_length,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        pin_memory=args.pin_memory,
        drop_last=True,
    )

    path_out_model = os.path.join(
        OUTPUT_FOLDER, "model_dvff_last_epoch.pth"
    )
    path_logger = os.path.join(OUTPUT_FOLDER, "train_dvff.log")

    fusion_levels = tuple(
        int(x.strip()) for x in args.fusion_levels.split(",") if x.strip() != ""
    )
    if not fusion_levels:
        raise ValueError("--fusion_levels must list at least one pyramid level")

    print(f"Training on device: {device}")
    print(f"data_root: {args.data_root}")
    print(f"masks_root: {train_dataset.masks_root}")
    print(f"metadata_path: {args.metadata_path}")
    print(f"Valid dual-view groups: {len(train_dataset.groups)}")
    print(f"use_fusion: {args.use_fusion}")
    print(f"fusion_levels: {fusion_levels}")
    print(f"fusion_warmup_epochs: {args.fusion_warmup_epochs}")
    print(f"coarse_affine: {not args.coarse_deform}")
    print(f"pretrained_path: {args.pretrained_path}")
    if args.resume:
        print(f"Resuming from: {args.resume}")
    print(f"Model will be saved to: {path_out_model}")

    train_dvff(
        train_loader,
        device,
        args.learning_rate_fusion,
        args.learning_rate_decoder,
        args.weight_decay,
        args.num_epochs,
        path_logger,
        path_out_model,
        OUTPUT_FOLDER,
        args.use_scheduler,
        args.max_iterations,
        lambda_regu=args.lambda_regu,
        lambda_cvc=args.lambda_cvc,
        lambda_affine=args.lambda_affine,
        pretrained_path=args.pretrained_path,
        use_fusion=args.use_fusion,
        resume_path=args.resume,
        coarse_affine=not args.coarse_deform,
        fusion_warmup_epochs=args.fusion_warmup_epochs,
        learning_rate_encoder=args.learning_rate_encoder,
        freeze_encoder=args.freeze_encoder,
        fusion_levels=fusion_levels,
    )


if __name__ == "__main__":
    main()
