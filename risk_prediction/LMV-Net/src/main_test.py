import argparse
import os
import random
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator

from dataloaders import BreastCancerRiskDatasetCSAWCC
from evaluate import test_risk
from registration.cli import add_registration_arguments, dataset_registration_kwargs, resolve_registration_settings


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--path_out_dir", type=str, required=True)
    add_registration_arguments(parser)
    parser.add_argument("--mirai_encoder_path", type=str, default=None, help="Path to MIRAI encoder weights (optional)")
    parser.add_argument("--id_training", type=int, required=True)
    parser.add_argument("--path_test_folder", type=str, required=True)
    parser.add_argument("--num_epoch", type=int)
    parser.add_argument("--dataset", type=str, default="CSAW", choices=["CSAW"])

    parser.add_argument("--early_stop", type=str, default="False")
    parser.add_argument("--best_model", type=str, default="False")
    parser.add_argument("--use_checkppoint", type=str, default="False")

    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", default=2023, type=int)

    return parser.parse_args()


def main():
    args = parse_arguments()
    resolve_registration_settings(args)
    accelerator = Accelerator()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    if args.dataset != "CSAW":
        raise ValueError(
            f"--dataset must be 'CSAW', got {args.dataset!r}. The downstream "
            "experiment reported in the paper runs on CSAW-CC only."
        )
    print("Use CSAW-CC dataset")
    ds_reg_kwargs = dataset_registration_kwargs(args)
    test_dataset = BreastCancerRiskDatasetCSAWCC(
        args.csv_file, args.data_root, "test", transforms=None, **ds_reg_kwargs
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        pin_memory=args.pin_memory,
    )

    model_filename = f"best_model_risk_prediction_id-{args.id_training}.pth"
    path_model_risk = os.path.join(args.path_out_dir, model_filename)
    logg_filename = f"test_risk_prediction_training_id_{args.id_training}.log"
    path_logger = os.path.join(args.path_test_folder, logg_filename)

    if accelerator.is_main_process:
        os.makedirs(args.path_test_folder, exist_ok=True)
        print("Model path:", path_model_risk)
        print("Logger path:", path_logger)

    test_risk(
        args=args,
        test_loader=test_loader,
        path_model=path_model_risk,
        out_dir=args.path_out_dir,
        path_logger=path_logger,
        accelerator=accelerator,
    )


if __name__ == "__main__":
    main()
