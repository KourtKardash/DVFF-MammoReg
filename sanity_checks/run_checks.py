#!/usr/bin/env python
"""
Offline sanity checks for the repository.

These checks answer one question: does the code still run after the repository
was cleaned up for release? They exercise imports, model forward and backward
passes, the loss terms, the data loaders, the command-line entry points, and a
one-iteration training step — all on synthetic data, on the CPU, at a scale that
fits on a laptop.

They are NOT a test of the method and they produce no scientific result. Nothing
in ``Methods/`` or ``risk_prediction/`` is imported in a modified form, patched,
or written to: every check either imports the production modules as they are or
runs them as a subprocess inside a temporary working directory.

Usage:
    python sanity_checks/run_checks.py                 # everything
    python sanity_checks/run_checks.py --fast          # skip the slow subprocess checks
    python sanity_checks/run_checks.py --list          # show check names
    python sanity_checks/run_checks.py -k forward      # run checks matching a substring
    python sanity_checks/run_checks.py --report out.md # write the report elsewhere

Exit code is 0 only if every selected check passed.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SANITY_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SANITY_DIR))

# Everything runs on the CPU: the checks must not compete with a real training
# run for GPU memory, and they are sized for a laptop.
# Leave no trace in the working tree: the `artifacts` check asserts that no
# bytecode is present, and it must not be able to trip over its own siblings.
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["WANDB_MODE"] = "disabled"
os.environ.setdefault("OMP_NUM_THREADS", "4")

CHECKS = []


def check(name, description, slow=False):
    def decorator(fn):
        CHECKS.append({"name": name, "description": description, "slow": slow, "fn": fn})
        return fn
    return decorator


class CheckFailure(AssertionError):
    """Raised by a check to report a specific, actionable failure."""


def require(condition, message):
    if not condition:
        raise CheckFailure(message)


# --------------------------------------------------------------------------- #
# Static checks
# --------------------------------------------------------------------------- #

SOURCE_DIRS = ("Methods", "analysis", "risk_prediction", "segmentation", "sanity_checks")


def _python_files():
    for directory in SOURCE_DIRS:
        yield from sorted((REPO_ROOT / directory).rglob("*.py"))


@check("compile", "Every Python file parses")
def check_compile():
    """Compile in memory, so that the check itself leaves no __pycache__ behind."""
    failures = []
    count = 0
    for path in _python_files():
        count += 1
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except SyntaxError as exc:
            failures.append(f"{path.relative_to(REPO_ROOT)}:{exc.lineno}: {exc.msg}")
    require(not failures, "files that do not parse:\n  " + "\n  ".join(failures))
    return f"{count} files compile"


@check("imports", "Public modules import cleanly")
def check_imports():
    import importlib

    # Modules that are imported as ``Methods.<...>`` from the repository root.
    modules = [
        "Methods.DVFF.models.dvff_net",
        "Methods.DVFF.dataloaders.dataset_dvff",
        "Methods.DVFF.train.losses_dvff",
        "Methods.DVFF.train.train_dvff",
        "Methods.MammoRegNet.models.MammoRegNet",
        "Methods.MammoRegNet.dataloaders.dataset_pairs",
        "Methods.MammoRegNet.train.losses_mammoregnet",
        "Methods.MammoRegNet.utils.utils",
    ]
    for module in modules:
        importlib.import_module(module)

    # Vendored packages use top-level imports and are only importable from their
    # own root, which is how their entry points put them on sys.path. The
    # vendored OncoNet tree is only needed to unpickle the Mirai encoder, but it
    # must still import after the unused dataset loaders were removed.
    vendored = [
        ("Methods/IDIR", "import models.models, networks.networks, objectives.ncc"),
        ("Methods/transmorph", "import models.TransMorph, losses, utils"),
        ("Methods/voxelmorph", "import voxelmorph"),
        ("risk_prediction/LMV-Net/src", "import onconet.models, onconet.datasets.factory"),
    ]
    for directory, program in vendored:
        env = {"VXM_BACKEND": "pytorch"}
        result = subprocess.run(
            [sys.executable, "-c", program],
            cwd=str(REPO_ROOT / directory),
            env={**os.environ, **env},
            capture_output=True, text=True,
        )
        require(
            result.returncode == 0,
            f"importing the vendored package in {directory} failed:\n" + result.stderr[-2000:],
        )

    return f"{len(modules)} project modules + {len(vendored)} vendored packages"


# --------------------------------------------------------------------------- #
# Model checks
# --------------------------------------------------------------------------- #

SMALL_HW = (160, 128)  # divisible by 32, five encoder levels


def _small_batch(batch=1, hw=SMALL_HW):
    import torch

    torch.manual_seed(0)
    return torch.rand(batch, 1, *hw)


@check("forward", "DVFFNet runs in every ablation configuration")
def check_forward():
    import torch

    from Methods.DVFF.models.dvff_net import DVFFNet

    lines = []
    for use_fusion in (True, False):
        for coarse_affine in (True, False):
            model = DVFFNet(
                use_fusion=use_fusion,
                coarse_affine=coarse_affine,
                fusion_levels=(1, 2, 3),
                use_checkpoint=False,
            ).eval()
            x = _small_batch()
            with torch.no_grad():
                out = model(x, x, x, x, moving_mask_mlo=x, moving_mask_cc=x)
            for key in ("warped_mlo", "warped_cc", "warped_mask_mlo", "warped_mask_cc"):
                require(tuple(out[key].shape) == (1, 1, *SMALL_HW), f"{key} has shape {out[key].shape}")
            for key in ("flow_mlo", "flow_cc"):
                require(tuple(out[key].shape) == (1, 2, *SMALL_HW), f"{key} has shape {out[key].shape}")
                require(torch.isfinite(out[key]).all(), f"{key} contains non-finite values")
            trainable = sum(p.numel() for p in model.parameters())
            lines.append(
                f"fusion={use_fusion}, coarse_affine={coarse_affine}: "
                f"{trainable / 1e6:.2f}M params, flows finite"
            )
    return "; ".join(lines)


@check("fusion_off", "Disabling fusion reproduces single-projection registration")
def check_fusion_off():
    """
    The paper claims the fusion blocks can be switched off at inference without
    retraining, so that each projection is registered on its own. That claim is
    only true if the dual-view forward pass with fusion disabled agrees exactly
    with ``register_pair`` on the same inputs.
    """
    import torch

    from Methods.DVFF.models.dvff_net import DVFFNet

    model = DVFFNet(use_fusion=True, coarse_affine=False,
                    fusion_levels=(1, 2, 3), use_checkpoint=False).eval()
    model.set_fusion_enabled(False)

    mlo, cc = _small_batch(), _small_batch() * 0.5
    with torch.no_grad():
        dual = model(mlo, cc, mlo, cc)
        single_mlo = model.register_pair(mlo, mlo)
        single_cc = model.register_pair(cc, cc)

    require(torch.allclose(dual["flow_mlo"], single_mlo["flow"], atol=1e-6),
            "MLO flow differs between the dual-view pass with fusion off and register_pair")
    require(torch.allclose(dual["flow_cc"], single_cc["flow"], atol=1e-6),
            "CC flow differs between the dual-view pass with fusion off and register_pair")

    model.set_fusion_enabled(True)
    with torch.no_grad():
        fused = model(mlo, cc, mlo, cc)
    require(not torch.allclose(fused["flow_mlo"], dual["flow_mlo"], atol=1e-6),
            "enabling fusion did not change the predicted flow")
    return "fusion off == register_pair; fusion on changes the flow"


@check("transfer", "A single-view checkpoint loads into the dual-view network")
def check_transfer():
    import torch

    from Methods.DVFF.models.dvff_net import DVFFNet
    from Methods.MammoRegNet.models.MammoRegNet import MammoRegNet

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp) / "model_249.pth"
        torch.save(MammoRegNet(coarse_affine=False).state_dict(), checkpoint)

        model = DVFFNet(use_fusion=True, coarse_affine=False, fusion_levels=(1, 2, 3))
        new_modules, unexpected = model.load_pretrained_mammoregnet(str(checkpoint))

    require(new_modules, "no fusion parameters were reported as newly initialised")
    require(all(k.startswith(("fusion_blocks.", "view_embed.")) for k in new_modules),
            f"unexpected modules left uninitialised: {new_modules}")
    require(not unexpected, f"checkpoint keys the model did not consume: {unexpected}")

    model.set_encoder_trainable(False)
    encoder_trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    require(encoder_trainable == 0, "freezing the encoder left trainable parameters")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return (
        f"{len(new_modules)} fusion tensors initialised from scratch; "
        f"{trainable / 1e6:.2f}M of {total / 1e6:.2f}M parameters trainable with the encoder frozen"
    )


@check("jacobian", "The Jacobian-determinant penalty behaves as documented")
def check_jacobian():
    """
    Guards the channel order of the penalty. The determinant must be 1 for a
    constant translation, must stay positive for a mild expansion, and must be
    negative where the field folds the image onto itself.
    """
    import torch

    from Methods.MammoRegNet.train.losses_mammoregnet import NJD

    njd = NJD(Lambda=1.0)
    height, width = 16, 16
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )

    # Constant translation: no local deformation, determinant identically 1.
    translation = torch.zeros(1, 2, height, width)
    translation[:, 0] = 3.0
    translation[:, 1] = -2.0
    det = njd.get_Ja(translation.permute(0, 2, 3, 1))
    require(torch.allclose(det, torch.ones_like(det), atol=1e-5),
            f"translation should give determinant 1, got range [{det.min():.4f}, {det.max():.4f}]")
    require(float(njd.loss(translation)) == 0.0, "translation was penalised")

    # Uniform expansion: dx = 0.5 x, dy = 0.5 y  ->  det = 1.5 * 1.5 = 2.25.
    expansion = torch.zeros(1, 2, height, width)
    expansion[:, 0] = 0.5 * yy      # channel 0 is dy
    expansion[:, 1] = 0.5 * xx      # channel 1 is dx
    det = njd.get_Ja(expansion.permute(0, 2, 3, 1))
    require(torch.allclose(det, torch.full_like(det, 2.25), atol=1e-4),
            f"expansion should give determinant 2.25, got mean {det.mean():.4f}")
    require(float(njd.loss(expansion)) == 0.0, "an expansion with no folding was penalised")

    # Folding: dx = -2 x inverts the x axis, det = (1 - 2) * 1 = -1.
    folding = torch.zeros(1, 2, height, width)
    folding[:, 1] = -2.0 * xx
    det = njd.get_Ja(folding.permute(0, 2, 3, 1))
    require(torch.allclose(det, torch.full_like(det, -1.0), atol=1e-4),
            f"folding should give determinant -1, got mean {det.mean():.4f}")
    penalty = float(njd.loss(folding))
    expected = float((det.numel()))
    require(abs(penalty - expected) < 1e-3,
            f"folding penalty should be the summed magnitude {expected}, got {penalty}")
    return "determinant 1 / 2.25 / -1 on translation, expansion and folding"


@check("loss", "The training loss is finite and produces gradients")
def check_loss():
    import torch

    from Methods.DVFF.models.dvff_net import DVFFNet
    from Methods.DVFF.train.losses_dvff import dvff_registration_loss

    results = []
    for coarse_affine in (True, False):
        model = DVFFNet(use_fusion=True, coarse_affine=coarse_affine,
                        fusion_levels=(1, 2, 3), use_checkpoint=False)
        fixed_mlo, fixed_cc = _small_batch(), _small_batch()
        out = model(fixed_mlo, fixed_cc, _small_batch(), _small_batch())
        loss, metrics = dvff_registration_loss(
            out, fixed_mlo, fixed_cc, lambda_regu=1.0, coarse_affine=coarse_affine
        )
        require(torch.isfinite(loss), "loss is not finite")
        for key in ("total", "ncc", "ncc_mlo", "ncc_cc", "regu"):
            require(key in metrics, f"metrics is missing {key!r}")
        loss.backward()
        grads = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
        require(grads, "backward produced no gradients")
        require(all(torch.isfinite(p.grad).all() for p in grads), "non-finite gradients")
        results.append(f"coarse_affine={coarse_affine}: loss={float(loss.detach()):.4f}, {len(grads)} tensors with gradients")
    return "; ".join(results)


@check("segmentation_unet", "The breast-segmentation U-Net and its loss behave")
def check_segmentation_unet():
    """
    Covers the network shape contract and the composite loss of the supplementary
    material: 0.5*BCE + 0.5*(1 - Dice). A perfect prediction must drive the Dice
    term to zero, and an inverted one must drive it to one.
    """
    import torch

    sys.path.insert(0, str(REPO_ROOT / "segmentation"))
    from nn import UNet, BCEDiceLoss, DiceLoss

    model = UNet(n_channels=1, n_classes=1).eval()
    x = torch.rand(1, 1, 64, 48)
    with torch.no_grad():
        logits = model(x)
    require(tuple(logits.shape) == (1, 1, 64, 48),
            f"the U-Net changed the spatial size: {tuple(logits.shape)}")

    dice = DiceLoss()
    target = (torch.rand(2, 1, 32, 32) > 0.5).float()
    perfect = float(dice(target, target))
    inverted = float(dice(1.0 - target, target))
    require(perfect < 1e-2, f"Dice loss on a perfect prediction should be ~0, got {perfect:.4f}")
    require(inverted > 0.99, f"Dice loss on an inverted prediction should be ~1, got {inverted:.4f}")

    criterion = BCEDiceLoss()
    require(abs(criterion.bce_weight - 0.5) < 1e-9 and abs(criterion.dice_weight - 0.5) < 1e-9,
            "the composite loss no longer weights BCE and Dice equally, which is "
            "what the supplementary material specifies")
    loss = criterion(torch.zeros(2, 1, 32, 32, requires_grad=True), target)
    require(torch.isfinite(loss), "the composite loss is not finite")

    # Every return path of predict() must agree on the output size, including the
    # one taken when the thresholded prediction has no connected component.
    import numpy as np
    from PIL import Image

    from predict_for_train import predict

    with tempfile.TemporaryDirectory() as tmp:
        image_path = Path(tmp) / "case.png"
        Image.fromarray(np.zeros((70, 55), dtype=np.uint8)).save(image_path)
        empty = predict(model, str(image_path), device="cpu", threshold=1e9)
        require(
            empty.size == (55, 70),
            f"an empty prediction came back as {empty.size}, the image is (55, 70); "
            "every return path of predict() must use the original image size",
        )
        require(
            not np.array(empty).any(),
            "an empty prediction is not all zeros",
        )

    return (
        f"U-Net preserves the input size; Dice loss {perfect:.3f} on a perfect "
        f"and {inverted:.3f} on an inverted prediction; BCE/Dice weights 0.5/0.5; "
        "empty predictions keep the original image size"
    )


# --------------------------------------------------------------------------- #
# Data checks
# --------------------------------------------------------------------------- #

@check("dataloaders", "The loaders read the on-disk layout described in the README")
def check_dataloaders():
    import pandas as pd
    import torch

    import mock_data
    from Methods.DVFF.dataloaders.dataset_dvff import DVFFDataset, build_dvff_groups
    from Methods.MammoRegNet.dataloaders.dataset_pairs import MammoDataset

    cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as tmp:
        mock_data.build(Path(tmp))
        # Both loaders resolve metadata paths relative to the working directory,
        # exactly as they do when a training run is launched from the data root.
        os.chdir(tmp)
        try:
            metadata_path = "Dataset/affine_train/metadata_train_correct.csv"
            dvff = DVFFDataset(
                "Dataset/affine_train", metadata_path=metadata_path, length=4
            )
            require(
                len(dvff.groups) == len(mock_data.TRAIN_SPEC),
                f"expected {len(mock_data.TRAIN_SPEC)} breast groups, found {len(dvff.groups)}",
            )
            sample = dvff[0]
            expected = (1, mock_data.IMAGE_SIZE[1], mock_data.IMAGE_SIZE[0])
            for key in ("fixed_mlo", "fixed_cc", "moving_mlo", "moving_cc"):
                tensor = sample[key]
                require(tuple(tensor.shape) == expected, f"{key} has shape {tuple(tensor.shape)}")
                require(tensor.dtype == torch.float32, f"{key} has dtype {tensor.dtype}")
                require(float(tensor.max()) <= 1.0, f"{key} is not scaled to [0, 1]")
            require(
                float(sample["fixed_mlo"].min()) == 0.0,
                "the breast mask was not applied: the background is not zero",
            )

            groups = build_dvff_groups(pd.read_csv(metadata_path))
            require(groups, "build_dvff_groups found nothing in the mock metadata")

            single = MammoDataset("Dataset/affine_train")
            item = single[0]
            require(tuple(item["img_fix"].shape) == expected,
                    f"img_fix has shape {tuple(item['img_fix'].shape)}")
            require(tuple(item["img_mov"].shape) == expected,
                    f"img_mov has shape {tuple(item['img_mov'].shape)}")
            groups_found = len(dvff.groups)
        finally:
            os.chdir(cwd)

    return (
        f"DVFFDataset: {groups_found} breast groups, masked samples at "
        f"{mock_data.IMAGE_SIZE[0]}x{mock_data.IMAGE_SIZE[1]}; MammoDataset: pair loaded"
    )


# --------------------------------------------------------------------------- #
# Subprocess checks
# --------------------------------------------------------------------------- #

def _run(command, cwd, env=None, timeout=900):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT)] + ([env["PYTHONPATH"]] if env and "PYTHONPATH" in env else [])
    )
    if env:
        environment.update(env)
    return subprocess.run(
        [sys.executable, *command], cwd=str(cwd), env=environment,
        capture_output=True, text=True, timeout=timeout,
    )


ENTRY_POINTS = [
    ("Methods/DVFF/train/main_train_dvff.py", REPO_ROOT),
    ("Methods/DVFF/evaluate/inference_dvff.py", REPO_ROOT),
    ("Methods/DVFF/evaluate/inference_dvff_fusion_off.py", REPO_ROOT),
    ("Methods/MammoRegNet/train/main_train_mammoregnet.py", REPO_ROOT),
    ("Methods/MammoRegNet/evaluate/inference_original.py", REPO_ROOT),
    ("Methods/IDIR/run.py", REPO_ROOT),
    ("Methods/voxelmorph/scripts/torch/train.py", REPO_ROOT),
    ("risk_prediction/scripts/cache_affine_transforms.py", REPO_ROOT),
    ("analysis/wilcoxon_rTRE_patients.py", REPO_ROOT),
    ("segmentation/train.py", REPO_ROOT),
    ("segmentation/predict_for_train.py", REPO_ROOT),
    ("main_train.py", REPO_ROOT / "risk_prediction/LMV-Net/src"),
    ("main_test.py", REPO_ROOT / "risk_prediction/LMV-Net/src"),
]


# Methods/transmorph/train_TransMorph.py is deliberately absent: it selects a CUDA
# device at module level, before its argument parser is reached, so it cannot be
# probed on a CPU-only machine. Its syntax and imports are covered by the
# `compile` and `imports` checks.
@check("entrypoints", "Every command-line entry point starts and prints its usage", slow=True)
def check_entrypoints():
    failures = []
    for script, cwd in ENTRY_POINTS:
        result = _run([script, "--help"], cwd=cwd, timeout=300)
        if result.returncode != 0 or "usage:" not in result.stdout.lower():
            failures.append(f"{script}\n    {result.stderr.strip()[-600:]}")

    # A couple of scripts create their output directory at module level, before
    # the argument parser is reached, and resolve it against the repository root
    # regardless of the working directory. Remove what they left, but only if it
    # is empty, so that a real run's results are never touched.
    for directory in sorted((REPO_ROOT / "outputs").glob("**/"), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass

    require(not failures, "entry points that do not start:\n  " + "\n  ".join(failures))
    return f"{len(ENTRY_POINTS)} entry points respond to --help"


@check("shell_scripts", "Shell scripts are syntactically valid", slow=False)
def check_shell_scripts():
    scripts = sorted(REPO_ROOT.rglob("*.sh"))
    scripts = [s for s in scripts if ".venv" not in s.parts]
    require(scripts, "no shell scripts found")
    failures = []
    for script in scripts:
        result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        if result.returncode != 0:
            failures.append(f"{script.relative_to(REPO_ROOT)}: {result.stderr.strip()}")
    require(not failures, "shell syntax errors:\n  " + "\n  ".join(failures))
    return f"{len(scripts)} shell scripts parse"


@check("train_step", "One training iteration of the proposed method completes", slow=True)
def check_train_step():
    """
    Runs the real training entry point, unmodified, for a single iteration at the
    native 896x1152 resolution on the CPU. This is the check that would catch a
    broken import path, a renamed argument, or a checkpoint that cannot be
    written after the rename to DVFF.
    """
    import mock_data

    with tempfile.TemporaryDirectory() as tmp:
        dataset_dir = mock_data.build(Path(tmp))
        started = time.time()
        result = _run(
            [
                str(REPO_ROOT / "Methods/DVFF/train/main_train_dvff.py"),
                "--data_root", str(dataset_dir / "affine_train"),
                "--metadata_path", str(dataset_dir / "affine_train" / "metadata_train_correct.csv"),
                "--num_epochs", "1",
                "--max_iterations", "1",
                "--batch_size", "1",
                "--dataset_length", "1",
                "--num_workers", "0",
                "--fusion_warmup_epochs", "0",
                "--fusion_levels", "1,2,3",
                "--coarse_deform",
            ],
            cwd=tmp,
            timeout=1800,
        )
        elapsed = time.time() - started
        require(
            result.returncode == 0,
            "training entry point failed:\n"
            + result.stdout[-1500:] + "\n" + result.stderr[-2500:],
        )
        checkpoints = list(Path(tmp).rglob("model_0.pth"))
        require(checkpoints, "no checkpoint was written by the training step")
        logs = list(Path(tmp).rglob("train_dvff.log"))
        require(logs, "no training log was written")

    return f"one epoch, one iteration at 896x1152 in {elapsed:.0f}s; checkpoint and log written"


@check("unreg_eval", "The evaluation path runs end to end and reports rTRE", slow=True)
def check_unreg_eval():
    """
    ``Methods/Unreg/inference.py`` is the cheapest method that exercises the whole
    evaluation path: landmark rescaling, image and mask loading, the similarity
    metrics, and the rTRE computation shared by every other method.
    """
    import json

    import mock_data

    with tempfile.TemporaryDirectory() as tmp:
        mock_data.build(Path(tmp))
        result = _run([str(REPO_ROOT / "Methods/Unreg/inference.py")], cwd=tmp, timeout=600)
        require(
            result.returncode == 0,
            "unregistered evaluation failed:\n" + result.stdout[-1500:] + "\n" + result.stderr[-2000:],
        )
        summary = Path(tmp) / "outputs" / "Unreg" / "Unreg.json"
        require(summary.is_file(), "no metrics JSON was written")
        metrics = json.loads(summary.read_text())
        require("Mean Metrics" in metrics, "the metrics JSON has no aggregate entry")
        for key in ("MSE", "SSIM", "MI", "CC", "DSC"):
            require(key in metrics["Mean Metrics"], f"aggregate metrics are missing {key}")
        require("Overall rTRE" in result.stdout or "rTRE" in result.stdout,
                "rTRE was not reported:\n" + result.stdout[-800:])

    return "metrics JSON written, rTRE reported for the mock evaluation set"


@check("segmentation_roundtrip", "Segmentation trains for an epoch and predicts masks", slow=True)
def check_segmentation_roundtrip():
    """
    Runs both segmentation entry points, unmodified, on a handful of synthetic
    images: one training epoch writes a checkpoint, and prediction reads that
    checkpoint back and writes one mask per image at the original resolution.
    """
    import numpy as np
    from PIL import Image

    import mock_data

    with tempfile.TemporaryDirectory() as tmp:
        images_dir, masks_dir = mock_data.build_segmentation(Path(tmp))
        checkpoint = Path(tmp) / "unet.pth"

        train = _run(
            [
                str(REPO_ROOT / "segmentation/train.py"),
                "--images_dir", str(images_dir),
                "--masks_dir", str(masks_dir),
                "--checkpoint", str(checkpoint),
                "--num_epochs", "1",
                "--batch_size", "2",
                "--num_workers", "0",
            ],
            cwd=tmp, timeout=900,
        )
        require(
            train.returncode == 0,
            "segmentation training failed:\n" + train.stdout[-1200:] + "\n" + train.stderr[-2000:],
        )
        require(checkpoint.is_file(), "training wrote no checkpoint")
        require("Train Loss" in train.stdout, "training reported no loss:\n" + train.stdout[-600:])

        # The threshold is applied to raw logits, and one epoch on six synthetic
        # images is nowhere near enough to push them above the default of 0.5.
        # A permissive threshold keeps every prediction non-empty, so that this
        # check exercises the normal post-processing path — morphology, hole
        # filling, largest connected component, resize back — rather than the
        # degenerate "nothing was found" branch. See the note in the report about
        # that branch.
        predicted_dir = Path(tmp) / "predicted_masks"
        predict = _run(
            [
                str(REPO_ROOT / "segmentation/predict_for_train.py"),
                "--images_dir", str(images_dir),
                "--masks_dir", str(predicted_dir),
                "--checkpoint", str(checkpoint),
                # Written as one token: argparse would read a bare "-1e9" as a flag.
                "--threshold=-1e9",
            ],
            cwd=tmp, timeout=900,
        )
        require(
            predict.returncode == 0,
            "segmentation prediction failed:\n" + predict.stdout[-1200:] + "\n" + predict.stderr[-2000:],
        )

        sources = sorted(images_dir.rglob("*.png"))
        produced = sorted(predicted_dir.rglob("*.png"))
        require(
            len(produced) == len(sources),
            f"expected {len(sources)} masks, found {len(produced)}",
        )
        for source, mask_path in zip(sources, produced):
            with Image.open(source) as source_image, Image.open(mask_path) as mask_image:
                require(
                    mask_image.size == source_image.size,
                    f"{mask_path.name} is {mask_image.size}, the image is {source_image.size}; "
                    "masks must come back at the original resolution",
                )
                values = set(np.unique(np.array(mask_image)))
                require(
                    values <= {0, 255},
                    f"{mask_path.name} is not binary, it holds {sorted(values)[:5]}",
                )

    return f"one epoch trained, {len(produced)} binary masks written at the original size"


@check("holm", "The Holm correction matches a worked example")
def check_holm():
    """
    The statistics are the main argument of the paper, so the correction is
    checked against a case that can be verified by hand rather than only for
    plausibility. For p = [0.01, 0.02, 0.03, 0.04] with n = 4, Holm multiplies the
    sorted values by 4, 3, 2, 1 and then enforces monotonicity, which gives
    [0.04, 0.06, 0.06, 0.06].
    """
    sys.path.insert(0, str(REPO_ROOT / "analysis"))
    from wilcoxon_rTRE_patients import holm

    adjusted = holm([0.01, 0.02, 0.03, 0.04])
    expected = [0.04, 0.06, 0.06, 0.06]
    for got, want in zip(adjusted, expected):
        require(abs(got - want) < 1e-12, f"Holm gave {adjusted}, expected {expected}")

    require(holm([0.5, 0.5]) == [1.0, 1.0], "Holm did not clamp adjusted values at 1")
    single = holm([0.03])
    require(abs(single[0] - 0.03) < 1e-12, f"Holm on one p value changed it: {single}")

    # Order of the input must not matter beyond a matching permutation.
    forward = holm([0.001, 0.2, 0.04])
    backward = holm([0.04, 0.2, 0.001])
    require(
        abs(forward[0] - backward[2]) < 1e-12 and abs(forward[2] - backward[0]) < 1e-12,
        "Holm is not permutation-equivariant",
    )
    return "worked example, clamping at 1, and permutation-equivariance all hold"


@check("analysis", "The patient-level comparison runs and covers every pair", slow=True)
def check_analysis():
    """
    Runs the analysis entry point on synthetic landmark files for all conditions.
    Checks the two numbers the paper depends on: the aggregation is per patient,
    not per image, and every condition is compared with every other.
    """
    import csv
    import itertools

    import mock_data

    sys.path.insert(0, str(REPO_ROOT / "analysis"))
    from wilcoxon_rTRE_patients import FAMILIES, METHODS

    seed = 11
    folders = [template.format(seed=seed) for _label, template, _families in METHODS]
    n_patients = 8
    family_sizes = {
        family: sum(family in families for _label, _template, families in METHODS)
        for family in FAMILIES
    }
    family_pairs = {family: n * (n - 1) // 2 for family, n in family_sizes.items()}
    require(
        family_pairs == {"benchmark": 55, "ablation": 21},
        f"the families do not match the paper (55 and 21 comparisons): {family_pairs}",
    )
    expected_pairs = sum(family_pairs.values())

    with tempfile.TemporaryDirectory() as tmp:
        outputs = mock_data.build_analysis(Path(tmp), folders, n_patients=n_patients)
        result = _run(
            [
                str(REPO_ROOT / "analysis/wilcoxon_rTRE_patients.py"),
                "--outputs_root", str(outputs),
                "--seed", str(seed),
            ],
            cwd=tmp, timeout=600,
        )
        require(
            result.returncode == 0,
            "the analysis script failed:\n" + result.stdout[-1200:] + "\n" + result.stderr[-2000:],
        )
        require(
            f"Patients: {n_patients}" in result.stdout,
            f"expected {n_patients} patients from {2 * n_patients} images; "
            "the aggregation is not per patient:\n" + result.stdout[-800:],
        )

        csv_path = outputs / "wilcoxon_rTRE_all_pairwise.csv"
        md_path = outputs / "wilcoxon_rTRE_all_pairwise.md"
        require(csv_path.is_file(), "no CSV of pairwise results was written")
        require(md_path.is_file(), "no Markdown report was written")

        with csv_path.open() as handle:
            rows = list(csv.DictReader(handle))
        require(
            len(rows) == expected_pairs,
            f"expected {expected_pairs} pairs from {len(METHODS)} conditions, got {len(rows)}",
        )
        for row in rows:
            raw, adjusted = float(row["p_two_sided"]), float(row["p_holm"])
            require(
                adjusted >= raw - 1e-12,
                f"Holm-adjusted p is below the raw p for {row['A']} vs {row['B']}",
            )
            require(0.0 <= adjusted <= 1.0, f"adjusted p out of range: {adjusted}")

        report = md_path.read_text()
        for family in FAMILIES:
            header = (
                f"## {family.capitalize()} family "
                f"({family_sizes[family]} conditions, {family_pairs[family]} comparisons)"
            )
            require(header in report,
                    f"the report does not state the {family} family and its comparisons")
            n_rows = sum(row["family"] == family for row in rows)
            require(n_rows == family_pairs[family],
                    f"expected {family_pairs[family]} {family} rows, got {n_rows}")

    return (
        f"{len(METHODS)} conditions, {family_pairs['benchmark']} + "
        f"{family_pairs['ablation']} pairs, "
        f"{n_patients} patients from {2 * n_patients} images; CSV and report written"
    )


@check("risk_backends", "The downstream pipeline resolves the renamed registration backend", slow=True)
def check_risk_backends():
    """
    The rename from the pre-release method name to DVFF reaches into the
    downstream risk-prediction code, where the backend is selected by a string.
    This runs inside ``src/`` because that package expects to be imported from
    there, exactly as the shell scripts do it.
    """
    src = REPO_ROOT / "risk_prediction" / "LMV-Net" / "src"
    program = f"""
import sys, torch, tempfile, os
sys.path.insert(0, {str(REPO_ROOT)!r})
from registration.factory import (
    REGISTRATION_METHODS, normalize_registration_method,
    load_registration_backend, default_ants_prewarp, default_coarse_affine,
)
assert "dvff" in REGISTRATION_METHODS, REGISTRATION_METHODS
assert normalize_registration_method("DVFF") == "dvff"
assert default_ants_prewarp("dvff") is True
assert default_coarse_affine("dvff") is False

from Methods.DVFF.models.dvff_net import DVFFNet
model = DVFFNet(use_fusion=True, coarse_affine=False, fusion_levels=(1, 2, 3))
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "dvff.pth")
    torch.save(model.state_dict(), path)
    backend = load_registration_backend(
        "dvff",
        checkpoint_path=path,
        map_location="cpu",
        coarse_affine=default_coarse_affine("dvff"),
    )
print("backend:", type(backend).__name__)
backend_unreg = load_registration_backend("unreg")
print("unreg:", type(backend_unreg).__name__)
"""
    result = _run(["-c", program], cwd=src, timeout=600)
    require(
        result.returncode == 0,
        "registration backend resolution failed:\n" + result.stdout[-1000:] + "\n" + result.stderr[-2500:],
    )
    require("DVFFBackend" in result.stdout, "the DVFF backend class was not returned:\n" + result.stdout)
    return result.stdout.strip().replace("\n", "; ")


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fast", action="store_true", help="skip the slow subprocess checks")
    parser.add_argument("--list", action="store_true", help="list the checks and exit")
    parser.add_argument("-k", dest="pattern", default=None, help="only run checks whose name contains this")
    parser.add_argument("--report", default=str(SANITY_DIR / "report.md"), help="where to write the markdown report")
    args = parser.parse_args()

    if args.list:
        for entry in CHECKS:
            print(f"{entry['name']:16s} {'[slow] ' if entry['slow'] else '       '}{entry['description']}")
        return 0

    selected = [
        entry for entry in CHECKS
        if (not args.fast or not entry["slow"])
        and (args.pattern is None or args.pattern in entry["name"])
    ]
    if not selected:
        print("No checks matched.")
        return 1

    print(f"Running {len(selected)} of {len(CHECKS)} checks on {sys.version.split()[0]}, CPU only.\n")
    results = []
    for entry in selected:
        started = time.time()
        print(f"  {entry['name']:16s} ... ", end="", flush=True)
        try:
            detail = entry["fn"]()
            status, message = "pass", detail or ""
        except CheckFailure as exc:
            status, message = "FAIL", str(exc)
        except Exception:
            status, message = "ERROR", traceback.format_exc()
        elapsed = time.time() - started
        print(f"{status:5s} ({elapsed:5.1f}s)")
        if status != "pass":
            print("      " + message.replace("\n", "\n      ")[:4000])
        results.append({**entry, "status": status, "message": message, "elapsed": elapsed})

    passed = sum(r["status"] == "pass" for r in results)
    print(f"\n{passed}/{len(results)} checks passed "
          f"({sum(r['elapsed'] for r in results):.0f}s total).")

    report = Path(args.report)
    report.write_text(_render_report(results), encoding="utf-8")
    print(f"Report written to {report}")
    return 0 if passed == len(results) else 1


def _render_report(results):
    lines = [
        "# Sanity check report",
        "",
        f"- Date (UTC): {time.strftime('%Y-%m-%d %H:%M', time.gmtime())}",
        f"- Python: {sys.version.split()[0]}",
        "- Device: CPU only (`CUDA_VISIBLE_DEVICES=\"\"`)",
        "- Data: synthetic, generated in a temporary directory",
        "",
        "| Check | Result | Time | What it shows |",
        "|---|---|---:|---|",
    ]
    for result in results:
        mark = {"pass": "pass", "FAIL": "**FAIL**", "ERROR": "**ERROR**"}[result["status"]]
        detail = result["message"].splitlines()[0] if result["status"] == "pass" else "see below"
        lines.append(
            f"| `{result['name']}` | {mark} | {result['elapsed']:.1f}s | {detail} |"
        )

    failures = [r for r in results if r["status"] != "pass"]
    if failures:
        lines += ["", "## Failures", ""]
        for result in failures:
            lines += [f"### `{result['name']}` — {result['description']}", "",
                      "```", result["message"].strip()[:4000], "```", ""]
    else:
        lines += ["", "All checks passed.", ""]

    lines += [
        "## What each check covers",
        "",
    ]
    for result in results:
        lines.append(f"- **`{result['name']}`** — {result['description']}.")
        if result["status"] == "pass" and result["message"]:
            lines.append(f"  Observed: {result['message']}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
