# Sanity checks

Offline smoke tests for this repository. They answer one question: **does the
code still run?** They do not evaluate the method, they produce no scientific
result, and they must never be read as one.

Everything runs on synthetic data, on the CPU, in a few minutes on a laptop.

```bash
python sanity_checks/run_checks.py          # everything (~4 min)
python sanity_checks/run_checks.py --fast   # skip the subprocess checks (~30 s)
python sanity_checks/run_checks.py --list   # what each check does
python sanity_checks/run_checks.py -k loss  # one check by name
```

The exit code is 0 only if every selected check passed. A markdown report is
written to `sanity_checks/report.md`.

## They do not touch the production code

This is a deliberate constraint, because the point of the checks is to test the
code that produced the paper's numbers, not a variant of it.

- Nothing in `Methods/`, `analysis/`, `segmentation/` or `risk_prediction/` is patched, monkeypatched, or
  imported in a modified form. The checks either import those modules exactly as
  they are, or run their entry points as subprocesses.
- `mock_data.py` writes its synthetic dataset into a temporary directory, and the
  subprocess checks use that directory as their working directory. The
  production scripts therefore resolve their usual relative paths (`Dataset/…`,
  `outputs/…`) inside the temporary directory, and nothing is written into the
  repository or into any real data root.
- `CUDA_VISIBLE_DEVICES` is cleared and `PYTHONDONTWRITEBYTECODE` is set for the
  whole run, so the checks cannot take GPU memory from a real training job and
  cannot leave `__pycache__` behind.
- The one training check runs a single iteration, with a batch of one, for one
  epoch, at the pipeline's native 896x1152 resolution. About 30 s on CPU.

## What is covered

| Check | What it would catch |
|---|---|
| `compile` | a file left unparseable by the rename or the deletions |
| `imports` | a stale import path; the vendored OncoNet tree broken by the removal of unused dataset loaders |
| `forward` | the network failing in any of the four fusion / decoder-head configurations |
| `fusion_off` | the dual-view pass with fusion disabled disagreeing with `register_pair`, which is the claim that fusion can be switched off at inference |
| `transfer` | the single-view checkpoint no longer loading into the dual-view network, or the encoder freeze not taking effect |
| `jacobian` | a regression in the Jacobian-determinant penalty, whose channel order differs from upstream; asserts determinant 1 for a translation, 2.25 for a known expansion, and −1 where the field folds |
| `loss` | a non-finite loss or a missing gradient |
| `segmentation_unet` | the mask U-Net changing the spatial size, the composite loss drifting from the 0.5/0.5 BCE + Dice of the supplementary material, or a return path of `predict()` disagreeing on the output size |
| `holm` | the multiple-comparison correction drifting; checked against a worked example, not only for plausibility |
| `dataloaders` | the metadata layout, pairing, or breast masking breaking |
| `entrypoints` | a command-line entry point that no longer starts |
| `shell_scripts` | a shell syntax error |
| `train_step` | the training entry point failing to run an iteration or write a checkpoint |
| `unreg_eval` | the evaluation path — landmark rescaling, metrics, rTRE — failing end to end |
| `segmentation_roundtrip` | the mask U-Net failing to train an epoch, or prediction failing to write binary masks at the original image resolution |
| `analysis` | the patient-level comparison aggregating per image instead of per patient, or dropping pairs from the all-vs-all comparison |
| `risk_backends` | the downstream pipeline failing to resolve or load the renamed registration backend |

`segmentation_roundtrip` passes `--threshold=-1e9` to the prediction step. The
threshold is applied to raw logits, and one epoch on six synthetic images leaves
them far below the default of `0.5`; a permissive threshold keeps the prediction
non-empty so that the check exercises the normal post-processing path — morphology,
hole filling, largest connected component, resize back. The empty-mask path is
covered separately, in process.

`Methods/transmorph/train_TransMorph.py` is not probed by `entrypoints`: it
selects a CUDA device at module level, before its argument parser is reached, so
it cannot be started on a CPU-only machine. Its syntax and imports are covered by
`compile` and `imports`.

## Not covered

These are functional checks only. They say nothing about numerical agreement with
the paper: reproducing the reported tables needs the real evaluation data and the
trained weights, see `scripts/reproduce_table3.sh`. They also make no claim about
repository hygiene or about what may or may not be published.
