"""Safe checkpoint loading on cluster filesystems (avoid NFS mmap SIGBUS)."""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
import time
from pathlib import Path


def _checkpoint_sizes_match(src: Path, dst: Path) -> bool:
    try:
        return dst.is_file() and dst.stat().st_size == src.stat().st_size
    except OSError:
        return False


@contextlib.contextmanager
def _exclusive_file_lock(lock_path: Path, timeout_s: float = 600.0):
    """
    Cross-process lock for multi-GPU jobs (all ranks stage the same checkpoint).
    Falls back to a no-op lock when fcntl is unavailable (e.g. Windows dev).
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError:
        yield
        return

    deadline = time.monotonic() + timeout_s
    with open(lock_path, "a+b") as lock_file:
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for checkpoint staging lock: {lock_path}"
                    )
                time.sleep(0.25)

        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def ensure_local_checkpoint(checkpoint_path: str | Path) -> str:
    """
    Copy a registration checkpoint to local disk before ``torch.load``.

    Reading large ``.pt`` / ``.pth.tar`` files directly from NFS/Lustre with
    multiple DDP ranks often triggers SIGBUS. Set ``STAGE_CHECKPOINT_LOCAL=0``
    to skip (e.g. when the path is already on node-local storage).
    """
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Registration checkpoint not found: {path}")

    if os.environ.get("STAGE_CHECKPOINT_LOCAL", "1") != "1":
        return str(path)

    local_root = os.environ.get(
        "LMV_CHECKPOINT_STAGE_DIR",
        os.path.join(
            os.environ.get("TMPDIR", tempfile.gettempdir()),
            f"lmvnet_ckpt_{os.environ.get('SLURM_JOB_ID', os.getpid())}",
        ),
    )
    os.makedirs(local_root, exist_ok=True)
    local_path = Path(local_root) / path.name
    lock_path = local_path.with_suffix(local_path.suffix + ".lock")

    if _checkpoint_sizes_match(path, local_path):
        return str(local_path)

    with _exclusive_file_lock(lock_path):
        if _checkpoint_sizes_match(path, local_path):
            return str(local_path)

        tmp_path = local_path.with_name(f"{local_path.name}.{os.getpid()}.partial")
        if tmp_path.exists():
            tmp_path.unlink()
        try:
            # copyfile avoids xattr/metadata races seen with copy2 on /tmp.
            shutil.copyfile(path, tmp_path)
            os.replace(tmp_path, local_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    if not _checkpoint_sizes_match(path, local_path):
        raise RuntimeError(
            f"Failed to stage checkpoint to local disk: {local_path}"
        )
    return str(local_path)


def torch_load_checkpoint(path: str, map_location="cpu"):
    """``torch.load`` with CPU map_location and optional ``mmap=False``."""
    import torch

    kwargs = {"map_location": map_location, "weights_only": False}
    try:
        return torch.load(path, mmap=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)
