"""Reproducibility helpers: seeding, provenance, atomic checkpoints."""

from __future__ import annotations

import json
import os
import platform
import random
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Deterministic kernels are slower; enable unless the user disables them.
        if os.environ.get("DOSESYN_DETERMINISTIC", "1") == "1":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover
        pass


def git_commit(root: str | Path = ".") -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(root), stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return None


def git_status_porcelain(root: str | Path = ".") -> str:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(root), stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return ""


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def environment_record() -> Dict[str, Any]:
    import pandas as pd

    rec: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    try:
        import sklearn

        rec["sklearn"] = sklearn.__version__
    except ImportError:
        pass
    try:
        import torch

        rec["torch"] = torch.__version__
        rec["cuda_available"] = bool(torch.cuda.is_available())
        rec["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            rec["gpu"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        pass
    try:
        import torch_geometric

        rec["torch_geometric"] = torch_geometric.__version__
    except ImportError:
        pass
    return rec


def save_json_atomic(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2, default=_json_default)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def save_torch_atomic(payload: Dict[str, Any], path: str | Path) -> Path:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path
