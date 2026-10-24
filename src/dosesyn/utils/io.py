"""IO helpers: hashing, format signatures, atomic JSON, Parquet, stable IDs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import pandas as pd

# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------
# ZIP local file header: OOXML (.xlsx/.xlsm/.docx...) is a ZIP container.
OOXML_MAGIC = b"PK\x03\x04"
# OLE2 / Compound File Binary: legacy BIFF .xls.
OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
GZIP_MAGIC = b"\x1f\x8b"
XMZ_MAGIC = b"MZ"


def read_signature(path: str | Path, nbytes: int = 16) -> bytes:
    with open(path, "rb") as fh:
        return fh.read(nbytes)


def detect_container(path: str | Path) -> Dict[str, Any]:
    """Detect the true container family from magic bytes (never the suffix)."""
    path = Path(path)
    sig = read_signature(path)
    suffix = path.suffix.lower()
    info: Dict[str, Any] = {
        "signature_hex": sig.hex(),
        "extension": suffix,
        "family": "unknown",
        "container": "unknown",
        "is_ooxml": False,
        "is_legacy_xls": False,
        "extension_matches_content": None,
        "suggested_reader": None,
        "note": "",
    }
    if sig.startswith(OOXML_MAGIC):
        info["family"] = "ooxml"
        info["container"] = "zip"
        info["is_ooxml"] = True
        info["suggested_reader"] = "openpyxl (read_only)"
        if suffix in (".xlsx", ".xlsm"):
            info["extension_matches_content"] = True
            info["note"] = "OOXML/ZIP signature and .xlsx-compatible suffix."
        else:
            info["extension_matches_content"] = False
            info["note"] = (
                f"File is OOXML/ZIP despite the '{suffix}' suffix; it must NOT "
                "be passed to a legacy BIFF/xlrd reader."
            )
    elif sig.startswith(OLE2_MAGIC):
        info["family"] = "ole2"
        info["container"] = "compound-file"
        info["is_legacy_xls"] = True
        info["suggested_reader"] = "xlrd (legacy BIFF)"
        info["note"] = "Legacy BIFF .xls container."
        info["extension_matches_content"] = suffix == ".xls"
    elif sig.startswith(GZIP_MAGIC):
        info["family"] = "gzip"
        info["container"] = "gzip"
        info["note"] = "Gzip-compressed payload; decompress before reading."
    elif sig.startswith(XMZ_MAGIC):
        info["family"] = "pe-executable"
        info["container"] = "exe"
        info["note"] = "Windows executable, not a spreadsheet."
    else:
        info["note"] = "Unrecognised file signature."
    return info


def file_size(path: str | Path) -> int:
    return Path(path).stat().st_size


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Stream a file through SHA256 without loading it into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# JSON / Parquet
# ---------------------------------------------------------------------------
def _json_default(obj: Any) -> Any:
    if isinstance(obj, (Path,)):
        return str(obj)
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    if hasattr(obj, "tolist"):  # numpy array
        return obj.tolist()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def write_json(obj: Any, path: str | Path, indent: int = 2) -> Path:
    """Atomically write JSON (UTF-8, readable unicode, trailing newline)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=indent, default=_json_default)
            fh.write("\n")
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_parquet(df: pd.DataFrame, path: str | Path, compression: str = "snappy") -> Path:
    """Write a DataFrame to Parquet, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression=compression)
    return path


def read_parquet(path: str | Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=list(columns) if columns else None)


# ---------------------------------------------------------------------------
# Stable identifiers
# ---------------------------------------------------------------------------
def slugify(value: Any) -> str:
    """Turn an experimental name into a stable ASCII token."""
    text = str(value).strip()
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_").upper()
    return text or "NA"


def stable_hash(parts: Iterable[Any], length: int = 12) -> str:
    """Deterministic short hash over an ordered/canonical sequence of values."""
    payload = json.dumps(
        list(parts),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def make_id(prefix: str, parts: Iterable[Any], length: int = 12) -> str:
    return f"{prefix}_{stable_hash(parts, length=length)}"


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def environment_snapshot() -> Dict[str, Any]:
    """Runtime versions relevant for reproducibility."""
    import numpy
    import openpyxl
    import pyarrow

    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "pandas": pd.__version__,
        "openpyxl": openpyxl.__version__,
        "pyarrow": pyarrow.__version__,
        "cwd": os.getcwd(),
    }
