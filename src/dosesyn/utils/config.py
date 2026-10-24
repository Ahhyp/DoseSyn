"""Configuration loading with project-root-relative path resolution."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

_PATH_KEYS = (
    "raw_dir",
    "interim_dir",
    "processed_dir",
    "splits_dir",
    "audit_dir",
    "logs_dir",
    "docs_dir",
    "work_dir",
)


def project_root() -> Path:
    """Return the DoseSyn project root.

    Order: ``DOSESYN_ROOT`` env var, else the directory three levels above
    ``src/dosesyn/utils/config.py`` (i.e. the repo checkout root).
    """
    env = os.environ.get("DOSESYN_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


def load_config(path: str | os.PathLike) -> Dict[str, Any]:
    """Load a YAML config and resolve all declared paths against the root."""
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = (project_root() / cfg_path).resolve()
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"config {cfg_path} is not a mapping")

    root = project_root()
    cfg["_root"] = str(root)
    cfg["_config_path"] = str(cfg_path)

    paths = cfg.setdefault("paths", {})
    for key in _PATH_KEYS:
        if key in paths:
            p = Path(paths[key])
            paths[key] = str(p if p.is_absolute() else (root / p))

    # Convenience absolute paths for the declared raw files.
    files = cfg.setdefault("files", {})
    for role, name in list(files.items()):
        fname = Path(name)
        files[f"{role}_path"] = str(
            fname if fname.is_absolute() else (Path(paths["raw_dir"]) / fname)
        )
    return cfg


def ensure_directories(cfg: Dict[str, Any]) -> None:
    """Create every configured output directory (idempotent)."""
    for key in _PATH_KEYS:
        p = cfg.get("paths", {}).get(key)
        if p:
            Path(p).mkdir(parents=True, exist_ok=True)


def resolve_path(cfg: Dict[str, Any], value: str) -> Path:
    """Resolve *value* relative to the project root."""
    p = Path(value)
    return p if p.is_absolute() else Path(cfg["_root"]) / p
