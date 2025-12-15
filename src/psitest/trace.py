from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def _now_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def get_output_base_dir() -> Path:
    """Return the required output base dir.

    By default this is `1 - RESEARCH/PSITest` (relative to CWD), but can be
    overridden via PSITEST_OUTPUT_DIR.
    """
    base = os.environ.get("PSITEST_OUTPUT_DIR", "1 - RESEARCH/PSITest")
    return Path(base)


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def get_env_info() -> Dict[str, Any]:
    """Collect environment information for reproducibility."""
    info: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    # Optional: pip freeze (best-effort)
    try:
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        info["pip_freeze"] = freeze.strip().splitlines()
    except Exception:
        info["pip_freeze"] = None

    # Optional: git commit (best-effort)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        info["git_commit"] = commit
    except Exception:
        info["git_commit"] = None

    return info


def snapshot_code(repo_root: Path, dest_dir: Path) -> None:
    """Copy relevant source files into dest_dir for verification."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        repo_root / "app" / "streamlit_app.py",
        repo_root / "requirements.txt",
        repo_root / "Dockerfile",
        repo_root / "docker-compose.yml",
        repo_root / "docker-compose.gpu.yml",
        repo_root / "src" / "psitest",
    ]

    for p in paths:
        if not p.exists():
            continue
        if p.is_dir():
            shutil.copytree(p, dest_dir / p.name, dirs_exist_ok=True)
        else:
            shutil.copy2(p, dest_dir / p.name)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def zip_dir(src_dir: Path, zip_path: Path) -> None:
    """Create a zip archive of src_dir."""
    import zipfile

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in src_dir.rglob("*"):
            if file.is_dir():
                continue
            zf.write(file, arcname=str(file.relative_to(src_dir)))


@dataclass
class RunContext:
    """A run directory with standard subfolders + trace helpers."""

    run_dir: Path

    @property
    def data_dir(self) -> Path:
        return self.run_dir / "data"

    @property
    def figures_dir(self) -> Path:
        return self.run_dir / "figures"

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"

    @property
    def ckpt_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def code_dir(self) -> Path:
        return self.run_dir / "code_snapshot"

    def mkdirs(self) -> None:
        for p in [self.data_dir, self.figures_dir, self.logs_dir, self.ckpt_dir, self.code_dir]:
            p.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def create(kind: str, tag: str = "") -> "RunContext":
        base = get_output_base_dir()
        run_id = _now_id()
        name = f"{run_id}_{_safe_name(kind)}"
        if tag:
            name += f"_{_safe_name(tag)}"
        run_dir = base / "runs" / name
        ctx = RunContext(run_dir=run_dir)
        ctx.mkdirs()
        return ctx

    def save_env(self) -> None:
        write_json(self.data_dir / "env.json", get_env_info())

    def save_config(self, config: Dict[str, Any]) -> None:
        write_json(self.data_dir / "config.json", config)

    def snapshot_repo(self, repo_root: Optional[Path] = None) -> None:
        if repo_root is None:
            # repo_root = .../src/psitest/trace.py -> .../ (repo)
            repo_root = Path(__file__).resolve().parents[3]
        snapshot_code(repo_root, self.code_dir)

    def zip_run(self) -> Path:
        zip_path = self.run_dir.with_suffix(".zip")
        zip_dir(self.run_dir, zip_path)
        return zip_path
