"""Small, secret-free runtime fingerprints attached to evaluation traces."""
from __future__ import annotations

import hashlib
from importlib import metadata
from pathlib import Path
import platform


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verifier_sha256(root: Path | None = None) -> str:
    """Hash relative names and content, so local and Modal mounts are comparable."""
    root = root or Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".md"}:
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def runtime_fingerprint(*, capture_gpu: bool = False) -> dict:
    versions = {}
    for package in ("torch", "triton", "numpy", "openai", "anthropic"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "verifier_sha256": verifier_sha256(),
    }
    if capture_gpu:
        try:
            import torch
            result["cuda_version"] = torch.version.cuda
            result["gpu_devices"] = [
                {"name": torch.cuda.get_device_name(i),
                 "capability": list(torch.cuda.get_device_capability(i))}
                for i in range(torch.cuda.device_count())
            ]
        except (ImportError, RuntimeError) as exc:
            result["gpu_metadata_error"] = type(exc).__name__
    return result
