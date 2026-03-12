from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Sequence


def _resolve_default_base_dir() -> Path:
    """Resolve certificate directory from env var or common workspace locations."""
    env_base_dir = os.getenv("DATALOAD_CERT_DIR")

    candidates: list[Path] = []
    if env_base_dir:
        candidates.append(Path(env_base_dir).expanduser())

    workspace_root = Path(__file__).resolve().parents[4]
    candidates.extend(
        [
            workspace_root / "ForNIshimura" / "SaveAnnotation" / "certs-tai-morita",
            Path.cwd() / "ForNIshimura" / "SaveAnnotation" / "certs-tai-morita",
            Path(r"D:\\_study\\ImageProcessing\\ForNIshimura\\SaveAnnotation\\certs-tai-morita"),
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Keep deterministic behavior even if certs are absent.
    return candidates[0]


@dataclass(frozen=True)
class DataLoadConfig:
    """Runtime configuration for volume/label download."""

    account: str
    base_dir: Path
    url_base: str
    catalogue_key: str
    catalogue_tags: Sequence[str]
    select_index: int = 0

    @property
    def ca_path(self) -> Path:
        return self.base_dir / "ca.crt"

    @property
    def client_crt(self) -> Path:
        return self.base_dir / f"{self.account}.client.crt"

    @property
    def client_key(self) -> Path:
        return self.base_dir / f"{self.account}.client.key"

    @classmethod
    def from_defaults(cls, index: int = 0) -> "DataLoadConfig":
        return cls(
            account="tai-morita",
            base_dir=_resolve_default_base_dir(),
            url_base="https://192.168.17.106:50050/api",
            catalogue_key="3af616826f78654f5d6755e047df971cf296bba908e5e668699a9e3029b6f33b",
            catalogue_tags=["segmentation", "test", "phase1", "上顎洞", "DentalSegmentator", "label"],
            select_index=index,
        )
