"""Pinned, auditable downloads for official benchmark source files."""
from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen


BFCL_REVISION = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
TOOLHOP_REVISION = "b439d7279af359fda46e8117ae4f0245b75f5c6b"


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urlopen(url) as response, temporary.open("wb") as handle:
        while block := response.read(1024 * 1024):
            handle.write(block)
    temporary.replace(destination)

