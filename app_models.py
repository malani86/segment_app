from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PreviewBundle:
    display_image: Any
    num_pages: int
    source_mode: str


@dataclass
class BatchRunResult:
    input_path: str
    out_dir: str
    log_text: str
    summary_rows: list[dict[str, str]]
    stats_rows: list[dict[str, str]]
    droplet_rows: list[dict[str, str]]
    histogram_path: str | None
