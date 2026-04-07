from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence

from PySide6.QtCore import QThread, Signal

from backend import read_csv_rows
from app_models import BatchRunResult


class BatchProcessWorker(QThread):
    output = Signal(str)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        *,
        input_path: Path,
        command: Sequence[str],
        out_dir: Path,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.input_path = input_path
        self.command = list(command)
        self.out_dir = out_dir

    def run(self) -> None:  # type: ignore[override]
        try:
            process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return

        assert process.stdout is not None
        collected: list[str] = []
        try:
            for line in process.stdout:
                cleaned = line.rstrip()
                collected.append(cleaned)
                self.output.emit(cleaned)
        finally:
            process.stdout.close()

        return_code = process.wait()
        log_text = "\n".join(collected)
        if return_code != 0:
            tail = "\n".join(filter(None, collected[-20:]))
            self.failed.emit(tail or f"Process exited with status {return_code}")
            return

        try:
            summary_rows = read_csv_rows(self.out_dir / "summary_per_image.csv")
            stats_rows = read_csv_rows(self.out_dir / "droplet_size_stats.csv")
            histogram_path = str(self.out_dir / "size_histogram.png") if (self.out_dir / "size_histogram.png").exists() else None
            droplet_rows: list[dict[str, str]] = []
            combined_csv = self.out_dir / "all_droplets.csv"
            if combined_csv.exists():
                droplet_rows = read_csv_rows(combined_csv)

            result = BatchRunResult(
                input_path=str(self.input_path),
                out_dir=str(self.out_dir),
                log_text=log_text,
                summary_rows=summary_rows,
                stats_rows=stats_rows,
                droplet_rows=droplet_rows,
                histogram_path=histogram_path,
            )
            self.succeeded.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))
