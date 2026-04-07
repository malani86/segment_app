from __future__ import annotations

import csv
import sys
from pathlib import Path

from config import SCRIPT_BASENAME, VALID_EXTS


def _resolve_batch_script() -> Path:
    start = Path(__file__).resolve()
    exe_path = Path(sys.argv[0]).resolve()
    frozen_dir = Path(getattr(sys, "_MEIPASS", "")) if getattr(sys, "frozen", False) else None

    if frozen_dir and frozen_dir.exists():
        search_roots = (frozen_dir, exe_path.parent, Path.cwd(), start.parent, *start.parents)
    else:
        search_roots = (exe_path.parent, Path.cwd(), start.parent, *start.parents)

    exe_name = SCRIPT_BASENAME + (".exe" if getattr(sys, "frozen", False) else ".py")
    candidates = []
    seen = set()
    for directory in search_roots:
        if directory in seen:
            continue
        seen.add(directory)
        candidates.append(directory / exe_name)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    locations = "\n - ".join(str(path.parent) for path in candidates)
    raise FileNotFoundError(f"Could not locate {exe_name}. Looked in:\n -  " + locations)


def build_batch_command(
    *,
    input_dir: Path,
    ckpt_path: Path,
    out_dir: Path,
    batch_size: int,
    prob_thresh: float,
    min_area: int,
    background_radius: int,
    px_per_micron: float | None,
    save_overlays: bool,
    excel_enabled: bool,
    histogram_enabled: bool,
) -> list[str]:
    script_path = _resolve_batch_script()

    args = ([str(script_path)] if getattr(sys, "frozen", False) else [sys.executable, str(script_path)]) + [
        "--img_dir", str(input_dir),
        "--ckpt_path", str(ckpt_path),
        "--out_dir", str(out_dir),
        "--batch", str(batch_size),
        "--prob_thresh", str(prob_thresh),
        "--min_area", str(min_area),
        "--background_radius", str(background_radius),
    ]

    if px_per_micron is not None and px_per_micron > 0:
        args.extend(["--px_per_micron", str(px_per_micron)])
    if save_overlays:
        args.append("--save_overlays")
    if not excel_enabled:
        args.append("--skip_excel")
    if not histogram_enabled:
        args.append("--skip_histogram")
    return args


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return [dict(row) for row in reader]
    except Exception:
        return []


def prepare_run(
    *,
    folder_text: str,
    ckpt_text: str,
    out_dir_text: str,
    batch_value: int,
    threshold_value: float,
    min_area_value: int,
    radius_value: int,
    px_per_micron_value: float,
    save_overlays: bool,
    excel_enabled: bool,
    histogram_enabled: bool,
) -> tuple[Path, list[str], Path]:
    ckpt_path = Path(ckpt_text.strip())
    if not ckpt_path.is_file():
        raise ValueError("Checkpoint file does not exist.")

    base_out_dir = Path(out_dir_text.strip()) if out_dir_text.strip() else Path("quant_results")
    base_out_dir.mkdir(parents=True, exist_ok=True)

    px_per_micron = px_per_micron_value if px_per_micron_value > 0 else None
    folder_path = Path(folder_text.strip())
    if not folder_path.is_dir():
        raise ValueError("Please select a valid input folder.")
    image_count = sum(1 for p in folder_path.iterdir() if p.suffix.lower() in VALID_EXTS)
    if image_count == 0:
        raise ValueError("The selected folder contains no supported images.")

    input_dir = folder_path
    out_dir = base_out_dir

    command = build_batch_command(
        input_dir=input_dir,
        ckpt_path=ckpt_path,
        out_dir=out_dir,
        batch_size=batch_value,
        prob_thresh=threshold_value,
        min_area=min_area_value,
        background_radius=radius_value,
        px_per_micron=px_per_micron,
        save_overlays=save_overlays,
        excel_enabled=excel_enabled,
        histogram_enabled=histogram_enabled,
    )

    return folder_path, command, out_dir
