from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageSequence
from skimage.measure import label, regionprops_table
from skimage.segmentation import clear_border
from tqdm import tqdm

from config import DEFAULT_BACKGROUND_RADIUS, DEFAULT_RESIZE_SIZE, DEFAULT_THRESHOLD
from models.model_2 import UNetDC
from utils.data_loader import (
    ensure_training_rgb_uint8,
    preprocess_rgb_like_training,
)

matplotlib.use("Agg")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TIFF_MODE_CURRENT_SLICE = "current_slice"
TIFF_MODE_MAX_PROJECTION = "max_projection"
TIFF_MODE_ALL_SLICES = "all_slices"
DEFAULT_TIFF_PROB_THRESHOLD = 0.7


@dataclass
class InferenceTarget:
    source_path: Path
    display_name: str
    output_stem: str
    rgb_image: np.ndarray
    corrected_rgb_image: np.ndarray
    preprocessed_image: np.ndarray
    tensor: torch.Tensor
    original_size: tuple[int, int]


def load_model(ckpt: str) -> UNetDC:
    model = UNetDC(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    return model.to(DEVICE).eval()


def read_image_pages(path: Path) -> list[np.ndarray]:
    with Image.open(path) as image:
        if is_tiff_path(path):
            return [ensure_training_rgb_uint8(frame.copy()) for frame in ImageSequence.Iterator(image)]
        return [ensure_training_rgb_uint8(image.copy())]


def is_tiff_path(path: Path) -> bool:
    return path.suffix.lower() in {".tif", ".tiff"}


def max_project_pages(pages: list[np.ndarray]) -> np.ndarray:
    rgb_pages = [ensure_training_rgb_uint8(page).astype(np.float32) for page in pages]
    base_shape = rgb_pages[0].shape
    if any(page.shape != base_shape for page in rgb_pages[1:]):
        raise ValueError("Cannot max-project TIFF pages with inconsistent shapes.")
    return np.max(np.stack(rgb_pages, axis=0), axis=0).clip(0, 255).astype(np.uint8)


def preprocess_rgb(image_rgb: np.ndarray, background_radius: int, resize_size: int) -> tuple[torch.Tensor, tuple[int, int], np.ndarray, np.ndarray]:
    preprocessed_image, original_size, corrected_image = preprocess_rgb_like_training(
        image_rgb,
        radius=background_radius,
        size=resize_size,
    )
    tensor = torch.from_numpy(preprocessed_image).permute(2, 0, 1)
    return tensor, original_size, corrected_image, preprocessed_image


def build_slice_stem(path: Path, slice_index: int) -> str:
    return f"{path.stem}_z{slice_index:03d}"


def build_inference_target(
    path: Path,
    *,
    display_name: str,
    output_stem: str,
    image_data: np.ndarray,
    background_radius: int,
    resize_size: int,
) -> InferenceTarget:
    rgb_image = ensure_training_rgb_uint8(image_data)
    tensor, original_size, corrected_rgb_image, preprocessed_image = preprocess_rgb(
        rgb_image,
        background_radius,
        resize_size,
    )
    return InferenceTarget(
        source_path=path,
        display_name=display_name,
        output_stem=output_stem,
        rgb_image=rgb_image,
        corrected_rgb_image=corrected_rgb_image,
        preprocessed_image=preprocessed_image,
        tensor=tensor,
        original_size=original_size,
    )


def save_debug_preprocessed_image(target: InferenceTarget, debug_dir: Path) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_rgb = np.clip(target.preprocessed_image * 255.0, 0, 255).astype(np.uint8)
    debug_bgr = cv2.cvtColor(debug_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(debug_dir / f"{target.output_stem}_preprocessed_input.png"), debug_bgr)
    np.save(debug_dir / f"{target.output_stem}_preprocessed_input.npy", target.preprocessed_image)


def debug_target_matches(query: str, target: InferenceTarget) -> bool:
    needle = query.strip().lower()
    if not needle:
        return False

    candidates = {
        target.source_path.name.lower(),
        target.source_path.stem.lower(),
        target.display_name.lower(),
        target.output_stem.lower(),
    }
    return any(needle == candidate or needle in candidate for candidate in candidates)


def threshold_for_target(target: InferenceTarget, requested_thresh: float) -> float:
    if is_tiff_path(target.source_path) and abs(requested_thresh - DEFAULT_THRESHOLD) < 1e-8:
        return DEFAULT_TIFF_PROB_THRESHOLD
    return requested_thresh


def remove_border_touching_components(mask: np.ndarray) -> np.ndarray:
    cleaned = clear_border(mask.astype(bool))
    return cleaned.astype(np.uint8)


def build_inference_targets(
    path: Path,
    *,
    tiff_mode: str,
    slice_index: int,
    background_radius: int,
    resize_size: int,
) -> list[InferenceTarget]:
    pages = read_image_pages(path)
    if not pages:
        return []

    if not is_tiff_path(path) or len(pages) == 1:
        return [
            build_inference_target(
                path,
                display_name=path.name,
                output_stem=path.stem,
                image_data=pages[0],
                background_radius=background_radius,
                resize_size=resize_size,
            )
        ]

    if tiff_mode == TIFF_MODE_MAX_PROJECTION:
        # Max projection is supported for exploration, but the training pipeline
        # uses one 2D RGB image per sample rather than projected stack inputs.
        return [
            build_inference_target(
                path,
                display_name=f"{path.name} [max projection]",
                output_stem=f"{path.stem}_maxproj",
                image_data=max_project_pages(pages),
                background_radius=background_radius,
                resize_size=resize_size,
            )
        ]

    if tiff_mode == TIFF_MODE_ALL_SLICES:
        targets: list[InferenceTarget] = []
        for index, page in enumerate(pages):
            targets.append(
                build_inference_target(
                    path,
                    display_name=f"{path.name} [slice {index}]",
                    output_stem=build_slice_stem(path, index),
                    image_data=page,
                    background_radius=background_radius,
                    resize_size=resize_size,
                )
            )
        return targets

    safe_index = max(0, min(len(pages) - 1, slice_index))
    return [
        build_inference_target(
            path,
            display_name=f"{path.name} [slice {safe_index}]",
            output_stem=build_slice_stem(path, safe_index),
            image_data=pages[safe_index],
            background_radius=background_radius,
            resize_size=resize_size,
        )
    ]


def quantify(binary_mask: np.ndarray, min_area: int, px_per_um: float | None) -> pd.DataFrame:
    labeled = label(binary_mask, connectivity=1)
    for component in np.unique(labeled):
        if component and (labeled == component).sum() < min_area:
            labeled[labeled == component] = 0
    labeled = label(labeled, connectivity=1)
    if labeled.max() == 0:
        return pd.DataFrame()

    props = regionprops_table(
        labeled,
        properties=["label", "area", "equivalent_diameter", "centroid"],
    )
    table = pd.DataFrame(props)
    if px_per_um is not None and px_per_um > 0 and not table.empty:
        table["area_sqmicron"] = table["area"] / (px_per_um ** 2)
        table["eq_diam_micron"] = table["equivalent_diameter"] / px_per_um
    return table


@torch.no_grad()
def run_batch(
    batch_targets: list[InferenceTarget],
    model: UNetDC,
    mask_dir: Path,
    overlay_dir: Path | None,
    thresh: float,
    min_area: int,
    px_per_um: float | None,
    per_image_rows: list[dict[str, object]],
    all_props: list[pd.DataFrame],
    overlay_alpha: float,
    save_masks: bool,
    automatic_quantification: bool,
) -> None:
    batch = torch.stack([target.tensor for target in batch_targets]).to(DEVICE)
    logits = model(batch)

    for index, target in enumerate(batch_targets):
        original_height, original_width = target.original_size
        target_thresh = threshold_for_target(target, thresh)
        mask512 = (logits[index, 0].cpu().numpy() > target_thresh).astype(np.uint8)
        mask = cv2.resize(mask512, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
        mask = remove_border_touching_components(mask)

        if save_masks:
            cv2.imwrite(str(mask_dir / f"{target.output_stem}_mask.png"), mask * 255)

        if automatic_quantification:
            frame_table = quantify(mask, min_area, px_per_um)
            frame_table.insert(0, "filename", target.display_name)
            frame_table.to_csv(mask_dir.parent / f"{target.output_stem}_droplets.csv", index=False)
            all_props.append(frame_table)
            per_image_rows.append(
                {
                    "filename": target.display_name,
                    "droplet_count": len(frame_table),
                    "total_area_px": frame_table["area"].sum() if not frame_table.empty else 0,
                }
            )

        if overlay_dir is not None:
            source_bgr = cv2.cvtColor(target.rgb_image, cv2.COLOR_RGB2BGR)
            overlay = source_bgr.copy()
            overlay[mask > 0] = (0, 255, 0)
            blended = cv2.addWeighted(overlay, overlay_alpha, source_bgr, 1.0 - overlay_alpha, 0.0)
            cv2.imwrite(str(overlay_dir / f"{target.output_stem}_overlay.png"), blended)


def main() -> None:
    parser = argparse.ArgumentParser("Segment lipid droplets and build a report")
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--ckpt_path", default="best_UNetDC_focal_model.pth")
    parser.add_argument("--out_dir", default="quant_results")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--prob_thresh", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--min_area", type=int, default=1, help="ignore objects smaller than this (pixels²)")
    parser.add_argument("--resize_size", type=int, default=DEFAULT_RESIZE_SIZE)
    parser.add_argument("--overlay_alpha", type=float, default=0.45)
    parser.add_argument("--px_per_micron", type=float, help="pixels per micron for physical-unit columns")
    parser.add_argument("--save_overlays", action="store_true")
    parser.add_argument("--skip_mask_save", action="store_true")
    parser.add_argument("--skip_quantification", action="store_true")
    parser.add_argument(
        "--background_radius",
        type=int,
        default=DEFAULT_BACKGROUND_RADIUS,
        help="radius for rolling ball background correction",
    )
    parser.add_argument("--skip_excel", action="store_true", help="skip generation of the Excel workbook")
    parser.add_argument("--skip_histogram", action="store_true", help="skip histogram plot generation")
    parser.add_argument(
        "--debug_preprocessed_match",
        type=str,
        default="",
        help="save the preprocessed model input for the first image/slice whose name or stem matches this value",
    )
    parser.add_argument(
        "--debug_preprocessed_dir",
        type=str,
        default="",
        help="optional directory for preprocessed debug outputs; defaults to <out_dir>/debug_preprocessed",
    )
    parser.add_argument(
        "--tiff_mode",
        choices=[TIFF_MODE_CURRENT_SLICE, TIFF_MODE_MAX_PROJECTION, TIFF_MODE_ALL_SLICES],
        default=TIFF_MODE_CURRENT_SLICE,
        help="how to segment multi-slice TIFF stacks",
    )
    parser.add_argument(
        "--slice_index",
        type=int,
        default=0,
        help="0-based slice index used when TIFF mode is current_slice",
    )
    parser.add_argument(
        "--tiff_slice_index",
        type=int,
        dest="slice_index_legacy",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.slice_index_legacy is not None:
        args.slice_index = args.slice_index_legacy

    input_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    mask_dir = out_dir / "predicted_masks"
    overlay_dir = out_dir / "overlays" if args.save_overlays else None
    debug_dir = Path(args.debug_preprocessed_dir) if args.debug_preprocessed_dir else out_dir / "debug_preprocessed"
    debug_saved = False

    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_mask_save:
        mask_dir.mkdir(exist_ok=True)
    if overlay_dir is not None:
        overlay_dir.mkdir(exist_ok=True)

    model = load_model(args.ckpt_path)

    input_images = sorted(
        path for path in input_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    )
    has_tiff_inputs = any(is_tiff_path(path) for path in input_images)
    print(f"Found {len(input_images)} input image(s).")
    print(f"Multi-slice TIFF mode: {args.tiff_mode}")
    if has_tiff_inputs and args.tiff_mode == TIFF_MODE_CURRENT_SLICE:
        print(f"Current slice index: {args.slice_index}")
    if has_tiff_inputs and abs(args.prob_thresh - DEFAULT_THRESHOLD) < 1e-8:
        print(f"Using adaptive TIFF threshold: TIFF inputs will use {DEFAULT_TIFF_PROB_THRESHOLD} instead of {DEFAULT_THRESHOLD}.")

    pending_targets: list[InferenceTarget] = []
    per_image_rows: list[dict[str, object]] = []
    all_props: list[pd.DataFrame] = []

    for image_path in tqdm(input_images, desc="Inference"):
        targets = build_inference_targets(
            image_path,
            tiff_mode=args.tiff_mode,
            slice_index=args.slice_index,
            background_radius=args.background_radius,
            resize_size=args.resize_size,
        )
        if args.debug_preprocessed_match and not debug_saved:
            for target in targets:
                if debug_target_matches(args.debug_preprocessed_match, target):
                    save_debug_preprocessed_image(target, debug_dir)
                    print(f"Saved preprocessed debug input for {target.display_name} to {debug_dir}")
                    debug_saved = True
                    break
        if len(targets) > 1:
            print(f"Expanded {image_path.name} into {len(targets)} slice(s) for inference.")
        pending_targets.extend(targets)
        while len(pending_targets) >= args.batch:
            run_batch(
                pending_targets[:args.batch],
                model,
                mask_dir,
                overlay_dir,
                args.prob_thresh,
                args.min_area,
                args.px_per_micron,
                per_image_rows,
                all_props,
                args.overlay_alpha,
                not args.skip_mask_save,
                not args.skip_quantification,
            )
            pending_targets = pending_targets[args.batch:]

    if args.debug_preprocessed_match and not debug_saved:
        print(f"No inference target matched debug_preprocessed_match={args.debug_preprocessed_match!r}.")

    if pending_targets:
        run_batch(
            pending_targets,
            model,
            mask_dir,
            overlay_dir,
            args.prob_thresh,
            args.min_area,
            args.px_per_micron,
            per_image_rows,
            all_props,
            args.overlay_alpha,
            not args.skip_mask_save,
            not args.skip_quantification,
        )

    if args.skip_quantification:
        print("\nSegmentation complete. Quantification was skipped. Outputs are in", out_dir)
        raise SystemExit(0)

    summary_df = pd.DataFrame(per_image_rows)
    summary_df.to_csv(out_dir / "summary_per_image.csv", index=False)

    if all_props:
        combined = pd.concat(all_props, ignore_index=True)
        combined.to_csv(out_dir / "all_droplets.csv", index=False)

        if not args.skip_excel:
            try:
                import xlsxwriter  # noqa: F401

                with pd.ExcelWriter(out_dir / "all_droplets.xlsx", engine="xlsxwriter") as writer:
                    combined.to_excel(writer, index=False, sheet_name="droplets")
                    summary_df.to_excel(writer, index=False, sheet_name="per_image")
            except (ImportError, AttributeError):
                combined.to_csv(out_dir / "all_droplets_noexcel.csv", index=False)
                print("Skipped Excel file; install compatible xlsxwriter if .xlsx output is needed.")

        size_column = "eq_diam_micron" if "eq_diam_micron" in combined.columns else "equivalent_diameter"
        stats = combined[size_column].describe()[["mean", "50%", "std"]].rename({"50%": "median"})
        stats.to_csv(out_dir / "droplet_size_stats.csv")

        if not args.skip_histogram:
            plt.figure(figsize=(6, 4))
            plt.hist(combined[size_column], bins=40)
            plt.xlabel("Diameter (µm)" if "micron" in size_column else "Diameter (pixels)")
            plt.ylabel("Count")
            plt.title("Droplet size distribution")
            plt.tight_layout()
            plt.savefig(out_dir / "size_histogram.png", dpi=300)
            plt.close()

    print("\nAll done. Outputs are in", out_dir)


if __name__ == "__main__":
    main()
