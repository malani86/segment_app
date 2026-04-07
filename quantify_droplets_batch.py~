# quantify_droplets_batch.py — inference + extended quantification report
# --------------------------------------------------------------------
# 1. Run UNetDC (batch inference) exactly as in train_DC_focal.py
# 2. Save per‑image predicted masks (+ optional overlays)
# 3. Save per‑droplet CSV per image **AND** a master CSV/Excel sheet
# 4. Plot a global size‑distribution histogram and compute summary stats
# --------------------------------------------------------------------

import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.measure import label, regionprops_table

from pathlib import Path
import argparse
import cv2


import matplotlib
matplotlib.use("Agg")              # headless servers
import matplotlib.pyplot as plt

import torch
from PIL import Image

from models.model_2 import UNetDC
from utils.data_loader import rolling_ball_correction_rgb

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 512  # matches A.Resize in training

# ------------------- helpers -------------------------------------------------

def load_model(ckpt):
    m = UNetDC(in_channels=3, out_channels=1)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    return m.to(DEVICE).eval()


def preprocess(path: Path, background_radius: int):
    im = np.array(Image.open(path).convert("RGB"))
    oh, ow = im.shape[:2]
    im = rolling_ball_correction_rgb(im, background_radius)
    im = cv2.resize(im, (IMG_SIZE, IMG_SIZE), cv2.INTER_AREA)
    im = im.astype(np.float32) / 255.0
    return torch.from_numpy(im).permute(2, 0, 1), (oh, ow)

@torch.no_grad()
def run_batch(tensors, meta, model, mask_dir, overlay_dir,
              thresh, min_area, px_per_um, per_image_rows, all_props):
    batch = torch.stack(tensors).to(DEVICE)
    logits = model(batch)
    for i in range(len(tensors)):
        fpath, (oh, ow) = meta[i]
        name = Path(fpath).stem
        mask512 = (logits[i, 0].cpu().numpy() > thresh).astype(np.uint8)
        mask    = cv2.resize(mask512, (ow, oh), cv2.INTER_NEAREST)
        cv2.imwrite(str(mask_dir / f"{name}_pred.png"), mask * 255)

        # ----- quantify ------------------------------------------------------
        df = quantify(mask, min_area, px_per_um)
        df.insert(0, "filename", Path(fpath).name)
        df.to_csv(mask_dir.parent / f"{name}_droplets.csv", index=False)
        all_props.append(df)


        per_image_row = {
            "filename": Path(fpath).name,
            "droplet_count": len(df),
            "total_area_px": df["area"].sum() if not df.empty else 0,
        }
        per_image_rows.append(per_image_row)
        # overlay
        if overlay_dir is not None:
            img = cv2.imread(str(fpath))
            if img is not None:
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
                cv2.imwrite(str(overlay_dir / f"{name}_overlay.png"), img)

def quantify(bin_mask, min_area, px_per_um):
    lbl = label(bin_mask, connectivity=1)
    for l in np.unique(lbl):
        if l and (lbl == l).sum() < min_area:
            lbl[lbl == l] = 0
    lbl = label(lbl, connectivity=1)
    if lbl.max() == 0:
        return pd.DataFrame()
    props = regionprops_table(lbl, properties=[
        "label", "area", "equivalent_diameter", "centroid"])
    df = pd.DataFrame(props)
    if px_per_um is not None and not df.empty:
        df["area_sqmicron"]  = df["area"] / (px_per_um ** 2)
        df["eq_diam_micron"] = df["equivalent_diameter"] / px_per_um
    return df


# ------------------------- main ---------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser("Segment lipid droplets and build a report")
    p.add_argument("--img_dir", required=True)
    p.add_argument("--ckpt_path", default="best_UNetDC_focal_model.pth")
    p.add_argument("--out_dir",  default="quant_results")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--prob_thresh", type=float, default=0.3)
    p.add_argument("--min_area",  type=int, default=1,
                   help="ignore objects smaller than this (pixels²)")
    p.add_argument("--px_per_micron", type=float,
                   help="pixels per micron for physical‑unit columns")
    p.add_argument("--save_overlays", action="store_true")
    p.add_argument(
        "--background_radius",
        type=int,
        default=50,
        help="radius for rolling ball background correction",
    )
    p.add_argument(
        "--skip_excel",
        action="store_true",
        help="skip generation of the Excel workbook",
    )
    p.add_argument(
        "--skip_histogram",
        action="store_true",
        help="skip histogram plot generation",
    )
    args = p.parse_args()

    in_dir  = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    mask_dir = out_dir / "predicted_masks"
    overlay_dir = out_dir / "overlays" if args.save_overlays else None
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(exist_ok=True)
    if overlay_dir: overlay_dir.mkdir(exist_ok=True)

    model = load_model(args.ckpt_path)

    tensors, meta = [], []
    per_image_rows, all_props = [], []

    images = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in
                     {".png", ".jpg", ".jpeg", ".tif", ".tiff"}])

    for img in tqdm(images, desc="Inference"):
       
        t, osize = preprocess(img, args.background_radius)
        tensors.append(t)
        meta.append((str(img), osize))
        if len(tensors) == args.batch:
            run_batch(tensors, meta, model, mask_dir, overlay_dir,
                      args.prob_thresh, args.min_area, args.px_per_micron,
                      per_image_rows, all_props)
            tensors, meta = [], []

    if tensors:
        run_batch(tensors, meta, model, mask_dir, overlay_dir,
                  args.prob_thresh, args.min_area, args.px_per_micron,
                  per_image_rows, all_props)

    # ---------- master CSV/Excel & stats ------------------------------------
    summary_df = pd.DataFrame(per_image_rows)
    summary_df.to_csv(out_dir / "summary_per_image.csv", index=False)
    if all_props:
        combined = pd.concat(all_props, ignore_index=True)
        combined.to_csv(out_dir / "all_droplets.csv", index=False)
       
        if not args.skip_excel:
            # --- try to write Excel workbook (requires xlsxwriter ≥1.2 and Py≥3.7) ----
            try:
                import xlsxwriter  # noqa: F401  (trigger ImportError / AttributeError early)
                with pd.ExcelWriter(out_dir / "all_droplets.xlsx", engine="xlsxwriter") as xw:
                    combined.to_excel(xw, index=False, sheet_name="droplets")
                    summary_df.to_excel(xw, index=False, sheet_name="per_image")
            except (ImportError, AttributeError):
                # fall back to plain CSV if xlsxwriter missing or too new for Py3.6
                combined.to_csv(out_dir / "all_droplets_noexcel.csv", index=False)
                print(
                    "⚠️  Skipped Excel file; install 'xlsxwriter<3.1.0' or use Python ≥3.7 if you need .xlsx output."
                )


        # choose size column
        size_col = "eq_diam_micron" if "eq_diam_micron" in combined.columns else "equivalent_diameter"
        stats = combined[size_col].describe()[["mean", "50%", "std"]].rename({"50%": "median"})
        stats.to_csv(out_dir / "droplet_size_stats.csv")

       
        if not args.skip_histogram:
            # histogram
            plt.figure(figsize=(6, 4))
            plt.hist(combined[size_col], bins=40)
            plt.xlabel("Diameter (µm)" if "micron" in size_col else "Diameter (pixels)")
            plt.ylabel("Count")
            plt.title("Droplet size distribution")
            plt.tight_layout()
            plt.savefig(out_dir / "size_histogram.png", dpi=300)
            plt.close()

    print("\n All done. Outputs are in ", out_dir)
