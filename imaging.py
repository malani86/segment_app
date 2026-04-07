from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence
from PySide6.QtGui import QImage, QPixmap

from app_models import PreviewBundle


def safe_normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)

    if img.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    if img.dtype == np.uint8:
        return img.copy()

    img = img.astype(np.float32)
    mn = float(np.min(img))
    mx = float(np.max(img))

    if mx - mn < 1e-8:
        return np.zeros(img.shape, dtype=np.uint8)

    img = (img - mn) / (mx - mn)
    return (img * 255.0).clip(0, 255).astype(np.uint8)


def _read_all_tiff_pages(path: str) -> list[np.ndarray]:
    pages: list[np.ndarray] = []
    with Image.open(path) as im:
        for frame in ImageSequence.Iterator(im):
            pages.append(np.array(frame))
    return pages


def _as_hw_or_hwc(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)

    while arr.ndim > 3 and 1 in arr.shape:
        arr = np.squeeze(arr)

    if arr.ndim > 3:
        arr = np.squeeze(arr)

    return arr


def _to_preview_gray(arr: np.ndarray) -> np.ndarray:
    arr = _as_hw_or_hwc(arr)

    if arr.ndim == 2:
        return safe_normalize_to_uint8(arr)

    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            return safe_normalize_to_uint8(arr[..., 0])

        if arr.shape[-1] >= 3:
            rgb = safe_normalize_to_uint8(arr[..., :3]).astype(np.float32)
            gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
            return gray.clip(0, 255).astype(np.uint8)

        return safe_normalize_to_uint8(np.squeeze(arr))

    raise ValueError(f"Unsupported image shape for preview: {arr.shape}")


def _to_preview_image(arr: np.ndarray) -> np.ndarray:
    arr = _as_hw_or_hwc(arr)

    if arr.ndim == 2:
        return safe_normalize_to_uint8(arr)

    if arr.ndim != 3:
        raise ValueError(f"Unsupported image shape for preview: {arr.shape}")

    channels = arr.shape[-1]
    if channels == 1:
        return safe_normalize_to_uint8(arr[..., 0])
    if channels == 3:
        return safe_normalize_to_uint8(arr[..., :3])
    if channels >= 4:
        return safe_normalize_to_uint8(arr[..., :4])

    squeezed = np.squeeze(arr)
    if squeezed.ndim == 2:
        return safe_normalize_to_uint8(squeezed)
    raise ValueError(f"Unsupported image shape for preview: {arr.shape}")


def _load_pages_via_pil(path: str) -> list[np.ndarray]:
    suffix = Path(path).suffix.lower()
    if suffix in {".tif", ".tiff"}:
        return [_as_hw_or_hwc(page) for page in _read_all_tiff_pages(path)]

    with Image.open(path) as im:
        return [_as_hw_or_hwc(np.array(im))]


def load_preview_bundle(path: str) -> PreviewBundle:
    pages = _load_pages_via_pil(path)

    if not pages:
        raise ValueError("No image data found.")

    num_pages = len(pages)
    first = pages[0]

    if num_pages == 1:
        display = _to_preview_image(first)
        if display.ndim == 2:
            mode = "single-gray"
        else:
            mode = f"single-color-{display.shape[-1]}ch"
        return PreviewBundle(
            display_image=display,
            num_pages=num_pages,
            source_mode=mode,
        )

    gray_pages = [_to_preview_gray(page) for page in pages]
    base_shape = gray_pages[0].shape
    if any(page.shape != base_shape for page in gray_pages[1:]):
        raise ValueError("Preview failed: TIFF stack pages do not all share the same 2D shape.")

    stack = np.stack([page.astype(np.float32) for page in gray_pages], axis=0)
    display = safe_normalize_to_uint8(np.max(stack, axis=0))
    mode = f"z-stack-maxproj-{len(gray_pages)}pages"

    return PreviewBundle(
        display_image=display,
        num_pages=num_pages,
        source_mode=mode,
    )


def numpy_gray_to_qpixmap(img: np.ndarray) -> QPixmap:
    img = np.ascontiguousarray(safe_normalize_to_uint8(img))
    h, w = img.shape
    qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
    return QPixmap.fromImage(qimg.copy())


def numpy_rgb_to_qpixmap(img: np.ndarray) -> QPixmap:
    img = np.ascontiguousarray(safe_normalize_to_uint8(img))
    h, w, c = img.shape
    if c != 3:
        raise ValueError("Expected RGB image with 3 channels.")
    qimg = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def numpy_rgba_to_qpixmap(img: np.ndarray) -> QPixmap:
    img = np.ascontiguousarray(safe_normalize_to_uint8(img))
    h, w, c = img.shape
    if c != 4:
        raise ValueError("Expected RGBA image with 4 channels.")
    qimg = QImage(img.data, w, h, 4 * w, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


def numpy_image_to_qpixmap(img: np.ndarray) -> QPixmap:
    arr = _to_preview_image(img)
    if arr.ndim == 2:
        return numpy_gray_to_qpixmap(arr)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return numpy_rgb_to_qpixmap(arr)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        return numpy_rgba_to_qpixmap(arr)
    raise ValueError(f"Unsupported image shape for QPixmap conversion: {arr.shape}")


def image_file_to_qpixmap(path: Path) -> QPixmap:
    try:
        bundle = load_preview_bundle(str(path))
        return numpy_image_to_qpixmap(bundle.display_image)
    except Exception:
        return QPixmap()
