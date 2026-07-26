"""Scan preprocessing.

Recognition quality on old Ethiopian press books is dominated by what happens before the
recognizer ever runs. Fidel glyphs distinguish themselves by small appendages, so a
binarization that thickens strokes or a two-degree skew will turn ቀ into ቁ far more often
than any model tuning will fix.

The core is NumPy only, so preprocessing works in a bare install. OpenCV, when present,
is used for the operations where it is markedly faster.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:  # optional acceleration
    import cv2

    HAS_CV2 = True
except ImportError:  # pragma: no cover - depends on install extras
    cv2 = None  # type: ignore[assignment]
    HAS_CV2 = False


class BinarizeMethod(str, Enum):
    NONE = "none"
    OTSU = "otsu"
    #: Local thresholding. The right default for aged paper, where illumination and
    #: staining vary across the page and a single global threshold eats faint strokes.
    SAUVOLA = "sauvola"
    ADAPTIVE_MEAN = "adaptive_mean"


@dataclass
class PreprocessOptions:
    grayscale: bool = True
    autocontrast: bool = False
    deskew: bool = True
    max_skew_degrees: float = 6.0
    binarize: BinarizeMethod = BinarizeMethod.SAUVOLA
    sauvola_window: int = 31
    sauvola_k: float = 0.2
    despeckle: bool = True
    despeckle_max_area: int = 12
    remove_border: bool = True
    border_fraction: float = 0.04
    upscale_to_dpi: int = 0  # 0 disables; 300 is the usual target for Tesseract
    source_dpi: int = 0
    invert_if_dark: bool = True


@dataclass
class PreprocessReport:
    skew_degrees: float = 0.0
    binarized: str = "none"
    speckles_removed: int = 0
    border_cropped: bool = False
    scaled_by: float = 1.0
    ink_coverage: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> str:
        bits = []
        if abs(self.skew_degrees) > 0.05:
            bits.append(f"deskewed {self.skew_degrees:+.2f}°")
        if self.binarized != "none":
            bits.append(self.binarized)
        if self.speckles_removed:
            bits.append(f"{self.speckles_removed} speckles removed")
        if self.border_cropped:
            bits.append("border cropped")
        if abs(self.scaled_by - 1.0) > 0.01:
            bits.append(f"scaled ×{self.scaled_by:.2f}")
        bits.append(f"{self.ink_coverage:.1%} ink")
        return ", ".join(bits)


# --------------------------------------------------------------------------------------
# Conversions
# --------------------------------------------------------------------------------------


def load_image(path: str | Path) -> Image.Image:
    image = Image.open(path)
    return ImageOps.exif_transpose(image) or image


def to_gray_array(image: Image.Image) -> np.ndarray:
    if image.mode != "L":
        image = image.convert("L")
    return np.asarray(image, dtype=np.uint8)


def to_image(array: np.ndarray) -> Image.Image:
    if array.dtype == bool:
        array = (array.astype(np.uint8)) * 255
    return Image.fromarray(array.astype(np.uint8), mode="L")


def image_dpi(image: Image.Image, default: int = 0) -> int:
    dpi = image.info.get("dpi")
    if isinstance(dpi, tuple) and dpi:
        try:
            return int(round(float(dpi[0])))
        except (TypeError, ValueError):
            return default
    return default


# --------------------------------------------------------------------------------------
# Binarization
# --------------------------------------------------------------------------------------


def _integral_stats(gray: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Local mean and standard deviation via summed-area tables."""
    radius = window // 2
    padded = np.pad(gray.astype(np.float64), radius + 1, mode="reflect")
    integral = padded.cumsum(axis=0).cumsum(axis=1)
    integral_sq = (padded**2).cumsum(axis=0).cumsum(axis=1)

    h, w = gray.shape
    y0 = np.arange(h)
    x0 = np.arange(w)
    top = y0[:, None]
    bottom = (y0 + window)[:, None]
    left = x0[None, :]
    right = (x0 + window)[None, :]

    def window_sum(table: np.ndarray) -> np.ndarray:
        return (
            table[bottom, right] - table[top, right] - table[bottom, left] + table[top, left]
        )

    area = float(window * window)
    mean = window_sum(integral) / area
    mean_sq = window_sum(integral_sq) / area
    variance = np.clip(mean_sq - mean**2, 0.0, None)
    return mean, np.sqrt(variance)


def sauvola_threshold(gray: np.ndarray, window: int = 31, k: float = 0.2, r: float = 128.0) -> np.ndarray:
    """Sauvola local thresholding: ``T = m * (1 + k * (s / R - 1))``.

    Chosen over Otsu because it keeps faint diacritics alive on stained paper, and those
    diacritics are exactly what distinguishes one vowel order from another.
    """
    if window % 2 == 0:
        window += 1
    mean, std = _integral_stats(gray, window)
    threshold = mean * (1.0 + k * ((std / r) - 1.0))
    return (gray > threshold).astype(np.uint8) * 255


def otsu_threshold(gray: np.ndarray) -> np.ndarray:
    if HAS_CV2:
        _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return out
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = histogram.sum()
    if total == 0:
        return gray
    levels = np.arange(256)
    weight_bg = np.cumsum(histogram)
    weight_fg = total - weight_bg
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_bg = np.cumsum(histogram * levels) / weight_bg
        mean_fg = (np.cumsum((histogram * levels)[::-1])[::-1] - histogram * levels) / weight_fg
        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    between = np.nan_to_num(between)
    return (gray > int(np.argmax(between))).astype(np.uint8) * 255


def adaptive_mean_threshold(gray: np.ndarray, window: int = 31, offset: float = 10.0) -> np.ndarray:
    mean, _ = _integral_stats(gray, window if window % 2 else window + 1)
    return (gray > (mean - offset)).astype(np.uint8) * 255


def binarize(gray: np.ndarray, options: PreprocessOptions) -> tuple[np.ndarray, str]:
    method = options.binarize
    if method is BinarizeMethod.NONE:
        return gray, "none"
    if method is BinarizeMethod.OTSU:
        return otsu_threshold(gray), "otsu"
    if method is BinarizeMethod.ADAPTIVE_MEAN:
        return adaptive_mean_threshold(gray, options.sauvola_window), "adaptive mean"
    return sauvola_threshold(gray, options.sauvola_window, options.sauvola_k), "sauvola"


# --------------------------------------------------------------------------------------
# Skew
# --------------------------------------------------------------------------------------


def estimate_skew(gray: np.ndarray, max_degrees: float = 6.0, coarse_step: float = 0.5) -> float:
    """Estimate page skew by maximizing the variance of the horizontal ink profile.

    Text lines produce sharp peaks in the row-sum profile only when they are level, so the
    angle with the highest profile variance is the one that straightens the page. A coarse
    sweep followed by a fine sweep keeps this fast on full-resolution scans.
    """
    work = _downsample_for_analysis(gray)
    ink = 255 - work  # ink positive

    def profile_variance(angle: float) -> float:
        rotated = _rotate_array(ink, angle)
        profile = rotated.sum(axis=1, dtype=np.float64)
        return float(np.var(np.diff(profile)))

    coarse = np.arange(-max_degrees, max_degrees + coarse_step, coarse_step)
    best = max(coarse, key=profile_variance)
    # The refinement window straddles the coarse winner, so at the ends of the sweep it
    # would otherwise run past the limit the caller asked for. That limit is a safety
    # bound on how far a page may be rotated, not a hint.
    fine = np.clip(
        np.arange(best - coarse_step, best + coarse_step + 0.05, 0.1), -max_degrees, max_degrees
    )
    return float(max(fine, key=profile_variance))


def _downsample_for_analysis(gray: np.ndarray, target_height: int = 900) -> np.ndarray:
    h = gray.shape[0]
    if h <= target_height:
        return gray
    step = int(np.ceil(h / target_height))
    return gray[::step, ::step]


def _rotate_array(array: np.ndarray, degrees: float) -> np.ndarray:
    if abs(degrees) < 1e-3:
        return array
    if HAS_CV2:
        h, w = array.shape
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
        return cv2.warpAffine(array, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    return np.asarray(
        Image.fromarray(array).rotate(degrees, resample=Image.BILINEAR, fillcolor=0)
    )


def deskew(gray: np.ndarray, degrees: float, fill: int = 255) -> np.ndarray:
    if abs(degrees) < 1e-3:
        return gray
    if HAS_CV2:
        h, w = gray.shape
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
        return cv2.warpAffine(
            gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
            borderValue=fill,
        )
    return np.asarray(
        Image.fromarray(gray).rotate(degrees, resample=Image.BICUBIC, fillcolor=fill)
    )


# --------------------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------------------


def despeckle(binary: np.ndarray, max_area: int = 12) -> tuple[np.ndarray, int]:
    """Erase connected ink blobs smaller than ``max_area`` pixels.

    The threshold is kept low on purpose: Ethiopic diacritic marks are themselves small
    components, and an aggressive filter would strip the very detail that identifies a
    vowel order.
    """
    ink = binary < 128
    if not ink.any():
        return binary, 0

    labels, count = _connected_components(ink)
    if count == 0:
        return binary, 0

    sizes = np.bincount(labels.ravel())
    small = np.flatnonzero(sizes <= max_area)
    small = small[small != 0]
    if small.size == 0:
        return binary, 0

    mask = np.isin(labels, small)
    out = binary.copy()
    out[mask] = 255
    return out, int(small.size)


def _connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    if HAS_CV2:
        count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        return labels, count - 1
    try:
        from scipy import ndimage

        labels, count = ndimage.label(mask, structure=np.ones((3, 3)))
        return labels, count
    except ImportError:
        return np.zeros_like(mask, dtype=np.int32), 0


def remove_border(binary: np.ndarray, fraction: float = 0.04) -> tuple[np.ndarray, bool]:
    """Trim the black scanner gutter that frames many book scans.

    Only edge bands are cleared, and only when they are overwhelmingly dark, so genuine
    marginalia survive.
    """
    h, w = binary.shape
    band_h = max(1, int(h * fraction))
    band_w = max(1, int(w * fraction))
    out = binary.copy()
    touched = False

    regions = (
        (slice(0, band_h), slice(None)),
        (slice(h - band_h, h), slice(None)),
        (slice(None), slice(0, band_w)),
        (slice(None), slice(w - band_w, w)),
    )
    for rows, cols in regions:
        region = out[rows, cols]
        if region.size and float((region < 128).mean()) > 0.55:
            out[rows, cols] = 255
            touched = True
    return out, touched


def ink_coverage(binary: np.ndarray) -> float:
    return float((binary < 128).mean())


# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------


def preprocess(
    image: Image.Image, options: PreprocessOptions | None = None
) -> tuple[Image.Image, PreprocessReport]:
    """Run the full preprocessing chain, returning the result and a report of what it did."""
    options = options or PreprocessOptions()
    report = PreprocessReport()

    gray = to_gray_array(image)

    if options.invert_if_dark and gray.mean() < 100:
        gray = 255 - gray
        report.warnings.append("Image looked inverted; flipped to dark ink on light paper")

    if options.autocontrast:
        gray = np.asarray(ImageOps.autocontrast(Image.fromarray(gray), cutoff=1))

    if options.upscale_to_dpi and options.source_dpi:
        scale = options.upscale_to_dpi / options.source_dpi
        if scale > 1.05:
            new_size = (int(gray.shape[1] * scale), int(gray.shape[0] * scale))
            gray = np.asarray(Image.fromarray(gray).resize(new_size, Image.LANCZOS))
            report.scaled_by = scale

    if options.deskew:
        angle = estimate_skew(gray, options.max_skew_degrees)
        if abs(angle) > 0.05:
            gray = deskew(gray, angle)
        report.skew_degrees = angle

    binary, method = binarize(gray, options)
    report.binarized = method

    if options.remove_border and method != "none":
        binary, cropped = remove_border(binary, options.border_fraction)
        report.border_cropped = cropped

    if options.despeckle and method != "none":
        binary, removed = despeckle(binary, options.despeckle_max_area)
        report.speckles_removed = removed

    report.ink_coverage = ink_coverage(binary)
    if report.ink_coverage < 0.005:
        report.warnings.append("Almost no ink detected — the page may be blank or over-thresholded")
    elif report.ink_coverage > 0.45:
        report.warnings.append("Very heavy ink — the page may be under-thresholded or a photograph")

    return to_image(binary), report


def make_thumbnail(image: Image.Image, max_side: int = 240) -> Image.Image:
    thumb = image.copy()
    thumb.thumbnail((max_side, max_side), Image.LANCZOS)
    return thumb.convert("L")


def crop_box(image: Image.Image, box: tuple[int, int, int, int], padding: int = 2) -> Image.Image:
    """Crop ``(x, y, w, h)`` with a little padding, clamped to the image."""
    x, y, w, h = box
    left = max(0, x - padding)
    top = max(0, y - padding)
    right = min(image.width, x + w + padding)
    bottom = min(image.height, y + h + padding)
    return image.crop((left, top, right, bottom))


def detect_text_lines(binary: np.ndarray, min_height: int = 6, threshold: float = 0.008) -> list[tuple[int, int]]:
    """Find text line bands as ``(top, bottom)`` from the horizontal ink profile.

    A fallback for recognizers that do not return line geometry, and the basis for
    cropping line images for training data.
    """
    ink = (binary < 128).astype(np.float64)
    profile = ink.mean(axis=1)
    active = profile > threshold

    bands: list[tuple[int, int]] = []
    start: int | None = None
    for y, is_ink in enumerate(active):
        if is_ink and start is None:
            start = y
        elif not is_ink and start is not None:
            if y - start >= min_height:
                bands.append((start, y))
            start = None
    if start is not None and len(active) - start >= min_height:
        bands.append((start, len(active)))
    return bands
