"""Scan preprocessing.

Synthetic pages are used deliberately: a rendered page has a known skew, a known amount
of noise and a known amount of ink, so each stage can be checked for actually improving
the thing it claims to improve rather than merely running without raising.

The array-level helpers work on uint8 grayscale where 0 is ink and 255 is paper.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from PIL import Image, ImageDraw

from amharic_studio.core import imaging
from amharic_studio.core.imaging import BinarizeMethod, PreprocessOptions


def page_image(width: int = 600, height: int = 400, skew: float = 0.0, lines: int = 6):
    """A white page with black text-like bars, optionally rotated."""
    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)
    for i in range(lines):
        y = 40 + i * 50
        draw.rectangle([60, y, width - 60, y + 18], fill=0)
    if skew:
        img = img.rotate(skew, resample=Image.BICUBIC, fillcolor=255)
    return img


def page(**kwargs) -> np.ndarray:
    return imaging.to_gray_array(page_image(**kwargs))


def speckled(gray: np.ndarray, amount: float = 0.02, seed: int = 0) -> np.ndarray:
    out = gray.copy()
    out[np.random.default_rng(seed).random(out.shape) < amount] = 0
    return out


def binary(gray: np.ndarray) -> np.ndarray:
    return imaging.otsu_threshold(gray)


class TestConversion:
    def test_to_gray_array_shape_is_row_major(self):
        assert imaging.to_gray_array(page_image(120, 80)).shape == (80, 120)

    def test_round_trip_through_an_array(self):
        original = page_image(120, 80)
        assert imaging.to_image(imaging.to_gray_array(original)).size == original.size

    def test_colour_input_is_accepted(self):
        colour = Image.new("RGB", (60, 40), (200, 100, 50))
        assert imaging.to_gray_array(colour).shape == (40, 60)


class TestBinarization:
    @pytest.mark.parametrize(
        "threshold_fn",
        [imaging.otsu_threshold, imaging.sauvola_threshold, imaging.adaptive_mean_threshold],
    )
    def test_output_is_strictly_two_valued(self, threshold_fn):
        assert set(np.unique(threshold_fn(page()))) <= {0, 255}

    @pytest.mark.parametrize("method", list(BinarizeMethod))
    def test_every_method_is_wired_up(self, method):
        result, name = imaging.binarize(page(), PreprocessOptions(binarize=method))
        assert name.replace(" ", "_") == method.value
        assert result.shape == page().shape

    def test_binarization_preserves_the_text(self):
        result = imaging.otsu_threshold(page())
        assert 0.05 < imaging.ink_coverage(result) < 0.5

    def test_sauvola_survives_an_uneven_background(self):
        # Faded ink plus a heavy shading gradient is the case no global threshold can
        # solve: shaded paper on one side is darker than the ink on the other, so any
        # single cut either floods the dark edge or loses the pale text. Sauvola is
        # windowed precisely so that it does not have to choose.
        img = Image.new("L", (600, 400), 255)
        draw = ImageDraw.Draw(img)
        for i in range(6):
            draw.rectangle([60, 40 + i * 50, 540, 58 + i * 50], fill=110)  # faded ink
        gray = imaging.to_gray_array(img).astype(np.float32)
        gray *= np.linspace(0.35, 1.0, gray.shape[1], dtype=np.float32)[None, :]
        shaded = gray.astype(np.uint8)

        truth = imaging.ink_coverage(imaging.otsu_threshold(imaging.to_gray_array(img)))
        sauvola = imaging.ink_coverage(imaging.sauvola_threshold(shaded))
        otsu = imaging.ink_coverage(imaging.otsu_threshold(shaded))
        assert abs(sauvola - truth) < abs(otsu - truth)

    def test_a_blank_page_does_not_become_all_ink(self):
        blank = np.full((200, 200), 255, dtype=np.uint8)
        assert imaging.ink_coverage(imaging.sauvola_threshold(blank)) < 0.02


class TestSkew:
    def test_a_straight_page_measures_near_zero(self):
        assert abs(imaging.estimate_skew(page())) < 0.5

    @pytest.mark.parametrize("angle", [-3.0, -1.5, 1.5, 3.0])
    def test_deskewing_by_the_measured_angle_straightens_the_page(self, angle):
        # The contract that matters is that measure-then-correct converges; this also
        # pins the sign convention, which is the part that is easy to get backwards.
        crooked = page(skew=angle)
        measured = imaging.estimate_skew(crooked)
        assert abs(measured) > 0.7, f"failed to detect a {angle}° skew"
        straightened = imaging.deskew(crooked, measured)
        assert abs(imaging.estimate_skew(straightened)) < abs(measured) / 2

    def test_skew_search_is_bounded(self):
        assert abs(imaging.estimate_skew(page(skew=3.0), max_degrees=2.0)) <= 2.0


class TestCleanup:
    def test_despeckle_removes_isolated_dots_and_keeps_the_text(self):
        clean = binary(page())
        noisy = binary(speckled(page()))
        despeckled, removed = imaging.despeckle(noisy)
        assert removed > 0
        assert imaging.ink_coverage(despeckled) < imaging.ink_coverage(noisy)
        assert imaging.ink_coverage(despeckled) > imaging.ink_coverage(clean) * 0.8

    def test_despeckle_reports_nothing_removed_from_clean_text(self):
        _result, removed = imaging.despeckle(binary(page()))
        assert removed == 0

    def test_remove_border_clears_a_black_scan_edge(self):
        img = page_image()
        ImageDraw.Draw(img).rectangle([0, 0, 24, img.height], fill=0)  # the platen edge
        framed = binary(imaging.to_gray_array(img))
        cleaned, was_cleared = imaging.remove_border(framed)
        assert was_cleared
        assert imaging.ink_coverage(cleaned[:, :24]) < imaging.ink_coverage(framed[:, :24])

    def test_remove_border_keeps_the_page_geometry(self):
        # Whitening rather than cropping is what keeps the recognizer's word boxes valid
        # against the stored page image.
        original = binary(page())
        cleaned, _ = imaging.remove_border(original)
        assert cleaned.shape == original.shape

    def test_remove_border_leaves_a_clean_page_alone(self):
        original = binary(page())
        cleaned, was_cleared = imaging.remove_border(original)
        assert not was_cleared
        assert np.array_equal(cleaned, original)

    def test_ink_coverage_is_a_fraction(self):
        assert 0.0 <= imaging.ink_coverage(binary(page())) <= 1.0
        assert imaging.ink_coverage(np.full((50, 50), 255, dtype=np.uint8)) == 0.0


class TestLineDetection:
    def test_finds_every_text_line(self):
        assert len(imaging.detect_text_lines(binary(page(lines=6)))) == 6

    def test_lines_are_returned_top_to_bottom(self):
        tops = [top for top, _bottom in imaging.detect_text_lines(binary(page(lines=6)))]
        assert tops == sorted(tops)

    def test_line_bands_do_not_overlap(self):
        bands = imaging.detect_text_lines(binary(page(lines=6)))
        for (_t1, b1), (t2, _b2) in itertools.pairwise(bands):
            assert b1 <= t2

    def test_a_blank_page_has_no_lines(self):
        assert imaging.detect_text_lines(np.full((200, 200), 255, dtype=np.uint8)) == []


class TestPipeline:
    def test_preprocess_reports_what_it_did(self):
        _result, report = imaging.preprocess(page_image(skew=2.0))
        described = report.describe()
        assert "deskewed" in described
        assert "ink" in described

    def test_preprocess_straightens_and_binarizes(self):
        source = imaging.to_image(speckled(page(skew=2.0)))
        result, report = imaging.preprocess(source)
        assert abs(report.skew_degrees) > 1.0
        assert report.binarized == BinarizeMethod.SAUVOLA.value
        assert set(np.unique(imaging.to_gray_array(result))) <= {0, 255}

    def test_every_stage_can_be_switched_off(self):
        options = PreprocessOptions(
            deskew=False,
            binarize=BinarizeMethod.NONE,
            despeckle=False,
            remove_border=False,
        )
        original = page_image()
        result, report = imaging.preprocess(original, options)
        assert result.size == original.size
        assert report.skew_degrees == 0.0
        assert report.binarized == "none"
        assert not report.border_cropped

    def test_a_tiny_image_does_not_crash_anything(self):
        result, _report = imaging.preprocess(Image.new("L", (3, 3), 255))
        assert result.size == (3, 3)

    def test_upscaling_a_low_resolution_scan(self):
        options = PreprocessOptions(source_dpi=150, upscale_to_dpi=300)
        result, report = imaging.preprocess(page_image(300, 200), options)
        assert report.scaled_by > 1.5
        assert result.width > 300


class TestThumbnails:
    def test_thumbnail_fits_the_box_and_keeps_the_aspect_ratio(self):
        thumb = imaging.make_thumbnail(page_image(600, 400), 120)
        assert max(thumb.size) <= 120
        assert abs(thumb.width / thumb.height - 1.5) < 0.05

    def test_a_small_image_is_not_blown_up(self):
        small = Image.new("L", (80, 60), 255)
        assert imaging.make_thumbnail(small, 240).size == small.size
