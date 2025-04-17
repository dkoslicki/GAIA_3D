#!/usr/bin/env python3
"""
Enhanced Stellar Parallax Animation Generator
=============================================
Creates an animated GIF that simulates stellar parallax by shifting actual
star pixels according to *Gaia* distances.

Changes in this version
-----------------------
* ``create_shifted_star_image_enhanced`` now honours the parameters
  ``parallax_mode``, ``contrast_factor`` and ``power``.
* ``create_transparent_parallax_frames`` no longer hard‑codes
  ``threshold_percentile`` and ``enhancement_factor`` – both are parameters
  (default 75 / 2) and are wired to the existing CLI flags.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from tqdm import tqdm

from gif_helpers import save_efficient_gif, save_optimized_gif

try:
    from astropy.io import fits  # type: ignore
except ImportError:  # pragma: no cover
    fits = None  # type: ignore[assignment]

try:
    from skimage import exposure, filters, measure, morphology  # type: ignore
except ImportError:  # pragma: no cover
    exposure = filters = measure = morphology = None  # type: ignore[assignment]

from scipy import ndimage

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("parallax_animator")


# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #
def check_dependencies() -> bool:
    """Return **True** if *astropy* and *scikit‑image* are both present."""
    missing: list[str] = []
    if fits is None:
        missing.append("astropy")
    if exposure is None:
        missing.append("scikit-image")

    if missing:  # pragma: no cover
        logger.warning(
            "Missing optional dependencies (%s).  Functionality will be limited.",
            ", ".join(missing),
        )
    return not missing


def load_star_data(csv_path: str) -> pd.DataFrame:
    """Load a star catalogue and normalise the expected column names."""
    logger.info("Loading star data from %s", csv_path)
    df = pd.read_csv(csv_path)

    canonical = ["image_x", "image_y", "distance_ly"]
    alt = ["image_x_coordinate", "image_y_coordinate", "distance_light_years"]
    if not all(c in df.columns for c in canonical):
        if all(c in df.columns for c in alt):
            df = df.rename(
                columns={
                    "image_x_coordinate": "image_x",
                    "image_y_coordinate": "image_y",
                    "distance_light_years": "distance_ly",
                }
            )
        else:  # pragma: no cover
            raise ValueError(f"CSV missing one of {canonical} or {alt}")

    before = len(df)
    df = df[df["distance_ly"].gt(0) & df["distance_ly"].notna()]
    logger.info("Loaded %d stars (filtered %d invalid rows).", len(df), before - len(df))
    return df


def load_image_file(file_path: str | Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Read an image (FITS or common raster) and return an 8‑bit numpy array."""
    path = Path(file_path)
    meta: Dict[str, Any] = {}
    if path.suffix.lower() in {".fits", ".fit"}:
        if fits is None:  # pragma: no cover
            raise ImportError("FITS support requires astropy.")
        with fits.open(path) as hdul:
            data = hdul[0].data
            meta["header"] = dict(hdul[0].header)
        if data.ndim > 2:
            data = np.mean(data, axis=0) if data.shape[0] <= 3 else data[0]
        p1, p99 = np.percentile(data, (1, 99))
        data = np.clip((data - p1) / (p99 - p1), 0, 1)
        image = (data * 255).astype(np.uint8)
    else:
        pil = Image.open(path)
        meta.update({"mode": pil.mode, "format": pil.format})
        if pil.mode == "P":
            pil = pil.convert("RGBA")
        elif pil.mode not in {"RGB", "RGBA", "L"}:
            pil = pil.convert("RGB")
        image = np.array(pil)
    logger.info("Loaded %s with shape %s", path.name, image.shape)
    return image, meta


# --------------------------------------------------------------------------- #
# Star extraction and region association
# --------------------------------------------------------------------------- #
def extract_stars(
    image_data: np.ndarray,
    starless_data: Optional[np.ndarray] = None,
    threshold_factor: float = 2.0,
    min_size: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return *(mask, stars_only)* isolating stellar pixels."""
    if image_data.ndim == 3 and image_data.shape[2] >= 3:
        gray = np.mean(image_data[..., :3], axis=2)
        has_alpha = image_data.shape[2] == 4
    else:
        gray = image_data
        has_alpha = False

    # ----- path A: difference against supplied starless frame ----------------
    if starless_data is not None:
        gray_bg = (
            np.mean(starless_data[..., :3], axis=2)
            if starless_data.ndim == 3 and starless_data.shape[2] >= 3
            else starless_data
        )
        if gray_bg.shape != gray.shape:
            gray_bg = np.array(Image.fromarray(gray_bg).resize(gray.shape[::-1]))

        diff = np.clip(gray.astype(float) - gray_bg.astype(float), 0, None)
        diff = (diff / diff.max() * 255).astype(np.uint8) if diff.max() else diff

        if exposure is not None and filters is not None:
            thresh = filters.threshold_otsu(diff) * 0.5
            mask = diff > thresh
            if morphology is not None:
                mask = morphology.remove_small_objects(mask, min_size=min_size)
        else:
            thresh = np.percentile(diff, 95)
            mask = ndimage.binary_opening(diff > thresh, structure=np.ones((2, 2)))

    # ----- path B: simple threshold -----------------------------------------
    else:  # pragma: no cover
        if exposure is not None and filters is not None:
            p2, p98 = np.percentile(gray, (2, 98))
            rescaled = exposure.rescale_intensity(gray, in_range=(p2, p98))
            thresh = filters.threshold_otsu(rescaled) * threshold_factor
            mask = rescaled > thresh
            if morphology is not None:
                mask = morphology.binary_opening(
                    morphology.remove_small_objects(mask, min_size=min_size),
                    morphology.disk(1),
                )
        else:
            p2, p98 = np.percentile(gray, (2, 98))
            rescaled = np.clip((gray - p2) / (p98 - p2) * 255, 0, 255)
            thresh = np.percentile(rescaled, 95) * threshold_factor / 100
            mask = ndimage.binary_opening(rescaled > thresh)

    stars_only = np.zeros_like(image_data)
    if image_data.ndim == 3:
        stars_only[..., :3] = np.where(mask[..., None], image_data[..., :3], 0)
        if has_alpha and stars_only.shape[2] == 4:
            stars_only[..., 3] = np.where(mask, image_data[..., 3], 0)
    else:
        stars_only = np.where(mask, image_data, 0)

    return mask, stars_only


def analyze_star_regions(
    stars_mask: np.ndarray, star_data: pd.DataFrame
) -> Dict[int, Dict[str, Any]]:
    """Associate connected star blobs with the nearest catalogue entry."""
    if measure is None:  # pragma: no cover
        out: Dict[int, Dict[str, Any]] = {}
        for idx, row in star_data.iterrows():
            y, x = int(round(row["image_y"])), int(round(row["image_x"]))
            out[idx + 1] = {
                "centroid": (y, x),
                "coords": [(y, x)],
                "area": 1,
                "star_data": row,
                "distance": row["distance_ly"],
            }
        return out

    labels, _ = measure.label(stars_mask, return_num=True)
    props = measure.regionprops(labels)
    cat_xy = np.array([[r.image_y, r.image_x] for _, r in star_data.iterrows()])
    out: Dict[int, Dict[str, Any]] = {}
    for rp in props:
        yc, xc = rp.centroid
        if not len(cat_xy):
            break
        dists = np.sqrt(((cat_xy - (yc, xc)) ** 2).sum(1))
        idx = int(np.argmin(dists))
        if dists[idx] <= max(rp.area ** 0.5 * 2, 30):
            row = star_data.iloc[idx]
            out[rp.label] = {
                "centroid": rp.centroid,
                "coords": rp.coords,
                "area": rp.area,
                "star_data": row,
                "distance": row["distance_ly"],
            }
    return out


# --------------------------------------------------------------------------- #
# Core pixel‑shifting helper
# --------------------------------------------------------------------------- #
def _scale_parallax(
    norm_dist: float,
    mode: str,
    contrast: float,
    power_exp: float,
    enhancement_exp: float,
) -> float:
    """Return a parallax multiplier ∊ [0 … 1] according to *mode*."""
    if mode == "linear":
        return max(0.0, (1.0 - norm_dist) * contrast)
    if mode == "inverse":
        return contrast / (norm_dist + contrast)
    if mode == "logarithmic":
        eps = 1e-4
        return np.log1p(contrast) - np.log1p(contrast * norm_dist + eps)
    if mode == "power":
        return (1.0 - norm_dist) ** power_exp
    # default / "enhanced"
    return (1.0 - norm_dist) ** enhancement_exp


def create_shifted_star_image_enhanced(
    stars_image: np.ndarray,
    star_regions: Dict[int, Dict[str, Any]],
    offset_factor: float,
    direction: str,
    width: int,
    height: int,
    threshold_percentile: float = 75.0,
    enhancement_factor: float = 2.0,
    *,
    parallax_mode: str = "enhanced",
    contrast_factor: float = 1.0,
    power: float = 2.0,
) -> np.ndarray:
    """Return a *stars‑only* layer with per‑star parallax applied."""
    shifted = np.zeros_like(stars_image)

    distances = np.array([r["distance"] for r in star_regions.values()])
    if distances.size == 0:
        return shifted

    max_d = np.percentile(distances, threshold_percentile)
    min_d = distances[distances <= max_d].min()

    for region in star_regions.values():
        d = region["distance"]
        if d > max_d:
            continue

        nd = (d - min_d) / (max_d - min_d) if max_d != min_d else 0.0
        parallax = _scale_parallax(
            nd,
            parallax_mode,
            contrast_factor,
            power,
            enhancement_factor,
        )
        dx = offset_factor * parallax if direction in {"horizontal", "both"} else 0.0
        dy = offset_factor * parallax * 0.3 if direction in {"vertical", "both"} else 0.0

        for y, x in region["coords"]:
            ny, nx = int(round(y + dy)), int(round(x + dx))
            if 0 <= ny < height and 0 <= nx < width:
                shifted[ny, nx] = stars_image[y, x]

    from scipy.ndimage import gaussian_filter

    if shifted.ndim == 3:
        for c in range(shifted.shape[2]):
            shifted[..., c] = gaussian_filter(shifted[..., c], 0.5)
    else:
        shifted = gaussian_filter(shifted, 0.5)
    return shifted


# --------------------------------------------------------------------------- #
# Frame generators
# --------------------------------------------------------------------------- #
def create_parallax_frames(
    star_data: pd.DataFrame,
    original_image: np.ndarray,
    starless_image: Optional[np.ndarray] = None,
    num_frames: int = 30,
    parallax_amplitude: float = 10.0,
    direction: str = "horizontal",
    blur_stars: bool = True,
    parallax_mode: str = "logarithmic",
    contrast_factor: float = 1.0,
    power: float = 2.0,
    threshold_percentile: float = 75.0,
    enhancement_factor: float = 2.0,
) -> List[Image.Image]:
    """Generate frames using the *generic* parallax model (any mode)."""
    height, width = original_image.shape[:2]
    smask, sonly = (
        extract_stars(original_image, starless_image)
        if starless_image is not None
        else extract_stars(original_image)
    )
    background = (
        starless_image.copy()
        if starless_image is not None
        else np.where(smask[..., None], 0, original_image)
    )
    regions = analyze_star_regions(smask, star_data)
    bg_pil = Image.fromarray(background.astype(np.uint8)).convert("RGBA")

    frames: List[Image.Image] = []
    for k in tqdm(range(num_frames), desc="Building frames"):
        phase = math.sin(2 * math.pi * k / num_frames)
        offset = phase * parallax_amplitude
        shifted = create_shifted_star_image_enhanced(
            sonly,
            regions,
            offset,
            direction,
            width,
            height,
            threshold_percentile,
            enhancement_factor,
            parallax_mode=parallax_mode,
            contrast_factor=contrast_factor,
            power=power,
        )

        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[..., :3] = shifted[..., :3] if shifted.ndim == 3 else np.repeat(
            shifted[..., None], 3, axis=2
        )
        alpha = shifted[..., 3] if shifted.ndim == 3 and shifted.shape[2] == 4 else (
            shifted.max(axis=2) if shifted.ndim == 3 else shifted
        )
        rgba[..., 3] = np.where(alpha > 0, 255, 0)

        stars_pil = Image.fromarray(rgba, "RGBA")
        if blur_stars:
            stars_pil = stars_pil.filter(ImageFilter.GaussianBlur(0.5))
        frame = bg_pil.copy()
        frame.paste(stars_pil, (0, 0), stars_pil)
        frames.append(frame)
    return frames


def create_parallax_frames_enhanced(
    star_data: pd.DataFrame,
    original_image: np.ndarray,
    starless_image: Optional[np.ndarray] = None,
    num_frames: int = 30,
    parallax_amplitude: float = 10.0,
    direction: str = "horizontal",
    blur_stars: bool = True,
    threshold_percentile: float = 75.0,
    enhancement_factor: float = 2.0,
) -> List[Image.Image]:
    """Convenience wrapper that calls *create_parallax_frames* with mode “enhanced”."""
    return create_parallax_frames(
        star_data,
        original_image,
        starless_image,
        num_frames,
        parallax_amplitude,
        direction,
        blur_stars,
        parallax_mode="enhanced",
        contrast_factor=1.0,
        power=2.0,
        threshold_percentile=threshold_percentile,
        enhancement_factor=enhancement_factor,
    )


def create_transparent_parallax_frames(
    star_data: pd.DataFrame,
    original_image: np.ndarray,
    starless_image: Optional[np.ndarray] = None,
    num_frames: int = 30,
    parallax_amplitude: float = 10.0,
    direction: str = "horizontal",
    blur_stars: bool = True,
    parallax_mode: str = "logarithmic",
    contrast_factor: float = 1.0,
    power: float = 2.0,
    threshold_percentile: float = 75.0,
    enhancement_factor: float = 2.0,
) -> List[Image.Image]:
    """Generate *transparent* frames containing only the parallax‑shifted stars."""
    height, width = original_image.shape[:2]
    smask, sonly = (
        extract_stars(original_image, starless_image)
        if starless_image is not None
        else extract_stars(original_image)
    )
    regions = analyze_star_regions(smask, star_data)

    frames: List[Image.Image] = []
    for k in tqdm(range(num_frames), desc="Transparent frames"):
        phase = math.sin(2 * math.pi * k / num_frames)
        offset = phase * parallax_amplitude
        shifted = create_shifted_star_image_enhanced(
            sonly,
            regions,
            offset,
            direction,
            width,
            height,
            threshold_percentile,
            enhancement_factor,
            parallax_mode=parallax_mode,
            contrast_factor=contrast_factor,
            power=power,
        )

        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[..., :3] = shifted[..., :3] if shifted.ndim == 3 else np.repeat(
            shifted[..., None], 3, axis=2
        )
        alpha = shifted[..., 3] if shifted.ndim == 3 and shifted.shape[2] == 4 else (
            shifted.max(axis=2) if shifted.ndim == 3 else shifted
        )
        rgba[..., 3] = np.where(alpha > 0, 255, 0)

        frame = Image.fromarray(rgba, "RGBA")
        if blur_stars:
            frame = frame.filter(ImageFilter.GaussianBlur(0.5))
        frames.append(frame)
    return frames


# --------------------------------------------------------------------------- #
# Misc helpers
# --------------------------------------------------------------------------- #
def create_background_from_starless(
    starless_image: np.ndarray, original_shape: Tuple[int, int, int]
) -> np.ndarray:
    """Resize *starless_image* to the shape of *original_shape* if needed."""
    if starless_image.shape[:2] == original_shape[:2]:
        return starless_image
    mode = (
        "RGBA"
        if starless_image.ndim == 3 and starless_image.shape[2] == 4
        else "RGB"
        if starless_image.ndim == 3
        else "L"
    )
    pil = Image.fromarray(starless_image.astype(np.uint8), mode)
    pil = pil.resize((original_shape[1], original_shape[0]))
    return np.array(pil)


def save_gif(
    frames: List[Image.Image],
    output_path: str | Path,
    duration: int = 100,
    loop: int = 0,
    optimize: bool = True,
) -> None:
    """Write *frames* as an animated GIF to *output_path*."""
    frames = [f.convert("RGBA") for f in frames]
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop,
        optimize=optimize,
    )
    logger.info("Saved %s", output_path)


# --------------------------------------------------------------------------- #
# CLI parsing
# --------------------------------------------------------------------------- #
def parse_arguments() -> argparse.Namespace:
    """Return parsed CLI arguments."""
    p = argparse.ArgumentParser(
        prog="parallax_animator",
        description="Create a depth‑aware parallax animation from stellar data.",
    )
    p.add_argument("csv_file")
    p.add_argument("-i", "--image", required=True)
    p.add_argument("-s", "--starless")
    p.add_argument("-o", "--output", default="parallax_animation.gif")
    p.add_argument("-f", "--frames", type=int, default=30)
    p.add_argument("-a", "--amplitude", type=float, default=10.0)
    p.add_argument("-d", "--duration", type=int, default=50)
    p.add_argument("--direction", choices=["horizontal", "vertical", "both"], default="horizontal")

    p.add_argument("--threshold", type=float, default=2.0)
    p.add_argument("--min-size", type=int, default=3)
    p.add_argument("--no-blur", action="store_true")

    p.add_argument(
        "--parallax-mode",
        choices=["logarithmic", "inverse", "power", "linear", "enhanced"],
        default="enhanced",
    )
    p.add_argument("--distance-threshold-percentile", type=float, default=75.0)
    p.add_argument("--enhancement-factor", type=float, default=2.0)
    p.add_argument("--contrast", type=float, default=1.0)
    p.add_argument("--power", type=float, default=2.0)

    p.add_argument(
        "--optimization-level", choices=["low", "medium", "high"], default="medium"
    )
    p.add_argument("--lossy", type=int, default=80, choices=range(0, 201))
    p.add_argument("--debug", action="store_true")
    p.add_argument("--save-stars")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:  # pragma: no cover
    check_dependencies()
    args = parse_arguments()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    star_df = load_star_data(args.csv_file)
    orig_img, _ = load_image_file(args.image)

    starless_img = None
    if args.starless:
        s_img, _ = load_image_file(args.starless)
        starless_img = create_background_from_starless(s_img, orig_img.shape)

    mask, stars_only = extract_stars(
        orig_img, starless_img, args.threshold, args.min_size
    )
    if args.save_stars:
        mode = "RGBA" if stars_only.ndim == 3 and stars_only.shape[2] == 4 else "RGB"
        Image.fromarray(stars_only.astype(np.uint8), mode).save(args.save_stars)

    builder = (
        create_parallax_frames_enhanced
        if args.parallax_mode == "enhanced"
        else create_parallax_frames
    )

    frames = builder(
        star_df,
        orig_img,
        starless_img,
        args.frames,
        args.amplitude,
        args.direction,
        not args.no_blur,
        threshold_percentile=args.distance_threshold_percentile,
        enhancement_factor=args.enhancement_factor,
        parallax_mode=args.parallax_mode,
        contrast_factor=args.contrast,
        power=args.power,
    )

    if args.optimization_level == "high":
        save_efficient_gif(frames, args.output, duration=args.duration)
    elif args.optimization_level == "medium":
        save_optimized_gif(frames, args.output, duration=args.duration)
    else:
        save_gif(frames, args.output, duration=args.duration)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
