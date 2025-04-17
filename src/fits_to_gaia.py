#!/usr/bin/env python3
"""
FITS to Gaia Star Distances with ASTAP Plate Solving

This script processes a raw astrophotography FITS file without WCS information,
plate solves it using ASTAP, detects stars, queries the Gaia catalog for matching stars,
and outputs a CSV file with image coordinates and distances.

Requirements:
- astropy
- astroquery
- photutils
- numpy
- pandas
- scipy
- matplotlib (for visualization)
"""

import os
import subprocess
import tempfile
import shutil
import numpy as np
import pandas as pd
import astropy
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from astroquery.vizier import Vizier
import logging
from typing import List, Optional, Tuple
from astropy.table import Table
from play_with_star_finding import (
    multi_scale_star_detection,
    enhanced_star_detection,
    detect_bright_bloated_stars,
    combine_star_catalogs,
)

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('fits_to_gaia')


def detect_stars(
    image_data: np.ndarray,
    use_multi_scale: bool = True,
) -> Optional[Table]:
    """Detect stars in *image_data* using multi-scale or enhanced methods.

    The default strategy uses the multi‑scale detection routine defined in
    *play_with_star_finding.py*, which applies two passes of IRAFStarFinder
    tuned for normal and bloated PSFs, then merges the results while removing
    near‑duplicates.

    Parameters
    ----------
    image_data : numpy.ndarray
        2‑D array of pixel values. If the array has more than two dimensions
        (e.g., multi‑extension FITS or RGB data), it is collapsed to 2‑D.
    use_multi_scale : bool, optional
        If True (default), run `multi_scale_star_detection`; otherwise run
        one standard and one bloated detection pass via `enhanced_star_detection`
        and merge them with `combine_star_catalogs`.

    Returns
    -------
    astropy.table.Table or None
        Table of detected sources, or None if no sources were found.
    """
    # Ensure 2‑D data
    if image_data.ndim > 2:
        logger.debug(
            "Converting multi‑dimensional image data (shape %s) to 2‑D", image_data.shape
        )
        image_data = (
            np.mean(image_data, axis=0)
            if image_data.ndim == 3
            else image_data[0]
        )

    # Run detection
    if use_multi_scale:
        logger.debug("Running multi‑scale star detection (IRAFStarFinder)")
        sources = multi_scale_star_detection(image_data)
    else:
        logger.debug("Running two‑pass enhanced star detection (standard + bloated)")
        sources_standard = enhanced_star_detection(
            image_data,
            fwhm=5.0,
            threshold_sigma=3.0,
            use_iraf=True,
            sharplo=0.2,
            sharphi=1.0,
        )
        sources_bloated = enhanced_star_detection(
            image_data,
            fwhm=12.0,
            threshold_sigma=5.0,
            use_iraf=True,
            sharplo=0.01,
            sharphi=2.0,
        )
        sources = combine_star_catalogs(sources_standard, sources_bloated)

    # Optionally look for additional very bloated stars missed earlier
    if sources is not None and len(sources) > 0:
        bloated_extra = detect_bright_bloated_stars(image_data, sources)
        sources = combine_star_catalogs(sources, bloated_extra)

    # Final sanity check
    if sources is None or len(sources) == 0:
        logger.warning("No stars detected in the image with the IRAF approach")
        return None

    logger.info("Detected %d stars in the image (IRAFStarFinder)", len(sources))
    return sources


def detect_stars_old(
    image_data: np.ndarray,
) -> Optional[Table]:
    """Detect stars in the image using DAOStarFinder.

    This function applies a sigma‑clipped background estimation and uses
    DAOStarFinder to identify star‑like sources.

    Parameters
    ----------
    image_data : numpy.ndarray
        2‑D array of pixel values. If the array has more than two dimensions
        (e.g., RGB or multi‑extension FITS), it is collapsed to 2‑D.

    Returns
    -------
    astropy.table.Table or None
        Table of detected sources, or None if no sources were found.
    """
    # NOTE: Redundant image_data dimensionality handling also present in detect_stars and process_fits_file
    if image_data.ndim > 2:
        logger.info("Converting multi‑dimensional image data to 2D")
        image_data = np.mean(image_data, axis=0) if image_data.ndim == 3 else image_data[0]

    # Compute image statistics with sigma clipping to ignore outliers
    mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
    logger.info(f"Image statistics - Mean: {mean:.2f}, Median: {median:.2f}, Std: {std:.2f}")

    # Configure star finder
    daofind = DAOStarFinder(fwhm=7.0, threshold=5. * std)  # TODO: adjust parameters

    # Find stars
    sources = daofind(image_data - median)

    if sources is None or len(sources) == 0:
        logger.warning("No stars detected in the image")
        return None

    logger.info(f"Detected {len(sources)} stars in the image")
    return sources


def find_astap_executable() -> Optional[str]:
    """Find the ASTAP executable on the system.

    Searches common installation paths and returns the first path found.

    Returns
    -------
    str or None
        Path to ASTAP executable if found, None otherwise.
    """
    common_paths = [
        # Windows paths
        "C:\\Program Files\\ASTAP\\astap.exe",
        "C:\\Program Files (x86)\\ASTAP\\astap.exe",
        os.path.expanduser("~\\AppData\\Local\\Programs\\ASTAP\\astap.exe"),
        os.path.expanduser("~\\ASTAP\\astap.exe"),
        "astap.exe",
        # Linux/macOS paths
        "/usr/bin/astap",
        "/usr/local/bin/astap",
        os.path.expanduser("~/ASTAP/astap"),
    ]

    for path in common_paths:
        if os.path.isfile(path):
            logger.info(f"Found ASTAP at: {path}")
            return path

    logger.warning("ASTAP executable not found in common locations")
    return None


def plate_solve_with_astap(
    fits_path: str,
    astap_path: Optional[str] = None,
) -> Optional[WCS]:
    """Perform plate solving on a FITS file using ASTAP.

    Parameters
    ----------
    fits_path : str
        Path to the FITS file to plate solve.
    astap_path : str, optional
        Path to the ASTAP executable. If None, attempts to find it.

    Returns
    -------
    astropy.wcs.WCS or None
        WCS solution if successful, None otherwise.
    """
    logger.info(f"Attempting to plate solve with ASTAP: {fits_path}")

    if astap_path is None:
        astap_path = find_astap_executable()
        if astap_path is None:
            logger.error("ASTAP executable not found. Please specify the path to astap.exe.")
            return None

    with tempfile.TemporaryDirectory() as temp_dir:
        base_name = os.path.splitext(os.path.basename(fits_path))[0]
        wcs_fits_path = os.path.join(temp_dir, f"{base_name}_solved.fits")

        command = [
            astap_path,
            "-f",
            "-r", "360",
            "-speed", "0",
            "-solve", fits_path,
            "-o", wcs_fits_path,
        ]

        logger.info(f"Running ASTAP command: {' '.join(command)}")

        try:
            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=300,
            )

            if process.returncode != 0:
                logger.error(f"ASTAP failed with return code {process.returncode}")
                logger.error(f"ASTAP stderr: {process.stderr}")
                return None

            if not os.path.isfile(wcs_fits_path):
                logger.error(f"ASTAP did not create the output file: {wcs_fits_path}")
                return None

            with fits.open(wcs_fits_path) as hdul:
                header = hdul[0].header
                try:
                    wcs = WCS(header, naxis=2)
                    _ = wcs.pixel_to_world(0, 0)
                    logger.info("Successfully extracted WCS from ASTAP solution")

                    # Copy WCS keywords back into original FITS
                    with fits.open(fits_path, mode='update') as original_hdul:
                        wcs_keywords = ['CTYPE1', 'CTYPE2', 'CRVAL1', 'CRVAL2',
                                        'CRPIX1', 'CRPIX2', 'CD1_1', 'CD1_2',
                                        'CD2_1', 'CD2_2', 'CDELT1', 'CDELT2',
                                        'CROTA1', 'CROTA2', 'PC1_1', 'PC1_2',
                                        'PC2_1', 'PC2_2', 'RADESYS', 'EQUINOX']
                        for keyword in wcs_keywords:
                            if keyword in header:
                                original_hdul[0].header[keyword] = header[keyword]
                        original_hdul.flush()

                    return wcs
                except Exception as e:
                    logger.error(f"Failed to extract WCS from ASTAP solution: {e}")
                    return None

        except subprocess.TimeoutExpired:
            logger.error("ASTAP plate solving timed out after 5 minutes")
            return None
        except Exception as e:
            logger.error(f"Error running ASTAP: {e}")
            return None


def query_gaia_for_region(
    center_coord: SkyCoord,
    radius: float = 0.5,
) -> Optional[Table]:
    """Query Gaia DR3 for stars in a specific region.

    Parameters
    ----------
    center_coord : astropy.coordinates.SkyCoord
        Center coordinates of the search region.
    radius : float, optional
        Search radius in degrees (default is 0.5).

    Returns
    -------
    astropy.table.Table or None
        Table of Gaia stars if successful, None otherwise.
    """
    logger.info(
        f"Querying Gaia DR3 at RA={center_coord.ra.degree:.6f}, "
        f"Dec={center_coord.dec.degree:.6f}, radius={radius}°"
    )
    Vizier.ROW_LIMIT = -1
    gaia_catalog = "I/355/gaiadr3"

    try:
        gaia_results = Vizier.query_region(
            center_coord,
            radius=radius * u.degree,
            catalog=gaia_catalog,
            column_filters={"Plx": ">0"},
        )

        if not gaia_results or len(gaia_results) == 0:
            logger.warning("No Gaia matches found in the field of view")
            return None

        gaia_table = gaia_results[0]
        logger.info(f"Found {len(gaia_table)} Gaia stars in the field of view")
        return gaia_table

    except Exception as e:
        logger.error(f"Error querying Gaia: {e}")
        return None


def match_stars(
    detected_sources: Table,
    wcs: WCS,
    gaia_table: Table,
    max_separation: float = 25.0,
) -> List[Tuple[float, float, float]]:
    """Match detected stars with Gaia catalog entries and compute distances.

    Parameters
    ----------
    detected_sources : astropy.table.Table
        Table of detected sources with 'xcentroid' and 'ycentroid' columns.
    wcs : astropy.wcs.WCS
        WCS solution for the image.
    gaia_table : astropy.table.Table
        Table of Gaia stars with 'RAJ2000', 'DEJ2000', and 'Plx'.
    max_separation : float, optional
        Maximum separation for matching in arcseconds (default is 25.0).

    Returns
    -------
    list of (float, float, float)
        List of tuples (x, y, distance_ly) for matched stars.
    """
    # Convert detected positions to sky coordinates
    x_coords = detected_sources['xcentroid']
    y_coords = detected_sources['ycentroid']
    ra, dec = wcs.pixel_to_world_values(x_coords, y_coords)
    sky_coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')

    # Gaia coordinates
    gaia_coords = SkyCoord(
        ra=gaia_table['RAJ2000'],
        dec=gaia_table['DEJ2000'],
        unit='deg',
        frame='icrs',
    )

    idx, d2d, _ = sky_coords.match_to_catalog_sky(gaia_coords)
    matched_indices = np.where(d2d < max_separation * u.arcsec)[0]

    if len(matched_indices) == 0:
        logger.warning("No matches found within the separation limit")
        return []

    logger.info(f"Found {len(matched_indices)} matches within {max_separation} arcseconds")
    results: List[Tuple[float, float, float]] = []
    for i in matched_indices:
        gaia_idx = idx[i]
        parallax_mas = gaia_table['Plx'][gaia_idx]

        distance_pc = 1000.0 / parallax_mas
        distance_ly = distance_pc * 3.26156

        results.append((float(x_coords[i]), float(y_coords[i]), float(distance_ly)))

    logger.info(f"Successfully matched {len(results)} stars with valid distances")
    return results


def process_fits_file(
    fits_path: str,
    astap_path: Optional[str] = None,
) -> Optional[List[Tuple[float, float, float]]]:
    """Process a FITS file to detect stars, plate solve, query Gaia, and match distances.

    Parameters
    ----------
    fits_path : str
        Path to the FITS file.
    astap_path : str, optional
        Path to the ASTAP executable. If None, attempts to find it.

    Returns
    -------
    list of (float, float, float) or None
        List of (x, y, distance_ly) tuples if successful, None otherwise.
    """
    try:
        # Open FITS and extract image_data and header
        logger.info(f"Opening FITS file: {fits_path}")
        with fits.open(fits_path) as hdul:
            image_data = hdul[0].data
            header = hdul[0].header

            # NOTE: Redundant image_data dimensionality handling also in detect_stars_old and generate_distance_visualization
            if image_data.ndim > 2:
                logger.info(f"Image has shape {image_data.shape}, converting to 2D")
                image_data = (
                    np.mean(image_data, axis=0)
                    if image_data.ndim == 3
                    else image_data[0]
                )

            # Validate or solve WCS
            try:
                wcs = WCS(header, naxis=2)
                _ = wcs.pixel_to_world(0, 0)
                logger.info("Valid WCS information found in the FITS file")
                has_wcs = True
            except Exception as e:
                logger.info(f"No valid WCS in header: {e}, will attempt plate solving")
                has_wcs = False

        if not has_wcs:
            wcs = plate_solve_with_astap(fits_path, astap_path)
            if wcs is None:
                logger.error("Plate solving failed, cannot proceed")
                return None

        # Detect stars
        with fits.open(fits_path) as hdul:
            image_data = hdul[0].data
            if image_data.ndim > 2:
                image_data = np.mean(image_data, axis=0) if image_data.ndim == 3 else image_data[0]

            sources = detect_stars(image_data)
            if not sources:
                logger.error("No stars detected in the image")
                return None

        # Determine center coordinate for Gaia query
        center_x = image_data.shape[1] / 2
        center_y = image_data.shape[0] / 2
        center_coord = wcs.pixel_to_world(center_x, center_y)
        if not isinstance(center_coord, SkyCoord):
            center_coord = SkyCoord(ra=center_coord[0], dec=center_coord[1], frame="icrs")

        # Compute search radius
        try:
            ra1, dec1 = wcs.pixel_to_world_values(0, 0)
            ra2, dec2 = wcs.pixel_to_world_values(
                image_data.shape[1] - 1,
                image_data.shape[0] - 1,
            )
            corner1 = SkyCoord(ra=ra1 * u.deg, dec=dec1 * u.deg, frame='icrs')
            corner2 = SkyCoord(ra=ra2 * u.deg, dec=dec2 * u.deg, frame='icrs')
            radius = corner1.separation(corner2).degree / 2
        except Exception as e:
            logger.warning(f"Error calculating field of view: {e}, using default radius")
            radius = 0.5

        logger.info(f"Using search radius of {radius:.4f} degrees")

        gaia_table = query_gaia_for_region(center_coord, radius)
        if gaia_table is None:
            return None

        results = match_stars(sources, wcs, gaia_table)
        if not results:
            logger.warning("No valid star matches found")
            return None

        return results
    except Exception as e:
        logger.error(f"Error processing FITS file: {e}")
        return None


def save_to_csv(
    results: List[Tuple[float, float, float]],
    output_path: str,
) -> pd.DataFrame:
    """Save matched star distances to a CSV file.

    Parameters
    ----------
    results : list of (float, float, float)
        List of (x, y, distance_ly) tuples.
    output_path : str
        Path to save the CSV file.

    Returns
    -------
    pandas.DataFrame
        DataFrame of results saved.
    """
    df = pd.DataFrame(results, columns=['image_x', 'image_y', 'distance_ly'])
    df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")
    return df


def generate_distance_visualization(
    fits_path: str,
    results_df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> None:
    """Generate a visualization of star distances overlaid on the original image.

    Parameters
    ----------
    fits_path : str
        Path to the FITS file.
    results_df : pandas.DataFrame
        DataFrame with 'image_x', 'image_y', and 'distance_ly' columns.
    output_path : str, optional
        Path to save the visualization image. If None, displays the plot.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        logger.info("Generating distance visualization")

        with fits.open(fits_path) as hdul:
            image_data = hdul[0].data
            # NOTE: Redundant image_data dimensionality handling also in process_fits_file and detect_stars_old
            if image_data.ndim > 2:
                image_data = np.mean(image_data, axis=0) if image_data.ndim == 3 else image_data[0]

        plt.figure(figsize=(12, 10))
        vmin = np.percentile(image_data, 5)
        vmax = np.percentile(image_data, 99)
        plt.imshow(image_data, cmap='gray', norm=LogNorm(vmin=max(0.1, vmin), vmax=vmax))

        scatter = plt.scatter(
            results_df['image_x'],
            results_df['image_y'],
            c=results_df['distance_ly'],
            cmap='viridis',
            s=50,
            alpha=0.5,
            edgecolor='white',
            norm=LogNorm(),
        )

        cbar = plt.colorbar(scatter)
        cbar.set_label('Distance (light years)', fontsize=12)

        plt.title(f'Star Distances from Gaia DR3 - {os.path.basename(fits_path)}', fontsize=14)
        plt.xlabel('X (pixels)', fontsize=12)
        plt.ylabel('Y (pixels)', fontsize=12)

        if output_path:
            plt.tight_layout()
            plt.savefig(output_path, dpi=150)
            logger.info(f"Visualization saved to {output_path}")
        else:
            plt.tight_layout()
            plt.show()

    except Exception as e:
        logger.error(f"Error generating visualization: {e}")


def main() -> None:
    """Command-line interface for processing FITS files and visualizing star distances."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Process a FITS file to extract star distances from Gaia data'
    )
    parser.add_argument('fits_file', help='Path to the FITS file')
    parser.add_argument('--astap', help='Path to ASTAP executable')
    parser.add_argument('--output', '-o', help='Output CSV file path', default='star_distances.csv')
    parser.add_argument('--visualize', '-v', action='store_true', help='Generate a visualization of the results')
    parser.add_argument('--viz-output', help='Path to save the visualization (requires --visualize)')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    results = process_fits_file(args.fits_file, args.astap)

    if results:
        results_df = save_to_csv(results, args.output)
        logger.info(f"Found distances for {len(results)} stars")

        if args.visualize:
            generate_distance_visualization(args.fits_file, results_df, args.viz_output)
    else:
        logger.error("Could not extract star distances from the FITS file")


if __name__ == "__main__":
    main()
