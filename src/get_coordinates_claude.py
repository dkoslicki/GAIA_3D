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
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from astroquery.vizier import Vizier
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('fits_to_gaia')


def detect_stars(image_data):
    """
    Detect stars in the image using DAOStarFinder

    Parameters:
    -----------
    image_data : numpy.ndarray
        The image data array

    Returns:
    --------
    photutils.detection.DAOStarFinder.starfind
        Table of detected sources
    """
    # Ensure we're working with a 2D array
    if len(image_data.shape) > 2:
        # If RGB, convert to grayscale by averaging channels or taking one channel
        logger.info("Converting multi-dimensional image data to 2D")
        image_data = np.mean(image_data, axis=0) if len(image_data.shape) == 3 else image_data[0]

    # Compute image statistics with sigma clipping to ignore outliers
    mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
    logger.info(f"Image statistics - Mean: {mean:.2f}, Median: {median:.2f}, Std: {std:.2f}")

    # Configure star finder
    # The FWHM (full width at half maximum) parameter should be adjusted based on your seeing conditions
    # The threshold parameter determines how many sigma above background a peak must be
    daofind = DAOStarFinder(fwhm=2.0, threshold=5. * std)

    # Find stars
    sources = daofind(image_data - median)

    if sources is None or len(sources) == 0:
        logger.warning("No stars detected in the image")
        return None

    logger.info(f"Detected {len(sources)} stars in the image")
    return sources


def find_astap_executable():
    """
    Find the ASTAP executable on the system

    Returns:
    --------
    str or None
        Path to ASTAP executable if found, None otherwise
    """
    # Common paths where ASTAP might be installed
    common_paths = [
        # Windows paths
        "C:\\Program Files\\ASTAP\\astap.exe",
        "C:\\Program Files (x86)\\ASTAP\\astap.exe",
        os.path.expanduser("~\\AppData\\Local\\Programs\\ASTAP\\astap.exe"),
        # Custom path - you can modify this to match your installation
        os.path.expanduser("~\\ASTAP\\astap.exe"),
        # Path if it's in the system PATH
        "astap.exe",
        # Linux/macOS paths
        "/usr/bin/astap",
        "/usr/local/bin/astap",
        os.path.expanduser("~/ASTAP/astap")
    ]

    for path in common_paths:
        if os.path.isfile(path):
            logger.info(f"Found ASTAP at: {path}")
            return path

    logger.warning("ASTAP executable not found in common locations")
    return None


def plate_solve_with_astap(fits_path, astap_path=None):
    """
    Perform plate solving on a FITS file using ASTAP

    Parameters:
    -----------
    fits_path : str
        Path to the FITS file
    astap_path : str, optional
        Path to the ASTAP executable, default is None (will search for it)

    Returns:
    --------
    astropy.wcs.WCS or None
        WCS solution if successful, None otherwise
    """
    logger.info(f"Attempting to plate solve with ASTAP: {fits_path}")

    # Find ASTAP executable if not provided
    if astap_path is None:
        astap_path = find_astap_executable()
        if astap_path is None:
            logger.error("ASTAP executable not found. Please specify the path to astap.exe.")
            return None

    # Create a temporary directory for output files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Get basename without extension
        base_name = os.path.splitext(os.path.basename(fits_path))[0]

        # Define paths for output files
        wcs_fits_path = os.path.join(temp_dir, f"{base_name}_solved.fits")

        # Build ASTAP command
        # -f: Force overwrite of existing files
        # -r 360: Search radius 360 degrees (all sky)
        # -speed 0: Best detection, slower
        command = [
            astap_path,
            "-f",
            "-r", "360",
            "-speed", "0",
            "-solve", fits_path,
            "-o", wcs_fits_path
        ]

        logger.info(f"Running ASTAP command: {' '.join(command)}")

        try:
            # Run ASTAP
            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=300  # 5 minutes timeout
            )

            # Check if the process ran successfully
            if process.returncode != 0:
                logger.error(f"ASTAP failed with return code {process.returncode}")
                logger.error(f"ASTAP stderr: {process.stderr}")
                return None

            # Check if the output file exists
            if not os.path.isfile(wcs_fits_path):
                logger.error(f"ASTAP did not create the output file: {wcs_fits_path}")
                return None

            # Open the solved FITS file and extract WCS
            with fits.open(wcs_fits_path) as hdul:
                header = hdul[0].header
                try:
                    wcs = WCS(header)
                    # Test the WCS to see if it's valid
                    _ = wcs.pixel_to_world(0, 0)
                    logger.info("Successfully extracted WCS from ASTAP solution")

                    # Create a copy of the original FITS file with the WCS header
                    with fits.open(fits_path, mode='update') as original_hdul:
                        # Copy WCS keywords
                        wcs_keywords = ['CTYPE1', 'CTYPE2', 'CRVAL1', 'CRVAL2',
                                        'CRPIX1', 'CRPIX2', 'CD1_1', 'CD1_2',
                                        'CD2_1', 'CD2_2', 'CDELT1', 'CDELT2',
                                        'CROTA1', 'CROTA2', 'PC1_1', 'PC1_2',
                                        'PC2_1', 'PC2_2', 'RADESYS', 'EQUINOX']

                        for keyword in wcs_keywords:
                            if keyword in header:
                                original_hdul[0].header[keyword] = header[keyword]

                        # Save the updated file
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


def query_gaia_for_region(center_coord, radius=0.5):
    """
    Query Gaia DR3 for stars in a specific region

    Parameters:
    -----------
    center_coord : astropy.coordinates.SkyCoord
        Center coordinates of the region
    radius : float, optional
        Search radius in degrees, default is 0.5

    Returns:
    --------
    astropy.table.Table or None
        Table of Gaia stars if successful, None otherwise
    """
    logger.info(
        f"Querying Gaia DR3 for region centered at RA={center_coord.ra.degree:.6f}, Dec={center_coord.dec.degree:.6f}, radius={radius} degrees")

    # Set up Vizier for Gaia DR3 queries
    Vizier.ROW_LIMIT = -1  # No row limit

    # Gaia DR3 catalog identifier
    gaia_catalog = "I/355/gaiadr3"

    try:
        # Query the catalog
        gaia_results = Vizier.query_region(
            center_coord,
            radius=radius * u.degree,
            catalog=gaia_catalog,
            column_filters={"Plx": ">0"}  # Only stars with positive parallax
        )

        if not gaia_results or len(gaia_results) == 0:
            logger.warning("No Gaia matches found in the field of view")
            return None

        # Get the Gaia table from the results
        gaia_table = gaia_results[0]
        logger.info(f"Found {len(gaia_table)} Gaia stars in the field of view")
        return gaia_table

    except Exception as e:
        logger.error(f"Error querying Gaia: {e}")
        return None


def match_stars(detected_sources, wcs, gaia_table, max_separation=15.0):
    """
    Match detected stars with Gaia catalog entries

    Parameters:
    -----------
    detected_sources : astropy.table.Table
        Table of detected sources
    wcs : astropy.wcs.WCS
        WCS solution for the image
    gaia_table : astropy.table.Table
        Table of Gaia stars
    max_separation : float, optional
        Maximum separation for matching in arcseconds, default is 5.0

    Returns:
    --------
    list of tuple
        List of (x, y, distance_ly) tuples for matched stars
    """
    # Get pixel coordinates of detected stars
    x_coords = detected_sources['xcentroid']
    y_coords = detected_sources['ycentroid']

    # Convert to RA/Dec using pixel_to_world_values to obtain numeric arrays
    ra, dec = wcs.pixel_to_world_values(x_coords, y_coords)

    # Create a SkyCoord object from the numeric RA/Dec values
    sky_coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')

    # Create SkyCoord objects for Gaia stars (ensure units are properly set)
    gaia_coords = SkyCoord(ra=gaia_table['RAJ2000'], dec=gaia_table['DEJ2000'], unit='deg', frame='icrs')

    # Now you can match the catalogs
    idx, d2d, _ = sky_coords.match_to_catalog_sky(gaia_coords)

    # Filter to only include matches within the separation limit
    matched_indices = np.where(d2d < max_separation * u.arcsec)[0]

    if len(matched_indices) == 0:
        logger.warning("No matches found within the separation limit")
        return []

    logger.info(f"Found {len(matched_indices)} matches within {max_separation} arcseconds")

    # Create a list of tuples with (x, y, distance)
    results = []
    for i in matched_indices:
        gaia_idx = idx[i]
        parallax_mas = gaia_table['Plx'][gaia_idx]  # Parallax in milliarcseconds

        # Skip stars with negative, zero, or very small parallax
        #if parallax_mas <= 0.1:  # Added threshold to avoid unrealistic distances
        #    continue

        # Convert parallax to distance in light years
        # 1 parsec = 3.26156 light years, parallax in mas = 1000/distance in parsecs
        distance_pc = 1000.0 / parallax_mas
        distance_ly = distance_pc * 3.26156

        # Skip unrealistically large distances (e.g., over 100,000 light years)
        #if distance_ly > 100000:
        #    continue

        # Add to results
        results.append((x_coords[i], y_coords[i], distance_ly))

    logger.info(f"Successfully matched {len(results)} stars with valid distances")
    return results


def process_fits_file(fits_path, astap_path=None):
    """
    Process a FITS file to extract star coordinates and match with Gaia data

    Parameters:
    -----------
    fits_path : str
        Path to the FITS file
    astap_path : str, optional
        Path to the ASTAP executable, default is None (will search for it)

    Returns:
    --------
    list of tuple or None
        List of (x, y, distance_ly) tuples if successful, None otherwise
    """
    try:
        # Open the FITS file
        logger.info(f"Opening FITS file: {fits_path}")
        with fits.open(fits_path) as hdul:
            # Get the image data (assuming it's in the primary HDU)
            image_data = hdul[0].data
            header = hdul[0].header

            # Handle different image formats
            if len(image_data.shape) > 2:
                logger.info(f"Image has shape {image_data.shape}, converting to 2D")
                image_data = np.mean(image_data, axis=0) if len(image_data.shape) == 3 else image_data[0]

            # Try to get WCS from the header first
            try:
                wcs = WCS(header)
                # Check if the WCS is valid by converting a test point
                test_coord = wcs.pixel_to_world(0, 0)
                test_coord = SkyCoord(ra=test_coord[0], dec=test_coord[1], frame="icrs")
                # This will raise an exception if the conversion fails
                _ = test_coord.ra.degree
                _ = test_coord.dec.degree
                logger.info("Valid WCS information found in the FITS file")
                has_wcs = True
            except Exception as e:
                logger.info(f"No valid WCS information in the header: {e}, will attempt plate solving")
                has_wcs = False

        # If no WCS in header, perform plate solving
        if not has_wcs:
            wcs = plate_solve_with_astap(fits_path, astap_path)
            if wcs is None:
                logger.error("Plate solving failed, cannot proceed")
                return None

        # Detect stars in the image
        with fits.open(fits_path) as hdul:
            image_data = hdul[0].data
            if len(image_data.shape) > 2:
                image_data = np.mean(image_data, axis=0) if len(image_data.shape) == 3 else image_data[0]

            sources = detect_stars(image_data)

            if sources is None or len(sources) == 0:
                logger.error("No stars detected in the image")
                return None

        # Get the center coordinates of the image for the Gaia query
        center_x = image_data.shape[1] / 2
        center_y = image_data.shape[0] / 2
        center_coord = wcs.pixel_to_world(center_x, center_y)
        center_coord = SkyCoord(ra=center_coord[0], dec=center_coord[1], frame="icrs")

        # Ensure center_coord is a proper SkyCoord object
        if not isinstance(center_coord, SkyCoord):
            logger.error(f"WCS pixel_to_world returned unexpected type: {type(center_coord)}")
            # Try to create a SkyCoord from the returned values
            try:
                # This handles cases where pixel_to_world returns a tuple of coordinates
                if isinstance(center_coord, tuple):
                    center_coord = SkyCoord(center_coord[0], center_coord[1])
                else:
                    # If we can't fix it, we can't proceed
                    logger.error("Cannot convert center coordinates to SkyCoord")
                    return None
            except Exception as e:
                logger.error(f"Error creating SkyCoord: {e}")
                return None

        # Log the center coordinates for debugging
        logger.debug(f"Image center coordinates: RA={center_coord.ra.degree:.6f}, Dec={center_coord.dec.degree:.6f}")

        # Estimate the field of view
        try:
            # Try to calculate the diagonal field of view in degrees
            #corner1 = wcs.pixel_to_world(0, 0)
            #corner2 = wcs.pixel_to_world(image_data.shape[1] - 1, image_data.shape[0] - 1)
            #radius = corner1.separation(corner2).degree / 2
            # Get the numeric RA and Dec values using pixel_to_world_values
            ra1, dec1 = wcs.pixel_to_world_values(0, 0)
            ra2, dec2 = wcs.pixel_to_world_values(image_data.shape[1] - 1, image_data.shape[0] - 1)

            # Create SkyCoord objects from these values
            corner1 = SkyCoord(ra=ra1 * u.deg, dec=dec1 * u.deg, frame='icrs')
            corner2 = SkyCoord(ra=ra2 * u.deg, dec=dec2 * u.deg, frame='icrs')

            # Now you can compute the separation
            #radius = corner1.separation(corner2).degree / 2
            radius = corner1.separation(corner2).degree / 2

            # Add some margin
            #radius = min(radius * 1.2, 1.0)  # Cap at 1 degree to avoid excessive query size
        except Exception as e:
            # Default radius
            logger.warning(f"Error calculating field of view: {e}, using default radius")
            radius = 0.5  # degrees

        logger.info(f"Using search radius of {radius:.4f} degrees")

        # Query Gaia for the region
        gaia_table = query_gaia_for_region(center_coord, radius)
        if gaia_table is None:
            return None

        # Match detected stars with Gaia stars
        results = match_stars(sources, wcs, gaia_table)

        if not results:
            logger.warning("No valid star matches found")
            return None

        return results

    except Exception as e:
        logger.error(f"Error processing FITS file: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def save_to_csv(results, output_path):
    """
    Save the results to a CSV file

    Parameters:
    -----------
    results : list of tuple
        List of (x, y, distance_ly) tuples
    output_path : str
        Path to save the CSV file

    Returns:
    --------
    pandas.DataFrame
        DataFrame of results
    """
    df = pd.DataFrame(results, columns=['image_x', 'image_y', 'distance_ly'])
    df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")
    return df


def generate_distance_visualization(fits_path, results_df, output_path=None):
    """
    Generate a visualization of the distance data overlaid on the original image

    Parameters:
    -----------
    fits_path : str
        Path to the FITS file
    results_df : pandas.DataFrame
        DataFrame with image_x, image_y, and distance_ly columns
    output_path : str, optional
        Path to save the visualization, default is None (show instead of save)
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        logger.info("Generating distance visualization")

        # Open the FITS file
        with fits.open(fits_path) as hdul:
            image_data = hdul[0].data
            if len(image_data.shape) > 2:
                image_data = np.mean(image_data, axis=0) if len(image_data.shape) == 3 else image_data[0]

        # Create the figure
        plt.figure(figsize=(12, 10))

        # Display the image with log scaling
        vmin = np.percentile(image_data, 5)
        vmax = np.percentile(image_data, 99)
        plt.imshow(image_data, cmap='gray', norm=LogNorm(vmin=max(0.1, vmin), vmax=vmax))

        # Overlay the stars, color-coded by distance
        scatter = plt.scatter(
            results_df['image_x'],
            results_df['image_y'],
            c=results_df['distance_ly'],
            cmap='viridis',
            s=50,
            alpha=0.5,
            edgecolor='white',
            norm=LogNorm()
        )

        # Add a colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Distance (light years)', fontsize=12)

        plt.title(f'Star Distances from Gaia DR3 - {os.path.basename(fits_path)}', fontsize=14)
        plt.xlabel('X (pixels)', fontsize=12)
        plt.ylabel('Y (pixels)', fontsize=12)

        # Save or show the figure
        if output_path:
            plt.tight_layout()
            plt.savefig(output_path, dpi=150)
            logger.info(f"Visualization saved to {output_path}")
        else:
            plt.tight_layout()
            plt.show()

    except Exception as e:
        logger.error(f"Error generating visualization: {e}")
        import traceback
        logger.error(traceback.format_exc())


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Process a FITS file to extract star distances from Gaia data')
    parser.add_argument('fits_file', help='Path to the FITS file')
    parser.add_argument('--astap', help='Path to ASTAP executable')
    parser.add_argument('--output', '-o', help='Output CSV file path', default='star_distances.csv')
    parser.add_argument('--visualize', '-v', action='store_true', help='Generate a visualization of the results')
    parser.add_argument('--viz-output', help='Path to save the visualization (requires --visualize)')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    # Set debug level if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Process the file
    results = process_fits_file(args.fits_file, args.astap)

    if results:
        results_df = save_to_csv(results, args.output)
        logger.info(f"Found distances for {len(results)} stars")

        # Generate visualization if requested
        if args.visualize:
            generate_distance_visualization(args.fits_file, results_df, args.viz_output)
    else:
        logger.error("Could not extract star distances from the FITS file")


if __name__ == "__main__":
    main()