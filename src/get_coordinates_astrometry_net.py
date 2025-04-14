#!/usr/bin/env python3
"""
FITS to Gaia Star Distances with Astrometry.net Output Files

This script processes output files from Astrometry.net web service,
detects stars, queries the Gaia catalog for matching stars,
and outputs a CSV file with image coordinates and distances.

Requirements:
- astropy
- astroquery
- photutils
- numpy
- pandas
- matplotlib (for visualization)
"""

import os
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table
from astroquery.vizier import Vizier
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('fits_to_gaia')


def load_astrometry_files(wcs_path, image_radec_path=None, axy_path=None):
    """
    Load the relevant files from Astrometry.net output

    Parameters:
    -----------
    wcs_path : str
        Path to the WCS file (wcs.fits)
    image_radec_path : str, optional
        Path to image-radec.fits (stars detected with RA/Dec)
    axy_path : str, optional
        Path to axy.fits (stars detected with x/y)

    Returns:
    --------
    tuple
        (wcs, detected_stars_xy, detected_stars_radec)
    """
    logger.info(f"Loading Astrometry.net output files")

    # Load WCS file
    try:
        with fits.open(wcs_path) as hdul:
            header = hdul[0].header
            wcs = WCS(header)
            logger.info("Successfully loaded WCS from Astrometry.net solution")
    except Exception as e:
        logger.error(f"Failed to load WCS from {wcs_path}: {e}")
        return None, None, None

    # Load detected stars with x,y coordinates if available
    detected_stars_xy = None
    if axy_path and os.path.exists(axy_path):
        try:
            with fits.open(axy_path) as hdul:
                # The format is typically a binary table with X and Y columns
                data = Table(hdul[1].data)
                logger.info(f"Loaded {len(data)} detected stars with X,Y coordinates from {axy_path}")
                detected_stars_xy = data
        except Exception as e:
            logger.warning(f"Failed to load detected stars from {axy_path}: {e}")

    # Load detected stars with RA,Dec coordinates if available
    detected_stars_radec = None
    if image_radec_path and os.path.exists(image_radec_path):
        try:
            with fits.open(image_radec_path) as hdul:
                # The format is typically a binary table with RA and DEC columns
                data = Table(hdul[1].data)
                logger.info(f"Loaded {len(data)} detected stars with RA,DEC coordinates from {image_radec_path}")
                detected_stars_radec = data
        except Exception as e:
            logger.warning(f"Failed to load detected stars from {image_radec_path}: {e}")

    return wcs, detected_stars_xy, detected_stars_radec


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


def match_stars_with_radec(detected_stars_radec, gaia_table, max_separation=5.0):
    """
    Match detected stars with Gaia catalog entries using RA/Dec coordinates

    Parameters:
    -----------
    detected_stars_radec : astropy.table.Table
        Table of detected sources with RA/Dec coordinates
    gaia_table : astropy.table.Table
        Table of Gaia stars
    max_separation : float, optional
        Maximum separation for matching in arcseconds, default is 5.0

    Returns:
    --------
    dict
        Dictionary with original indices as keys and (ra, dec, distance_ly) as values
    """
    # Check if the table includes RA/Dec columns
    ra_col = None
    dec_col = None

    # Check common column names for RA/Dec in astrometry.net output
    for ra_name in ['RA', 'ra', 'ALPHA_J2000', 'alpha']:
        if ra_name in detected_stars_radec.colnames:
            ra_col = ra_name
            break

    for dec_name in ['DEC', 'dec', 'DELTA_J2000', 'delta']:
        if dec_name in detected_stars_radec.colnames:
            dec_col = dec_name
            break

    if ra_col is None or dec_col is None:
        logger.error(
            f"Could not find RA/Dec columns in the detected stars table. Available columns: {detected_stars_radec.colnames}")
        return {}

    # Create SkyCoord for detected stars
    detected_coords = SkyCoord(ra=detected_stars_radec[ra_col], dec=detected_stars_radec[dec_col], unit='deg',
                               frame='icrs')

    # Create SkyCoord for Gaia stars
    gaia_coords = SkyCoord(ra=gaia_table['RAJ2000'], dec=gaia_table['DEJ2000'], unit='deg', frame='icrs')

    # Match the catalogs
    idx, d2d, _ = detected_coords.match_to_catalog_sky(gaia_coords)

    # Filter to only include matches within the separation limit
    matched_indices = np.where(d2d < max_separation * u.arcsec)[0]

    if len(matched_indices) == 0:
        logger.warning("No matches found within the separation limit")
        return {}

    logger.info(f"Found {len(matched_indices)} matches within {max_separation} arcseconds")

    # Create a dictionary with results
    results = {}
    for i in matched_indices:
        gaia_idx = idx[i]
        parallax_mas = gaia_table['Plx'][gaia_idx]  # Parallax in milliarcseconds

        # Convert parallax to distance in light years
        # 1 parsec = 3.26156 light years, parallax in mas = 1000/distance in parsecs
        distance_pc = 1000.0 / parallax_mas
        distance_ly = distance_pc * 3.26156

        # Add to results - store original index, ra, dec, and distance
        results[i] = (detected_stars_radec[ra_col][i], detected_stars_radec[dec_col][i], distance_ly)

    logger.info(f"Successfully matched {len(results)} stars with valid distances")
    return results


def match_stars_with_xy(detected_stars_xy, wcs, gaia_table, max_separation=5.0):
    """
    Match detected stars (X,Y) with Gaia catalog entries

    Parameters:
    -----------
    detected_stars_xy : astropy.table.Table
        Table of detected sources with X,Y coordinates
    wcs : astropy.wcs.WCS
        WCS solution for the image
    gaia_table : astropy.table.Table
        Table of Gaia stars
    max_separation : float, optional
        Maximum separation for matching in arcseconds, default is 5.0

    Returns:
    --------
    dict
        Dictionary with original indices as keys and (x, y, distance_ly) as values
    """
    # Check if the table includes X/Y columns
    x_col = None
    y_col = None

    # Check common column names for X/Y in astrometry.net output
    for x_name in ['X', 'x', 'XIMAGE', 'ximage']:
        if x_name in detected_stars_xy.colnames:
            x_col = x_name
            break

    for y_name in ['Y', 'y', 'YIMAGE', 'yimage']:
        if y_name in detected_stars_xy.colnames:
            y_col = y_name
            break

    if x_col is None or y_col is None:
        logger.error(
            f"Could not find X/Y columns in the detected stars table. Available columns: {detected_stars_xy.colnames}")
        return {}

    # Get pixel coordinates of detected stars
    x_coords = detected_stars_xy[x_col]
    y_coords = detected_stars_xy[y_col]

    # Convert to RA/Dec using pixel_to_world_values to obtain numeric arrays
    ra, dec = wcs.pixel_to_world_values(x_coords, y_coords)

    # Create a SkyCoord object from the numeric RA/Dec values
    sky_coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')

    # Create SkyCoord objects for Gaia stars
    gaia_coords = SkyCoord(ra=gaia_table['RAJ2000'], dec=gaia_table['DEJ2000'], unit='deg', frame='icrs')

    # Match the catalogs
    idx, d2d, _ = sky_coords.match_to_catalog_sky(gaia_coords)

    # Filter to only include matches within the separation limit
    matched_indices = np.where(d2d < max_separation * u.arcsec)[0]

    if len(matched_indices) == 0:
        logger.warning("No matches found within the separation limit")
        return {}

    logger.info(f"Found {len(matched_indices)} matches within {max_separation} arcseconds")

    # Create a dictionary with results
    results = {}
    for i in matched_indices:
        gaia_idx = idx[i]
        parallax_mas = gaia_table['Plx'][gaia_idx]  # Parallax in milliarcseconds

        # Convert parallax to distance in light years
        # 1 parsec = 3.26156 light years, parallax in mas = 1000/distance in parsecs
        distance_pc = 1000.0 / parallax_mas
        distance_ly = distance_pc * 3.26156

        # Add to results - store original index, x, y, and distance
        results[i] = (x_coords[i], y_coords[i], distance_ly)

    logger.info(f"Successfully matched {len(results)} stars with valid distances")
    return results


def process_astrometry_files(wcs_path, fits_path=None, image_radec_path=None, axy_path=None):
    """
    Process Astrometry.net output files to match stars with Gaia data

    Parameters:
    -----------
    wcs_path : str
        Path to the WCS file (wcs.fits)
    fits_path : str, optional
        Path to the original or new FITS image (for visualization)
    image_radec_path : str, optional
        Path to image-radec.fits (stars detected with RA/Dec)
    axy_path : str, optional
        Path to axy.fits (stars detected with x/y)

    Returns:
    --------
    list of tuple or None
        List of (x, y, distance_ly) tuples if successful, None otherwise
    """
    try:
        # Load Astrometry.net output files
        wcs, detected_stars_xy, detected_stars_radec = load_astrometry_files(
            wcs_path, image_radec_path, axy_path
        )

        if wcs is None:
            logger.error("Could not load WCS information, cannot proceed")
            return None

        # Define image dimensions for calculating FoV
        # If we have the FITS file, get dimensions from it
        image_width, image_height = 0, 0
        if fits_path and os.path.exists(fits_path):
            try:
                with fits.open(fits_path) as hdul:
                    image_data = hdul[0].data
                    if len(image_data.shape) > 2:
                        image_data = np.mean(image_data, axis=0) if len(image_data.shape) == 3 else image_data[0]
                    image_height, image_width = image_data.shape
            except Exception as e:
                logger.warning(f"Could not get image dimensions from FITS file: {e}")

        # If we couldn't get dimensions from FITS file, try to get them from WCS or use defaults
        if image_width == 0 or image_height == 0:
            try:
                # Try to get dimensions from WCS header
                if 'NAXIS1' in wcs.to_header() and 'NAXIS2' in wcs.to_header():
                    image_width = wcs.to_header()['NAXIS1']
                    image_height = wcs.to_header()['NAXIS2']
                else:
                    # Use default dimensions if we can't get them from anywhere
                    logger.warning("Could not determine image dimensions, using defaults")
                    image_width = 1000
                    image_height = 1000
            except Exception as e:
                logger.warning(f"Could not get image dimensions from WCS: {e}, using defaults")
                image_width = 1000
                image_height = 1000

        logger.info(f"Using image dimensions: {image_width} x {image_height}")

        # Calculate the center coordinates
        center_x = image_width / 2
        center_y = image_height / 2

        # Convert center pixel to RA/Dec
        ra_center, dec_center = wcs.pixel_to_world_values(center_x, center_y)
        center_coord = SkyCoord(ra=ra_center * u.deg, dec=dec_center * u.deg, frame='icrs')

        # Calculate field of view for GAIA query
        try:
            # Calculate the diagonal field of view in degrees
            ra1, dec1 = wcs.pixel_to_world_values(0, 0)
            ra2, dec2 = wcs.pixel_to_world_values(image_width - 1, image_height - 1)

            corner1 = SkyCoord(ra=ra1 * u.deg, dec=dec1 * u.deg, frame='icrs')
            corner2 = SkyCoord(ra=ra2 * u.deg, dec=dec2 * u.deg, frame='icrs')

            radius = corner1.separation(corner2).degree / 2

            # Add some margin and cap the radius
            radius = min(radius * 1.2, 1.0)
        except Exception as e:
            logger.warning(f"Error calculating field of view: {e}, using default radius")
            radius = 0.5  # degrees

        logger.info(f"Using search radius of {radius:.4f} degrees")

        # Query Gaia for the region
        gaia_table = query_gaia_for_region(center_coord, radius)
        if gaia_table is None:
            logger.error("Could not query Gaia catalog")
            return None

        # Process results based on available files
        results = []

        # If we have X,Y detected stars, match them with Gaia
        if detected_stars_xy is not None:
            xy_matches = match_stars_with_xy(detected_stars_xy, wcs, gaia_table)
            if xy_matches:
                for _, (x, y, distance) in xy_matches.items():
                    results.append((x, y, distance))
                logger.info(f"Added {len(xy_matches)} matches from X,Y coordinates")

        # If we have RA,Dec detected stars and no X,Y matches, use those
        elif detected_stars_radec is not None:
            radec_matches = match_stars_with_radec(detected_stars_radec, gaia_table)
            if radec_matches:
                for _, (ra, dec, distance) in radec_matches.items():
                    # Convert RA,Dec back to pixel coordinates
                    x, y = wcs.world_to_pixel_values(ra, dec)
                    results.append((x, y, distance))
                logger.info(f"Added {len(radec_matches)} matches from RA,Dec coordinates")

        # If we have results, return them
        if results:
            return results
        else:
            logger.warning("No star matches found")
            return None

    except Exception as e:
        logger.error(f"Error processing Astrometry.net files: {e}")
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

    parser = argparse.ArgumentParser(
        description='Process Astrometry.net output files to extract star distances from Gaia data')
    parser.add_argument('--wcs', required=True, help='Path to the WCS file (wcs.fits) from Astrometry.net')
    parser.add_argument('--fits', help='Path to the original or new FITS image (for visualization)')
    parser.add_argument('--image-radec', help='Path to image-radec.fits file (detected stars with RA/Dec)')
    parser.add_argument('--axy', help='Path to axy.fits file (detected stars with X/Y)')
    parser.add_argument('--output', '-o', help='Output CSV file path', default='star_distances.csv')
    parser.add_argument('--visualize', '-v', action='store_true', help='Generate a visualization of the results')
    parser.add_argument('--viz-output', help='Path to save the visualization (requires --visualize)')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    # Set debug level if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Process the files
    results = process_astrometry_files(
        args.wcs,
        args.fits,
        args.image_radec,
        args.axy
    )

    if results:
        results_df = save_to_csv(results, args.output)
        logger.info(f"Found distances for {len(results)} stars")

        # Generate visualization if requested
        if args.visualize and args.fits:
            generate_distance_visualization(args.fits, results_df, args.viz_output)
        elif args.visualize and not args.fits:
            logger.warning("Cannot generate visualization without a FITS image file")
    else:
        logger.error("Could not extract star distances from the Astrometry.net files")


if __name__ == "__main__":
    main()