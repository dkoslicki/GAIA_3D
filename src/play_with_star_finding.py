#!/usr/bin/env python3
"""
Enhanced Star Detection and Visualization

This script extends the functionality of the original FITS processing script
by improving detection of bright, bloated stars and providing a simple
visualization of detected stars overlaid on the original image.
"""

import os
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder, IRAFStarFinder
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('star_detection')


def enhanced_star_detection(image_data, fwhm=3.0, threshold_sigma=3.0,
                            use_iraf=True, sharplo=0.2, sharphi=1.0,
                            roundlo=-1.0, roundhi=1.0):
    """
    Enhanced star detection to better capture bright, bloated stars

    Parameters:
    -----------
    image_data : numpy.ndarray
        The image data array
    fwhm : float, optional
        Full width at half maximum of the PSF, default is 3.0
    threshold_sigma : float, optional
        Detection threshold in sigma above background, default is 3.0
    use_iraf : bool, optional
        Whether to use IRAFStarFinder instead of DAOStarFinder for better handling
        of bright stars, default is True
    sharplo : float, optional
        Lower bound on sharpness for IRAFStarFinder, default is 0.2
    sharphi : float, optional
        Upper bound on sharpness for IRAFStarFinder, default is 1.0
    roundlo : float, optional
        Lower bound on roundness for IRAFStarFinder, default is -1.0
    roundhi : float, optional
        Upper bound on roundness for IRAFStarFinder, default is 1.0

    Returns:
    --------
    photutils.detection Table
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

    # Background subtracted data
    background_subtracted = image_data - median

    # Choose the star finder based on input
    if use_iraf:
        # IRAFStarFinder is often better at detecting bright, bloated stars
        # Adjust sharplo/sharphi for star bloat tolerance (lower sharplo helps with bloated stars)
        # Adjust roundlo/roundhi for how circular the stars should be
        finder = IRAFStarFinder(
            threshold=threshold_sigma * std,
            fwhm=fwhm,
            sharplo=sharplo,  # Lower this to detect more bloated stars
            sharphi=sharphi,  # May need to increase for very bloated stars
            roundlo=roundlo,  # Adjust for non-circular stars
            roundhi=roundhi
        )
        logger.info(f"Using IRAFStarFinder with fwhm={fwhm}, threshold={threshold_sigma}*sigma, "
                    f"sharplo={sharplo}, sharphi={sharphi}")
    else:
        # DAOStarFinder is the standard choice but may miss very bright/bloated stars
        finder = DAOStarFinder(
            fwhm=fwhm,
            threshold=threshold_sigma * std
        )
        logger.info(f"Using DAOStarFinder with fwhm={fwhm}, threshold={threshold_sigma}*sigma")

    # Find stars
    sources = finder(background_subtracted)

    if sources is None or len(sources) == 0:
        logger.warning("No stars detected in the image")
        return None

    logger.info(f"Detected {len(sources)} stars in the image")
    return sources


def detect_bright_bloated_stars(image_data, regular_sources=None):
    """
    Specifically target bright, bloated stars that might be missed by regular detection

    Parameters:
    -----------
    image_data : numpy.ndarray
        The image data array
    regular_sources : photutils.detection Table or None
        Table of already detected sources to avoid duplicates

    Returns:
    --------
    photutils.detection Table
        Table of additional bright, bloated stars detected
    """
    # Ensure we're working with a 2D array
    if len(image_data.shape) > 2:
        image_data = np.mean(image_data, axis=0) if len(image_data.shape) == 3 else image_data[0]

    # Compute image statistics
    mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)

    # Create a smoothed version of the image to better detect bloated stars
    from scipy.ndimage import gaussian_filter
    smoothed_data = gaussian_filter(image_data, sigma=3.0)

    # Use a higher FWHM and lower threshold specifically for bloated stars
    bloat_finder = IRAFStarFinder(
        threshold=5.0 * std,  # Higher threshold to focus on bright stars
        fwhm=10.0,  # Much larger FWHM for bloated stars
        sharplo=0.01,  # Very low sharpness threshold for bloated stars
        sharphi=2.0,  # Higher upper limit for sharpness
        roundlo=-2.0,  # More tolerant roundness parameters
        roundhi=2.0
    )

    logger.info("Looking for additional bright, bloated stars with specialized parameters")
    bloated_sources = bloat_finder(smoothed_data - median)

    if bloated_sources is None or len(bloated_sources) == 0:
        logger.info("No additional bright, bloated stars detected")
        return None

    # If we have regular sources, filter out duplicates within a certain pixel distance
    if regular_sources is not None:
        # Convert regular sources to a set of (x,y) locations
        reg_positions = set(zip(regular_sources['xcentroid'], regular_sources['ycentroid']))

        # Filter bloated sources to keep only those not close to regular sources
        from astropy.table import Table
        new_rows = []
        for row in bloated_sources:
            x, y = row['xcentroid'], row['ycentroid']

            # Check if this bloated star is near any regular star
            is_duplicate = any(np.hypot(x - rx, y - ry) < 10.0 for rx, ry in reg_positions)

            if not is_duplicate:
                new_rows.append(row)

        if not new_rows:
            logger.info("All bloated stars were already detected")
            return None

        # Create a new table with just the unique bloated stars
        bloated_sources = Table(rows=new_rows, names=bloated_sources.colnames)

    logger.info(f"Detected {len(bloated_sources)} additional bright, bloated stars")
    return bloated_sources


def combine_star_catalogs(catalog1, catalog2):
    """
    Combine two star catalogs into one

    Parameters:
    -----------
    catalog1 : photutils.detection Table
        First catalog of stars
    catalog2 : photutils.detection Table
        Second catalog of stars

    Returns:
    --------
    photutils.detection Table
        Combined catalog
    """
    if catalog1 is None:
        return catalog2
    if catalog2 is None:
        return catalog1

    from astropy.table import vstack
    combined = vstack([catalog1, catalog2])
    logger.info(f"Combined catalog contains {len(combined)} stars")
    return combined


def multi_scale_star_detection(image_data):
    """
    Detect stars across multiple scales to catch both small and bloated stars

    Parameters:
    -----------
    image_data : numpy.ndarray
        The image data array

    Returns:
    --------
    photutils.detection Table
        Combined table of detected sources across scales
    """
    # Standard detection for average stars
    sources_standard = enhanced_star_detection(
        image_data,
        fwhm=5.0,  # Standard FWHM
        threshold_sigma=3.0,  # Standard threshold
        use_iraf=True,
        sharplo=0.2,  # Standard sharpness
        sharphi=1.0
    )

    # Detection optimized for bright, bloated stars
    sources_bloated = enhanced_star_detection(
        image_data,
        fwhm=12.0,  # Much larger FWHM for bloated stars
        threshold_sigma=5.0,  # Higher threshold for bright stars
        use_iraf=True,
        sharplo=0.01,  # Very permissive sharpness for bloated stars
        sharphi=2.0
    )

    # Detection optimized for dim, small stars
    #sources_dim = enhanced_star_detection(
    #    image_data,
    #    fwhm=2.5,  # Smaller FWHM for dim stars
    #    threshold_sigma=2.5,  # Lower threshold to catch dimmer stars
    #    use_iraf=True,
    #    sharplo=0.3,  # Higher sharpness for small stars
    #    sharphi=1.2
    #)
    sources_dim = None  # Placeholder for dim star detection

    # Combine the catalogs, removing duplicates
    # Start with brightest/bloated stars
    combined = sources_bloated

    # Add standard stars not already in the combined catalog
    if sources_standard is not None and combined is not None:
        from astropy.table import vstack

        # Get current positions
        combined_pos = set(zip(combined['xcentroid'], combined['ycentroid']))

        # Filter standard sources to keep only those not close to combined sources
        from astropy.table import Table
        new_rows = []
        for row in sources_standard:
            x, y = row['xcentroid'], row['ycentroid']

            # Check if this star is near any already detected star
            is_duplicate = any(np.hypot(x - cx, y - cy) < 5.0 for cx, cy in combined_pos)

            if not is_duplicate:
                new_rows.append(row)

        if new_rows:
            # Add these new stars to the combined catalog
            new_standard = Table(rows=new_rows, names=sources_standard.colnames)
            combined = vstack([combined, new_standard])

    # Add dim stars not already in the combined catalog
    if sources_dim is not None and combined is not None:
        # Get updated combined positions
        combined_pos = set(zip(combined['xcentroid'], combined['ycentroid']))

        # Filter dim sources to keep only those not close to combined sources
        new_rows = []
        for row in sources_dim:
            x, y = row['xcentroid'], row['ycentroid']

            # Check if this star is near any already detected star
            is_duplicate = any(np.hypot(x - cx, y - cy) < 5.0 for cx, cy in combined_pos)

            if not is_duplicate:
                new_rows.append(row)

        if new_rows:
            # Add these new stars to the combined catalog
            new_dim = Table(rows=new_rows, names=sources_dim.colnames)
            combined = vstack([combined, new_dim])

    logger.info(f"Multi-scale detection found a total of {len(combined)} stars")
    return combined


def visualize_star_detection(fits_path, sources, output_path=None, flux_key='flux'):
    """
    Create a visualization of detected stars overlaid on the original image

    Parameters:
    -----------
    fits_path : str
        Path to the FITS file
    sources : photutils.detection Table
        Table of detected sources
    output_path : str, optional
        Path to save the visualization, default is None (show instead of save)
    flux_key : str, optional
        Column name in sources table for flux/brightness values, default is 'flux'
    """
    try:
        # Open the FITS file
        with fits.open(fits_path) as hdul:
            image_data = hdul[0].data
            if len(image_data.shape) > 2:
                image_data = np.mean(image_data, axis=0) if len(image_data.shape) == 3 else image_data[0]

        # Create the figure
        plt.figure(figsize=(14, 12))

        # Display the image with log scaling
        vmin = np.percentile(image_data, 1)  # Lower percentile to show more detail
        vmax = np.percentile(image_data, 99)
        plt.imshow(image_data, cmap='gray', norm=LogNorm(vmin=max(0.1, vmin), vmax=vmax))

        # Extract coordinates and flux values
        x_coords = sources['xcentroid']
        y_coords = sources['ycentroid']

        # Get flux values if available, otherwise use a constant
        if flux_key in sources.colnames:
            fluxes = sources[flux_key]
            # Log scale for size to handle extreme brightness variations
            sizes = 10 + 20 * np.log1p(fluxes / np.min(fluxes[fluxes > 0]))
        else:
            sizes = 30  # Constant size if flux not available

        # Add markers for detected stars with varying size based on flux
        scatter = plt.scatter(
            x_coords,
            y_coords,
            s=sizes,
            facecolors='none',  # Transparent face
            edgecolors='cyan',  # Cyan edge for visibility
            linewidths=1.5,  # Thicker edge
            alpha=0.8  # Slight transparency
        )

        # Add labels
        plt.title(f'Detected Stars - {os.path.basename(fits_path)}', fontsize=16)
        plt.xlabel('X (pixels)', fontsize=14)
        plt.ylabel('Y (pixels)', fontsize=14)

        # Add some info text
        plt.annotate(
            f"Total stars detected: {len(sources)}",
            xy=(0.02, 0.02),
            xycoords='axes fraction',
            fontsize=12,
            color='white',
            bbox=dict(facecolor='black', alpha=0.7)
        )

        # Save or show the figure
        if output_path:
            plt.tight_layout()
            plt.savefig(output_path, dpi=200)
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

    parser = argparse.ArgumentParser(description='Enhanced star detection and visualization')
    parser.add_argument('fits_file', help='Path to the FITS file')
    parser.add_argument('--output', '-o', help='Output visualization path', default='detected_stars.png')
    parser.add_argument('--multi_scale', '-m', action='store_true',
                        help='Use multi-scale detection for various star sizes')
    parser.add_argument('--fwhm', type=float, default=5.0,
                        help='FWHM parameter for star detection (default: 5.0)')
    parser.add_argument('--threshold', type=float, default=3.0,
                        help='Threshold in sigma for detection (default: 3.0)')
    parser.add_argument('--sharplo', type=float, default=0.1,
                        help='Lower bound on sharpness (default: 0.1)')
    parser.add_argument('--sharphi', type=float, default=1.5,
                        help='Upper bound on sharpness (default: 1.5)')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    # Set debug level if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Open the FITS file
    try:
        with fits.open(args.fits_file) as hdul:
            image_data = hdul[0].data

            # Detect stars based on chosen method
            if args.multi_scale:
                logger.info("Using multi-scale star detection")
                sources = multi_scale_star_detection(image_data)
            else:
                logger.info("Using standard star detection")
                # Use the basic enhanced detection
                sources = enhanced_star_detection(
                    image_data,
                    fwhm=args.fwhm,
                    threshold_sigma=args.threshold,
                    use_iraf=True,
                    sharplo=args.sharplo,
                    sharphi=args.sharphi
                )

                # Also look for bright bloated stars and combine the results
                bloated_sources = detect_bright_bloated_stars(image_data, sources)
                sources = combine_star_catalogs(sources, bloated_sources)

            if sources:
                # Visualize the results
                visualize_star_detection(args.fits_file, sources, args.output)
                logger.info(f"Detected {len(sources)} stars")
            else:
                logger.error("No stars detected in the image")

    except Exception as e:
        logger.error(f"Error processing FITS file: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()