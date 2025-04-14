#!/usr/bin/env python
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import astropy.units as u

def process_fits(fits_filename, output_csv):
    # Load the FITS file
    with fits.open(fits_filename) as hdul:
        header = hdul[0].header
        data = hdul[0].data

    # Create a WCS (World Coordinate System) object from the header
    wcs = WCS(header)

    # Estimate image background statistics and detect stars.
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    print(f"Image statistics: mean = {mean:.2f}, median = {median:.2f}, std = {std:.2f}")

    # Use DAOStarFinder to detect stars; adjust fwhm and threshold as needed.

    print("Detecting stars in the image...")
    daofind = DAOStarFinder(fwhm=3.0, threshold=5.0 * std)
    sources = daofind(data - median)

    if sources is None:
        print("No stars detected in the image.")
        return

    # Extract the pixel positions of the detected stars
    x_pix = sources['xcentroid']
    y_pix = sources['ycentroid']

    # Convert pixel coordinates to sky coordinates (RA, Dec)
    print("Converting pixel coordinates to sky coordinates...")
    ra, dec = wcs.all_pix2world(x_pix, y_pix, 0)

    # Prepare a list to store distances (in light years)
    distances_ly = []

    # Loop over each star and query the Gaia catalog using a cone search.
    # (Note: querying one-by-one is simple but may be slow if many stars are detected.)
    for r, d in zip(ra, dec):
        coord = SkyCoord(ra=r*u.deg, dec=d*u.deg, frame='icrs')
        try:
            # Query with a 2 arcsecond radius (adjustable)
            radius = 2 * u.arcsec
            job = Gaia.cone_search_async(coord, radius)
            result = job.get_results()
        except Exception as e:
            print(f"Error querying Gaia for star at RA={r:.5f}, Dec={d:.5f}: {e}")
            result = None

        if result is None or len(result) == 0:
            # No Gaia match found
            distances_ly.append(np.nan)
        else:
            # Assume the first returned Gaia entry is the best match.
            star = result[0]
            parallax = star['parallax']  # in milliarcseconds
            if parallax <= 0:
                distances_ly.append(np.nan)
            else:
                # Calculate distance in parsecs (1/(parallax in arcsec)); here parallax is in mas.
                distance_pc = 1000.0 / parallax
                distance_ly = distance_pc * 3.26156  # Convert parsecs to light years.
                distances_ly.append(distance_ly)

    # Create a DataFrame to organize the output.
    df = pd.DataFrame({
        'image_x_coordinate': x_pix,
        'image_y_coordinate': y_pix,
        'distance_light_years': distances_ly
    })

    # Save the results to CSV.
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python get_coordinates.py <input_fits_file> <output_csv_file>")
    else:
        fits_filename = sys.argv[1]
        output_csv = sys.argv[2]
        process_fits(fits_filename, output_csv)
