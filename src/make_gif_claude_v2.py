#!/usr/bin/env python3
"""
Enhanced Stellar Parallax Animation Generator

This script creates an animated GIF that simulates stellar parallax by shifting stars
based on their distances from Earth. Stars closer to Earth will appear to move more
than distant stars when the observer's position changes.

Features:
- Supports both FITS and TIFF input images
- Automatically extracts stars from original image
- Uses real star images from the input rather than synthetic renderings
- Simulates parallax motion based on actual star distances

Requirements:
- numpy
- pandas
- pillow
- astropy (for FITS file handling)
- scikit-image (for image processing)
- tqdm (for progress bar)
"""

import os
import argparse
import logging
import math
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageChops, ImageEnhance
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage

# Import conditional dependencies
try:
    from astropy.io import fits
except ImportError:
    fits = None

try:
    from skimage import exposure, filters, morphology, measure
except ImportError:
    exposure = filters = morphology = measure = None

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('parallax_animator')


def check_dependencies():
    """Check if all required dependencies are installed"""
    missing = []
    if fits is None:
        missing.append("astropy")
    if exposure is None:
        missing.append("scikit-image")

    if missing:
        logger.warning(f"Missing optional dependencies: {', '.join(missing)}")
        logger.warning("Some functionality may be limited.")

        if "astropy" in missing:
            logger.warning("FITS file support will be disabled. Install with: pip install astropy")
        if "scikit-image" in missing:
            logger.warning("Advanced image processing will be limited. Install with: pip install scikit-image")

    return len(missing) == 0


def load_star_data(csv_path):
    """
    Load star position and distance data from CSV file

    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing star data

    Returns:
    --------
    pandas.DataFrame
        DataFrame with star positions and distances
    """
    logger.info(f"Loading star data from {csv_path}")
    try:
        df = pd.read_csv(csv_path)

        # Ensure expected columns exist
        required_columns = ['image_x', 'image_y', 'distance_ly']
        if not all(col in df.columns for col in required_columns):
            # Try with alternative column names
            alt_columns = ['image_x_coordinate', 'image_y_coordinate', 'distance_light_years']
            if all(col in df.columns for col in alt_columns):
                logger.info("Using alternative column names from CSV")
                df = df.rename(columns={
                    'image_x_coordinate': 'image_x',
                    'image_y_coordinate': 'image_y',
                    'distance_light_years': 'distance_ly'
                })
            else:
                logger.error(f"CSV file missing required columns. Expected either {required_columns} or {alt_columns}")
                raise ValueError("Invalid CSV format")

        # Filter out invalid distances
        original_count = len(df)
        df = df[df['distance_ly'].notna() & (df['distance_ly'] > 0)]
        if len(df) < original_count:
            logger.info(f"Filtered out {original_count - len(df)} stars with invalid distances")

        logger.info(f"Loaded {len(df)} stars with valid distance data")
        logger.info(f"Distance range: {df['distance_ly'].min():.1f} to {df['distance_ly'].max():.1f} light years")
        return df

    except Exception as e:
        logger.error(f"Error loading star data: {e}")
        raise


def load_image_file(file_path):
    """
    Load image file (supports FITS and common image formats)

    Parameters:
    -----------
    file_path : str
        Path to the image file

    Returns:
    --------
    numpy.ndarray
        Image data as numpy array
    dict
        Metadata information
    """
    file_path = str(file_path)  # Convert Path to string if needed
    file_ext = os.path.splitext(file_path)[1].lower()

    logger.info(f"Loading image from {file_path}")
    metadata = {}

    try:
        if file_ext == '.fits' or file_ext == '.fit':
            if fits is None:
                raise ImportError("astropy is required for FITS file support")

            with fits.open(file_path) as hdul:
                # Get primary HDU
                data = hdul[0].data
                header = hdul[0].header

                # Store important metadata
                metadata['header'] = header

                # Handle multi-dimensional data (take first plane or average color channels)
                if len(data.shape) > 2:
                    logger.info(f"FITS image has shape {data.shape}, converting to 2D")
                    if len(data.shape) == 3 and data.shape[0] <= 3:
                        # Likely color channels first
                        data = np.mean(data, axis=0)
                    else:
                        # Take first plane
                        data = data[0]

                # Normalize FITS data for PIL
                min_val = np.percentile(data, 1)
                max_val = np.percentile(data, 99)

                if max_val > min_val:
                    data = (data - min_val) / (max_val - min_val)
                    data = np.clip(data, 0, 1)

                # Convert to 8-bit
                data = (data * 255).astype(np.uint8)

                logger.info(f"Loaded FITS image with shape {data.shape}")
                return data, metadata
        else:
            # Use PIL for other formats
            with Image.open(file_path) as img:
                metadata['mode'] = img.mode
                metadata['format'] = img.format

                # Convert to RGB or RGBA if needed
                if img.mode == 'P':
                    img = img.convert('RGBA')
                elif img.mode not in ('RGB', 'RGBA', 'L'):
                    img = img.convert('RGB')

                data = np.array(img)
                logger.info(f"Loaded image with shape {data.shape}")
                return data, metadata

    except Exception as e:
        logger.error(f"Error loading image file: {e}")
        raise


def extract_stars(image_data, starless_data=None, threshold_factor=2.0, min_size=3):
    """
    Extract stars from an image by comparing with starless version
    or by threshold detection if starless not provided

    Parameters:
    -----------
    image_data : numpy.ndarray
        Original image data with stars
    starless_data : numpy.ndarray, optional
        Starless version of the image
    threshold_factor : float, optional
        Threshold factor for star detection
    min_size : int, optional
        Minimum size of star detection

    Returns:
    --------
    numpy.ndarray
        Mask of star locations
    numpy.ndarray
        Stars-only image
    """
    logger.info("Extracting stars from image")

    # Handle different image types
    if len(image_data.shape) == 3 and image_data.shape[2] >= 3:
        # Convert color image to grayscale for detection
        gray_image = np.mean(image_data[:, :, :3], axis=2)
        has_alpha = image_data.shape[2] == 4
    else:
        # Already grayscale
        gray_image = image_data.copy()
        has_alpha = False

    if starless_data is not None:
        # Method 1: Use difference between original and starless images
        logger.info("Using difference between original and starless images")

        # Convert starless to grayscale if needed
        if len(starless_data.shape) == 3 and starless_data.shape[2] >= 3:
            gray_starless = np.mean(starless_data[:, :, :3], axis=2)
        else:
            gray_starless = starless_data.copy()

        # Ensure they have the same shape
        if gray_image.shape != gray_starless.shape:
            logger.warning("Original and starless images have different dimensions!")
            # Resize to match
            from PIL import Image
            gray_starless = np.array(Image.fromarray(gray_starless.astype(np.uint8))
                                     .resize(gray_image.shape[::-1]))

        # Calculate difference
        diff = gray_image.astype(np.float32) - gray_starless.astype(np.float32)

        # Only positive differences (stars are brighter than background)
        diff = np.clip(diff, 0, None)

        # Normalize and threshold
        if diff.max() > 0:
            diff = diff / diff.max() * 255
        diff = diff.astype(np.uint8)

        # Use a threshold to identify stars
        if exposure and filters:
            # Use scikit-image for better results
            threshold = filters.threshold_otsu(diff) * 0.5
            stars_mask = diff > threshold

            # Remove small objects
            if morphology:
                stars_mask = morphology.remove_small_objects(stars_mask, min_size=min_size)
        else:
            # Simple threshold
            threshold = np.percentile(diff, 95)
            stars_mask = diff > threshold

            # Simple cleanup
            from scipy import ndimage
            stars_mask = ndimage.binary_opening(stars_mask, structure=np.ones((2, 2)))

    else:
        # Method 2: Direct threshold detection
        logger.info("Using direct threshold detection for stars")

        if exposure and filters:
            # Enhance contrast
            p2, p98 = np.percentile(gray_image, (2, 98))
            rescaled = exposure.rescale_intensity(gray_image, in_range=(p2, p98))

            # Detect bright spots (stars)
            threshold = filters.threshold_otsu(rescaled) * threshold_factor
            stars_mask = rescaled > threshold

            # Clean up the mask
            if morphology:
                stars_mask = morphology.remove_small_objects(stars_mask, min_size=min_size)
                stars_mask = morphology.binary_opening(stars_mask, morphology.disk(1))
        else:
            # Simple threshold
            p2, p98 = np.percentile(gray_image, (2, 98))
            rescaled = np.clip((gray_image - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
            threshold = np.percentile(rescaled, 95) * threshold_factor / 100
            stars_mask = rescaled > threshold

            # Simple cleanup
            from scipy import ndimage
            stars_mask = ndimage.binary_opening(stars_mask)

    # Create stars-only image
    stars_only = np.zeros_like(image_data)

    # Copy original pixel values for stars
    if len(image_data.shape) == 3:
        for c in range(min(3, image_data.shape[2])):
            stars_only[:, :, c] = np.where(stars_mask, image_data[:, :, c], 0)

        # Handle alpha channel if present
        if has_alpha and stars_only.shape[2] == 4:
            stars_only[:, :, 3] = np.where(stars_mask, image_data[:, :, 3], 0)
    else:
        # Grayscale
        stars_only = np.where(stars_mask, image_data, 0)

    logger.info(f"Extracted {np.sum(stars_mask)} star pixels")
    return stars_mask, stars_only


def analyze_star_regions_ChatGPT(stars_mask, star_data):
    # Label connected regions
    labeled_stars, num_labels = measure.label(stars_mask, return_num=True)
    logger.info(f"Identified {num_labels} distinct star regions")

    # Create a dictionary to hold associations: region label -> list of star data (for cases with multiple stars)
    associations = {}

    for idx, star in star_data.iterrows():
        # Note: star_data columns: 'image_x', 'image_y'
        y = int(round(star['image_y']))
        x = int(round(star['image_x']))

        # Ensure the coordinates are within bounds
        if 0 <= y < labeled_stars.shape[0] and 0 <= x < labeled_stars.shape[1]:
            region_label = labeled_stars[y, x]
            if region_label != 0:  # Skip background
                if region_label not in associations:
                    associations[region_label] = {
                        'centroid': (y, x),
                        'coords': [],  # Optionally, you could combine coordinates or region info later
                        'star_data': []
                    }
                # Append the star record from CSV
                associations[region_label]['star_data'].append(star)

    logger.info(f"Associated {len(associations)} star regions with star distance data")
    return associations


def analyze_star_regions_Claude(stars_mask, star_data):  # TODO: super weird that ChatGPT 03-mini-high and Claude 3.7
    # Sonnet gave the EXACT same answer, down to the comments.
    """
    Analyze star regions by associating each star from the CSV with a labeled region

    Parameters:
    -----------
    stars_mask : numpy.ndarray
        Binary mask of star locations
    star_data : pandas.DataFrame
        DataFrame with star positions and distances

    Returns:
    --------
    dict
        Mapping from region labels to star data
    """
    # Label connected regions
    labeled_stars, num_labels = measure.label(stars_mask, return_num=True)
    logger.info(f"Identified {num_labels} distinct star regions")

    # Create a dictionary to hold associations: region label -> list of star data (for cases with multiple stars)
    associations = {}

    for idx, star in star_data.iterrows():
        # Note: star_data columns: 'image_x', 'image_y'
        y = int(round(star['image_y']))
        x = int(round(star['image_x']))

        # Ensure the coordinates are within bounds
        if 0 <= y < labeled_stars.shape[0] and 0 <= x < labeled_stars.shape[1]:
            region_label = labeled_stars[y, x]
            if region_label != 0:  # Skip background
                if region_label not in associations:
                    associations[region_label] = {
                        'centroid': (y, x),
                        'coords': [],  # Optionally, you could combine coordinates or region info later
                        'star_data': []
                    }
                # Append the star record from CSV
                associations[region_label]['star_data'].append(star)

    logger.info(f"Associated {len(associations)} star regions with star distance data")
    return associations


def analyze_star_regions(stars_mask, star_data):
    """
    Analyze star regions to associate detected regions with star data

    Parameters:
    -----------
    stars_mask : numpy.ndarray
        Binary mask of star locations
    star_data : pandas.DataFrame
        DataFrame with star positions and distances

    Returns:
    --------
    dict
        Mapping from region labels to star data
    """
    if measure is None:
        logger.warning("scikit-image measure module not available, using simplified star regions")
        # Use simple circular regions around known star positions
        star_regions = {}
        for idx, star in star_data.iterrows():
            x, y = int(round(star['image_x'])), int(round(star['image_y']))
            star_regions[idx + 1] = {
                'centroid': (y, x),  # Row, column format for indexing
                'coords': [(y, x)],
                'area': 1,
                'star_data': star
            }
        return star_regions

    # Label connected regions
    labeled_stars, num_labels = measure.label(stars_mask, return_num=True)
    logger.info(f"Identified {num_labels} distinct star regions")

    # Get region properties
    regions = measure.regionprops(labeled_stars)

    # Associate regions with star data by finding closest matches
    star_regions = {}
    points = np.array([[s['image_y'], s['image_x']] for _, s in star_data.iterrows()])

    for i, region in enumerate(regions):
        label = region.label
        centroid = region.centroid

        # Find nearest star to this region
        if len(points) > 0:
            distances = np.sqrt(np.sum((points - centroid) ** 2, axis=1))
            nearest_idx = np.argmin(distances)
            min_distance = distances[nearest_idx]

            # Only associate if reasonably close
            max_distance = max(region.area ** 0.5 * 2, 30)  # Adjust threshold based on region size

            if min_distance <= max_distance:
                star_row = star_data.iloc[nearest_idx]

                star_regions[label] = {
                    'centroid': centroid,
                    'coords': region.coords,
                    'area': region.area,
                    'star_data': star_row,
                    'distance': star_row['distance_ly']
                }

    logger.info(f"Associated {len(star_regions)} star regions with distance data")
    return star_regions


def create_shifted_star_image(stars_image, star_regions, offset_factor, direction, width, height,
                              parallax_mode="logarithmic", contrast_factor=1.0, power=2.0):
    """
    Create a new image with stars shifted according to their parallax

    Parameters:
    -----------
    stars_image : numpy.ndarray
        Original stars-only image data
    star_regions : dict
        Mapping from region labels to star data
    offset_factor : float
        Base offset factor (will be scaled by parallax)
    direction : str
        Direction of parallax motion ('horizontal', 'vertical', or 'both')
    width : int
        Width of the output image
    height : int
        Height of the output image
    parallax_mode : str, optional
        Mode for scaling parallax with distance ('logarithmic', 'inverse', 'power')
    contrast_factor : float, optional
        Factor to enhance contrast between near and far star movements (higher = more contrast)
    power : float, optional
        Power factor for 'power' parallax mode

    Returns:
    --------
    numpy.ndarray
        New image with shifted stars
    """
    # Create output array same shape as input
    shifted_stars = np.zeros_like(stars_image)

    # Get distance range for scaling
    distances = np.array(
        [r['star_data']['distance_ly'] for r in star_regions.values() if 'distance_ly' in r['star_data']])
    if len(distances) == 0:
        logger.warning("No valid distances found for star regions")
        return shifted_stars

    min_distance = np.min(distances)
    max_distance = np.max(distances)

    # For each star region, apply shift based on distance
    for label, region in star_regions.items():
        if 'star_data' not in region or 'distance_ly' not in region['star_data']:
            continue

        # Get distance
        distance = region['star_data']['distance_ly']

        # Calculate parallax factor based on chosen mode
        if parallax_mode == "logarithmic":
            # Logarithmic scaling (original)
            log_min = np.log10(min_distance)
            log_max = np.log10(max_distance)
            log_distance = np.log10(distance)
            parallax_factor = (log_max - log_distance) / (log_max - log_min)

        elif parallax_mode == "inverse":
            # Simple inverse scaling (1/distance)
            inv_min = 1.0 / max_distance
            inv_max = 1.0 / min_distance
            parallax_factor = (1.0 / distance - inv_min) / (inv_max - inv_min)

        elif parallax_mode == "power":
            # Power function scaling
            # Normalize to 0-1 range first
            norm_distance = (distance - min_distance) / (max_distance - min_distance)
            parallax_factor = 1.0 - pow(norm_distance, power)

        else:
            # Default to linear scaling
            parallax_factor = 1.0 - (distance - min_distance) / (max_distance - min_distance)

        # Apply contrast enhancement
        if contrast_factor != 1.0:
            # Rescale from 0-1 to ensure contrast is centered
            # This makes the middle distances shift more when contrast > 1
            parallax_factor = 0.5 + (parallax_factor - 0.5) * contrast_factor
            # Clip to 0-1 range
            parallax_factor = max(0, min(1, parallax_factor))

        # Calculate pixel shifts
        dx = offset_factor * parallax_factor if direction in ('horizontal', 'both') else 0
        dy = offset_factor * parallax_factor * 0.3 if direction in ('vertical', 'both') else 0

        # For each pixel in the region, shift it
        for y, x in region['coords']:
            # Calculate new position
            new_x = int(round(x + dx))
            new_y = int(round(y + dy))

            # Check bounds
            if 0 <= new_y < height and 0 <= new_x < width:
                # Copy star pixel to new position
                if len(stars_image.shape) == 3:
                    shifted_stars[new_y, new_x] = stars_image[y, x]
                else:
                    shifted_stars[new_y, new_x] = stars_image[y, x]

    # Apply a slight blur to smooth the shifted stars
    from scipy.ndimage import gaussian_filter
    if len(shifted_stars.shape) == 3:
        for c in range(shifted_stars.shape[2]):
            shifted_stars[:, :, c] = gaussian_filter(shifted_stars[:, :, c], sigma=0.5)
    else:
        shifted_stars = gaussian_filter(shifted_stars, sigma=0.5)

    # save the parallax factors to a CSV file named "temp"

    #np.save("temp.npy", shifted_stars)

    return shifted_stars


def create_shifted_star_image_log_scaling(stars_image, star_regions, offset_factor, direction, width, height):
    """
    Create a new image with stars shifted according to their parallax

    Parameters:
    -----------
    stars_image : numpy.ndarray
        Original stars-only image data
    star_regions : dict
        Mapping from region labels to star data
    offset_factor : float
        Base offset factor (will be scaled by parallax)
    direction : str
        Direction of parallax motion ('horizontal', 'vertical', or 'both')
    width : int
        Width of the output image
    height : int
        Height of the output image

    Returns:
    --------
    numpy.ndarray
        New image with shifted stars
    """
    # Create output array same shape as input
    shifted_stars = np.zeros_like(stars_image)

    # Get distance range for scaling
    distances = np.array(
        [r['star_data']['distance_ly'] for r in star_regions.values() if 'distance_ly' in r['star_data']])
    if len(distances) == 0:
        logger.warning("No valid distances found for star regions")
        return shifted_stars

    min_distance = np.min(distances)
    max_distance = np.max(distances)
    log_min = np.log10(min_distance)
    log_max = np.log10(max_distance)

    # For each star region, apply shift based on distance
    for label, region in star_regions.items():
        if 'star_data' not in region or 'distance_ly' not in region['star_data']:
            continue

        # Get parallax factor (inverse relationship with distance)
        distance = region['star_data']['distance_ly']

        # Logarithmic scaling for better visual effect
        log_distance = np.log10(distance)
        parallax_factor = (log_max - log_distance) / (log_max - log_min)

        # Calculate pixel shifts
        dx = offset_factor * parallax_factor if direction in ('horizontal', 'both') else 0
        dy = offset_factor * parallax_factor * 0.3 if direction in ('vertical', 'both') else 0

        # For each pixel in the region, shift it
        for y, x in region['coords']:
            # Calculate new position
            new_x = int(round(x + dx))
            new_y = int(round(y + dy))

            # Check bounds
            if 0 <= new_y < height and 0 <= new_x < width:
                # Copy star pixel to new position
                if len(stars_image.shape) == 3:
                    shifted_stars[new_y, new_x] = stars_image[y, x]
                else:
                    shifted_stars[new_y, new_x] = stars_image[y, x]

    # Apply a slight blur to smooth the shifted stars
    from scipy.ndimage import gaussian_filter
    if len(shifted_stars.shape) == 3:
        for c in range(shifted_stars.shape[2]):
            shifted_stars[:, :, c] = gaussian_filter(shifted_stars[:, :, c], sigma=0.5)
    else:
        shifted_stars = gaussian_filter(shifted_stars, sigma=0.5)

    return shifted_stars


def create_shifted_star_image_enhanced(stars_image, star_regions, offset_factor, direction, width, height,
                                       threshold_percentile=75, enhancement_factor=2.0):
    """
    Create a new image with stars shifted according to enhanced parallax algorithm

    Parameters:
    -----------
    stars_image : numpy.ndarray
        Original stars-only image data
    star_regions : dict
        Mapping from region labels to star data
    offset_factor : float
        Base offset factor (will be scaled by parallax)
    direction : str
        Direction of parallax motion ('horizontal', 'vertical', or 'both')
    width : int
        Width of the output image
    height : int
        Height of the output image
    threshold_percentile : float, optional
        Percentile value (0-100) for distance threshold calculation
    enhancement_factor : float, optional
        Factor to enhance the movement of closer stars

    Returns:
    --------
    numpy.ndarray
        New image with shifted stars
    """
    # Create output array same shape as input
    shifted_stars = np.zeros_like(stars_image)

    # Collect all valid distances
    distances = np.array([r['star_data']['distance_ly'] for r in star_regions.values()
                          if 'star_data' in r and 'distance_ly' in r['star_data']])

    if len(distances) == 0:
        logger.warning("No valid distances found for star regions")
        return shifted_stars

    # Calculate distance threshold using percentile
    distance_threshold = np.percentile(distances, threshold_percentile)
    logger.info(f"Using distance threshold of {distance_threshold:.1f} light years "
                f"({threshold_percentile}th percentile)")

    # Normalize distances for stars below threshold
    foreground_distances = distances[distances <= distance_threshold]
    if len(foreground_distances) > 0:
        min_distance = np.min(foreground_distances)

        # For each star region, apply shift based on distance
        for label, region in star_regions.items():
            if 'star_data' not in region or 'distance_ly' not in region['star_data']:
                continue

            # Get distance
            distance = region['star_data']['distance_ly']

            # Calculate parallax factor - only for stars within threshold
            if distance <= distance_threshold:
                # Normalized distance (0=closest, 1=threshold)
                norm_distance = (distance - min_distance) / (distance_threshold - min_distance)

                # Apply enhancement (stronger effect for closer stars)
                # Using power function for non-linear scaling
                parallax_factor = pow(1.0 - norm_distance, enhancement_factor)
            else:
                # No movement for stars beyond threshold
                parallax_factor = 0

            # Calculate pixel shifts
            dx = offset_factor * parallax_factor if direction in ('horizontal', 'both') else 0
            dy = offset_factor * parallax_factor * 0.3 if direction in ('vertical', 'both') else 0

            # For each pixel in the region, shift it
            for y, x in region['coords']:
                # Calculate new position
                new_x = int(round(x + dx))
                new_y = int(round(y + dy))

                # Check bounds
                if 0 <= new_y < height and 0 <= new_x < width:
                    # Copy star pixel to new position
                    if len(stars_image.shape) == 3:
                        shifted_stars[new_y, new_x] = stars_image[y, x]
                    else:
                        shifted_stars[new_y, new_x] = stars_image[y, x]

    # Apply a slight blur to smooth the shifted stars
    from scipy.ndimage import gaussian_filter
    if len(shifted_stars.shape) == 3:
        for c in range(shifted_stars.shape[2]):
            shifted_stars[:, :, c] = gaussian_filter(shifted_stars[:, :, c], sigma=0.5)
    else:
        shifted_stars = gaussian_filter(shifted_stars, sigma=0.5)

    return shifted_stars


def create_parallax_frames(
        star_data,
        original_image,
        starless_image=None,
        num_frames=30,
        parallax_amplitude=10,
        direction='horizontal',
        blur_stars=True,
        parallax_mode="logarithmic",
        contrast_factor=1.0,
        power=2.0
):
    """
    Create animation frames showing parallax effect using actual star images

    Parameters:
    -----------
    star_data : pandas.DataFrame
        DataFrame with star positions and distances
    original_image : numpy.ndarray
        Original image with stars
    starless_image : numpy.ndarray, optional
        Starless version of the image (if available)
    num_frames : int, optional
        Number of frames to generate
    parallax_amplitude : float, optional
        Maximum pixel offset for closest stars
    direction : str, optional
        Direction of parallax motion ('horizontal', 'vertical', or 'both')
    blur_stars : bool, optional
        Whether to apply slight blur to stars for smoother animation
    parallax_mode : str, optional
        Mode for scaling parallax with distance ('logarithmic', 'inverse', 'power')
    contrast_factor : float, optional
        Factor to enhance contrast between near and far star movements (higher = more contrast)
    power : float, optional
        Power factor for 'power' parallax mode

    Returns:
    --------
    list
        List of PIL.Image frames
    """
    height, width = original_image.shape[:2]
    frames = []

    # Extract stars if needed
    if starless_image is not None:
        # Extract stars by difference
        stars_mask, stars_only = extract_stars(original_image, starless_image)
        background = starless_image.copy()  # Make a copy to ensure we don't modify the original
    else:
        # Extract stars by threshold
        stars_mask, stars_only = extract_stars(original_image)

        # Create background by removing stars
        background = original_image.copy()
        if len(background.shape) == 3:
            for c in range(background.shape[2]):
                background[:, :, c] = np.where(stars_mask, 0, background[:, :, c])
        else:
            background = np.where(stars_mask, 0, background)

    # Analyze star regions
    star_regions = analyze_star_regions(stars_mask, star_data)

    # Convert numpy arrays to PIL images for the final composition
    # Always convert background to RGBA to ensure proper alpha compositing
    if len(background.shape) == 3 and background.shape[2] == 4:
        bg_pil = Image.fromarray(background.astype(np.uint8), 'RGBA')
    elif len(background.shape) == 3:
        bg_pil = Image.fromarray(background.astype(np.uint8), 'RGB').convert('RGBA')
    else:
        bg_pil = Image.fromarray(background.astype(np.uint8), 'L').convert('RGBA')

    logger.info(f"Generating {num_frames} animation frames")
    for frame_idx in tqdm(range(num_frames)):
        # Calculate normalized offset (-1 to 1) using sine wave
        t = frame_idx / num_frames
        offset_norm = math.sin(2 * math.pi * t)

        # Base offset for the closest star
        offset = offset_norm * parallax_amplitude

        # Create shifted star image
        shifted_stars = create_shifted_star_image(
            stars_only, star_regions, offset, direction, width, height,
            parallax_mode=parallax_mode, contrast_factor=contrast_factor, power=power
        )

        # Create a transparent image for stars with alpha channel
        stars_rgba = np.zeros((height, width, 4), dtype=np.uint8)

        # Copy RGB channels from shifted stars
        if len(shifted_stars.shape) == 3 and shifted_stars.shape[2] >= 3:
            stars_rgba[:, :, :3] = shifted_stars[:, :, :3]
        else:
            # For grayscale, copy to all RGB channels
            for c in range(3):
                stars_rgba[:, :, c] = shifted_stars

        # Create alpha channel - any non-zero pixel becomes fully opaque
        if len(shifted_stars.shape) == 3 and shifted_stars.shape[2] == 4:
            # Use existing alpha if available
            stars_rgba[:, :, 3] = shifted_stars[:, :, 3]
        else:
            # Create alpha channel from luminance
            if len(shifted_stars.shape) == 3:
                luminance = np.max(shifted_stars[:, :, :3], axis=2)
            else:
                luminance = shifted_stars
            stars_rgba[:, :, 3] = np.where(luminance > 0, 255, 0)

        # Convert to PIL Image
        stars_pil = Image.fromarray(stars_rgba, 'RGBA')

        # Apply blur if requested
        if blur_stars:
            stars_pil = stars_pil.filter(ImageFilter.GaussianBlur(0.5))

        # Create a new blank image with the background
        frame = bg_pil.copy()

        # Paste stars with transparency
        frame.paste(stars_pil, (0, 0), stars_pil)

        frames.append(frame)

    logger.info(f"Generated {len(frames)} animation frames")
    return frames


def create_parallax_frames_log_scaling(
        star_data,
        original_image,
        starless_image=None,
        num_frames=30,
        parallax_amplitude=10,
        direction='horizontal',
        blur_stars=True
):
    """
    Create animation frames showing parallax effect using actual star images

    Parameters:
    -----------
    star_data : pandas.DataFrame
        DataFrame with star positions and distances
    original_image : numpy.ndarray
        Original image with stars
    starless_image : numpy.ndarray, optional
        Starless version of the image (if available)
    num_frames : int, optional
        Number of frames to generate
    parallax_amplitude : float, optional
        Maximum pixel offset for closest stars
    direction : str, optional
        Direction of parallax motion ('horizontal', 'vertical', or 'both')
    blur_stars : bool, optional
        Whether to apply slight blur to stars for smoother animation

    Returns:
    --------
    list
        List of PIL.Image frames
    """
    height, width = original_image.shape[:2]
    frames = []

    # Extract stars if needed
    if starless_image is not None:
        # Extract stars by difference
        stars_mask, stars_only = extract_stars(original_image, starless_image)
        background = starless_image.copy()  # Make a copy to ensure we don't modify the original
    else:
        # Extract stars by threshold
        stars_mask, stars_only = extract_stars(original_image)

        # Create background by removing stars
        background = original_image.copy()
        if len(background.shape) == 3:
            for c in range(background.shape[2]):
                background[:, :, c] = np.where(stars_mask, 0, background[:, :, c])
        else:
            background = np.where(stars_mask, 0, background)

    # Analyze star regions
    star_regions = analyze_star_regions(stars_mask, star_data)

    # Convert numpy arrays to PIL images for the final composition
    # Always convert background to RGBA to ensure proper alpha compositing
    if len(background.shape) == 3 and background.shape[2] == 4:
        bg_pil = Image.fromarray(background.astype(np.uint8), 'RGBA')
    elif len(background.shape) == 3:
        bg_pil = Image.fromarray(background.astype(np.uint8), 'RGB').convert('RGBA')
    else:
        bg_pil = Image.fromarray(background.astype(np.uint8), 'L').convert('RGBA')

    logger.info(f"Generating {num_frames} animation frames")
    for frame_idx in tqdm(range(num_frames)):
        # Calculate normalized offset (-1 to 1) using sine wave
        t = frame_idx / num_frames
        offset_norm = math.sin(2 * math.pi * t)

        # Base offset for the closest star
        offset = offset_norm * parallax_amplitude

        # Create shifted star image
        shifted_stars = create_shifted_star_image(
            stars_only, star_regions, offset, direction, width, height
        )

        # Create a transparent image for stars with alpha channel
        stars_rgba = np.zeros((height, width, 4), dtype=np.uint8)

        # Copy RGB channels from shifted stars
        if len(shifted_stars.shape) == 3 and shifted_stars.shape[2] >= 3:
            stars_rgba[:, :, :3] = shifted_stars[:, :, :3]
        else:
            # For grayscale, copy to all RGB channels
            for c in range(3):
                stars_rgba[:, :, c] = shifted_stars

        # Create alpha channel - any non-zero pixel becomes fully opaque
        if len(shifted_stars.shape) == 3 and shifted_stars.shape[2] == 4:
            # Use existing alpha if available
            stars_rgba[:, :, 3] = shifted_stars[:, :, 3]
        else:
            # Create alpha channel from luminance
            if len(shifted_stars.shape) == 3:
                luminance = np.max(shifted_stars[:, :, :3], axis=2)
            else:
                luminance = shifted_stars
            stars_rgba[:, :, 3] = np.where(luminance > 0, 255, 0)

        # Convert to PIL Image
        stars_pil = Image.fromarray(stars_rgba, 'RGBA')

        # Apply blur if requested
        if blur_stars:
            stars_pil = stars_pil.filter(ImageFilter.GaussianBlur(0.5))

        # Create a new blank image with the background
        frame = bg_pil.copy()

        # Paste stars with transparency
        frame.paste(stars_pil, (0, 0), stars_pil)

        frames.append(frame)

    logger.info(f"Generated {len(frames)} animation frames")
    return frames


def create_parallax_frames_old(
        star_data,
        original_image,
        starless_image=None,
        num_frames=30,
        parallax_amplitude=10,
        direction='horizontal',
        blur_stars=True
):
    """
    Create animation frames showing parallax effect using actual star images

    Parameters:
    -----------
    star_data : pandas.DataFrame
        DataFrame with star positions and distances
    original_image : numpy.ndarray
        Original image with stars
    starless_image : numpy.ndarray, optional
        Starless version of the image (if available)
    num_frames : int, optional
        Number of frames to generate
    parallax_amplitude : float, optional
        Maximum pixel offset for closest stars
    direction : str, optional
        Direction of parallax motion ('horizontal', 'vertical', or 'both')
    blur_stars : bool, optional
        Whether to apply slight blur to stars for smoother animation

    Returns:
    --------
    list
        List of PIL.Image frames
    """
    height, width = original_image.shape[:2]
    frames = []

    # Extract stars if needed
    if starless_image is not None:
        # Extract stars by difference
        stars_mask, stars_only = extract_stars(original_image, starless_image)
        background = starless_image
    else:
        # Extract stars by threshold
        stars_mask, stars_only = extract_stars(original_image)

        # Create background by removing stars
        background = original_image.copy()
        if len(background.shape) == 3:
            for c in range(background.shape[2]):
                background[:, :, c] = np.where(stars_mask, 0, background[:, :, c])
        else:
            background = np.where(stars_mask, 0, background)

    # Analyze star regions
    star_regions = analyze_star_regions(stars_mask, star_data)

    # Convert numpy arrays to PIL images for the final composition
    if len(original_image.shape) == 3 and original_image.shape[2] == 4:
        bg_pil = Image.fromarray(background.astype(np.uint8), 'RGBA')
    elif len(original_image.shape) == 3:
        bg_pil = Image.fromarray(background.astype(np.uint8), 'RGB')
    else:
        bg_pil = Image.fromarray(background.astype(np.uint8), 'L')

    logger.info(f"Generating {num_frames} animation frames")
    for frame_idx in tqdm(range(num_frames)):
        # Calculate normalized offset (-1 to 1) using sine wave
        t = frame_idx / num_frames
        offset_norm = math.sin(2 * math.pi * t)

        # Base offset for the closest star
        offset = offset_norm * parallax_amplitude

        # Create shifted star image
        shifted_stars = create_shifted_star_image(
            stars_only, star_regions, offset, direction, width, height
        )

        # Convert to PIL Image
        if len(shifted_stars.shape) == 3 and shifted_stars.shape[2] == 4:
            stars_pil = Image.fromarray(shifted_stars.astype(np.uint8), 'RGBA')
        elif len(shifted_stars.shape) == 3:
            stars_pil = Image.fromarray(shifted_stars.astype(np.uint8), 'RGB')
        else:
            stars_pil = Image.fromarray(shifted_stars.astype(np.uint8), 'L')

        # Apply blur if requested
        if blur_stars:
            stars_pil = stars_pil.filter(ImageFilter.GaussianBlur(0.5))

        # Combine background and stars
        frame = Image.alpha_composite(bg_pil.convert('RGBA'), stars_pil.convert('RGBA'))
        frames.append(frame)

    logger.info(f"Generated {len(frames)} animation frames")
    return frames


def create_parallax_frames_enhanced(
        star_data,
        original_image,
        starless_image=None,
        num_frames=30,
        parallax_amplitude=10,
        direction='horizontal',
        blur_stars=True,
        threshold_percentile=75,
        enhancement_factor=2.0
):
    """
    Create animation frames showing enhanced parallax effect

    Parameters:
    -----------
    star_data : pandas.DataFrame
        DataFrame with star positions and distances
    original_image : numpy.ndarray
        Original image with stars
    starless_image : numpy.ndarray, optional
        Starless version of the image (if available)
    num_frames : int, optional
        Number of frames to generate
    parallax_amplitude : float, optional
        Maximum pixel offset for closest stars
    direction : str, optional
        Direction of parallax motion ('horizontal', 'vertical', or 'both')
    blur_stars : bool, optional
        Whether to apply slight blur to stars for smoother animation
    threshold_percentile : float, optional
        Percentile value (0-100) for distance threshold calculation
    enhancement_factor : float, optional
        Factor to enhance the movement of closer stars

    Returns:
    --------
    list
        List of PIL.Image frames
    """
    height, width = original_image.shape[:2]
    frames = []

    # Extract stars if needed
    if starless_image is not None:
        # Extract stars by difference
        stars_mask, stars_only = extract_stars(original_image, starless_image)
        background = starless_image.copy()  # Make a copy to ensure we don't modify the original
    else:
        # Extract stars by threshold
        stars_mask, stars_only = extract_stars(original_image)

        # Create background by removing stars
        background = original_image.copy()
        if len(background.shape) == 3:
            for c in range(background.shape[2]):
                background[:, :, c] = np.where(stars_mask, 0, background[:, :, c])
        else:
            background = np.where(stars_mask, 0, background)

    # Analyze star regions
    star_regions = analyze_star_regions(stars_mask, star_data)

    # Convert numpy arrays to PIL images for the final composition
    # Always convert background to RGBA to ensure proper alpha compositing
    if len(background.shape) == 3 and background.shape[2] == 4:
        bg_pil = Image.fromarray(background.astype(np.uint8), 'RGBA')
    elif len(background.shape) == 3:
        bg_pil = Image.fromarray(background.astype(np.uint8), 'RGB').convert('RGBA')
    else:
        bg_pil = Image.fromarray(background.astype(np.uint8), 'L').convert('RGBA')

    logger.info(f"Generating {num_frames} animation frames with enhanced parallax")
    for frame_idx in tqdm(range(num_frames)):
        # Calculate normalized offset (-1 to 1) using sine wave
        t = frame_idx / num_frames
        offset_norm = math.sin(2 * math.pi * t)

        # Base offset for the closest star
        offset = offset_norm * parallax_amplitude

        # Create shifted star image with enhanced parallax
        shifted_stars = create_shifted_star_image_enhanced(
            stars_only, star_regions, offset, direction, width, height,
            threshold_percentile=threshold_percentile,
            enhancement_factor=enhancement_factor
        )

        # Create a transparent image for stars with alpha channel
        stars_rgba = np.zeros((height, width, 4), dtype=np.uint8)

        # Copy RGB channels from shifted stars
        if len(shifted_stars.shape) == 3 and shifted_stars.shape[2] >= 3:
            stars_rgba[:, :, :3] = shifted_stars[:, :, :3]
        else:
            # For grayscale, copy to all RGB channels
            for c in range(3):
                stars_rgba[:, :, c] = shifted_stars

        # Create alpha channel - any non-zero pixel becomes fully opaque
        if len(shifted_stars.shape) == 3 and shifted_stars.shape[2] == 4:
            # Use existing alpha if available
            stars_rgba[:, :, 3] = shifted_stars[:, :, 3]
        else:
            # Create alpha channel from luminance
            if len(shifted_stars.shape) == 3:
                luminance = np.max(shifted_stars[:, :, :3], axis=2)
            else:
                luminance = shifted_stars
            stars_rgba[:, :, 3] = np.where(luminance > 0, 255, 0)

        # Convert to PIL Image
        stars_pil = Image.fromarray(stars_rgba, 'RGBA')

        # Apply blur if requested
        if blur_stars:
            stars_pil = stars_pil.filter(ImageFilter.GaussianBlur(0.5))

        # Create a new blank image with the background
        frame = bg_pil.copy()

        # Paste stars with transparency
        frame.paste(stars_pil, (0, 0), stars_pil)

        frames.append(frame)

    logger.info(f"Generated {len(frames)} animation frames")
    return frames


def create_background_from_starless(starless_image, original_shape):
    """
    Create a suitable background image from a starless image

    Parameters:
    -----------
    starless_image : numpy.ndarray
        Starless image data
    original_shape : tuple
        Shape of the original image

    Returns:
    --------
    numpy.ndarray
        Processed background image
    """
    # Ensure background has the right shape
    if starless_image.shape[:2] != original_shape[:2]:
        logger.warning(f"Resizing starless image from {starless_image.shape[:2]} to {original_shape[:2]}")
        from PIL import Image

        # Convert to PIL, resize, convert back
        if len(starless_image.shape) == 3 and starless_image.shape[2] == 4:
            mode = 'RGBA'
        elif len(starless_image.shape) == 3:
            mode = 'RGB'
        else:
            mode = 'L'

        pil_img = Image.fromarray(starless_image.astype(np.uint8), mode)
        pil_img = pil_img.resize((original_shape[1], original_shape[0]))
        starless_image = np.array(pil_img)

    return starless_image


def save_gif(frames, output_path, duration=100, loop=0, optimize=True):
    """
    Save frames as an animated GIF

    Parameters:
    -----------
    frames : list
        List of PIL.Image frames
    output_path : str
        Path to save the output GIF
    duration : int, optional
        Duration of each frame in milliseconds
    loop : int, optional
        Number of times to loop the animation (0 for infinite)
    optimize : bool, optional
        Whether to optimize the palette
    """
    logger.info(f"Saving animated GIF to {output_path} ({len(frames)} frames)")

    try:
        # Ensure all frames are in the same mode
        frames = [frame.convert('RGBA') for frame in frames]

        # Save as GIF
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=loop,
            optimize=optimize
        )
        logger.info(f"Successfully saved animation to {output_path}")
    except Exception as e:
        logger.error(f"Error saving animation: {e}")
        raise


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Create a parallax animation from star distance data')

    # Input files
    parser.add_argument('csv_file', help='Path to CSV file with star coordinates and distances')
    parser.add_argument('--image', '-i', required=True, help='Path to original image (FITS or TIFF/PNG)')
    parser.add_argument('--starless', '-s', help='Path to starless version of the image (optional)')

    # Output options
    parser.add_argument('--output', '-o', help='Output GIF file path', default='parallax_animation.gif')

    # Animation parameters
    parser.add_argument('--frames', '-f', type=int, help='Number of frames', default=30)
    parser.add_argument('--amplitude', '-a', type=float, help='Maximum parallax displacement in pixels', default=10.0)
    parser.add_argument('--duration', '-d', type=int, help='Frame duration in milliseconds', default=100)
    parser.add_argument('--direction', choices=['horizontal', 'vertical', 'both'],
                        help='Direction of parallax motion', default='horizontal')

    # Star extraction options
    parser.add_argument('--threshold', '-t', type=float, help='Star detection threshold factor', default=2.0)
    parser.add_argument('--min-size', type=int, help='Minimum star size in pixels', default=3)
    parser.add_argument('--no-blur', action='store_true', help='Disable star blurring')

    # Enhanced parallax options
    parser.add_argument('--parallax-mode', choices=['logarithmic', 'inverse', 'power', 'linear', 'enhanced'],
                        help='Mode for scaling parallax with distance', default='enhanced')
    parser.add_argument('--distance-threshold-percentile', type=float,
                        help='Percentile for distance threshold (0-100)', default=75.0)
    parser.add_argument('--enhancement-factor', type=float,
                        help='Factor to enhance parallax for close stars (higher = more dramatic)', default=2.0)
    parser.add_argument('--contrast', type=float,
                        help='Contrast factor for parallax (for non-enhanced modes)', default=1.0)
    parser.add_argument('--power', type=float,
                        help='Power factor for power scaling mode', default=2.0)

    # Debug options
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--save-stars', help='Save extracted stars image to this path')

    return parser.parse_args()


def parse_arguments_multiple_scaling():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Create a parallax animation from star distance data')

    # Input files
    parser.add_argument('csv_file', help='Path to CSV file with star coordinates and distances')
    parser.add_argument('--image', '-i', required=True, help='Path to original image (FITS or TIFF/PNG)')
    parser.add_argument('--starless', '-s', help='Path to starless version of the image (optional)')

    # Output options
    parser.add_argument('--output', '-o', help='Output GIF file path', default='parallax_animation.gif')

    # Animation parameters
    parser.add_argument('--frames', '-f', type=int, help='Number of frames', default=30)
    parser.add_argument('--amplitude', '-a', type=float, help='Maximum parallax displacement in pixels', default=10.0)
    parser.add_argument('--duration', '-d', type=int, help='Frame duration in milliseconds', default=100)
    parser.add_argument('--direction', choices=['horizontal', 'vertical', 'both'],
                        help='Direction of parallax motion', default='horizontal')

    # Star extraction options
    parser.add_argument('--threshold', '-t', type=float, help='Star detection threshold factor', default=2.0)
    parser.add_argument('--min-size', type=int, help='Minimum star size in pixels', default=3)
    parser.add_argument('--no-blur', action='store_true', help='Disable star blurring')

    # Debug options
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--save-stars', help='Save extracted stars image to this path')

    # New parallax control parameters
    parser.add_argument('--parallax-mode', choices=['logarithmic', 'inverse', 'power', 'linear'],
                        help='Mode for scaling parallax with distance', default='logarithmic')
    parser.add_argument('--contrast', type=float,
                        help='Contrast factor for parallax (higher = more difference between near and far stars)',
                        default=1.0)
    parser.add_argument('--power', type=float,
                        help='Power factor for power scaling mode', default=2.0)

    return parser.parse_args()


def parse_arguments_log_scaling():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Create a parallax animation from star distance data')

    # Input files
    parser.add_argument('csv_file', help='Path to CSV file with star coordinates and distances')
    parser.add_argument('--image', '-i', required=True, help='Path to original image (FITS or TIFF/PNG)')
    parser.add_argument('--starless', '-s', help='Path to starless version of the image (optional)')

    # Output options
    parser.add_argument('--output', '-o', help='Output GIF file path', default='parallax_animation.gif')

    # Animation parameters
    parser.add_argument('--frames', '-f', type=int, help='Number of frames', default=30)
    parser.add_argument('--amplitude', '-a', type=float, help='Maximum parallax displacement in pixels', default=10.0)
    parser.add_argument('--duration', '-d', type=int, help='Frame duration in milliseconds', default=100)
    parser.add_argument('--direction', choices=['horizontal', 'vertical', 'both'],
                        help='Direction of parallax motion', default='horizontal')

    # Star extraction options
    parser.add_argument('--threshold', '-t', type=float, help='Star detection threshold factor', default=2.0)
    parser.add_argument('--min-size', type=int, help='Minimum star size in pixels', default=3)
    parser.add_argument('--no-blur', action='store_true', help='Disable star blurring')

    # Debug options
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--save-stars', help='Save extracted stars image to this path')

    return parser.parse_args()


def main():
    """Main entry point"""
    # Check dependencies
    all_deps = check_dependencies()

    # Parse command line arguments
    args = parse_arguments()

    # Set logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)

    try:
        # Load star data
        star_data = load_star_data(args.csv_file)

        # Load original image
        original_data, original_meta = load_image_file(args.image)

        # Load starless image if provided
        starless_data = None
        if args.starless:
            starless_data, _ = load_image_file(args.starless)
            starless_data = create_background_from_starless(starless_data, original_data.shape)

        # Extract stars and save if requested
        stars_mask, stars_only = extract_stars(
            original_data,
            starless_data,
            threshold_factor=args.threshold,
            min_size=args.min_size
        )

        if args.save_stars:
            logger.info(f"Saving extracted stars to {args.save_stars}")
            if len(stars_only.shape) == 3 and stars_only.shape[2] == 4:
                Image.fromarray(stars_only.astype(np.uint8), 'RGBA').save(args.save_stars)
            elif len(stars_only.shape) == 3:
                Image.fromarray(stars_only.astype(np.uint8), 'RGB').save(args.save_stars)
            else:
                Image.fromarray(stars_only.astype(np.uint8), 'L').save(args.save_stars)

        # Generate animation frames
        if args.parallax_mode == 'enhanced':
            frames = create_parallax_frames_enhanced(
                star_data=star_data,
                original_image=original_data,
                starless_image=starless_data,
                num_frames=args.frames,
                parallax_amplitude=args.amplitude,
                direction=args.direction,
                blur_stars=not args.no_blur,
                threshold_percentile=args.distance_threshold_percentile,
                enhancement_factor=args.enhancement_factor
            )
        else:
            frames = create_parallax_frames(
                star_data=star_data,
                original_image=original_data,
                starless_image=starless_data,
                num_frames=args.frames,
                parallax_amplitude=args.amplitude,
                direction=args.direction,
                blur_stars=not args.no_blur,
                parallax_mode=args.parallax_mode,
                contrast_factor=args.contrast,
                power=args.power
            )

        # Save as GIF
        save_gif(frames, args.output, duration=args.duration)

        logger.info("Animation creation completed successfully")

    except Exception as e:
        logger.error(f"Error creating animation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    return 0


if __name__ == "__main__":
    exit(main())