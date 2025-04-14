#!/usr/bin/env python
import math
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# Load star data from the CSV.
# Expected columns: image_x_coordinate, image_y_coordinate, distance_light_years
star_data = pd.read_csv('star_data.csv')

# Load the background image (starless TIFF).
background = Image.open('background.tif').convert('RGBA')
width, height = background.size

# (Optional) Load the stars-only image if you plan to use it for more realistic star patches.
# For this tutorial we will simply render stars as small circles.
# stars_img = Image.open('stars_only.tif').convert('RGBA')

# Determine distance range from the CSV data (ignoring invalid distances).
valid_distances = star_data['distance_light_years'].dropna()
d_min = valid_distances.min()
d_max = valid_distances.max()

# Animation parameters
num_frames = 30  # total number of frames in the animation
amplitude = 10  # maximum horizontal offset in pixels for the closest stars
star_radius = 2  # radius of drawn stars

frames = []

# Loop over frames and compute a lateral camera offset for each frame.
for frame_idx in range(num_frames):
    # Compute normalized offset using a sine wave (oscillates between -1 and 1).
    offset_norm = math.sin(2 * math.pi * frame_idx / num_frames)
    # The raw camera offset in pixels (the stars will move proportionally)
    camera_offset = offset_norm * amplitude

    # Start this frame with a copy of the background.
    frame = background.copy()
    draw = ImageDraw.Draw(frame)

    # For each star, calculate its parallax displacement.
    # Closer stars (lower distance) will have a larger parallax_factor.
    for _, row in star_data.iterrows():
        # Get the original image coordinates of the star.
        x_orig = row['image_x_coordinate']
        y_orig = row['image_y_coordinate']
        d = row['distance_light_years']
        if np.isnan(d) or d <= 0:
            continue  # skip if distance is invalid

        # Compute a parallax factor between 0 and 1.
        parallax_factor = (d_max - d) / (d_max - d_min)

        # Compute displacement in x: the closer the star, the bigger the shift.
        dx = camera_offset * parallax_factor

        # Calculate the new (shifted) star coordinates.
        new_x = x_orig + dx
        new_y = y_orig  # For a horizontal parallax, vertical coordinate remains unchanged.

        # Draw the star on this frame; you can adjust color and size.
        draw.ellipse(
            (new_x - star_radius, new_y - star_radius, new_x + star_radius, new_y + star_radius),
            fill="white"
        )

    frames.append(frame)

# Save the frames as an animated GIF.
# The duration parameter (in milliseconds) controls the delay between frames.
output_filename = 'parallax_animation.gif'
frames[0].save(output_filename, save_all=True, append_images=frames[1:], duration=100, loop=0)

print(f"Animated GIF saved as {output_filename}")
