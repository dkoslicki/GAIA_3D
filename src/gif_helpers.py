import os
import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops, ImageEnhance
import math
import subprocess
import tempfile
from tqdm import tqdm
import shutil

logger = logging.getLogger('parallax_animator')


def create_delta_frames(frames):
    """
    Convert full frames to delta frames to reduce GIF size

    Parameters:
    -----------
    frames : list
        List of PIL.Image frames in RGB or RGBA format

    Returns:
    --------
    list
        List of PIL.Image frames with transparency for unchanged areas
    """
    logger.info("Converting to delta frames for smaller GIF size")

    # Ensure all frames are in RGBA mode for proper transparency handling
    frames = [f.convert('RGBA') for f in frames]

    # First frame remains as is
    delta_frames = [frames[0].copy()]

    # Convert subsequent frames to show only the changes
    previous = frames[0].copy()

    for i in range(1, len(frames)):
        current = frames[i].copy()

        # Find difference between current and previous frame
        diff = ImageChops.difference(current, previous)

        # Create mask where pixels have changed
        diff_array = np.array(diff)
        # Sum across RGB channels to detect any change
        change_mask = np.sum(diff_array[:, :, :3], axis=2) > 10  # Threshold for what counts as "changed"

        # Create a new frame with transparent background where pixels haven't changed
        new_frame = Image.new('RGBA', current.size, (0, 0, 0, 0))
        current_array = np.array(current)

        # Copy pixels from current frame
        new_frame_array = np.array(new_frame)
        for c in range(4):  # Copy all channels including alpha
            new_frame_array[:, :, c] = np.where(change_mask, current_array[:, :, c], 0)

        # For alpha channel: if pixel hasn't changed, make it transparent
        new_frame_array[:, :, 3] = np.where(change_mask, current_array[:, :, 3], 0)

        new_frame = Image.fromarray(new_frame_array, 'RGBA')
        delta_frames.append(new_frame)
        previous = current

    return delta_frames


def downsample_frames(frames, scale_factor=0.5):
    """
    Spatially downsample frames to reduce GIF size

    Parameters:
    -----------
    frames : list
        List of PIL.Image frames
    scale_factor : float, optional
        Factor to scale dimensions (e.g., 0.5 for half size)

    Returns:
    --------
    list
        List of downsampled frames
    """
    if scale_factor >= 1.0:
        return frames

    logger.info(f"Spatially downsampling frames by factor of {scale_factor}")

    # Get dimensions
    width, height = frames[0].size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Resize all frames
    return [frame.resize((new_width, new_height), Image.LANCZOS) for frame in frames]


def optimize_colors(frames, colors=256):
    """
    Optimize color palette across all frames for better GIF compression

    Parameters:
    -----------
    frames : list
        List of PIL.Image frames
    colors : int, optional
        Number of colors in the palette (max 256 for GIF)

    Returns:
    --------
    list
        List of frames with optimized palette
    """
    logger.info(f"Optimizing color palette to {colors} colors")

    # Convert all frames to RGB for consistent processing
    frames = [f.convert('RGB') for f in frames]

    # Create a single image containing samples from all frames
    sample_width = min(100, frames[0].width)
    sample_height = min(100, frames[0].height)

    # Sample regions from random frames
    samples = []
    for _ in range(min(10, len(frames))):
        frame_idx = np.random.randint(0, len(frames))
        x_offset = np.random.randint(0, max(1, frames[0].width - sample_width))
        y_offset = np.random.randint(0, max(1, frames[0].height - sample_height))

        region = frames[frame_idx].crop((x_offset, y_offset,
                                         x_offset + sample_width,
                                         y_offset + sample_height))
        samples.append(region)

    # Create a composite image from samples
    composite = Image.new('RGB', (sample_width * len(samples), sample_height))
    for i, sample in enumerate(samples):
        composite.paste(sample, (i * sample_width, 0))

    # Quantize to generate optimized palette
    palette = composite.quantize(colors=colors)

    # Apply palette to all frames
    return [f.quantize(palette=palette) for f in frames]


def save_optimized_gif(frames, output_path, duration=50, loop=0, optimization_level='medium',
                       lossy=80, colors=256, use_delta_frames=False, downscale=1.0):
    """
    Save frames as an optimized animated GIF with various techniques to reduce file size

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
    optimization_level : str, optional
        Level of optimization: 'low', 'medium', or 'high'
    lossy : int, optional
        Lossy compression level for gifsicle (0-200)
    colors : int, optional
        Number of colors to use in the palette
    use_delta_frames : bool, optional
        Whether to use delta frame encoding
    downscale : float, optional
        Factor to downscale images (e.g., 0.5 for half size)
    """
    logger.info(f"Saving optimized GIF to {output_path} with optimization level: {optimization_level}")

    # Apply spatial downsampling if requested
    if downscale < 1.0:
        frames = downsample_frames(frames, downscale)

    # Apply delta frame encoding if requested
    if use_delta_frames:
        frames = create_delta_frames(frames)

    # Apply color optimization
    if colors < 256:
        frames = optimize_colors(frames, colors)

    try:
        # Basic save with PIL's built-in optimization
        if use_delta_frames:
            # First frame needs to be complete
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=loop,
                optimize=True,
                disposal=2,  # Clear previous frame
                transparency=0  # Index for transparent color
            )
        else:
            # Normal save
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=loop,
                optimize=True
            )

        initial_size = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"Initial save size: {initial_size:.2f} MB")

        # If gifsicle is available, apply further optimization
        windows = False
        if optimization_level in ['medium', 'high']:
            try:
                # Check if gifsicle is installed
                try:
                    subprocess.run(['gifsicle', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    logger.info("Found gifsicle, applying additional optimization")
                except:
                    subprocess.run(['C:\\Program Files\\gifsicle\\gifsicle.exe', '--version'], stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
                    logger.info("Found gifsicle, applying additional optimization")
                    windows = True

                with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as temp_file:
                    temp_path = temp_file.name

                # Apply gifsicle optimization
                if windows:
                    cmd = ['C:\\Program Files\\gifsicle\\gifsicle.exe', '--optimize=3']
                else:
                    cmd = ['gifsicle', '--optimize=3']

                # Add lossy compression for 'high' optimization
                if optimization_level == 'high' and lossy > 0:
                    cmd.extend(['--lossy=' + str(lossy)])

                cmd.extend(['--colors=' + str(colors), '-o', temp_path, output_path])

                subprocess.run(cmd)

                # Check if optimization actually reduced size
                optimized_size = os.path.getsize(temp_path) / (1024 * 1024)
                logger.info(f"Size after gifsicle: {optimized_size:.2f} MB")

                if optimized_size < initial_size:
                    # Replace original with optimized version
                    #os.replace(temp_path, output_path)
                    shutil.move(temp_path, output_path)
                    logger.info(f"Replaced with optimized version, saved {initial_size - optimized_size:.2f} MB")
                else:
                    # Remove temp file if not better
                    os.unlink(temp_path)
                    logger.info("Kept original version as it was smaller")

            except (subprocess.SubprocessError, FileNotFoundError):
                logger.info("Gifsicle not available, skipping additional optimization")

        final_size = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"Final GIF size: {final_size:.2f} MB")

    except Exception as e:
        logger.error(f"Error saving optimized animation: {e}")
        raise



def create_subsampled_frames(frames, output_frames=30):
    """
    Create temporally subsampled frames with interpolation to maintain animation smoothness

    Parameters:
    -----------
    frames : list
        Original list of PIL.Image frames
    output_frames : int, optional
        Number of frames to output

    Returns:
    --------
    list
        Subsampled list of frames
    """
    import numpy as np
    from PIL import Image

    if len(frames) <= output_frames:
        return frames

    logger.info(f"Temporally subsampling from {len(frames)} to {output_frames} frames")

    # Calculate indices of frames to keep
    indices = np.linspace(0, len(frames) - 1, output_frames)
    indices = np.round(indices).astype(int)

    # Extract frames at these indices
    return [frames[i] for i in indices]


def save_efficient_gif(frames, output_path, duration=50, loop=0):
    """
    Save frames as an animated GIF using delta frames for efficiency

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
    """
    import tempfile
    import os
    import subprocess
    from PIL import Image

    logger.info(f"Saving efficient GIF to {output_path} ({len(frames)} frames)")

    # Convert to delta frames to minimize redundant information
    delta_frames = create_delta_frames(frames)

    try:
        # Save with transparency optimization
        delta_frames[0].save(
            output_path,
            save_all=True,
            append_images=delta_frames[1:],
            duration=duration,
            loop=loop,
            optimize=True,
            disposal=2,  # Clear previous frame
            transparency=0  # Index for transparent color
        )

        initial_size = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"Size after delta frame optimization: {initial_size:.2f} MB")

        # Try gifsicle optimization if available
        try:
            subprocess.run(['gifsicle', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as temp_file:
                temp_path = temp_file.name

            subprocess.run([
                'gifsicle',
                '--optimize=3',
                '--colors=256',
                '--lossy=80',
                '-o', temp_path,
                output_path
            ])

            optimized_size = os.path.getsize(temp_path) / (1024 * 1024)
            logger.info(f"Size after gifsicle: {optimized_size:.2f} MB")

            if optimized_size < initial_size:
                os.replace(temp_path, output_path)
                logger.info(f"Replaced with optimized version, saved {initial_size - optimized_size:.2f} MB")
            else:
                os.unlink(temp_path)
                logger.info("Kept original version as it was smaller")

        except (subprocess.SubprocessError, FileNotFoundError):
            logger.info("Gifsicle not available, skipping additional optimization")

    except Exception as e:
        logger.error(f"Error saving efficient animation: {e}")
        raise
