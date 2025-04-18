# GAIA_3D
3D visualization of GAIA parallax data

# Installation
Use the requirements.txt file to install the required packages. You can do this with:
```bash
pip install -r requirements.txt
```
Otherwise, you can use conda:
```bash
conda create -n myenv python=3.9 \
    numpy \
    pillow \
    tqdm \
    pandas \
    scipy \
    scikit-image \
    astropy \
    astroquery \
    photutils \
    -c conda-forge
conda activate myenv
```
1. Replace myenv with your desired environment name.
2. Activate the environment with conda activate myenv.
3. (Windows only) [Install gifsicle manually](https://eternallybored.org/misc/gifsicle/).
4. (Linux/macOS) Optionally install gifsicle via your system package manager or use `conda install gifsicle -c conda-forge`.

# How to run
After installation of required packages, you will need to:
1. Solve with ASTAP manually (code exists to call ASTAP, but it's not working currently)
2. Run with something like:
```bash
python .\src\fits_to_gaia.py "E:\Dropbox\Astrophotography\Pinwheel galaxy\March 2025\pinwheel_temp-HaRGB_2-csc-crop-St-Starless.fits" -o "E:\Dropbox\Astrophotography\Pinwheel galaxy\March 2025\distances_updated.csv" -v --viz-output "E:\Dropbox\Astrophotography\Pinwheel galaxy\March 2025\distance_viz.png" -d --astap "C:\Program Files\astap\astap.exe"
```
*PLEASE NOTE* that the `--astap` flag is not working currently. You will need to run ASTAP manually and then update the FITS header with the WCS information.

This script will create a CSV file with the coordinates of the stars in the image, and a PNG file with the 
visualization of the stars in the image. 

If you solve your image on Astrometry.net, download all the accessory files (eg. `wcs.fit`, `axy.fit`, etc.) and 
then run something like:
```bash
python .\src\get_coordinates_astrometry_net.py --wcs .\data\wcs.fits --fits .\data\new-image.fits --image-radec .\data\image-radec.fits --axy .\data\axy.fits --output .\data\test_astrometry.csv -v --viz-output .\data\test_astrometry.png -d
```
Taking the Astrometry.net route is straightforward, but does not give you that many found stars (max of 500 or 
something like that). If you want more stars, then just run the `fits_to_gaia.py` script.

Next, you will need a starless image before making the visualization. I just use StarNet++ to do this, but you can 
use whatever you would like.

Then you can make a GIF with something like:
```bash
python .\src\create_visualization.py "E:\Dropbox\Astrophotography\Pinwheel galaxy\March 2025\distances_updated.csv" -i "E:\Dropbox\Astrophotography\Pinwheel galaxy\March 2025\pinwheel_temp-HaRGB_2-csc-crop-St-Starless.tif" -s "E:\Dropbox\Astrophotography\Pinwheel galaxy\March 2025\pinwheel_temp-HaRGB_2-csc-crop-St-Starless-no-stars.tif" -o "E:\Dropbox\Astrophotography\Pinwheel galaxy\March 2025\pinwheel_star_parallax_updated.gif" --parallax-mode enhanced --amplitude 100 --duration 50 --frames 30 --enhancement-factor 5
```

I've found that funky things happen when making the visualization with FITS files, or mixing file formats (PNG and 
TIF, FITS and TIF, etc.). I recommend using TIF files when creating the visualization.

