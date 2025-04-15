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
python .\src\get_coordinates_claude.py .\data\pinwheel_temp_top_90p-Blue-session_1.fits -o .\data\test_stack.csv -v --viz-output .\data\test_viz_stack.png -d --astap "C:\Program Files\astap\astap.exe"
```
This will create a CSV file with the coordinates of the stars in the image, and a PNG file with the visualization of the stars in the image. 

If you solve your image on Astrometry.net, download all the accessory files (eg. `wcs.fit`, `axy.fit`, etc.) and 
then run something like:
```bash
python .\src\get_coordinates_astrometry_net.py --wcs .\data\wcs.fits --fits .\data\new-image.fits --image-radec .\data\image-radec.fits --axy .\data\axy.fits --output .\data\test_astrometry.csv -v --viz-output .\data\test_astrometry.png -d
```
Taking the Astrometry.net route is straightforward, but does not give you that many found stars (max of 500 or 
something like that). If you want more stars, then:

1. Solve with Astrometry.net
2. Download the `new-image.fits` file (essentially: convert your TIFF/JPEG to a FITS file)
3. Solve in ASTAP
4. Update the FITS Header
5. Then run the `get_coordinates_claude.py`



Then you can make a GIF with something like:
```bash
python .\src\make_gif_claude.py .\data\test_astrometry.csv -i .\data\pinwheel_temp-HaRGB_2-csc-crop-St.tiff -s .\data\pinwheel_temp-HaRGB_2-csc-crop-St-Starless.tiff -o .\data\test_astrometry.gif --debug --save-stars .\data\test_astrometry_stars.png
```
or
```bash
 python .\src\make_gif_claude.py data/test_astrometry.csv -i data/pinwheel_temp-HaRGB_2-csc-crop-St.tiff -s data/pinwheel_temp-HaRGB_2-csc-crop-St-Starless.tiff -o data/test_astrometry.gif --parallax-mode power --power 3.0 --contrast 2.5
```

