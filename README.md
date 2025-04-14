# GAIA_3D
3D visualization of GAIA paralax data

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

Then you can make a GIF with something like:
```bash
python .\src\make_gif_claude.py .\data\test_astrometry.csv -i .\data\pinwheel_temp-HaRGB_2-csc-crop-St.tiff -s .\data\pinwheel_temp-HaRGB_2-csc-crop-St-Starless.tiff -o .\data\test_astrometry.gif --debug --save-stars .\data\test_astrometry_stars.png
```
or
```bash
 python .\src\make_gif_claude.py data/test_astrometry.csv -i data/pinwheel_temp-HaRGB_2-csc-crop-St.tiff -s data/pinwheel_temp-HaRGB_2-csc-crop-St-Starless.tiff -o data/test_astrometry.gif --parallax-mode power --power 3.0 --contrast 2.5
```