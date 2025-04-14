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