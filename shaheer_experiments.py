import os
import numpy as np
import os
print(f"Python is currently looking in: {os.getcwd()}")
print(f"Is the file there? {os.path.exists('camera_params_fisheye_aruko.npz')}")

# Get the directory where the script is actually located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Join that with the filename
file_path = os.path.join(script_dir, 'camera_params_fisheye_aruko.npz')

# Now load it
with np.load(file_path) as data:
    # Access your params
    K = data['mtx']
    dist = data['dist']
    print("Successfully loaded calibration params.")