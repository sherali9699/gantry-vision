import numpy as np

with np.load("camera_params_non_fisheye.npz") as data:
    mtx = data['mtx']
    dist = data['dist']
    print("--- Calibration Check ---")
    print(f"Focal Length (fx, fy): {mtx[0,0]}, {mtx[1,1]}")
    print(f"Principal Point (cx, cy): {mtx[0,2]}, {mtx[1,2]}")