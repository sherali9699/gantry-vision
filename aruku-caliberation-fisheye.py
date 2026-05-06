#!/usr/bin/env python3
import cv2
import numpy as np
import glob
import os

MARKER_SIZE = 174.0

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(ARUCO_DICT, detector_params)

images = glob.glob("./calib-captures-fisheye-aruku/*.png")

print(f"Processing {len(images)} images...")

objpoints = []
imgpoints = []
image_size = None

marker_obj_points = np.array([
    [-MARKER_SIZE/2,  MARKER_SIZE/2, 0],
    [ MARKER_SIZE/2,  MARKER_SIZE/2, 0],
    [ MARKER_SIZE/2, -MARKER_SIZE/2, 0],
    [-MARKER_SIZE/2, -MARKER_SIZE/2, 0]
], dtype=np.float32)

for fname in images:

    img = cv2.imread(fname)
    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),0)

    if image_size is None:
        image_size = gray.shape[::-1]

    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        print(f"[FAIL] No marker found in {os.path.basename(fname)}")
        continue

    img_pts = corners[0].reshape(4,2)

    imgpoints.append(img_pts)
    objpoints.append(marker_obj_points)

    print(f"[OK] Detected marker in {os.path.basename(fname)}")


if len(objpoints) < 10:
    print("Not enough frames for calibration")
    exit()

print(f"\nUsing {len(objpoints)} frames for calibration")

ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    image_size,
    None,
    None
)

print("\nCalibration RMS:", ret)
print("\nCamera Matrix K:\n", K)
print("\nDistortion Coefficients D:\n", D)

#np.savez("camera_params_fisheye_aruko.npz", K=K, D=D)
np.savez("camera_params_fisheye_aruko.npz", mtx=K, dist=D)

print("\nSaved camera_params_fisheye_aruko.npz")