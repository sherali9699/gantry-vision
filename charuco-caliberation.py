#!/usr/bin/env python3
import cv2
import numpy as np
import glob
import os

# 1. ChArUco Board Configuration
# Use the exact parameters you used to generate/print your board
SQUARE_LENGTH = 48.0  # mm
MARKER_LENGTH = 35.0  # mm (usually 50-70% of square length)
CHARUCO_BOARD_SIZE = (6, 4) # Number of squares (Width, Height)
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# Create the board object
board = cv2.aruco.CharucoBoard(
    CHARUCO_BOARD_SIZE, SQUARE_LENGTH, MARKER_LENGTH, ARUCO_DICT
)
detector_params = cv2.aruco.DetectorParameters()
# The detector will use default charuco parameters automatically
charuco_detector = cv2.aruco.CharucoDetector(board)
charuco_detector.setDetectorParameters(detector_params)

# Storage
all_charuco_corners = []
all_charuco_ids = []
image_size = None

# 2. Process Images
images = glob.glob('./charuco-calib-fisheye-captures/*.png')

if not images:
    print("No images found in ./charuco-calib-fisheye-captures/")
    exit()

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]

    # Detect markers and interpolate chessboard corners
    charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)

    # We need at least 4 corners to contribute to calibration
    if charuco_ids is not None and len(charuco_ids) > 4:
        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)
        
        # Optional: Visualization
        cv2.aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids)
        cv2.imshow('ChArUco Detection', cv2.resize(img, (960, 540)))
        cv2.waitKey(100)
    else:
        print(f"Skipping {fname}: Not enough corners detected.")

cv2.destroyAllWindows()

# 3. Calibration
if len(all_charuco_corners) > 0:
    print(f"Calibrating with {len(all_charuco_corners)} frames...")
    
    # In newer OpenCV, we use the board to get the object points 
    # for the specific charuco corners we detected
    all_obj_points = []
    all_img_points = []
    
    for i in range(len(all_charuco_ids)):
        # This function matches detected charuco corners to their 3D board positions
        charuco_obj_points, charuco_img_points = board.matchImagePoints(
            all_charuco_corners[i], all_charuco_ids[i]
        )
        all_obj_points.append(charuco_obj_points)
        all_img_points.append(charuco_img_points)

    # Now use the standard calibration function
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        all_obj_points, 
        all_img_points, 
        image_size, 
        None, 
        None
    )

    if ret:
        print("\n[SUCCESS] Calibration Successful!")
        print("Reprojection Error:", ret)
        print("\nCamera Matrix (K):\n", mtx)
        print("\nDistortion Coefficients (D):\n", dist)
        
        np.savez("camera_params_fisheye.npz", mtx=mtx, dist=dist)
        print("\nParameters saved to camera_params_non_fisheye.npz")
    else:
        print("Calibration failed.")