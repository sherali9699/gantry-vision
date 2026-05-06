import numpy as np
import cv2
import glob

# 1. Configuration - CHANGE THESE TO MATCH YOUR PRINTED PATTERN
# If you have a 10x7 squares board, the INNER corners are 9x6
CHESSBOARD_SIZE = (8, 5) 
SQUARE_SIZE_MM = 30.0  # Measure one square with your calipers

# Stop criteria for sub-pixel refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (0,0,0), (1,0,0), (2,0,0) ... based on square size
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE_MM

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane

# Load your images
images = glob.glob('./calib-captures-fisheye-aruku/*.png')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if ret:
        objpoints.append(objp)
        # Refine corners to sub-pixel accuracy (Critical for Industrial Gantry)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        
        # Optional: Draw and display the corners to verify
        cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners2, ret)
        cv2.imshow('Verification', cv2.resize(img, (0,0), fx=0.5, fy=0.5))
        cv2.waitKey(100)

cv2.destroyAllWindows()

# 2. THE CALIBRATION MATH
# This calculates: 
# mtx (Camera Matrix), dist (Distortion Coeffs), rvecs (Rotation), tvecs (Translation)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

if ret:
    print("Calibration Successful!")
    print("\nCamera Matrix (K):\n", mtx)
    print("\nDistortion Coefficients (D):\n", dist)
    
    # SAVE THE DATA
    np.savez("camera_params_fisheye_aruku.npz", mtx=mtx, dist=dist)
    print("\nParameters saved to camera_params_fisheye_aruku.npz")
else:
    print("Calibration failed. Check your CHESSBOARD_SIZE.")