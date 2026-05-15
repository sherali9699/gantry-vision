#!/usr/bin/env python3

"""
Diagnostic script to check if camera intrinsics are correct.
Tests solvePnP with and without distortion correction.
"""

import cv2
import numpy as np
import sys

def diagnose_intrinsics(image_path, calib_file):
    """
    Diagnose camera intrinsics by checking calibration quality.
    """
    
    print("[INFO] Loading calibration file...")
    data = np.load(calib_file)
    camera_matrix = data["mtx"]
    dist_coeffs = data["dist"]
    
    print(f"\n[INFO] Camera Matrix:")
    print(camera_matrix)
    print(f"\n[INFO] Distortion Coefficients:")
    print(dist_coeffs)
    print(f"[INFO] Distortion shape: {dist_coeffs.shape}")
    
    # Check if it looks like fisheye distortion
    if dist_coeffs.shape[0] == 1 and dist_coeffs.shape[1] == 5:
        print("[WARN] This looks like STANDARD OpenCV distortion (not fisheye)")
        print("       Fisheye distortion should have 4 coefficients")
    elif dist_coeffs.shape == (4,):
        print("[WARN] This looks like FISHEYE distortion (4 coefficients)")
    
    # Extract focal lengths and principal point
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    print(f"\n[INFO] Focal length X (fx): {fx:.2f}")
    print(f"[INFO] Focal length Y (fy): {fy:.2f}")
    print(f"[INFO] Principal point (cx, cy): ({cx:.2f}, {cy:.2f})")
    
    # Load and check image
    print(f"\n[INFO] Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print("[ERROR] Could not load image")
        return False
    
    h, w = image.shape[:2]
    print(f"[INFO] Image size: {w}x{h}")
    
    # Detect checkerboard
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    success, corners = cv2.findChessboardCorners(gray, (7, 9), None)
    
    if not success:
        print("[ERROR] Could not detect checkerboard")
        return False
    
    print(f"[INFO] Detected {len(corners)} corners ✓")
    
    # Refine corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    
    # Build object points
    object_points = []
    for row in range(9):
        for col in range(7):
            x = col * 0.020
            y = row * 0.020
            z = 0.0
            object_points.append([x, y, z])
    object_points = np.array(object_points, dtype=np.float32)
    
    # Try solvePnP with distortion
    print(f"\n[TEST 1] solvePnP WITH distortion:")
    success1, rvec1, tvec1 = cv2.solvePnP(
        object_points, corners_refined,
        camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if success1:
        print(f"  ✓ Success")
        print(f"  tvec: {tvec1.flatten()}")
        print(f"  Distance: {np.linalg.norm(tvec1):.4f} m")
        
        # Reprojection error
        proj1, _ = cv2.projectPoints(object_points, rvec1, tvec1, camera_matrix, dist_coeffs)
        errors1 = np.linalg.norm(proj1.reshape(-1, 2) - corners_refined.reshape(-1, 2), axis=1)
        print(f"  Mean reprojection error: {np.mean(errors1):.3f} px")
    else:
        print(f"  ✗ Failed")
    
    # Try solvePnP without distortion
    print(f"\n[TEST 2] solvePnP WITHOUT distortion:")
    dist_coeffs_none = np.zeros_like(dist_coeffs)
    success2, rvec2, tvec2 = cv2.solvePnP(
        object_points, corners_refined,
        camera_matrix, dist_coeffs_none,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if success2:
        print(f"  ✓ Success")
        print(f"  tvec: {tvec2.flatten()}")
        print(f"  Distance: {np.linalg.norm(tvec2):.4f} m")
        
        # Reprojection error
        proj2, _ = cv2.projectPoints(object_points, rvec2, tvec2, camera_matrix, dist_coeffs_none)
        errors2 = np.linalg.norm(proj2.reshape(-1, 2) - corners_refined.reshape(-1, 2), axis=1)
        print(f"  Mean reprojection error: {np.mean(errors2):.3f} px")
    else:
        print(f"  ✗ Failed")
    
    # Comparison
    print(f"\n[DIAGNOSIS]")
    if success1 and success2:
        error1 = np.mean(errors1)
        error2 = np.mean(errors2)
        
        if error1 < 1.0:
            print("  ✓ WITH distortion: Good reprojection error (<1px)")
            print("  → Your intrinsics calibration is likely CORRECT")
        elif error2 < 1.0:
            print("  ✓ WITHOUT distortion: Good reprojection error (<1px)")
            print("  ⚠ WITH distortion: High reprojection error")
            print("  → Your distortion coefficients may be WRONG")
            print("  → Try recalibrating camera without fisheye assumption")
        else:
            print("  ✗ Both WITH and WITHOUT distortion have high error")
            print("  → Camera matrix (focal length, principal point) may be WRONG")
            print("  → Recalibrate the entire camera")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 diagnose_intrinsics.py <image.png> <calib.npz>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    calib_file = sys.argv[2]
    
    diagnose_intrinsics(image_path, calib_file)