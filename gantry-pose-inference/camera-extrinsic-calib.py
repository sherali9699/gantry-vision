#!/usr/bin/env python3

"""
Camera Extrinsics Calibration Script

Purpose:
  Calibrate camera position and orientation relative to gantry frame
  using a checkerboard at a known location (origin).

Workflow:
  1. Load camera intrinsics (mtx, dist from .npz)
  2. Load image of checkerboard
  3. Detect checkerboard corners
  4. Run solvePnP to get camera pose relative to checkerboard
  5. Build and validate T_camera_to_gantry transformation matrix
  6. Save transformation matrix for runtime use

Ground Truth (for validation):
  - Camera position in gantry frame: (-0.020m, -0.285m, 1.855m)
  - Camera rotation: 45° tilt along X-axis
  - Checkerboard at origin: (0, 0, 0)
"""

import cv2
import numpy as np
import argparse
import sys
import os
from pathlib import Path


# ============================================================
# Checkerboard Detection
# ============================================================

def detect_checkerboard(image, checkerboard_size=(3, 5)):
    """
    Detect checkerboard corners in image.
    
    Args:
        image: Input image (BGR)
        checkerboard_size: (cols, rows) of inner corners = (3, 5) for 4×6 board
    
    Returns:
        corners: Detected corners (15,) numpy array or None
        gray: Grayscale image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find checkerboard corners
    success, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
    
    if not success:
        return None, gray
    
    # Refine corners to sub-pixel accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    
    return corners_refined, gray


def visualize_detection(image, corners, title="Detected Corners"):
    """
    Visualize detected corners on image.
    
    Args:
        image: Input image
        corners: Detected corners array (shape: (N, 1, 2))
        title: Window title
    
    Returns:
        Image with drawn corners
    """
    vis = image.copy()
    
    if corners is not None:
        # Draw corners using OpenCV built-in function
        cv2.drawChessboardCorners(vis, (3, 5), corners, True)
        
        # Draw corner indices (skip to avoid clutter)
        # Uncomment if needed for debugging
        # corners_flat = corners.reshape(-1, 2)
        # for i, corner in enumerate(corners_flat):
        #     x, y = int(corner[0]), int(corner[1])
        #     cv2.putText(vis, str(i), (x + 5, y - 5),
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    
    return vis


# ============================================================
# 3D Object Points (Checkerboard Geometry)
# ============================================================

def build_checkerboard_object_points(square_size_mm=40.0):
    """
    Build 3D points for checkerboard in object/gantry frame.
    
    Checkerboard: 4×6 squares (inner corners: 3×5 = 15)
    Each square: 40×40 mm (configurable)
    
    Args:
        square_size_mm: Size of each square in mm
    
    Returns:
        object_points: Shape (15, 3) - all corner positions in meters
    """
    square_size_m = square_size_mm / 1000.0  # Convert to meters
    
    # 3×5 inner corners = 15 points total (from 4×6 checkerboard)
    rows, cols = 5, 3
    
    # Create grid (in XY plane, Z=0)
    # Place origin at the center of the board
    object_points = []
    
    x_start = -cols * square_size_m / 2.0
    y_start = -rows * square_size_m / 2.0
    
    for row in range(rows):
        for col in range(cols):
            x = x_start + col * square_size_m
            y = y_start + row * square_size_m
            z = 0.0
            object_points.append([x, y, z])
    
    return np.array(object_points, dtype=np.float32)


# ============================================================
# SolvePnP and Transformation
# ============================================================

def calibrate_camera_extrinsics(image_points, object_points, camera_matrix, dist_coeffs):
    """
    Use solvePnP to get camera pose relative to checkerboard.
    
    Args:
        image_points: Detected corners in image (63, 2)
        object_points: 3D points in gantry frame (63, 3)
        camera_matrix: Camera intrinsics (3, 3)
        dist_coeffs: Distortion coefficients (may be ignored for better accuracy)
    
    Returns:
        rvec: Rotation vector (camera relative to gantry)
        tvec: Translation vector (camera relative to gantry)
        success: Boolean success flag
    """
    # Use zero distortion for better accuracy (distortion calibration may be poor)
    dist_coeffs_zero = np.zeros_like(dist_coeffs)
    
    success, rvec, tvec = cv2.solvePnP(
        object_points, image_points,
        camera_matrix, dist_coeffs_zero,
        flags=cv2.SOLVEPNP_IPPE
    )
    
    if not success:
        success, rvec, tvec = cv2.solvePnP(
            object_points, image_points,
            camera_matrix, dist_coeffs_zero,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
    
    return rvec, tvec, success


def build_transformation_matrix(rvec, tvec):
    """
    Build 4×4 homogeneous transformation matrix from rvec and tvec.
    
    Args:
        rvec: Rotation vector (3,)
        tvec: Translation vector (3,)
    
    Returns:
        T: 4×4 transformation matrix
    """
    R, _ = cv2.Rodrigues(rvec)
    
    T = np.eye(4, dtype=np.float32)
    T[0:3, 0:3] = R
    T[0:3, 3] = tvec.flatten()
    
    return T


def invert_transformation(T):
    """
    Invert a 4×4 homogeneous transformation matrix.
    
    Args:
        T: 4×4 transformation matrix
    
    Returns:
        T_inv: Inverse transformation
    """
    T_inv = np.eye(4, dtype=np.float32)
    R = T[0:3, 0:3]
    t = T[0:3, 3]
    
    T_inv[0:3, 0:3] = R.T
    T_inv[0:3, 3] = -R.T @ t
    
    return T_inv


def rotation_matrix_to_euler_degrees(R):
    """
    Convert 3×3 rotation matrix to Euler angles (degrees, ZYX convention).
    
    Args:
        R: 3×3 rotation matrix
    
    Returns:
        (roll, pitch, yaw) in degrees
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    
    if not singular:
        roll  = np.degrees(np.arctan2( R[2, 1], R[2, 2]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw   = np.degrees(np.arctan2( R[1, 0], R[0, 0]))
    else:
        roll  = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw   = 0.0
    
    return roll, pitch, yaw


# ============================================================
# Validation and Reporting
# ============================================================

def validate_calibration(image_points, object_points, rvec, tvec, camera_matrix, dist_coeffs):
    """
    Validate calibration by computing reprojection error.
    
    Args:
        image_points: Detected corners
        object_points: 3D points
        rvec: Rotation vector
        tvec: Translation vector
        camera_matrix: Camera intrinsics
        dist_coeffs: Distortion coefficients (will be zeroed for accuracy)
    
    Returns:
        Dictionary with validation metrics
    """
    # Use zero distortion for reprojection (for accuracy)
    dist_coeffs_zero = np.zeros_like(dist_coeffs)
    
    # Reproject 3D points to image
    proj_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs_zero)
    proj_points = proj_points.reshape(-1, 2)
    
    # Compute reprojection errors
    errors = np.linalg.norm(proj_points - image_points, axis=1)
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    
    # Distance from camera to checkerboard center
    distance = np.linalg.norm(tvec)
    
    # Rotation matrix and Euler angles
    R, _ = cv2.Rodrigues(rvec)
    roll, pitch, yaw = rotation_matrix_to_euler_degrees(R)
    
    return {
        "mean_reprojection_error_px": float(mean_error),
        "max_reprojection_error_px": float(max_error),
        "distance_to_checkerboard_m": float(distance),
        "tvec_x_m": float(tvec[0]),
        "tvec_y_m": float(tvec[1]),
        "tvec_z_m": float(tvec[2]),
        "roll_deg": float(roll),
        "pitch_deg": float(pitch),
        "yaw_deg": float(yaw),
        "proj_points": proj_points,
        "errors": errors,
    }


def print_calibration_report(validation, ground_truth_camera_pos=None):
    """
    Print detailed calibration report.
    
    Args:
        validation: Dictionary from validate_calibration()
        ground_truth_camera_pos: Tuple (x, y, z) for comparison
    """
    print("\n" + "="*70)
    print("CAMERA EXTRINSICS CALIBRATION REPORT")
    print("="*70)
    
    print(f"\n[REPROJECTION ERROR]")
    print(f"  Mean error    : {validation['mean_reprojection_error_px']:.3f} px")
    print(f"  Max error     : {validation['max_reprojection_error_px']:.3f} px")
    print(f"  {'✓ GOOD' if validation['mean_reprojection_error_px'] < 1.0 else '⚠ CHECK'}")
    
    print(f"\n[CAMERA POSITION IN GANTRY FRAME]")
    print(f"  X (left/right)  : {validation['tvec_x_m']:+.4f} m ({validation['tvec_x_m']*1000:+.1f} mm)")
    print(f"  Y (back/forward): {validation['tvec_y_m']:+.4f} m ({validation['tvec_y_m']*1000:+.1f} mm)")
    print(f"  Z (height)      : {validation['tvec_z_m']:+.4f} m ({validation['tvec_z_m']*1000:+.1f} mm)")
    print(f"  Distance        : {validation['distance_to_checkerboard_m']:.4f} m")
    
    if ground_truth_camera_pos is not None:
        gx, gy, gz = ground_truth_camera_pos
        dx = validation['tvec_x_m'] - gx
        dy = validation['tvec_y_m'] - gy
        dz = validation['tvec_z_m'] - (-gz)  # Note: tvec is camera→checkerboard, so Z is negative
        
        print(f"\n[GROUND TRUTH COMPARISON]")
        print(f"  Ground truth X    : {gx:+.4f} m")
        print(f"  Measured X        : {validation['tvec_x_m']:+.4f} m")
        print(f"  Difference        : {dx:+.4f} m ({dx*1000:+.1f} mm)")
        
        print(f"\n  Ground truth Y    : {gy:+.4f} m")
        print(f"  Measured Y        : {validation['tvec_y_m']:+.4f} m")
        print(f"  Difference        : {dy:+.4f} m ({dy*1000:+.1f} mm)")
        
        print(f"\n  Ground truth Z    : {gz:+.4f} m")
        print(f"  Measured Z (abs)  : {abs(validation['tvec_z_m']):+.4f} m")
        print(f"  Difference        : {dz:+.4f} m ({dz*1000:+.1f} mm)")
    
    print(f"\n[CAMERA ROTATION]")
    print(f"  Roll  (X-axis) : {validation['roll_deg']:+.2f}°")
    print(f"  Pitch (Y-axis) : {validation['pitch_deg']:+.2f}°  (Expected: ~-45°)")
    print(f"  Yaw   (Z-axis) : {validation['yaw_deg']:+.2f}°")
    
    print("\n" + "="*70)


# ============================================================
# Main Calibration Pipeline
# ============================================================

def run_calibration(
    image_path,
    calib_file,
    checkerboard_size=(3, 5),
    square_size_mm=40.0,
    ground_truth_camera_pos=None,
    output_dir=".",
):
    """
    Run complete calibration pipeline.
    
    Args:
        image_path: Path to checkerboard image
        calib_file: Path to camera calibration .npz file
        checkerboard_size: (cols, rows) of checkerboard inner corners = (3, 5) for 4×6 board
        square_size_mm: Size of each square in mm (default: 40.0)
        ground_truth_camera_pos: Tuple (x, y, z) for validation
        output_dir: Directory to save results
    """
    
    print("[INFO] Loading camera intrinsics...")
    if not os.path.exists(calib_file):
        print(f"[ERROR] Calibration file not found: {calib_file}")
        return False
    
    data = np.load(calib_file)
    camera_matrix = data["mtx"]
    dist_coeffs = data["dist"]
    print(f"[INFO] Camera matrix shape: {camera_matrix.shape}")
    print(f"[INFO] Distortion coefficients shape: {dist_coeffs.shape}")
    
    print("\n[INFO] Loading image...")
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return False
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Failed to load image: {image_path}")
        return False
    
    print(f"[INFO] Image size: {image.shape}")
    
    # DISPLAY RAW IMAGE FIRST
    print("\n[INFO] Displaying raw loaded image...")
    cv2.imshow("Raw Image - Press any key to continue", image)
    print("[INFO] Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n[INFO] Detecting checkerboard...")
    corners, gray = detect_checkerboard(image, checkerboard_size)
    
    if corners is None:
        print("[ERROR] Could not detect checkerboard. Check image quality.")
        return False
    
    print(f"[INFO] Detected {len(corners)} corners ✓")
    
    # VISUALIZE DETECTED CORNERS
    print("\n[INFO] Displaying detected checkerboard corners...")
    vis_corners = visualize_detection(image, corners, "Detected Checkerboard Corners")
    cv2.imshow("Detected Corners - Press any key to continue", vis_corners)
    print("[INFO] Press any key in the image window to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n[INFO] Building 3D object points...")
    object_points = build_checkerboard_object_points(square_size_mm)
    print(f"[INFO] Object points shape: {object_points.shape}")
    print(f"[INFO] Object points range:")
    print(f"       X: [{object_points[:, 0].min():.4f}, {object_points[:, 0].max():.4f}] m")
    print(f"       Y: [{object_points[:, 1].min():.4f}, {object_points[:, 1].max():.4f}] m")
    print(f"       Z: {object_points[0, 2]:.4f} m")
    
    print("\n[INFO] Running solvePnP...")
    rvec, tvec, success = calibrate_camera_extrinsics(
        corners, object_points, camera_matrix, dist_coeffs
    )
    
    if not success:
        print("[ERROR] solvePnP failed")
        return False
    
    print("[INFO] solvePnP successful ✓")
    
    print("\n[INFO] Validating calibration...")
    validation = validate_calibration(
        corners, object_points, rvec, tvec, camera_matrix, dist_coeffs
    )
    
    print_calibration_report(validation, ground_truth_camera_pos)
    
    print("\n[INFO] Building transformation matrix...")
    T_camera_to_gantry = build_transformation_matrix(rvec, tvec)
    print("[INFO] T_camera_to_gantry (camera → gantry frame):")
    print(T_camera_to_gantry)
    
    # Also compute the inverse (gantry → camera), useful for some operations
    T_gantry_to_camera = invert_transformation(T_camera_to_gantry)
    print("\n[INFO] T_gantry_to_camera (gantry → camera frame):")
    print(T_gantry_to_camera)
    
    print("\n[INFO] Saving results...")
    output_path = os.path.join(output_dir, "camera_extrinsics.npz")
    np.savez(
        output_path,
        T_camera_to_gantry=T_camera_to_gantry,
        T_gantry_to_camera=T_gantry_to_camera,
        rvec=rvec,
        tvec=tvec,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
    )
    print(f"[SAVED] {output_path}")
    
    print("\n[INFO] Creating visualizations...")
    
    # Visualization 1: Detected corners
    vis_corners = visualize_detection(image, corners, "Detected Checkerboard Corners")
    vis_path = os.path.join(output_dir, "01_detected_corners.png")
    cv2.imwrite(vis_path, vis_corners)
    print(f"[SAVED] {vis_path}")
    
    # Visualization 2: Reprojection overlay
    vis_reprojection = image.copy()
    detected_pts = corners.reshape(-1, 2)
    proj_pts = validation["proj_points"]
    
    # Draw detected corners (green)
    for pt in detected_pts:
        cv2.circle(vis_reprojection, tuple(pt.astype(int)), 4, (0, 255, 0), -1)
    
    # Draw reprojected corners (red)
    for pt in proj_pts:
        cv2.circle(vis_reprojection, tuple(pt.astype(int)), 3, (0, 0, 255), -1)
    
    # Draw lines showing error
    for det, proj in zip(detected_pts, proj_pts):
        cv2.line(vis_reprojection, tuple(det.astype(int)), tuple(proj.astype(int)), (255, 0, 0), 1)
    
    cv2.putText(vis_reprojection,
                f"Green=Detected  Red=Reprojected  Mean Error: {validation['mean_reprojection_error_px']:.2f}px",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    vis_path = os.path.join(output_dir, "02_reprojection_overlay.png")
    cv2.imwrite(vis_path, vis_reprojection)
    print(f"[SAVED] {vis_path}")
    
    # Visualization 3: Side-by-side comparison
    h, w = image.shape[:2]
    comparison = np.hstack([vis_corners, vis_reprojection])
    vis_path = os.path.join(output_dir, "03_full_comparison.png")
    cv2.imwrite(vis_path, comparison)
    print(f"[SAVED] {vis_path}")
    
    print("\n" + "="*70)
    print("CALIBRATION COMPLETE ✓")
    print("="*70)
    print(f"\nNext steps:")
    print(f"1. Review the visualizations to verify corner detection")
    print(f"2. Check that reprojection error is < 1.0 pixel")
    print(f"3. Verify measured camera position matches ground truth")
    print(f"4. If satisfied, copy camera_extrinsics.npz to your project")
    print(f"5. Modify pose estimation script to use this transformation")
    
    return True


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calibrate camera extrinsics relative to gantry using checkerboard"
    )
    
    parser.add_argument("--image", default="calib-image.png",
                       help="Path to checkerboard image (default: calib-image.png)")
    parser.add_argument("--calib", default="camera_params_fisheye_aruko.npz",
                       help="Path to camera calibration .npz file")
    parser.add_argument("--output", default=".",
                       help="Directory to save calibration results")
    
    # Ground truth for validation
    parser.add_argument("--gt-x", type=float, default=-0.020,
                       help="Ground truth camera X position (m)")
    parser.add_argument("--gt-y", type=float, default=-0.285,
                       help="Ground truth camera Y position (m)")
    parser.add_argument("--gt-z", type=float, default=1.855,
                       help="Ground truth camera Z position (m)")
    
    parser.add_argument("--square-size", type=float, default=20.0,
                       help="Size of each checkerboard square in mm (default: 20.0)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    success = run_calibration(
        image_path=args.image,
        calib_file=args.calib,
        square_size_mm=args.square_size,
        ground_truth_camera_pos=(args.gt_x, args.gt_y, args.gt_z),
        output_dir=args.output,
    )
    
    sys.exit(0 if success else 1)