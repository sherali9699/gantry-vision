"""
ChArUco fisheye camera calibration.

Workflow:
  1. Capture 20-30 images of the board with your camera (press C in ir_feed.py)
  2. Run this script to get K and distortion coefficients
  3. Use new_K in your extrinsic calibration

Run from ~/gantry-vision:
  python charuco_calibration.py
"""

import cv2
import numpy as np
import glob
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG — edit before running
# ─────────────────────────────────────────────

# Board geometry — checkerboard inner corners
CORNERS_X     = 5        # inner corners horizontally (squares - 1)
CORNERS_Y     = 3        # inner corners vertically (squares - 1)
SQUARE_SIZE_M = 0.0475
CHECKERBOARD  = (CORNERS_X, CORNERS_Y)
# Remove MARKER_SIZE_M and ARUCO_DICT lines

# Images captured with ir_feed.py (press C) — raw distorted, no IR filter
CALIB_IMAGES_GLOB = "calibration_images_clean/calib_raw_*.png"

# Output files
OUTPUT_PATH        = "new_fisheye_calibration.npy"
VIS_OUTPUT_DIR     = "calibration_images_distorted/detections"

# 'fisheye' for wide-angle/fisheye lenses, 'pinhole' for normal lenses
CAMERA_MODEL = 'fisheye'


# ─────────────────────────────────────────────
# STEP 1: DETECT CHARUCO CORNERS IN IMAGES
# ─────────────────────────────────────────────

def preprocess_for_detection(img):
    """Input already IR-corrected — just enhance contrast."""
    img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    
def detect_corners_in_images(image_paths, visualise=True):
    all_corners = []
    all_obj_pts = []
    image_size  = None
    used_paths  = []

    # 3D object points — same for every image
    obj_pts = np.zeros((CORNERS_X * CORNERS_Y, 3), np.float32)
    obj_pts[:, :2] = np.mgrid[0:CORNERS_X, 0:CORNERS_Y].T.reshape(-1, 2)
    obj_pts *= SQUARE_SIZE_M

    if visualise:
        Path(VIS_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
             cv2.CALIB_CB_NORMALIZE_IMAGE |
             cv2.CALIB_CB_FAST_CHECK)

    print(f"\nProcessing {len(image_paths)} images …")
    for idx, p in enumerate(image_paths):
        img = cv2.imread(str(p))
        if img is None:
            print(f"  [{idx:02d}] {Path(p).name}: cannot load — skipped")
            continue

        if image_size is None:
            image_size = img.shape[1], img.shape[0]

        img_proc = preprocess_for_detection(img)
        gray     = cv2.cvtColor(img_proc, cv2.COLOR_BGR2GRAY)

        ok, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)

        if not ok:
            print(f"  [{idx:02d}] {Path(p).name}: not found — skipped")
            continue

        # Subpixel refinement
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners  = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        all_corners.append(corners)
        all_obj_pts.append(obj_pts)
        used_paths.append(p)
        print(f"  [{idx:02d}] {Path(p).name}: {CORNERS_X*CORNERS_Y} corners ✓")

        if visualise:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, CHECKERBOARD, corners, ok)
            cv2.imwrite(str(Path(VIS_OUTPUT_DIR) / f"detect_{idx:02d}.png"), vis)

    print(f"\nUsable images: {len(all_corners)} / {len(image_paths)}")
    return all_corners, all_obj_pts, image_size, used_paths


# ─────────────────────────────────────────────
# STEP 2A: PINHOLE CALIBRATION
# ─────────────────────────────────────────────

def calibrate_pinhole(all_corners, all_obj_pts, image_size):
    print("\nRunning pinhole calibration …")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        all_obj_pts, all_corners, image_size, None, None
    )
    print(f"  RMS : {rms:.4f} px  (target < 1.0)")
    print(f"  K:\n{K}")
    print(f"  dist: {dist.ravel()}")
    return K, dist, rms


# ─────────────────────────────────────────────
# STEP 2B: FISHEYE CALIBRATION
# ─────────────────────────────────────────────

def calibrate_fisheye(all_corners, all_obj_pts, image_size):
    print("\nRunning fisheye calibration …")

    obj_pts_f = [o.reshape(-1, 1, 3) for o in all_obj_pts]
    img_pts_f = [c.reshape(-1, 1, 2) for c in all_corners]

    K     = np.zeros((3, 3))
    D     = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(obj_pts_f))]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(obj_pts_f))]

    flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
             cv2.fisheye.CALIB_FIX_SKEW |
             cv2.fisheye.CALIB_CHECK_COND)

    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        obj_pts_f, img_pts_f, image_size,
        K, D, rvecs, tvecs, flags,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-6)
    )

    print(f"  RMS : {rms:.4f} px  (target < 1.0)")
    print(f"  K:\n{K}")
    print(f"  D: {D.ravel()}")
    return K, D, rms


# ─────────────────────────────────────────────
# STEP 3: SAVE + UNDISTORT DEMO
# ─────────────────────────────────────────────

def save_calibration(K, dist, rms, image_size, model):
    cal = {
        'K':             K,
        'dist':          dist,
        'rms':           rms,
        'image_size':    image_size,
        'model':         model,
        'square_size_m': SQUARE_SIZE_M,
    }
    np.save(OUTPUT_PATH, cal, allow_pickle=True)
    print(f"\nCalibration saved → {OUTPUT_PATH}")


def demo_undistort(image_path, K, dist, image_size, model):
    """Apply undistortion to one image and show before/after."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Cannot load {image_path} for demo")
        return None

    if model == 'fisheye':
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, dist, image_size, np.eye(3), balance=1.0
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, dist, np.eye(3), new_K, image_size, cv2.CV_16SC2
        )
    else:
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, image_size, 0)
        map1, map2 = cv2.initUndistortRectifyMap(
            K, dist, None, new_K, image_size, cv2.CV_16SC2
        )

    undistorted = cv2.remap(img, map1, map2,
                             interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT)

    cv2.imwrite("undistort_before.png", img)
    cv2.imwrite("undistort_after.png", undistorted)

    print("\nUndistortion demo saved:")
    print("  undistort_before.png")
    print("  undistort_after.png")
    print(f"\n  new_K (use this as K in your extrinsic calibration):")
    print(f"  {new_K.tolist()}")

    return new_K


# ─────────────────────────────────────────────
# BATCH UNDISTORT UTILITY
# ─────────────────────────────────────────────

def undistort_all(src_glob, output_dir, K, dist, image_size, model):
    """
    Optional: undistort all images matching src_glob and save to output_dir.
    Call this after calibration to regenerate your test dataset.
    """
    paths = sorted(glob.glob(src_glob))
    if not paths:
        print(f"No images found: {src_glob}")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if model == 'fisheye':
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, dist, image_size, np.eye(3), balance=1.0
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, dist, np.eye(3), new_K, image_size, cv2.CV_16SC2
        )
    else:
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, image_size, 0)
        map1, map2 = cv2.initUndistortRectifyMap(
            K, dist, None, new_K, image_size, cv2.CV_16SC2
        )

    print(f"\nUndistorting {len(paths)} images → {output_dir}/")
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        out = cv2.remap(img, map1, map2,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT)
        out_path = str(Path(output_dir) / Path(p).name)
        cv2.imwrite(out_path, out)
    print("  Done.")
    return new_K


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 65)
    print("ChArUco Camera Calibration")
    print(f"  Board   : {CORNERS_X+1}x{CORNERS_Y+1} squares ({CORNERS_X}x{CORNERS_Y} inner corners), {SQUARE_SIZE_M*1000:.1f}mm each")    
    print(f"  Model   : {CAMERA_MODEL}")
    print(f"  Images  : {CALIB_IMAGES_GLOB}")
    print("=" * 65)

    # Detect corners
    image_paths = sorted(glob.glob(CALIB_IMAGES_GLOB))
    assert len(image_paths) > 0, (
        f"No images found at '{CALIB_IMAGES_GLOB}'\n"
        f"Make sure you captured images with ir_feed.py (press C) and the path is correct."
    )
    print(f"Found {len(image_paths)} images.")

    assert len(image_paths) >= 10, (
        f"Need at least 10 images for reliable calibration, found {len(image_paths)}.\n"
        f"Capture more images at varied angles."
    )


    all_corners, all_obj_pts, image_size, used = detect_corners_in_images(
        image_paths, visualise=True
    )

    assert len(all_corners) >= 10, (
        f"Only {len(all_corners)} usable images after detection.\n"
        f"Make sure the board is well-lit and fully visible in more images."
    )

    if CAMERA_MODEL == 'fisheye':
        K, dist, rms = calibrate_fisheye(all_corners, all_obj_pts, image_size)
    else:
        K, dist, rms = calibrate_pinhole(all_corners, all_obj_pts, image_size)
    # Save
    save_calibration(K, dist, rms, image_size, CAMERA_MODEL)

    # Demo undistort on first used image
    if used:
        new_K = demo_undistort(used[0], K, dist, image_size, CAMERA_MODEL)

    # ── Optional: batch undistort your working images ──────────
    # Uncomment and set paths to regenerate your test dataset with new calibration:
    #
    # new_K = undistort_all(
    #     src_glob   = "raw_test_images/*.png",
    #     output_dir = "undistorted_test_images",
    #     K=K, dist=dist, image_size=image_size, model=CAMERA_MODEL
    # )

    print("\n" + "=" * 65)
    print("DONE")
    print("=" * 65)
    print("\nNext steps:")
    print("  1. Check undistort_before.png vs undistort_after.png")
    print("  2. Copy the new_K printed above into Block 1 of your pipeline")
    print("  3. Undistort all your working images using the batch utility above")
    print("  4. Re-run extrinsic calibration with the undistorted images")


if __name__ == "__main__":
    main()