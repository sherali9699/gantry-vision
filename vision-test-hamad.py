#!/usr/bin/env python3
#(hamad)
"""
IR Camera Live Feed — removes washed-out pink cast from cameras without IR filters.
Includes frame capture functionality with timestamped filenames.
Applies undistortion using camera_params_fisheye.npz (pinhole model: mtx, dist).
"""

import cv2
import numpy as np
import argparse
import sys
import os
from datetime import datetime


# ──────────────────────────────────────────────
# Undistortion Setup (pinhole model)
# ──────────────────────────────────────────────
def load_fisheye_maps(npz_path: str, width: int, height: int):
    """
    Load calibration params and precompute undistortion maps.
    Expects keys: mtx (3x3), dist (1x5).
    Uses pinhole model — cv2.getOptimalNewCameraMatrix + initUndistortRectifyMap.
    """
    if not os.path.exists(npz_path):
        print(f"[ERROR] Calibration file not found: {npz_path}")
        sys.exit(1)

    data = np.load(npz_path)
    K = data["mtx"]
    D = data["dist"]

    print(f"[INFO] Loaded calibration params from: {npz_path}")
    print(f"[INFO] K =\n{K}")
    print(f"[INFO] D = {D.ravel()}")
    print(f"[INFO] D shape = {D.shape}  → using pinhole undistortion")

    # alpha=0.0 → tight crop, no black borders (best for YOLO training images)
    # alpha=1.0 → keep all pixels, black borders appear
    new_K, roi = cv2.getOptimalNewCameraMatrix(
        K, D, (width, height), alpha=0.5, newImgSize=(width, height)
    )

    map1, map2 = cv2.initUndistortRectifyMap(
        K, D, None, new_K, (width, height), cv2.CV_16SC2
    )

    print(f"[INFO] ROI after undistort: {roi}")
    return map1, map2


def undistort_frame(frame: np.ndarray, map1, map2) -> np.ndarray:
    return cv2.remap(frame, map1, map2,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT,
                     borderValue=(0, 0, 0))


# ──────────────────────────────────────────────
# IR Pink Correction
# ──────────────────────────────────────────────
def correct_ir_cast(frame: np.ndarray, strength: float = 1.0) -> np.ndarray:
    frame = frame.astype(np.float32)

    r, g, b = cv2.split(frame)
    r_corrected = r * (1.0 - 0.30 * strength)
    g_corrected = g * (1.0 + 0.05 * strength)
    b_corrected = b * (1.0 - 0.15 * strength)

    frame = cv2.merge([
        np.clip(r_corrected, 0, 255),
        np.clip(g_corrected, 0, 255),
        np.clip(b_corrected, 0, 255),
    ]).astype(np.uint8)

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b_ch = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    a = cv2.addWeighted(a, 1.0 - 0.25 * strength,
                        np.full_like(a, 128), 0.25 * strength, 0)

    lab = cv2.merge((l, a, b_ch))
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return frame


# ──────────────────────────────────────────────
# Camera Setup
# ──────────────────────────────────────────────
def open_camera(device: str, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)

    if not cap.isOpened():
        print(f"[ERROR] Could not open {device}")
        sys.exit(1)

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap


# ──────────────────────────────────────────────
# Main Loop
# ──────────────────────────────────────────────
def run(device: str, width: int, height: int, strength: float,
        show_original: bool, npz_path: str, no_undistort: bool):

    if not no_undistort:
        map1, map2 = load_fisheye_maps(npz_path, width, height)
    else:
        map1, map2 = None, None
        print("[INFO] Undistortion disabled.")

    cap = open_camera(device, width, height)

    output_dir = "yolo-train-images"
    os.makedirs(output_dir, exist_ok=True)

    win_name = "IR Feed — Q: Quit | S: Capture | O: Toggle Original | U: Toggle Undistort | +/-: Strength"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    undistort_enabled = not no_undistort

    print("\n[CONTROLS]")
    print("  Q / ESC    — quit")
    print("  S          — capture current corrected frame")
    print("  +/-        — adjust IR correction strength")
    print("  O          — toggle side-by-side original")
    print("  U          — toggle undistortion on/off")
    print(f"\n[INFO] Saving frames to: {os.path.abspath(output_dir)}")
    print(f"[INFO] Undistortion: {'ON' if undistort_enabled else 'OFF'}")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Step 1: undistortion (geometric, on raw frame)
        if undistort_enabled and map1 is not None:
            frame = undistort_frame(frame, map1, map2)

        # Step 2: IR pink correction
        corrected = correct_ir_cast(frame, strength=strength)

        if show_original:
            label_orig = frame.copy()
            label_corr = corrected.copy()
            cv2.putText(label_orig, "ORIGINAL", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(label_corr, f"CORRECTED str={strength:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            display = np.hstack([label_orig, label_corr])
        else:
            display = corrected.copy()
            undist_label = "UNDIST:ON" if undistort_enabled else "UNDIST:OFF"
            cv2.putText(display, f"STR:{strength:.2f}  {undist_label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(win_name, display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord('+') or key == ord('='):
            strength = min(strength + 0.05, 2.0)
        elif key == ord('-'):
            strength = max(strength - 0.05, 0.0)
        elif key == ord('o'):
            show_original = not show_original
        elif key == ord('u'):
            if map1 is not None:
                undistort_enabled = not undistort_enabled
                print(f"[INFO] Undistortion: {'ON' if undistort_enabled else 'OFF'}")
            else:
                print("[WARN] No calibration loaded — run without --no-undistort")
        elif key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            fname = os.path.join(output_dir, f"ir_frame_{timestamp}.png")
            cv2.imwrite(fname, corrected)
            print(f"[SAVED] {fname}")
        elif key == ord('c'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            fname = os.path.join(output_dir, f"calib_raw_{timestamp}.png")
            # Save undistorted but NO IR correction applied
            cv2.imwrite(fname, frame)
            print(f"[SAVED RAW] {fname}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IR camera feed with pink-cast correction and undistortion")
    parser.add_argument("--device",       default="/dev/video2")
    parser.add_argument("--width",        type=int, default=1920)
    parser.add_argument("--height",       type=int, default=1080)
    parser.add_argument("--strength",     type=float, default=1.0)
    parser.add_argument("--side-by-side", action="store_true")
    parser.add_argument("--npz",          default="camera_params_fisheye.npz",
                        help="Path to calibration .npz file (keys: mtx, dist)")
    parser.add_argument("--no-undistort", action="store_true",
                        help="Disable undistortion (pass-through mode)")
    args = parser.parse_args()

    run(args.device, args.width, args.height, args.strength,
        args.side_by_side, args.npz, args.no_undistort)