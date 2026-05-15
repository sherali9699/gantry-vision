#!/usr/bin/env python3

"""
Real-time Microwave Pose Estimation

Pipeline:
1. Read frame from IR camera
2. Apply IR pink-cast correction
3. Run YOLO top-4 microwave keypoint model
4. Use solvePnP with microwave top-face geometry
5. Draw keypoints, bbox, XYZ axes, reprojection error, and distance
6. Show live OpenCV window

Controls:
Q / ESC  = quit
S        = save current frame
+ / -    = adjust IR correction strength
O        = toggle original/corrected side-by-side
P        = toggle pose inference on/off
"""

import cv2
import numpy as np
import argparse
import sys
import os
from datetime import datetime
from ultralytics import YOLO


# ============================================================
# IR Pink Correction
# ============================================================

def correct_ir_cast(frame: np.ndarray, strength: float = 1.0) -> np.ndarray:
    frame = frame.astype(np.float32)

    b, g, r = cv2.split(frame)

    r_corrected = r * (1.0 - 0.30 * strength)
    g_corrected = g * (1.0 + 0.05 * strength)
    b_corrected = b * (1.0 - 0.15 * strength)

    frame = cv2.merge([
        np.clip(b_corrected, 0, 255),
        np.clip(g_corrected, 0, 255),
        np.clip(r_corrected, 0, 255),
    ]).astype(np.uint8)

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b_ch = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    a = cv2.addWeighted(
        a,
        1.0 - 0.25 * strength,
        np.full_like(a, 128),
        0.25 * strength,
        0
    )

    lab = cv2.merge((l, a, b_ch))
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return frame


# ============================================================
# Camera Setup
# ============================================================

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

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"[INFO] Camera device: {device}")
    print(f"[INFO] Resolution: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")

    return cap


# ============================================================
# Pose Estimation Helper
# ============================================================

def estimate_pose_and_draw(
    frame,
    model,
    camera_matrix,
    dist_coeffs,
    mw_width,
    mw_depth,
    conf_thresh=0.30,
    yolo_conf=0.25,
    imgsz=640
):
    """
    Runs YOLO top-4 keypoint inference and PnP.
    Returns annotated frame and info dictionary.
    """

    vis = frame.copy()

    results = model.predict(
        source=frame,
        imgsz=imgsz,
        conf=yolo_conf,
        verbose=False
    )

    res = results[0]

    if res.keypoints is None or len(res.keypoints.xy) == 0:
        cv2.putText(
            vis,
            "No microwave detected",
            (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3
        )
        return vis, None

    if res.boxes is None or len(res.boxes.xyxy) == 0:
        cv2.putText(
            vis,
            "No bbox detected",
            (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3
        )
        return vis, None

    # Use first detected microwave
    kpts_xy = res.keypoints.xy[0].cpu().numpy().astype(np.float32)
    kpts_conf = res.keypoints.conf[0].cpu().numpy().astype(np.float32)
    box = res.boxes.xyxy[0].cpu().numpy().astype(np.float32)

    valid_idx = np.where(kpts_conf > conf_thresh)[0]

    # Draw bbox
    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)

    # Draw keypoints
    for i, (x, y) in enumerate(kpts_xy):
        color = (0, 255, 255) if kpts_conf[i] > conf_thresh else (0, 0, 255)

        cv2.circle(vis, (int(x), int(y)), 7, color, -1)
        cv2.putText(
            vis,
            f"P{i}:{kpts_conf[i]:.2f}",
            (int(x) + 5, int(y) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

    if len(valid_idx) < 4:
        cv2.putText(
            vis,
            f"Need 4 keypoints, got {len(valid_idx)}",
            (40, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            3
        )
        return vis, None

    # Top-4 points:
    # 0 = top_front_left
    # 1 = top_front_right
    # 2 = top_back_left
    # 3 = top_back_right
    image_points = kpts_xy.astype(np.float32)

    W = mw_width
    D = mw_depth

    object_points = np.array([
        [0, 0, 0],   # top_front_left
        [W, 0, 0],   # top_front_right
        [0, D, 0],   # top_back_left
        [W, D, 0],   # top_back_right
    ], dtype=np.float32)

    # Planar PnP
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE
    )

    if not success:
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

    if not success:
        cv2.putText(
            vis,
            "solvePnP failed",
            (40, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            3
        )
        return vis, None

    # Reprojection error
    projected_points, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs
    )

    projected_points = projected_points.reshape(-1, 2)

    errors = np.linalg.norm(projected_points - image_points, axis=1)
    mean_error = float(np.mean(errors))

    # Distance to origin
    distance_origin_m = float(np.linalg.norm(tvec))

    # Distance to top-face center
    R, _ = cv2.Rodrigues(rvec)

    center_obj = np.array([[W / 2], [D / 2], [0]], dtype=np.float32)
    center_cam = R @ center_obj + tvec
    distance_center_m = float(np.linalg.norm(center_cam))

    # Draw reprojected points
    for i, (x, y) in enumerate(projected_points):
        if np.isfinite(x) and np.isfinite(y):
            cv2.circle(vis, (int(x), int(y)), 5, (255, 255, 255), -1)
            cv2.putText(
                vis,
                f"R{i}",
                (int(x) + 5, int(y) + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

    # Draw XYZ axes
    axis_len = 0.20  # 20 cm

    axis_3d = np.array([
        [0, 0, 0],
        [axis_len, 0, 0],
        [0, axis_len, 0],
        [0, 0, -axis_len],
    ], dtype=np.float32)

    axis_2d, _ = cv2.projectPoints(
        axis_3d,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs
    )

    axis_2d = axis_2d.reshape(-1, 2)

    origin = tuple(axis_2d[0].astype(int))
    x_axis = tuple(axis_2d[1].astype(int))
    y_axis = tuple(axis_2d[2].astype(int))
    z_axis = tuple(axis_2d[3].astype(int))

    # X = red, Y = green, Z = blue
    cv2.arrowedLine(vis, origin, x_axis, (0, 0, 255), 4, tipLength=0.15)
    cv2.arrowedLine(vis, origin, y_axis, (0, 255, 0), 4, tipLength=0.15)
    cv2.arrowedLine(vis, origin, z_axis, (255, 0, 0), 4, tipLength=0.15)

    cv2.putText(vis, "X", x_axis, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    cv2.putText(vis, "Y", y_axis, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    cv2.putText(vis, "Z", z_axis, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)

    # Text overlay
    cv2.putText(
        vis,
        f"Origin Dist: {distance_origin_m:.2f} m",
        (40, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        3
    )

    cv2.putText(
        vis,
        f"Center Dist: {distance_center_m:.2f} m",
        (40, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        3
    )

    cv2.putText(
        vis,
        f"Reproj Err: {mean_error:.2f} px",
        (40, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        3
    )

    info = {
        "rvec": rvec,
        "tvec": tvec,
        "distance_origin_m": distance_origin_m,
        "distance_center_m": distance_center_m,
        "mean_error": mean_error,
        "keypoints": kpts_xy,
        "keypoint_conf": kpts_conf,
        "box": box,
    }

    return vis, info


# ============================================================
# Main Loop
# ============================================================

def run(
    device,
    width,
    height,
    strength,
    show_original,
    model_path,
    calib_file,
    mw_width,
    mw_depth,
    conf_thresh,
    yolo_conf,
    imgsz
):
    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print("[INFO] Loading YOLO model...")
    model = YOLO(model_path)
    print("[INFO] YOLO model loaded.")

    # Load camera calibration
    if not os.path.exists(calib_file):
        raise FileNotFoundError(f"Calibration file not found: {calib_file}")

    data = np.load(calib_file)
    camera_matrix = data["mtx"]
    dist_coeffs = data["dist"]

    print("[INFO] Camera calibration loaded.")
    print("Camera matrix:")
    print(camera_matrix)
    print("Distortion coefficients:")
    print(dist_coeffs)

    cap = open_camera(device, width, height)

    output_dir = "realtime_pose_captures_version02"
    os.makedirs(output_dir, exist_ok=True)

    win_name = "Realtime Microwave Pose — Q quit | S save | P pose on/off | O original | +/- strength"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    pose_enabled = True

    print("\n[CONTROLS]")
    print("  Q / ESC    — quit")
    print("  S          — save current display frame")
    print("  P          — toggle pose inference on/off")
    print("  O          — toggle original/corrected side-by-side")
    print("  +/-        — adjust IR correction strength")
    print(f"\n[INFO] Saving frames to: {os.path.abspath(output_dir)}")

    frame_id = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("[WARN] Frame grab failed")
            continue

        corrected = correct_ir_cast(frame, strength=strength)

        if pose_enabled:
            pose_view, info = estimate_pose_and_draw(
                corrected,
                model,
                camera_matrix,
                dist_coeffs,
                mw_width,
                mw_depth,
                conf_thresh=conf_thresh,
                yolo_conf=yolo_conf,
                imgsz=imgsz
            )
        else:
            pose_view = corrected.copy()
            info = None

        # Add status text
        cv2.putText(
            pose_view,
            f"Pose: {'ON' if pose_enabled else 'OFF'} | STR: {strength:.2f}",
            (40, pose_view.shape[0] - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
        )

        if show_original:
            label_orig = frame.copy()
            label_pose = pose_view.copy()

            cv2.putText(
                label_orig,
                "ORIGINAL",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )

            cv2.putText(
                label_pose,
                "CORRECTED + POSE",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Resize if heights mismatch
            if label_orig.shape[0] != label_pose.shape[0]:
                label_orig = cv2.resize(label_orig, (label_pose.shape[1], label_pose.shape[0]))

            display = np.hstack([label_orig, label_pose])
        else:
            display = pose_view

        cv2.imshow(win_name, display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

        elif key == ord("+") or key == ord("="):
            strength = min(strength + 0.05, 2.0)
            print(f"[INFO] Strength: {strength:.2f}")

        elif key == ord("-"):
            strength = max(strength - 0.05, 0.0)
            print(f"[INFO] Strength: {strength:.2f}")

        elif key == ord("o"):
            show_original = not show_original
            print(f"[INFO] show_original: {show_original}")

        elif key == ord("p"):
            pose_enabled = not pose_enabled
            print(f"[INFO] pose_enabled: {pose_enabled}")

        elif key == ord("s"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            fname = os.path.join(output_dir, f"pose_frame_{timestamp}.png")
            cv2.imwrite(fname, display)
            print(f"[SAVED] {fname}")

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Realtime microwave pose estimation with YOLO top-4 keypoints + PnP")

    parser.add_argument("--device", default="/dev/video2")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)

    parser.add_argument("--model", default="best_26n_extended .pt")
    parser.add_argument("--calib", default="camera_params_fisheye_aruko.npz")

    parser.add_argument("--mw-width", type=float, default=0.44, help="Microwave top-face width in meters")
    parser.add_argument("--mw-depth", type=float, default=0.61, help="Microwave top-face depth in meters")

    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--side-by-side", action="store_true")

    parser.add_argument("--conf-thresh", type=float, default=0.30, help="Keypoint confidence threshold")
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO detection confidence")
    parser.add_argument("--imgsz", type=int, default=640)

    args = parser.parse_args()

    run(
        device=args.device,
        width=args.width,
        height=args.height,
        strength=args.strength,
        show_original=args.side_by_side,
        model_path=args.model,
        calib_file=args.calib,
        mw_width=args.mw_width,
        mw_depth=args.mw_depth,
        conf_thresh=args.conf_thresh,
        yolo_conf=args.yolo_conf,
        imgsz=args.imgsz
    )