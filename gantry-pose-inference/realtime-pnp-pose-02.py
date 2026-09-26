#!/usr/bin/env python3

"""
Real-time Microwave + Package Pose Estimation

Pipeline:
1. Read frame from IR camera
2. Apply IR pink-cast correction
3. Run YOLO top-4 keypoint model (2 classes)
4. Use solvePnP with per-class geometry
5. Draw keypoints, bbox, XYZ axes, per-axis distances, Euler angles,
   reprojection error, and Euclidean distance
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
    a = cv2.addWeighted(a, 1.0 - 0.25 * strength,
                        np.full_like(a, 128), 0.25 * strength, 0)
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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Camera device   : {device}")
    print(f"[INFO] Resolution      : {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")
    return cap


# ============================================================
# Per-class 3D geometry
# ============================================================

def build_class_dims(mw_width, mw_depth, pkg_width, pkg_depth):
    """
    Returns a dict keyed by class_id with object_points + metadata.
    Keypoint order for BOTH classes:
        0 = top_front_left
        1 = top_front_right
        2 = top_back_left
        3 = top_back_right
    All on the top face → Z = 0 (planar PnP).
    """
    return {
        0: {
            "name":          "microwave",
            "color":         (0, 255, 255),   # cyan
            "W":             mw_width,
            "D":             mw_depth,
            "object_points": np.array([
                [0,        0,        0],
                [mw_width, 0,        0],
                [0,        mw_depth, 0],
                [mw_width, mw_depth, 0],
            ], dtype=np.float32),
        },
        1: {
            "name":          "package",
            "color":         (0, 165, 255),   # orange
            "W":             pkg_width,
            "D":             pkg_depth,
            "object_points": np.array([
                [0,         0,         0],
                [pkg_width, 0,         0],
                [0,         pkg_depth, 0],
                [pkg_width, pkg_depth, 0],
            ], dtype=np.float32),
        },
    }


# ============================================================
# Euler Angles from Rotation Matrix  (ZYX convention)
# ============================================================

def rotation_matrix_to_euler(R):
    """
    Decomposes a 3x3 rotation matrix into Roll, Pitch, Yaw (degrees).
    Convention: ZYX (yaw → pitch → roll).
    Returns (roll, pitch, yaw) in degrees.
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
# Pose Estimation + Drawing
# ============================================================

def estimate_pose_and_draw(
    frame,
    model,
    camera_matrix,
    dist_coeffs,
    class_dims,
    conf_thresh=0.30,
    yolo_conf=0.25,
    imgsz=640
):
    vis = frame.copy()

    results = model.predict(source=frame, imgsz=imgsz, conf=yolo_conf, verbose=False)
    res     = results[0]

    if res.keypoints is None or len(res.keypoints.xy) == 0:
        cv2.putText(vis, "No objects detected", (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        return vis, []

    all_info = []

    for det_idx in range(len(res.boxes)):

        class_id  = int(res.boxes.cls[det_idx].item())
        det_conf  = float(res.boxes.conf[det_idx].item())
        kpts_xy   = res.keypoints.xy[det_idx].cpu().numpy().astype(np.float32)
        kpts_conf = res.keypoints.conf[det_idx].cpu().numpy().astype(np.float32)
        box       = res.boxes.xyxy[det_idx].cpu().numpy().astype(np.float32)

        # ── Skip unknown classes ──────────────────────────────────────────────
        if class_id not in class_dims:
            continue

        dims     = class_dims[class_id]
        obj_name = dims["name"]
        color    = dims["color"]
        obj_pts  = dims["object_points"]
        W, D     = dims["W"], dims["D"]

        # ── Bounding box ──────────────────────────────────────────────────────
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f"{obj_name} {det_conf:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        # ── Keypoints ─────────────────────────────────────────────────────────
        KP_NAMES  = ["TFL", "TFR", "TBL", "TBR"]
        valid_idx = np.where(kpts_conf > conf_thresh)[0]

        for i, (x, y) in enumerate(kpts_xy):
            kp_color = color if kpts_conf[i] > conf_thresh else (0, 0, 255)
            cv2.circle(vis, (int(x), int(y)), 7, kp_color, -1)
            cv2.putText(vis,
                        f"{KP_NAMES[i]}:{kpts_conf[i]:.2f}",
                        (int(x) + 5, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, kp_color, 2)

        if len(valid_idx) < 4:
            cv2.putText(vis,
                        f"{obj_name}: need 4 kpts, got {len(valid_idx)}",
                        (x1, y2 + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            continue

        image_points = kpts_xy.astype(np.float32)

        # ── solvePnP ──────────────────────────────────────────────────────────
        success, rvec, tvec = cv2.solvePnP(
            obj_pts, image_points,
            camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE
        )
        if not success:
            success, rvec, tvec = cv2.solvePnP(
                obj_pts, image_points,
                camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        if not success:
            cv2.putText(vis, f"{obj_name}: solvePnP failed",
                        (x1, y2 + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            continue

        # ── Reprojection error ────────────────────────────────────────────────
        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
        proj     = proj.reshape(-1, 2)
        errors   = np.linalg.norm(proj - image_points, axis=1)
        mean_err = float(np.mean(errors))

        # ── Translation components (cm) ───────────────────────────────────────
        tx_cm = float(tvec[0]) * 100   # left(-) / right(+)
        ty_cm = float(tvec[1]) * 100   # up(-)   / down(+)
        tz_cm = float(tvec[2]) * 100   # depth (toward camera = positive)

        # ── Euclidean distances ───────────────────────────────────────────────
        dist_origin = float(np.linalg.norm(tvec))
        R, _        = cv2.Rodrigues(rvec)
        center_obj  = np.array([[W / 2], [D / 2], [0]], dtype=np.float32)
        center_cam  = R @ center_obj + tvec
        dist_center = float(np.linalg.norm(center_cam))

        # ── Euler angles (ZYX convention) ─────────────────────────────────────
        roll, pitch, yaw = rotation_matrix_to_euler(R)

        # ── Reprojected points (white) ────────────────────────────────────────
        for i, (x, y) in enumerate(proj):
            if np.isfinite(x) and np.isfinite(y):
                cv2.circle(vis, (int(x), int(y)), 5, (255, 255, 255), -1)

        # ── XYZ axes ──────────────────────────────────────────────────────────
        axis_len = 0.20
        axis_3d  = np.array([
            [0, 0, 0],
            [axis_len, 0, 0],
            [0, axis_len, 0],
            [0, 0, -axis_len],
        ], dtype=np.float32)
        ax2d, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)
        ax2d    = ax2d.reshape(-1, 2)

        origin = tuple(ax2d[0].astype(int))
        x_pt   = tuple(ax2d[1].astype(int))
        y_pt   = tuple(ax2d[2].astype(int))
        z_pt   = tuple(ax2d[3].astype(int))

        cv2.arrowedLine(vis, origin, x_pt, (0, 0, 255),   4, tipLength=0.15)
        cv2.arrowedLine(vis, origin, y_pt, (0, 255, 0),   4, tipLength=0.15)
        cv2.arrowedLine(vis, origin, z_pt, (255, 0, 0),   4, tipLength=0.15)
        cv2.putText(vis, "X", x_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(vis, "Y", y_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(vis, "Z", z_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # ── HUD text (anchored near bbox) ─────────────────────────────────────
        hud = [
            f"[{obj_name}]  conf={det_conf:.2f}",
            f"Euclidean : {dist_center*100:.1f} cm",
            f"X         : {tx_cm:+.1f} cm   (L- / R+)",
            f"Y         : {ty_cm:+.1f} cm   (U- / D+)",
            f"Z (depth) : {tz_cm:+.1f} cm",
            f"Roll      : {roll:+.1f} deg",
            f"Pitch     : {pitch:+.1f} deg",
            f"Yaw       : {yaw:+.1f} deg",
            f"RpjErr    : {mean_err:.1f} px",
        ]
        line_h = 28
        for i, line in enumerate(hud):
            y_pos = y1 - 20 - (len(hud) - 1 - i) * line_h
            y_pos = max(y_pos, 20 + i * line_h)   # clamp to frame top
            # dark shadow for readability, then colored text
            cv2.putText(vis, line, (x1, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 0), 4)
            cv2.putText(vis, line, (x1, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2)

        all_info.append({
            "class_id":           class_id,
            "name":               obj_name,
            "rvec":               rvec,
            "tvec":               tvec,
            "tx_cm":              tx_cm,
            "ty_cm":              ty_cm,
            "tz_cm":              tz_cm,
            "roll_deg":           roll,
            "pitch_deg":          pitch,
            "yaw_deg":            yaw,
            "distance_origin_m":  dist_origin,
            "distance_center_m":  dist_center,
            "mean_error_px":      mean_err,
            "keypoints":          kpts_xy,
            "keypoint_conf":      kpts_conf,
            "box":                box,
        })

    return vis, all_info


# ============================================================
# Main Loop
# ============================================================

def run(
    device, width, height, strength, show_original,
    model_path, calib_file,
    mw_width, mw_depth,
    pkg_width, pkg_depth,
    conf_thresh, yolo_conf, imgsz
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print("[INFO] Loading YOLO model...")
    model = YOLO(model_path)
    print("[INFO] YOLO model loaded.")

    if not os.path.exists(calib_file):
        raise FileNotFoundError(f"Calibration file not found: {calib_file}")

    data          = np.load(calib_file)
    camera_matrix = data["mtx"]
    dist_coeffs   = data["dist"]
    print("[INFO] Calibration loaded.")

    class_dims = build_class_dims(mw_width, mw_depth, pkg_width, pkg_depth)
    print(f"[INFO] Microwave : {mw_width} m  x  {mw_depth} m")
    print(f"[INFO] Package   : {pkg_width} m  x  {pkg_depth} m")

    cap        = open_camera(device, width, height)
    output_dir = "realtime_pose_captures"
    os.makedirs(output_dir, exist_ok=True)

    win_name = (
        "Pose Estimation  —  Q quit | S save | P pose on/off | O side-by-side | +/- strength"
    )
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    pose_enabled = True
    frame_id     = 0

    print("\n[CONTROLS]")
    print("  Q / ESC  — quit")
    print("  S        — save frame")
    print("  P        — toggle pose on/off")
    print("  O        — toggle side-by-side original")
    print("  + / -    — IR correction strength")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed")
            continue

        corrected = correct_ir_cast(frame, strength=strength)

        if pose_enabled:
            pose_view, info_list = estimate_pose_and_draw(
                corrected, model,
                camera_matrix, dist_coeffs,
                class_dims,
                conf_thresh=conf_thresh,
                yolo_conf=yolo_conf,
                imgsz=imgsz,
            )
        else:
            pose_view = corrected.copy()
            info_list = []

        # ── Status bar ────────────────────────────────────────────────────────
        cv2.putText(
            pose_view,
            f"Pose: {'ON' if pose_enabled else 'OFF'} | "
            f"STR: {strength:.2f} | "
            f"Detections: {len(info_list)}",
            (40, pose_view.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2,
        )

        # ── Display ───────────────────────────────────────────────────────────
        if show_original:
            label_orig = frame.copy()
            label_pose = pose_view.copy()
            cv2.putText(label_orig, "ORIGINAL",         (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(label_pose, "CORRECTED + POSE", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),   2)
            if label_orig.shape[0] != label_pose.shape[0]:
                label_orig = cv2.resize(
                    label_orig, (label_pose.shape[1], label_pose.shape[0])
                )
            display = np.hstack([label_orig, label_pose])
        else:
            display = pose_view

        cv2.imshow(win_name, display)

        # ── Key handling ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key in (ord("+"), ord("=")):
            strength = min(strength + 0.05, 2.0)
            print(f"[INFO] IR strength: {strength:.2f}")
        elif key == ord("-"):
            strength = max(strength - 0.05, 0.0)
            print(f"[INFO] IR strength: {strength:.2f}")
        elif key == ord("o"):
            show_original = not show_original
        elif key == ord("p"):
            pose_enabled = not pose_enabled
            print(f"[INFO] Pose: {'ON' if pose_enabled else 'OFF'}")
        elif key == ord("s"):
            ts    = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            fname = os.path.join(output_dir, f"pose_frame_{ts}.png")
            cv2.imwrite(fname, display)
            print(f"[SAVED] {fname}")

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Realtime microwave + package pose estimation"
    )

    parser.add_argument("--device",       default="/dev/video2")
    parser.add_argument("--width",        type=int,   default=1920)
    parser.add_argument("--height",       type=int,   default=1080)
    parser.add_argument("--model",        default="best_26n_mv_pkg_v3.pt")
    parser.add_argument("--calib",        default="camera_params_fisheye_aruko.npz")

    # Microwave physical dims (metres)
    parser.add_argument("--mw-width",     type=float, default=0.44)
    parser.add_argument("--mw-depth",     type=float, default=0.61)

    # Package physical dims (metres)
    parser.add_argument("--pkg-width",    type=float, default=0.57)
    parser.add_argument("--pkg-depth",    type=float, default=0.67)

    parser.add_argument("--strength",     type=float, default=1.0)
    parser.add_argument("--side-by-side", action="store_true")
    parser.add_argument("--conf-thresh",  type=float, default=0.30)
    parser.add_argument("--yolo-conf",    type=float, default=0.25)
    parser.add_argument("--imgsz",        type=int,   default=640)

    args = parser.parse_args()

    run(
        device        = args.device,
        width         = args.width,
        height        = args.height,
        strength      = args.strength,
        show_original = args.side_by_side,
        model_path    = args.model,
        calib_file    = args.calib,
        mw_width      = args.mw_width,
        mw_depth      = args.mw_depth,
        pkg_width     = args.pkg_width,
        pkg_depth     = args.pkg_depth,
        conf_thresh   = args.conf_thresh,
        yolo_conf     = args.yolo_conf,
        imgsz         = args.imgsz,
    )





























# #!/usr/bin/env python3

# """
# Real-time Microwave + Package Pose Estimation

# Pipeline:
# 1. Read frame from IR camera
# 2. Apply IR pink-cast correction
# 3. Run YOLO top-4 keypoint model (2 classes)
# 4. Use solvePnP with per-class geometry
# 5. Draw keypoints, bbox, XYZ axes, reprojection error, and distance
# 6. Show live OpenCV window

# Controls:
# Q / ESC  = quit
# S        = save current frame
# + / -    = adjust IR correction strength
# O        = toggle original/corrected side-by-side
# P        = toggle pose inference on/off
# """

# import cv2
# import numpy as np
# import argparse
# import sys
# import os
# from datetime import datetime
# from ultralytics import YOLO


# # ============================================================
# # IR Pink Correction  (unchanged)
# # ============================================================

# def correct_ir_cast(frame: np.ndarray, strength: float = 1.0) -> np.ndarray:
#     frame = frame.astype(np.float32)
#     b, g, r = cv2.split(frame)
#     r_corrected = r * (1.0 - 0.30 * strength)
#     g_corrected = g * (1.0 + 0.05 * strength)
#     b_corrected = b * (1.0 - 0.15 * strength)
#     frame = cv2.merge([
#         np.clip(b_corrected, 0, 255),
#         np.clip(g_corrected, 0, 255),
#         np.clip(r_corrected, 0, 255),
#     ]).astype(np.uint8)
#     lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
#     l, a, b_ch = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#     l = clahe.apply(l)
#     a = cv2.addWeighted(a, 1.0 - 0.25 * strength, np.full_like(a, 128), 0.25 * strength, 0)
#     lab = cv2.merge((l, a, b_ch))
#     frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
#     return frame


# # ============================================================
# # Camera Setup  (unchanged)
# # ============================================================

# def open_camera(device: str, width: int, height: int) -> cv2.VideoCapture:
#     cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
#     if not cap.isOpened():
#         print(f"[ERROR] Could not open {device}")
#         sys.exit(1)
#     fourcc = cv2.VideoWriter_fourcc(*"MJPG")
#     cap.set(cv2.CAP_PROP_FOURCC, fourcc)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
#     cap.set(cv2.CAP_PROP_FPS, 30)
#     cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#     actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     actual_fps = cap.get(cv2.CAP_PROP_FPS)
#     print(f"[INFO] Camera device   : {device}")
#     print(f"[INFO] Resolution      : {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")
#     return cap


# # ============================================================
# # Per-class 3D geometry
# # ============================================================

# def build_class_dims(mw_width, mw_depth, pkg_width, pkg_depth):
#     """
#     Returns a dict keyed by class_id with object_points + metadata.
#     Keypoint order for BOTH classes:
#         0 = top_front_left
#         1 = top_front_right
#         2 = top_back_left
#         3 = top_back_right
#     All on the top face → Z = 0 (planar PnP).
#     """
#     return {
#         0: {
#             "name":          "microwave",
#             "color":         (0, 255, 255),   # cyan
#             "W":             mw_width,
#             "D":             mw_depth,
#             "object_points": np.array([
#                 [0,        0,        0],
#                 [mw_width, 0,        0],
#                 [0,        mw_depth, 0],
#                 [mw_width, mw_depth, 0],
#             ], dtype=np.float32),
#         },
#         1: {
#             "name":          "package",
#             "color":         (0, 165, 255),   # orange
#             "W":             pkg_width,
#             "D":             pkg_depth,
#             "object_points": np.array([
#                 [0,         0,         0],
#                 [pkg_width, 0,         0],
#                 [0,         pkg_depth, 0],
#                 [pkg_width, pkg_depth, 0],
#             ], dtype=np.float32),
#         },
#     }


# # ============================================================
# # Pose Estimation Helper  (updated for 2 classes)
# # ============================================================

# def estimate_pose_and_draw(
#     frame,
#     model,
#     camera_matrix,
#     dist_coeffs,
#     class_dims,          # ← dict from build_class_dims()
#     conf_thresh=0.30,
#     yolo_conf=0.25,
#     imgsz=640
# ):
#     vis = frame.copy()

#     results = model.predict(source=frame, imgsz=imgsz, conf=yolo_conf, verbose=False)
#     res     = results[0]

#     if res.keypoints is None or len(res.keypoints.xy) == 0:
#         cv2.putText(vis, "No objects detected", (40, 60),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
#         return vis, []

#     all_info = []

#     for det_idx in range(len(res.boxes)):

#         class_id   = int(res.boxes.cls[det_idx].item())
#         det_conf   = float(res.boxes.conf[det_idx].item())
#         kpts_xy    = res.keypoints.xy[det_idx].cpu().numpy().astype(np.float32)
#         kpts_conf  = res.keypoints.conf[det_idx].cpu().numpy().astype(np.float32)
#         box        = res.boxes.xyxy[det_idx].cpu().numpy().astype(np.float32)

#         # ── Skip unknown classes ──────────────────────────────────────────────
#         if class_id not in class_dims:
#             continue

#         dims       = class_dims[class_id]
#         obj_name   = dims["name"]
#         color      = dims["color"]
#         obj_pts    = dims["object_points"]
#         W, D       = dims["W"], dims["D"]

#         # ── Bounding box ──────────────────────────────────────────────────────
#         x1, y1, x2, y2 = box.astype(int)
#         cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
#         cv2.putText(vis, f"{obj_name} {det_conf:.2f}", (x1, y1 - 8),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

#         # ── Keypoints ─────────────────────────────────────────────────────────
#         KP_NAMES  = ["TFL", "TFR", "TBL", "TBR"]
#         valid_idx = np.where(kpts_conf > conf_thresh)[0]

#         for i, (x, y) in enumerate(kpts_xy):
#             kp_color = color if kpts_conf[i] > conf_thresh else (0, 0, 255)
#             cv2.circle(vis, (int(x), int(y)), 7, kp_color, -1)
#             cv2.putText(vis, f"{KP_NAMES[i]}:{kpts_conf[i]:.2f}",
#                         (int(x) + 5, int(y) - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.50, kp_color, 2)

#         if len(valid_idx) < 4:
#             cv2.putText(vis,
#                         f"{obj_name}: need 4 kpts, got {len(valid_idx)}",
#                         (x1, y2 + 24),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#             continue

#         image_points = kpts_xy.astype(np.float32)

#         # ── solvePnP ──────────────────────────────────────────────────────────
#         success, rvec, tvec = cv2.solvePnP(
#             obj_pts, image_points,
#             camera_matrix, dist_coeffs,
#             flags=cv2.SOLVEPNP_IPPE
#         )
#         if not success:
#             success, rvec, tvec = cv2.solvePnP(
#                 obj_pts, image_points,
#                 camera_matrix, dist_coeffs,
#                 flags=cv2.SOLVEPNP_ITERATIVE
#             )
#         if not success:
#             cv2.putText(vis, f"{obj_name}: solvePnP failed",
#                         (x1, y2 + 24),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#             continue

#         # ── Reprojection error ────────────────────────────────────────────────
#         proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
#         proj     = proj.reshape(-1, 2)
#         errors   = np.linalg.norm(proj - image_points, axis=1)
#         mean_err = float(np.mean(errors))

#         # ── Distances ─────────────────────────────────────────────────────────
#         dist_origin = float(np.linalg.norm(tvec))
#         R, _        = cv2.Rodrigues(rvec)
#         center_obj  = np.array([[W / 2], [D / 2], [0]], dtype=np.float32)
#         center_cam  = R @ center_obj + tvec
#         dist_center = float(np.linalg.norm(center_cam))

#         # ── Reprojected points (white) ────────────────────────────────────────
#         for i, (x, y) in enumerate(proj):
#             if np.isfinite(x) and np.isfinite(y):
#                 cv2.circle(vis, (int(x), int(y)), 5, (255, 255, 255), -1)

#         # ── XYZ axes ──────────────────────────────────────────────────────────
#         axis_len = 0.20
#         axis_3d  = np.array([
#             [0, 0, 0], [axis_len, 0, 0], [0, axis_len, 0], [0, 0, -axis_len]
#         ], dtype=np.float32)
#         ax2d, _  = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)
#         ax2d     = ax2d.reshape(-1, 2)

#         origin = tuple(ax2d[0].astype(int))
#         x_pt   = tuple(ax2d[1].astype(int))
#         y_pt   = tuple(ax2d[2].astype(int))
#         z_pt   = tuple(ax2d[3].astype(int))

#         cv2.arrowedLine(vis, origin, x_pt, (0, 0, 255),   4, tipLength=0.15)
#         cv2.arrowedLine(vis, origin, y_pt, (0, 255, 0),   4, tipLength=0.15)
#         cv2.arrowedLine(vis, origin, z_pt, (255, 0, 0),   4, tipLength=0.15)
#         cv2.putText(vis, "X", x_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
#         cv2.putText(vis, "Y", y_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#         cv2.putText(vis, "Z", z_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

#         # ── HUD text (anchored near bbox) ─────────────────────────────────────
#         hud = [
#             f"[{obj_name}] conf={det_conf:.2f}",
#             f"Origin : {dist_origin*100:.1f} cm",
#             f"Center : {dist_center*100:.1f} cm",
#             f"RpjErr : {mean_err:.1f} px",
#         ]
#         for i, line in enumerate(hud):
#             y_pos = y1 - 20 - (len(hud) - 1 - i) * 30
#             y_pos = max(y_pos, 20 + i * 30)   # clamp to frame top
#             # shadow then colored text for readability
#             cv2.putText(vis, line, (x1, y_pos),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
#             cv2.putText(vis, line, (x1, y_pos),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

#         all_info.append({
#             "class_id":         class_id,
#             "name":             obj_name,
#             "rvec":             rvec,
#             "tvec":             tvec,
#             "distance_origin_m": dist_origin,
#             "distance_center_m": dist_center,
#             "mean_error":       mean_err,
#             "keypoints":        kpts_xy,
#             "keypoint_conf":    kpts_conf,
#             "box":              box,
#         })

#     return vis, all_info


# # ============================================================
# # Main Loop  (updated for 2-class dims + new argparse args)
# # ============================================================

# def run(
#     device, width, height, strength, show_original,
#     model_path, calib_file,
#     mw_width, mw_depth,
#     pkg_width, pkg_depth,       # ← new
#     conf_thresh, yolo_conf, imgsz
# ):
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model not found: {model_path}")

#     print("[INFO] Loading YOLO model...")
#     model = YOLO(model_path)
#     print("[INFO] YOLO model loaded.")

#     if not os.path.exists(calib_file):
#         raise FileNotFoundError(f"Calibration file not found: {calib_file}")

#     data          = np.load(calib_file)
#     camera_matrix = data["mtx"]
#     dist_coeffs   = data["dist"]
#     print("[INFO] Calibration loaded.")

#     # Build per-class geometry once
#     class_dims = build_class_dims(mw_width, mw_depth, pkg_width, pkg_depth)
#     print(f"[INFO] Microwave : {mw_width}m x {mw_depth}m")
#     print(f"[INFO] Package   : {pkg_width}m x {pkg_depth}m")

#     cap        = open_camera(device, width, height)
#     output_dir = "realtime_pose_captures"
#     os.makedirs(output_dir, exist_ok=True)

#     win_name = "Pose Estimation — Q quit | S save | P pose on/off | O original | +/- strength"
#     cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

#     pose_enabled = True
#     frame_id     = 0

#     print("\n[CONTROLS]")
#     print("  Q / ESC  — quit")
#     print("  S        — save frame")
#     print("  P        — toggle pose on/off")
#     print("  O        — toggle side-by-side")
#     print("  +/-      — IR correction strength")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("[WARN] Frame grab failed")
#             continue

#         corrected = correct_ir_cast(frame, strength=strength)

#         if pose_enabled:
#             pose_view, info_list = estimate_pose_and_draw(
#                 corrected, model,
#                 camera_matrix, dist_coeffs,
#                 class_dims,
#                 conf_thresh=conf_thresh,
#                 yolo_conf=yolo_conf,
#                 imgsz=imgsz
#             )
#         else:
#             pose_view = corrected.copy()
#             info_list = []

#         # Status bar
#         cv2.putText(
#             pose_view,
#             f"Pose: {'ON' if pose_enabled else 'OFF'} | STR: {strength:.2f} | "
#             f"Detections: {len(info_list)}",
#             (40, pose_view.shape[0] - 30),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2
#         )

#         if show_original:
#             label_orig = frame.copy()
#             label_pose = pose_view.copy()
#             cv2.putText(label_orig, "ORIGINAL",         (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#             cv2.putText(label_pose, "CORRECTED + POSE", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),   2)
#             if label_orig.shape[0] != label_pose.shape[0]:
#                 label_orig = cv2.resize(label_orig, (label_pose.shape[1], label_pose.shape[0]))
#             display = np.hstack([label_orig, label_pose])
#         else:
#             display = pose_view

#         cv2.imshow(win_name, display)

#         key = cv2.waitKey(1) & 0xFF

#         if key in (ord("q"), 27):
#             break
#         elif key in (ord("+"), ord("=")):
#             strength = min(strength + 0.05, 2.0)
#             print(f"[INFO] Strength: {strength:.2f}")
#         elif key == ord("-"):
#             strength = max(strength - 0.05, 0.0)
#             print(f"[INFO] Strength: {strength:.2f}")
#         elif key == ord("o"):
#             show_original = not show_original
#         elif key == ord("p"):
#             pose_enabled = not pose_enabled
#             print(f"[INFO] Pose: {'ON' if pose_enabled else 'OFF'}")
#         elif key == ord("s"):
#             ts    = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
#             fname = os.path.join(output_dir, f"pose_frame_{ts}.png")
#             cv2.imwrite(fname, display)
#             print(f"[SAVED] {fname}")

#         frame_id += 1

#     cap.release()
#     cv2.destroyAllWindows()


# # ============================================================
# # Entry Point
# # ============================================================

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Realtime microwave + package pose estimation"
#     )

#     parser.add_argument("--device",      default="/dev/video2")
#     parser.add_argument("--width",       type=int,   default=1920)
#     parser.add_argument("--height",      type=int,   default=1080)
#     parser.add_argument("--model",       default="best_26n_mv_pkg.pt")
#     parser.add_argument("--calib",       default="camera_params_fisheye_aruko.npz")

#     # Microwave dims
#     parser.add_argument("--mw-width",    type=float, default=0.44)
#     parser.add_argument("--mw-depth",    type=float, default=0.61)

#     # Package dims  ← new, replace defaults with real measurements
#     parser.add_argument("--pkg-width",   type=float, default=0.57)
#     parser.add_argument("--pkg-depth",   type=float, default=0.67)

#     parser.add_argument("--strength",    type=float, default=1.0)
#     parser.add_argument("--side-by-side",action="store_true")
#     parser.add_argument("--conf-thresh", type=float, default=0.30)
#     parser.add_argument("--yolo-conf",   type=float, default=0.25)
#     parser.add_argument("--imgsz",       type=int,   default=640)

#     args = parser.parse_args()

#     run(
#         device       = args.device,
#         width        = args.width,
#         height       = args.height,
#         strength     = args.strength,
#         show_original= args.side_by_side,
#         model_path   = args.model,
#         calib_file   = args.calib,
#         mw_width     = args.mw_width,
#         mw_depth     = args.mw_depth,
#         pkg_width    = args.pkg_width,
#         pkg_depth    = args.pkg_depth,
#         conf_thresh  = args.conf_thresh,
#         yolo_conf    = args.yolo_conf,
#         imgsz        = args.imgsz
#     )