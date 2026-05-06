#!/usr/bin/env python3
import cv2
import numpy as np
import os
import sys

# ──────────────────────────────────────────────
# 1. Configuration & Calibration Data
# ──────────────────────────────────────────────
CALIB_FILE = "camera_params_fisheye.npz"

# Board Configuration (MUST match your calibration exactly)
SQUARE_LENGTH = 48.0 / 1000  
MARKER_LENGTH = 35.0 / 1000  
CHARUCO_BOARD_SIZE = (6, 4)
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

if os.path.exists(CALIB_FILE):
    with np.load(CALIB_FILE) as data:
        camera_matrix = data['mtx']
        dist_coeffs = data['dist']
    print(f"[SUCCESS] Loaded constants from {CALIB_FILE}")
else:
    print(f"[ERROR] Calibration file {CALIB_FILE} not found!")
    sys.exit(1)

# ──────────────────────────────────────────────
# 2. Setup ChArUco Detector
# ──────────────────────────────────────────────
board = cv2.aruco.CharucoBoard(
    CHARUCO_BOARD_SIZE, SQUARE_LENGTH, MARKER_LENGTH, ARUCO_DICT
)
charuco_params = cv2.aruco.CharucoParameters()
detector_params = cv2.aruco.DetectorParameters()
charuco_detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)

def run_charuco_pose(device="/dev/video2"):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    if not cap.isOpened():
        print("[ERROR] Camera not found.")
        return

    print(f"\n{'='*40}")
    print(f" TRACKING STARTING ")
    print(f"{'='*40}")
    print("Press 'Q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)

        if charuco_ids is not None and len(charuco_ids) >= 4:
            cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)
            obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
            valid, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_matrix, dist_coeffs)

            if valid:
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

                # 1. Extract XYZ
                x, y, z = tvec.flatten()
                
                # 2. Calculate Euclidean Distance (L2 Norm)
                euclidean_dist = np.linalg.norm(tvec)
                
                # Format strings for display
                pose_str = f"X: {x:6.3f} | Y: {y:6.3f} | Z: {z:6.3f} (m)"
                dist_str = f"Euclidean Distance: {euclidean_dist:.4f} m ({euclidean_dist*100:.2f} cm)"

                # Overlay on image
                cv2.putText(frame, pose_str, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, dist_str, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                # 3. Pretty Print to Terminal
                # Using \r (carriage return) to update the same line in terminal
                sys.stdout.write(f"\r[POSE] {pose_str} | [DIST] {euclidean_dist*100:6.2f} cm    ")
                sys.stdout.flush()

        cv2.imshow("IR ChArUco Pose", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n\nExiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_charuco_pose()