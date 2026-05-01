#!/usr/bin/env python3
import cv2
import numpy as np
import os
import sys

# ──────────────────────────────────────────────
# Load Calibration Data
# ──────────────────────────────────────────────
CALIB_FILE = "camera_params_non_fisheye.npz"

if os.path.exists(CALIB_FILE):
    with np.load(CALIB_FILE) as data:
        # Note: Ensure these keys ('mtx' and 'dist') match the names 
        # used when you saved the .npz file.
        camera_matrix = data['mtx']
        dist_coeffs = data['dist']
    print(f"[SUCCESS] Loaded constants from {CALIB_FILE}")
else:
    print(f"[ERROR] Calibration file {CALIB_FILE} not found!")
    sys.exit(1)

# Marker size in meters (Adjust this to your physical marker size)
MARKER_SIZE = 17.4/100

# ──────────────────────────────────────────────
# Setup ArUco
# ──────────────────────────────────────────────
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

def run_aruco_pose(device="/dev/video2"):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    
    # Matching the resolution used during calibration is best for accuracy
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    if not cap.isOpened():
        print("[ERROR] Camera not found.")
        return

    print(f"Tracking... Press 'Q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Detection works best on grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for i in range(len(ids)):
                # Define 3D points of marker in its own coordinate system
                obj_points = np.array([
                    [-MARKER_SIZE/2,  MARKER_SIZE/2, 0],
                    [ MARKER_SIZE/2,  MARKER_SIZE/2, 0],
                    [ MARKER_SIZE/2, -MARKER_SIZE/2, 0],
                    [-MARKER_SIZE/2, -MARKER_SIZE/2, 0]
                ], dtype=np.float32)

                # SolvePnP for Pose
                _, rvec, tvec = cv2.solvePnP(obj_points, corners[i], camera_matrix, dist_coeffs)

                # Draw Axis (X=Red, Y=Green, Z=Blue)
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

                # Translation vector (tvec) gives X, Y, Z coordinates in meters
                x, y, z = tvec.flatten()
                
                # Display pose info
                info_text = f"X:{x:.2f} Y:{y:.2f} Z:{z:.2f}m"
                cv2.putText(frame, info_text, (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("IR ArUco Pose", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_aruco_pose()