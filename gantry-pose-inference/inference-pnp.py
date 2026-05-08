# ============================================================
# inference_pnp.py
# YOLO Top-4 Microwave Keypoint Detection + solvePnP Pose
# ============================================================

from ultralytics import YOLO
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


# ============================================================
# USER CONFIG
# ============================================================

MODEL_PATH = "best.pt"
CALIB_FILE = "camera_params_fisheye_aruko.npz"
IMAGE_PATH = "test-image-02.png"

# Real microwave top-face dimensions in meters
# Change these if your measurements are different
MW_WIDTH = 0.44   # left-right width
MW_DEPTH = 0.61   # front-back depth

# YOLO/keypoint confidence threshold
CONF_THRESH = 0.30

# Output image
OUTPUT_PATH = "pnp_result.jpg"


# ============================================================
# 1. Load YOLO top-4 keypoint model
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

print("[INFO] Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("[INFO] Model loaded successfully.")


# ============================================================
# 2. Load camera calibration
# ============================================================

if not os.path.exists(CALIB_FILE):
    raise FileNotFoundError(f"Calibration file not found: {CALIB_FILE}")

print("[INFO] Loading camera calibration...")
data = np.load(CALIB_FILE)

camera_matrix = data["mtx"]
dist_coeffs = data["dist"]

print("\nCamera matrix:")
print(camera_matrix)

print("\nDistortion coefficients:")
print(dist_coeffs)


# ============================================================
# 3. Load image
# ============================================================

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Image file not found: {IMAGE_PATH}")

img = cv2.imread(IMAGE_PATH)

if img is None:
    raise ValueError("Could not read image. Check IMAGE_PATH.")

print(f"\n[INFO] Image loaded: {IMAGE_PATH}")
print("Image shape:", img.shape)


# ============================================================
# 4. Run YOLO inference
# ============================================================

print("\n[INFO] Running YOLO inference...")

results = model.predict(
    source=IMAGE_PATH,
    imgsz=640,
    conf=0.25,
    verbose=False
)

res = results[0]

if res.keypoints is None or len(res.keypoints.xy) == 0:
    raise ValueError("No microwave keypoints detected.")

if res.boxes is None or len(res.boxes.xyxy) == 0:
    raise ValueError("No microwave box detected.")

# Use first detected microwave
kpts_xy = res.keypoints.xy[0].cpu().numpy().astype(np.float32)       # shape: (4, 2)
kpts_conf = res.keypoints.conf[0].cpu().numpy().astype(np.float32)   # shape: (4,)
box = res.boxes.xyxy[0].cpu().numpy().astype(np.float32)

print("\nPredicted top-4 keypoints:")
print(kpts_xy)

print("\nKeypoint confidence:")
print(kpts_conf)

print("\nPredicted box:")
print(box)


# ============================================================
# 5. Check keypoint confidence
# ============================================================

valid_idx = np.where(kpts_conf > CONF_THRESH)[0]

print("\nValid keypoint indices:", valid_idx)

if len(valid_idx) < 4:
    raise ValueError(
        f"Need all 4 top keypoints for reliable PnP. "
        f"Only {len(valid_idx)} passed confidence threshold {CONF_THRESH}."
    )

# We need all 4 top keypoints for planar PnP
image_points = kpts_xy.astype(np.float32)


# ============================================================
# 6. Define real-world 3D microwave top-face points
# ============================================================

W = MW_WIDTH
D = MW_DEPTH

# Keypoint order must match training:
# 0 = top_front_left
# 1 = top_front_right
# 2 = top_back_left
# 3 = top_back_right
object_points = np.array([
    [0, 0, 0],   # top_front_left
    [W, 0, 0],   # top_front_right
    [0, D, 0],   # top_back_left
    [W, D, 0],   # top_back_right
], dtype=np.float32)

print("\nObject points 3D:")
print(object_points)

print("\nImage points 2D:")
print(image_points)


# ============================================================
# 7. Run solvePnP
# ============================================================

print("\n[INFO] Running solvePnP...")

success, rvec, tvec = cv2.solvePnP(
    object_points,
    image_points,
    camera_matrix,
    dist_coeffs,
    flags=cv2.SOLVEPNP_IPPE
)

if not success:
    print("[WARN] SOLVEPNP_IPPE failed. Trying SOLVEPNP_ITERATIVE...")

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

if not success:
    raise ValueError("solvePnP failed.")

print("\nPnP successful.")
print("Rotation vector rvec:")
print(rvec.ravel())

print("\nTranslation vector tvec:")
print(tvec.ravel())


# ============================================================
# 8. Calculate reprojection error
# ============================================================

projected_points, _ = cv2.projectPoints(
    object_points,
    rvec,
    tvec,
    camera_matrix,
    dist_coeffs
)

projected_points = projected_points.reshape(-1, 2)

errors = np.linalg.norm(projected_points - image_points, axis=1)
mean_error = np.mean(errors)

print("\nReprojection errors:")
for i, err in enumerate(errors):
    print(f"Point {i}: {err:.2f} px")

print(f"Mean reprojection error: {mean_error:.2f} px")


# ============================================================
# 9. Calculate distance from camera
# ============================================================

# Distance from camera to object origin
# Here origin = top_front_left point
distance_origin_m = np.linalg.norm(tvec)

# Convert rvec to rotation matrix
R, _ = cv2.Rodrigues(rvec)

# Center of microwave top face in object coordinate frame
center_obj = np.array([[W / 2], [D / 2], [0]], dtype=np.float32)

# Transform center point to camera coordinate frame
center_cam = R @ center_obj + tvec

# Distance from camera to top-face center
distance_center_m = np.linalg.norm(center_cam)

print(f"\nDistance from camera to microwave origin: {distance_origin_m:.3f} m")
print(f"Distance from camera to microwave origin: {distance_origin_m * 100:.1f} cm")

print("\nTop-face center in camera coordinates:")
print(center_cam.ravel())

print(f"Distance from camera to microwave top-face center: {distance_center_m:.3f} m")
print(f"Distance from camera to microwave top-face center: {distance_center_m * 100:.1f} cm")


# ============================================================
# 10. Draw pose axes
# ============================================================

axis_len = 0.20  # 20 cm

axis_3d = np.array([
    [0, 0, 0],              # origin
    [axis_len, 0, 0],       # X axis
    [0, axis_len, 0],       # Y axis
    [0, 0, -axis_len],      # Z axis downward from top face
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


# ============================================================
# 11. Visualize result
# ============================================================

vis = img.copy()

# Draw bounding box
x1, y1, x2, y2 = box.astype(int)
cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)

# Draw predicted keypoints
for i, (x, y) in enumerate(image_points):
    cv2.circle(vis, (int(x), int(y)), 8, (0, 255, 255), -1)
    cv2.putText(
        vis,
        f"P{i}",
        (int(x) + 5, int(y) - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

# Draw reprojected points
for i, (x, y) in enumerate(projected_points):
    cv2.circle(vis, (int(x), int(y)), 6, (255, 255, 255), -1)
    cv2.putText(
        vis,
        f"R{i}",
        (int(x) + 5, int(y) + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

# Draw XYZ axes
# OpenCV uses BGR:
# X = red
# Y = green
# Z = blue
cv2.arrowedLine(vis, origin, x_axis, (0, 0, 255), 4, tipLength=0.15)
cv2.arrowedLine(vis, origin, y_axis, (0, 255, 0), 4, tipLength=0.15)
cv2.arrowedLine(vis, origin, z_axis, (255, 0, 0), 4, tipLength=0.15)

cv2.putText(vis, "X", x_axis, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
cv2.putText(vis, "Y", y_axis, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
cv2.putText(vis, "Z", z_axis, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)

# Draw text info
cv2.putText(
    vis,
    f"Origin Distance: {distance_origin_m:.2f} m",
    (40, 60),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    (255, 255, 255),
    3
)

cv2.putText(
    vis,
    f"Center Distance: {distance_center_m:.2f} m",
    (40, 105),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    (255, 255, 255),
    3
)

cv2.putText(
    vis,
    f"Reproj Error: {mean_error:.2f} px",
    (40, 150),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    (255, 255, 255),
    3
)


# ============================================================
# 12. Save and display result
# ============================================================

cv2.imwrite(OUTPUT_PATH, vis)
print(f"\n[INFO] Saved result image: {OUTPUT_PATH}")

# Display using matplotlib
plt.figure(figsize=(13, 9))
plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("YOLO Top-4 Keypoints + PnP Pose + Distance")
plt.show()