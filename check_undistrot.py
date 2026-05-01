import cv2
import numpy as np
import os

# 1. Load the calibration parameters you just generated
if not os.path.exists("camera_params_non_fisheye.npz"):
    print("Error: camera_params_non_fisheye.npz not found!")
    exit()

data = np.load("camera_params_non_fisheye.npz")
mtx = data['mtx']
dist = data['dist']

# 2. Load your raw fisheye image
#img_path='./calib-images/im_09.png'  # Change this to your actual image path
img_path = './ir_captures/test.png'  # Change this to your actual image path
img = cv2.imread(img_path)

if img is None:
    print(f"Error: Could not load image at {img_path}")
    exit()

h, w = img.shape[:2]

# 3. Calculate the optimal matrix
# alpha=1 means we keep all pixels (black edges might appear)
# alpha=0 means OpenCV crops the image so all pixels are valid
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# 4. Apply the Correction
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# 5. Create a side-by-side comparison
# We resize them slightly so they fit on your laptop screen (Latitude 7490 is 1080p usually)
scale = 0.5 
img_small = cv2.resize(img, (0,0), fx=scale, fy=scale)
dst_small = cv2.resize(dst, (0,0), fx=scale, fy=scale)

# Stack images horizontally: [Original | Corrected]
comparison = np.hstack((img_small, dst_small))

# Add labels
cv2.putText(comparison, "BEFORE (Fisheye)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
cv2.putText(comparison, "AFTER (Rectilinear)", (int(w*scale) + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

# 6. Show and Save
cv2.imshow('Lens Correction Comparison', comparison)
print("Press any key on the image window to close.")
cv2.waitKey(0)
cv2.destroyAllWindows()