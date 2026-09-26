import cv2
import numpy as np
import glob
import os

data = np.load('camera_params_fisheye.npz')
K = data['mtx']
D = data['dist']

for split in ['train', 'val']:
    images = glob.glob(f'./gantry_mv_data_kaggle/{split}/images/*.png')
    out_dir = f'./gantry_mv_data_kaggle_undistorted/{split}/images'
    os.makedirs(out_dir, exist_ok=True)

    if not images:
        print(f"No images found for {split}")
        continue

    # Compute K_new once from first image
    img0 = cv2.imread(images[0])
    h, w = img0.shape[:2]
    K_new, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0.5)

    for img_path in images:
        img = cv2.imread(img_path)
        undistorted = cv2.undistort(img, K, D, None, K_new)
        out_path = os.path.join(out_dir, os.path.basename(img_path))
        cv2.imwrite(out_path, undistorted)

    print(f"{split}: {len(images)} images → {out_dir}")

print("\nK_new to use in Cube R-CNN:\n", K_new)

np.save('K_new.npy', K_new)