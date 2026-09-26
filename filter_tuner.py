"""
Interactive filter tuner for calibration images.
Shows original and filtered side by side.
Tune parameters live, save filtered images when happy.

Run from ~/gantry-vision:
  python filter_tuner.py

Controls:
  A / D          — previous / next image
  S              — save current filtered image to output folder
  B              — batch save ALL images with current settings
  Q / ESC        — quit

  1 / 2          — decrease / increase R multiplier (IR correction)
  3 / 4          — decrease / increase G multiplier
  5 / 6          — decrease / increase B multiplier
  7 / 8          — decrease / increase CLAHE clip limit
  9 / 0          — decrease / increase gamma
  [ / ]          — decrease / increase bilateral sigma
"""

import cv2
import numpy as np
import glob
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INPUT_GLOB  = "calibration_images_distorted/calib_raw_*.png"
OUTPUT_DIR  = "calibration_images_clean"
STEP        = 0.05   # adjustment step per keypress

# ─────────────────────────────────────────────
# FILTER PARAMS — tune these live with keys
# ─────────────────────────────────────────────
params = {
    'IR_R':         0.70,    # keys 1/2
    'IR_G':         1.05,    # keys 3/4
    'IR_B':         0.85,    # keys 5/6
    'CLAHE_CLIP':   4.0,     # keys 7/8
    'GAMMA':        1.0,     # keys 9/0
    'BILATERAL_S':  75.0,    # keys [/]
}


# ─────────────────────────────────────────────
# FILTER
# ─────────────────────────────────────────────

def apply_filter(img, p):
    # IR correction
    f = img.astype(np.float32)
    f[:, :, 2] *= p['IR_R']
    f[:, :, 1] *= p['IR_G']
    f[:, :, 0] *= p['IR_B']
    out = np.clip(f, 0, 255).astype(np.uint8)

    # Bilateral
    s = int(p['BILATERAL_S'])
    if s > 0:
        out = cv2.bilateralFilter(out, d=9, sigmaColor=s, sigmaSpace=s)

    # CLAHE
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=p['CLAHE_CLIP'], tileGridSize=(8, 8))
    l = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # Gamma
    if p['GAMMA'] != 1.0:
        lut = np.array([((i / 255.0) ** p['GAMMA']) * 255
                        for i in range(256)], dtype=np.uint8)
        out = cv2.LUT(out, lut)

    return out


# ─────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────

def make_display(img_orig, img_filt, p, idx, total, saved_count):
    h, w = img_orig.shape[:2]
    scale = min(1.0, 850.0 / w)
    dw, dh = int(w * scale), int(h * scale)

    left  = cv2.resize(img_orig, (dw, dh))
    right = cv2.resize(img_filt, (dw, dh))

    # Labels on left
    cv2.putText(left, "ORIGINAL", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(left, f"Image {idx+1}/{total}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Params on right
    cv2.putText(right, "FILTERED", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    lines = [
        f"R={p['IR_R']:.2f} [1/2]   G={p['IR_G']:.2f} [3/4]   B={p['IR_B']:.2f} [5/6]",
        f"CLAHE={p['CLAHE_CLIP']:.1f} [7/8]   Gamma={p['GAMMA']:.2f} [9/0]   Bilat={p['BILATERAL_S']:.0f} [[/]]",
        f"Saved: {saved_count}   |   S=save this   B=batch all   A/D=prev/next   Q=quit",
    ]
    for i, line in enumerate(lines):
        cv2.putText(right, line, (10, 60 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)

    return np.hstack([left, right])


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    image_paths = sorted(glob.glob(INPUT_GLOB))
    assert len(image_paths) > 0, f"No images found at {INPUT_GLOB}"
    print(f"Found {len(image_paths)} images.")
    print(f"Output → {OUTPUT_DIR}/")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    idx         = 0
    saved_count = 0
    p           = params.copy()

    win = "Filter Tuner"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1700, 520)

    images = {}   # cache loaded images

    def get_img(i):
        if i not in images:
            images[i] = cv2.imread(image_paths[i])
        return images[i]

    def save_one(i, filtered):
        out_name = Path(image_paths[i]).stem + "_clean.png"
        out_path = str(Path(OUTPUT_DIR) / out_name)
        cv2.imwrite(out_path, filtered)
        return out_path

    print("\nControls:")
    print("  A/D        — prev/next image")
    print("  S          — save this image")
    print("  B          — batch save ALL images")
    print("  1/2        — R multiplier  (IR red suppression)")
    print("  3/4        — G multiplier")
    print("  5/6        — B multiplier")
    print("  7/8        — CLAHE clip limit")
    print("  9/0        — gamma  (<1 = brighter)")
    print("  [/]        — bilateral sigma")
    print("  Q/ESC      — quit\n")

    while True:
        img_orig = get_img(idx)
        img_filt = apply_filter(img_orig, p)
        frame    = make_display(img_orig, img_filt, p, idx, len(image_paths), saved_count)

        cv2.imshow(win, frame)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q') or key == 27:
            break

        elif key == ord('a') or key == 81:
            idx = (idx - 1) % len(image_paths)
        elif key == ord('d') or key == 83:
            idx = (idx + 1) % len(image_paths)

        elif key == ord('s'):
            path = save_one(idx, img_filt)
            saved_count += 1
            print(f"Saved → {path}  (total: {saved_count})")

        elif key == ord('b'):
            print(f"\nBatch saving {len(image_paths)} images …")
            for i in range(len(image_paths)):
                raw  = get_img(i)
                filt = apply_filter(raw, p)
                save_one(i, filt)
            saved_count = len(image_paths)
            print(f"Done. {saved_count} images saved to {OUTPUT_DIR}/")

        # Param adjustments
        elif key == ord('1'): p['IR_R']        = max(0.1,  p['IR_R']        - STEP)
        elif key == ord('2'): p['IR_R']        = min(2.0,  p['IR_R']        + STEP)
        elif key == ord('3'): p['IR_G']        = max(0.1,  p['IR_G']        - STEP)
        elif key == ord('4'): p['IR_G']        = min(2.0,  p['IR_G']        + STEP)
        elif key == ord('5'): p['IR_B']        = max(0.1,  p['IR_B']        - STEP)
        elif key == ord('6'): p['IR_B']        = min(2.0,  p['IR_B']        + STEP)
        elif key == ord('7'): p['CLAHE_CLIP']  = max(1.0,  p['CLAHE_CLIP']  - 0.5)
        elif key == ord('8'): p['CLAHE_CLIP']  = min(20.0, p['CLAHE_CLIP']  + 0.5)
        elif key == ord('9'): p['GAMMA']       = max(0.2,  p['GAMMA']       - STEP)
        elif key == ord('0'): p['GAMMA']       = min(3.0,  p['GAMMA']       + STEP)
        elif key == ord('['): p['BILATERAL_S'] = max(0,    p['BILATERAL_S'] - 10)
        elif key == ord(']'): p['BILATERAL_S'] = min(200,  p['BILATERAL_S'] + 10)

    cv2.destroyAllWindows()
    print(f"\nFinal params: {p}")
    print(f"Total saved: {saved_count} images in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()