"""
ChArUco detection parameter tuner.
Reads all calibration images and lets you tune preprocessing + detector params
to maximise corner detection before running full calibration.

Run from ~/gantry-vision:
  python charuco_tuner.py

Controls:
  LEFT/RIGHT arrows  — cycle through images
  S                  — save current params to charuco_params.npy
  Q / ESC            — quit
  (tweak params in the TUNABLE CONFIG section and re-run)
"""

import cv2
import numpy as np
import glob
from pathlib import Path

# ─────────────────────────────────────────────
# FIXED CONFIG
# ─────────────────────────────────────────────
SQUARES_X     = 6
SQUARES_Y     = 4
SQUARE_SIZE_M = 0.0475
MARKER_SIZE_M = 0.035
ARUCO_DICT    = cv2.aruco.DICT_6X6_250

CALIB_IMAGES_GLOB = "calibration_images_distorted/calib_raw_*.png"
OUTPUT_PARAMS     = "charuco_params.npy"

# ─────────────────────────────────────────────
# TUNABLE CONFIG — edit these and re-run
# ─────────────────────────────────────────────

# IR correction — channel multipliers (BGR order)
IR_B = 0.85
IR_G = 1.05
IR_R = 0.70

# Bilateral filter (set to 0 to disable)
BILATERAL_D          = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75

# CLAHE
CLAHE_CLIP_LIMIT  = 6.0
CLAHE_TILE_SIZE   = 4      # tileGridSize = (N, N)

# Gamma correction (1.0 = no change, <1 = brighten, >1 = darken)
GAMMA = 0.8

# ArUco detector params
ARUCO_ADAPT_THRESH_WIN_SIZE_MIN  = 3
ARUCO_ADAPT_THRESH_WIN_SIZE_MAX  = 23
ARUCO_ADAPT_THRESH_WIN_SIZE_STEP = 10
ARUCO_MIN_MARKER_PERIM_RATE      = 0.02
ARUCO_MAX_MARKER_PERIM_RATE      = 4.0
ARUCO_CORNER_REFINEMENT_METHOD   = cv2.aruco.CORNER_REFINE_SUBPIX


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(img):
    # IR correction
    f = img.astype(np.float32)
    f[:, :, 0] *= IR_B
    f[:, :, 1] *= IR_G
    f[:, :, 2] *= IR_R
    img = np.clip(f, 0, 255).astype(np.uint8)

    # Bilateral filter
    if BILATERAL_D > 0:
        img = cv2.bilateralFilter(img, BILATERAL_D,
                                   BILATERAL_SIGMA_COLOR,
                                   BILATERAL_SIGMA_SPACE)

    # CLAHE on L channel
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT,
                              tileGridSize=(CLAHE_TILE_SIZE, CLAHE_TILE_SIZE))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # Gamma correction
    if GAMMA != 1.0:
        lut = np.array([((i / 255.0) ** GAMMA) * 255
                         for i in range(256)], dtype=np.uint8)
        img = cv2.LUT(img, lut)

    return img


# ─────────────────────────────────────────────
# DETECTION
# ─────────────────────────────────────────────

def make_detector():
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), SQUARE_SIZE_M, MARKER_SIZE_M, aruco_dict
    )

    det_params = cv2.aruco.DetectorParameters()
    det_params.adaptiveThreshWinSizeMin  = ARUCO_ADAPT_THRESH_WIN_SIZE_MIN
    det_params.adaptiveThreshWinSizeMax  = ARUCO_ADAPT_THRESH_WIN_SIZE_MAX
    det_params.adaptiveThreshWinSizeStep = ARUCO_ADAPT_THRESH_WIN_SIZE_STEP
    det_params.minMarkerPerimeterRate    = ARUCO_MIN_MARKER_PERIM_RATE
    det_params.maxMarkerPerimeterRate    = ARUCO_MAX_MARKER_PERIM_RATE
    det_params.cornerRefinementMethod    = ARUCO_CORNER_REFINEMENT_METHOD

    charuco_params   = cv2.aruco.CharucoParameters()
    charuco_detector = cv2.aruco.CharucoDetector(board, charuco_params,
                                                  det_params)
    return charuco_detector, board


def detect(img_raw, charuco_detector):
    img_proc = preprocess(img_raw)
    gray     = cv2.cvtColor(img_proc, cv2.COLOR_BGR2GRAY)
    ch_corners, ch_ids, markers, marker_ids = charuco_detector.detectBoard(gray)
    n_corners = len(ch_corners) if ch_corners is not None else 0
    n_markers = len(markers)    if markers    is not None else 0
    return ch_corners, ch_ids, markers, marker_ids, img_proc, n_corners, n_markers


# ─────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────

def draw_result(img_raw, img_proc, ch_corners, ch_ids,
                markers, marker_ids, n_corners, n_markers, idx, total):
    h, w = img_raw.shape[:2]

    # Left: preprocessed with detections
    vis_proc = img_proc.copy()
    if markers is not None and len(markers) > 0:
        cv2.aruco.drawDetectedMarkers(vis_proc, markers, marker_ids, (0, 200, 255))
    if ch_corners is not None and len(ch_corners) > 0:
        cv2.aruco.drawDetectedCornersCharuco(vis_proc, ch_corners, ch_ids, (0, 255, 0))

    # Right: raw with detections
    vis_raw = img_raw.copy()
    if markers is not None and len(markers) > 0:
        cv2.aruco.drawDetectedMarkers(vis_raw, markers, marker_ids, (0, 200, 255))
    if ch_corners is not None and len(ch_corners) > 0:
        cv2.aruco.drawDetectedCornersCharuco(vis_raw, ch_corners, ch_ids, (0, 255, 0))

    # Labels
    status_color = (0, 255, 0) if n_corners >= 6 else (0, 80, 255)
    for vis, label in [(vis_proc, "PREPROCESSED"), (vis_raw, "RAW")]:
        cv2.putText(vis, label, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        cv2.putText(vis,
                    f"Image {idx+1}/{total}  |  markers={n_markers}  corners={n_corners}/15",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        usable = "USABLE" if n_corners >= 6 else "SKIPPED"
        cv2.putText(vis, usable, (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

    # Scale down for display if image is large
    scale = min(1.0, 900 / w)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        vis_proc = cv2.resize(vis_proc, (new_w, new_h))
        vis_raw  = cv2.resize(vis_raw,  (new_w, new_h))

    return np.hstack([vis_proc, vis_raw])


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────

def print_summary(results):
    usable = [r for r in results if r['n_corners'] >= 6]
    print(f"\n{'='*60}")
    print(f"SUMMARY: {len(usable)} / {len(results)} images usable (≥6 corners)")
    print(f"{'='*60}")
    total_corners = sum(r['n_corners'] for r in usable)
    print(f"Total corners across usable images: {total_corners}")
    print(f"\nPer-image:")
    for r in results:
        status = "✓" if r['n_corners'] >= 6 else "✗"
        print(f"  {status} {Path(r['path']).name:<45} "
              f"markers={r['n_markers']:>2}  corners={r['n_corners']:>2}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    image_paths = sorted(glob.glob(CALIB_IMAGES_GLOB))
    assert len(image_paths) > 0, f"No images found at {CALIB_IMAGES_GLOB}"
    print(f"Found {len(image_paths)} images.")

    charuco_detector, board = make_detector()

    # Run detection on all images
    results = []
    for p in image_paths:
        img_raw = cv2.imread(p)
        if img_raw is None:
            continue
        ch_corners, ch_ids, markers, marker_ids, img_proc, n_corners, n_markers = \
            detect(img_raw, charuco_detector)
        results.append({
            'path':      p,
            'img_raw':   img_raw,
            'img_proc':  img_proc,
            'ch_corners': ch_corners,
            'ch_ids':     ch_ids,
            'markers':    markers,
            'marker_ids': marker_ids,
            'n_corners':  n_corners,
            'n_markers':  n_markers,
        })

    print_summary(results)

    # Interactive viewer
    idx = 0
    win = "ChArUco Tuner  |  LEFT/RIGHT: cycle  |  S: save params  |  Q: quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1800, 540)

    print("\nOpening viewer … use LEFT/RIGHT arrows to cycle images.")
    print("Edit TUNABLE CONFIG at top of script and re-run to see changes.\n")

    while True:
        r = results[idx]
        frame = draw_result(
            r['img_raw'], r['img_proc'],
            r['ch_corners'], r['ch_ids'],
            r['markers'], r['marker_ids'],
            r['n_corners'], r['n_markers'],
            idx, len(results)
        )
        cv2.imshow(win, frame)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == 81 or key == ord('a'):   # left arrow or a
            idx = (idx - 1) % len(results)
        elif key == 83 or key == ord('d'):   # right arrow or d
            idx = (idx + 1) % len(results)
        elif key == ord('s'):
            params = {
                'IR_B': IR_B, 'IR_G': IR_G, 'IR_R': IR_R,
                'BILATERAL_D': BILATERAL_D,
                'BILATERAL_SIGMA_COLOR': BILATERAL_SIGMA_COLOR,
                'BILATERAL_SIGMA_SPACE': BILATERAL_SIGMA_SPACE,
                'CLAHE_CLIP_LIMIT': CLAHE_CLIP_LIMIT,
                'CLAHE_TILE_SIZE': CLAHE_TILE_SIZE,
                'GAMMA': GAMMA,
            }
            np.save(OUTPUT_PARAMS, params, allow_pickle=True)
            print(f"Params saved → {OUTPUT_PARAMS}")

    cv2.destroyAllWindows()
    print_summary(results)


if __name__ == "__main__":
    main()