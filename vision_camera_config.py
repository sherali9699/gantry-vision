import numpy as np

class VisionConfig:

    # ===== COLORS =====
    TARGET_COLORS = ("black", "brown")

    # ===== MASK THRESHOLDS =====
    MIN_CONTOUR_AREA = 3000
    BLACK_BRIGHTNESS_LIMIT = 45

    # ===== DEBUG =====
    DEBUG = False

    # ===== CAMERA =====
    CAMERA_ID = 2
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    CAMERA_FPS = 120

    # ===== HOMOGRAPHY =====
    PTS_SRC = np.array([
        [234, 302],
        [234, 776],
        [840, 302],
        [840, 776]
    ], dtype=float)

    PTS_DST = np.array([
        [288, 111],
        [288, 718],
        [900, 111],
        [900, 718]
    ], dtype=float)

    OUTPUT_W = 960
    OUTPUT_H = 950