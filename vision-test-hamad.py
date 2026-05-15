#!/usr/bin/env python3
#(hamad)
"""
IR Camera Live Feed — removes washed-out pink cast from cameras without IR filters.
Includes frame capture functionality with timestamped filenames.
"""

import cv2
import numpy as np
import argparse
import sys
import os
from datetime import datetime

# ──────────────────────────────────────────────
# IR Pink Correction
# ──────────────────────────────────────────────
def correct_ir_cast(frame: np.ndarray, strength: float = 1.0) -> np.ndarray:
    frame = frame.astype(np.float32)

    r, g, b = cv2.split(frame)
    r_corrected = r * (1.0 - 0.30 * strength)
    g_corrected = g * (1.0 + 0.05 * strength)
    b_corrected = b * (1.0 - 0.15 * strength)

    frame = cv2.merge([
        np.clip(r_corrected, 0, 255),
        np.clip(g_corrected, 0, 255),
        np.clip(b_corrected, 0, 255),
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


# ──────────────────────────────────────────────
# Camera Setup
# ──────────────────────────────────────────────
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

    return cap


# ──────────────────────────────────────────────
# Main Loop
# ──────────────────────────────────────────────
def run(device: str, width: int, height: int, strength: float, show_original: bool):
    cap = open_camera(device, width, height)
    
    # Create a directory for captures
    #output_dir = "yolo-train-images-with-version-06"
    output_dir = "cam-caliberation"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    win_name = "IR Feed — Q: Quit | S: Capture | O: Toggle Original | +/-: Strength"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    
    print("\n[CONTROLS]")
    print("  Q / ESC    — quit")
    print("  S          — capture current corrected frame")
    print("  +/-        — adjust correction strength")
    print("  O          — toggle side-by-side original")
    print(f"\n[INFO] Saving frames to: {os.path.abspath(output_dir)}")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        corrected = correct_ir_cast(frame, strength=strength)

        if show_original:
            label_orig = frame.copy()
            label_corr = corrected.copy()
            cv2.putText(label_orig, "ORIGINAL", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(label_corr, f"CORRECTED str={strength:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            display = np.hstack([label_orig, label_corr])
        else:
            display = corrected.copy()
            cv2.putText(display, f"STR: {strength:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(win_name, display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord('+') or key == ord('='):
            strength = min(strength + 0.05, 2.0)
        elif key == ord('-'):
            strength = max(strength - 0.05, 0.0)
        elif key == ord('o'):
            show_original = not show_original
        elif key == ord('s'):
            # Generate filename based on current timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            fname = os.path.join(output_dir, f"ir_frame_{timestamp}.png")
            
            # Save the corrected frame (not the 'display' version with text overlays)
            cv2.imwrite(fname, corrected)
            print(f"[SAVED] {fname}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IR camera feed with pink-cast correction")
    parser.add_argument("--device",   default="/dev/video2")
    parser.add_argument("--width",    type=int, default=1920)
    parser.add_argument("--height",   type=int, default=1080)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--side-by-side", action="store_true")
    args = parser.parse_args()

    run(args.device, args.width, args.height, args.strength, args.side_by_side)















# #!/usr/bin/env python3
# #(hamad)
# """
# IR Camera Live Feed — removes washed-out pink cast from cameras without IR filters.
# Uses v4l2 capture with MJPEG, scales to 1920x1080, applies CLAHE + IR correction.

# Usage:
#     python3 ir_feed.py [--device /dev/video2] [--width 1920] [--height 1080]
    
# Dependencies:
#     pip install opencv-python numpy
#     sudo apt install v4l-utils  # optional, for v4l2-ctl diagnostics
# """

# import cv2
# import numpy as np
# import argparse
# import sys

# # ──────────────────────────────────────────────
# # IR Pink Correction
# # ──────────────────────────────────────────────
# def correct_ir_cast(frame: np.ndarray, strength: float = 1.0) -> np.ndarray:
#     """
#     Remove the pink/magenta cast introduced by IR light hitting a camera
#     without an IR-cut filter.

#     The pink cast comes from:
#       - Red channel over-saturated by near-IR (700–1000 nm)
#       - Blue channel slightly elevated
#       - Green channel relatively unaffected

#     Strategy:
#       1. Suppress R channel, slightly lift G, suppress B
#       2. Convert to LAB and apply CLAHE on L (brightness/haze)
#       3. Desaturate the A channel (red–green axis) to kill residual pink
#     """
#     frame = frame.astype(np.float32)

#     # Step 1 — channel mixing to neutralise IR pink
#     # These ratios were tuned for typical no-IR-cut cameras outdoors.
#     # Adjust `strength` (0.0 = no correction, 1.0 = full, >1.0 = aggressive)
#     r, g, b = cv2.split(frame)

#     r_corrected = r * (1.0 - 0.30 * strength)          # kill excess red
#     g_corrected = g * (1.0 + 0.05 * strength)           # slight green lift
#     b_corrected = b * (1.0 - 0.15 * strength)           # trim IR-lifted blue

#     frame = cv2.merge([
#         np.clip(r_corrected, 0, 255),
#         np.clip(g_corrected, 0, 255),
#         np.clip(b_corrected, 0, 255),
#     ]).astype(np.uint8)

#     # Step 2 — CLAHE on L channel (reduces haze / restores local contrast)
#     lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
#     l, a, b_ch = cv2.split(lab)

#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#     l = clahe.apply(l)

#     # Step 3 — pull A channel toward neutral to reduce residual pink cast
#     # A channel: 128 = neutral, >128 = more red/pink, <128 = more green
#     a = cv2.addWeighted(a, 1.0 - 0.25 * strength,
#                         np.full_like(a, 128), 0.25 * strength, 0)

#     lab = cv2.merge((l, a, b_ch))
#     frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

#     return frame


# # ──────────────────────────────────────────────
# # Camera Setup
# # ──────────────────────────────────────────────
# def open_camera(device: str, width: int, height: int) -> cv2.VideoCapture:
#     """
#     Open a v4l2 device, request MJPEG codec, and set resolution.
#     MJPEG offloads decompression from the USB bus — essential for HD framerates.
#     """
#     # V4L2 backend + MJPEG fourcc
#     cap = cv2.VideoCapture(device, cv2.CAP_V4L2)

#     if not cap.isOpened():
#         print(f"[ERROR] Could not open {device}")
#         sys.exit(1)

#     # Force MJPEG on the camera side (reduces USB bandwidth ~10x vs YUYV)
#     fourcc = cv2.VideoWriter_fourcc(*"MJPG")
#     cap.set(cv2.CAP_PROP_FOURCC, fourcc)

#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
#     cap.set(cv2.CAP_PROP_FPS, 30)

#     # Reduce internal buffer to 1 frame — keeps feed live, not buffered
#     cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

#     # Report what we actually got
#     actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     actual_fps = cap.get(cv2.CAP_PROP_FPS)
#     actual_cc = int(cap.get(cv2.CAP_PROP_FOURCC))
#     cc_str = "".join([chr((actual_cc >> (8 * i)) & 0xFF) for i in range(4)])

#     print(f"[INFO] Device  : {device}")
#     print(f"[INFO] Codec   : {cc_str}")
#     print(f"[INFO] Resolution: {actual_w}x{actual_h} @ {actual_fps:.1f} fps")

#     return cap


# # ──────────────────────────────────────────────
# # Main Loop
# # ──────────────────────────────────────────────
# def run(device: str, width: int, height: int, strength: float, show_original: bool):
#     cap = open_camera(device, width, height)

#     win_name = "IR Feed — press Q to quit | +/- to adjust strength | O toggle original"
#     cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
#     cv2.resizeWindow(win_name, min(width, 1280), min(height, 480 if show_original else 720))

#     print("\n[CONTROLS]")
#     print("  Q          — quit")
#     print("  +/-        — increase/decrease IR correction strength")
#     print("  O          — toggle side-by-side original")
#     print("  S          — save screenshot")
#     print(f"\n[INFO] IR correction strength: {strength:.2f}\n")

#     frame_count = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("[WARN] Frame grab failed — retrying...")
#             continue

#         corrected = correct_ir_cast(frame, strength=strength)

#         if show_original:
#             # Side-by-side: original | corrected
#             label_orig = frame.copy()
#             label_corr = corrected.copy()
#             cv2.putText(label_orig, "ORIGINAL", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#             cv2.putText(label_corr, f"CORRECTED  str={strength:.2f}", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             display = np.hstack([label_orig, label_corr])
#         else:
#             display = corrected
#             cv2.putText(display, f"IR correction: {strength:.2f}", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#         cv2.imshow(win_name, display)

#         key = cv2.waitKey(1) & 0xFF

#         if key == ord('q') or key == 27:
#             break
#         elif key == ord('+') or key == ord('='):
#             strength = min(strength + 0.05, 2.0)
#             print(f"[INFO] Strength → {strength:.2f}")
#         elif key == ord('-'):
#             strength = max(strength - 0.05, 0.0)
#             print(f"[INFO] Strength → {strength:.2f}")
#         elif key == ord('o'):
#             show_original = not show_original
#         elif key == ord('s'):
#             fname = f"ir_screenshot_{frame_count:04d}.png"
#             cv2.imwrite(fname, corrected)
#             print(f"[INFO] Saved {fname}")

#         frame_count += 1

#     cap.release()
#     cv2.destroyAllWindows()


# # ──────────────────────────────────────────────
# # Entry Point
# # ──────────────────────────────────────────────
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="IR camera feed with pink-cast correction")
#     parser.add_argument("--device",   default="/dev/video2", help="v4l2 device path")
#     parser.add_argument("--width",    type=int, default=1920, help="capture width")
#     parser.add_argument("--height",   type=int, default=1080, help="capture height")
#     parser.add_argument("--strength", type=float, default=1.0,
#                         help="IR correction strength 0.0–2.0 (default 1.0)")
#     parser.add_argument("--side-by-side", action="store_true",
#                         help="Show original and corrected side by side")
#     args = parser.parse_args()

#     run(
#         device=args.device,
#         width=args.width,
#         height=args.height,
#         strength=args.strength,
#         show_original=args.side_by_side,
#     )
