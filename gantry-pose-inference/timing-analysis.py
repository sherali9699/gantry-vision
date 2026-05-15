#!/usr/bin/env python3
"""
Pipeline Timing Benchmark — Single Test Image
=============================================
Measures wall-clock time for every stage of the pose estimation pipeline:

  Stage 0 : Image load / decode
  Stage 1 : IR pink-cast correction  (correct_ir_cast)
  Stage 2 : YOLO inference            (model.predict)
  Stage 3 : Per-detection loop
    Stage 3a : solvePnP
    Stage 3b : Reprojection + axis drawing
    Stage 3c : HUD text rendering
  Stage 4 : Final display composition (optional side-by-side)
  TOTAL   : Stage 0 → Stage 4

Runs N iterations and reports: min / mean / max / std for each stage.

Usage
-----
  python benchmark_pose.py --image test_frame.jpg \
      --model best_26n_mv_pkg.pt \
      --calib camera_params_fisheye_aruko.npz \
      --iters 50
"""

import cv2
import numpy as np
import argparse
import os
import sys
import time
from collections import defaultdict
from ultralytics import YOLO


# ============================================================
# Tiny high-res timer helper
# ============================================================

class Stopwatch:
    """Lightweight context-manager stopwatch (nanoseconds → ms)."""
    def __enter__(self):
        self._t0 = time.perf_counter_ns()
        return self
    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter_ns() - self._t0) / 1e6


# ============================================================
# IR Pink Correction  (unchanged from original)
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
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ============================================================
# Per-class 3D geometry  (unchanged)
# ============================================================

def build_class_dims(mw_width, mw_depth, pkg_width, pkg_depth):
    return {
        0: {
            "name":          "microwave",
            "color":         (0, 255, 255),
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
            "color":         (0, 165, 255),
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
# Instrumented pose estimation — returns vis frame + timing dict
# ============================================================

def estimate_pose_timed(
    frame,
    model,
    camera_matrix,
    dist_coeffs,
    class_dims,
    conf_thresh=0.30,
    yolo_conf=0.25,
    imgsz=640
):
    """
    Same logic as estimate_pose_and_draw(), but every sub-stage is timed.

    Returns
    -------
    vis       : annotated frame
    timings   : dict {stage_name: elapsed_ms}
    """
    timings = {}
    vis     = frame.copy()

    # ── Stage 2 : YOLO inference ─────────────────────────────────────────────
    with Stopwatch() as sw_yolo:
        results = model.predict(source=frame, imgsz=imgsz,
                                conf=yolo_conf, verbose=False)
    timings["yolo_inference_ms"] = sw_yolo.elapsed_ms

    res = results[0]

    # ── Stage 3 : Per-detection processing ───────────────────────────────────
    t_pnp_total  = 0.0
    t_proj_total = 0.0
    t_hud_total  = 0.0
    n_detections = 0

    if res.keypoints is None or len(res.keypoints.xy) == 0:
        timings.update({
            "pnp_total_ms":          0.0,
            "reprojection_draw_total_ms": 0.0,
            "hud_draw_total_ms":     0.0,
            "n_detections":          0,
        })
        return vis, timings

    for det_idx in range(len(res.boxes)):
        class_id   = int(res.boxes.cls[det_idx].item())
        det_conf   = float(res.boxes.conf[det_idx].item())
        kpts_xy    = res.keypoints.xy[det_idx].cpu().numpy().astype(np.float32)
        kpts_conf  = res.keypoints.conf[det_idx].cpu().numpy().astype(np.float32)
        box        = res.boxes.xyxy[det_idx].cpu().numpy().astype(np.float32)

        if class_id not in class_dims:
            continue

        n_detections += 1
        dims     = class_dims[class_id]
        obj_name = dims["name"]
        color    = dims["color"]
        obj_pts  = dims["object_points"]
        W, D     = dims["W"], dims["D"]

        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        KP_NAMES  = ["TFL", "TFR", "TBL", "TBR"]
        valid_idx = np.where(kpts_conf > conf_thresh)[0]
        for i, (x, y) in enumerate(kpts_xy):
            kp_color = color if kpts_conf[i] > conf_thresh else (0, 0, 255)
            cv2.circle(vis, (int(x), int(y)), 7, kp_color, -1)
            cv2.putText(vis, f"{KP_NAMES[i]}:{kpts_conf[i]:.2f}",
                        (int(x) + 5, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, kp_color, 2)

        if len(valid_idx) < 4:
            continue

        image_points = kpts_xy.astype(np.float32)

        # ── Stage 3a : solvePnP ───────────────────────────────────────────────
        with Stopwatch() as sw_pnp:
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
        t_pnp_total += sw_pnp.elapsed_ms

        if not success:
            continue

        # ── Stage 3b : Reprojection + axis drawing ────────────────────────────
        with Stopwatch() as sw_proj:
            proj, _ = cv2.projectPoints(obj_pts, rvec, tvec,
                                        camera_matrix, dist_coeffs)
            proj     = proj.reshape(-1, 2)
            errors   = np.linalg.norm(proj - image_points, axis=1)
            mean_err = float(np.mean(errors))

            dist_origin = float(np.linalg.norm(tvec))
            R, _        = cv2.Rodrigues(rvec)
            center_obj  = np.array([[W / 2], [D / 2], [0]], dtype=np.float32)
            center_cam  = R @ center_obj + tvec
            dist_center = float(np.linalg.norm(center_cam))

            for i, (x, y) in enumerate(proj):
                if np.isfinite(x) and np.isfinite(y):
                    cv2.circle(vis, (int(x), int(y)), 5, (255, 255, 255), -1)

            axis_len = 0.20
            axis_3d  = np.array([
                [0, 0, 0], [axis_len, 0, 0],
                [0, axis_len, 0], [0, 0, -axis_len]
            ], dtype=np.float32)
            ax2d, _  = cv2.projectPoints(axis_3d, rvec, tvec,
                                         camera_matrix, dist_coeffs)
            ax2d     = ax2d.reshape(-1, 2)
            origin   = tuple(ax2d[0].astype(int))
            x_pt     = tuple(ax2d[1].astype(int))
            y_pt     = tuple(ax2d[2].astype(int))
            z_pt     = tuple(ax2d[3].astype(int))
            cv2.arrowedLine(vis, origin, x_pt, (0, 0, 255),   4, tipLength=0.15)
            cv2.arrowedLine(vis, origin, y_pt, (0, 255, 0),   4, tipLength=0.15)
            cv2.arrowedLine(vis, origin, z_pt, (255, 0, 0),   4, tipLength=0.15)
            cv2.putText(vis, "X", x_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.putText(vis, "Y", y_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(vis, "Z", z_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        t_proj_total += sw_proj.elapsed_ms

        # ── Stage 3c : HUD text ───────────────────────────────────────────────
        with Stopwatch() as sw_hud:
            hud = [
                f"[{obj_name}] conf={det_conf:.2f}",
                f"Origin : {dist_origin*100:.1f} cm",
                f"Center : {dist_center*100:.1f} cm",
                f"RpjErr : {mean_err:.1f} px",
            ]
            for i, line in enumerate(hud):
                y_pos = y1 - 20 - (len(hud) - 1 - i) * 30
                y_pos = max(y_pos, 20 + i * 30)
                cv2.putText(vis, line, (x1, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
                cv2.putText(vis, line, (x1, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        t_hud_total += sw_hud.elapsed_ms

    timings["pnp_total_ms"]              = t_pnp_total
    timings["reprojection_draw_total_ms"] = t_proj_total
    timings["hud_draw_total_ms"]         = t_hud_total
    timings["n_detections"]              = n_detections

    return vis, timings


# ============================================================
# Stats helper
# ============================================================

def stats(arr):
    a = np.array(arr)
    return dict(min=a.min(), mean=a.mean(), max=a.max(), std=a.std())


def print_stats_table(label, s, indent=4):
    pad = " " * indent
    print(f"{pad}{label:<38} "
          f"min={s['min']:7.3f}  mean={s['mean']:7.3f}  "
          f"max={s['max']:7.3f}  std={s['std']:6.3f}  ms")


# ============================================================
# Main benchmark loop
# ============================================================

def run_benchmark(
    image_path,
    model_path,
    calib_file,
    mw_width, mw_depth,
    pkg_width, pkg_depth,
    strength,
    conf_thresh, yolo_conf, imgsz,
    iters,
    warmup,
    save_annotated
):
    # ── Load model & calibration ─────────────────────────────────────────────
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    print(f"\n[INFO] Loading YOLO model: {model_path}")
    model = YOLO(model_path)
    print("[INFO] YOLO model loaded.")

    if not os.path.exists(calib_file):
        raise FileNotFoundError(f"Calibration file not found: {calib_file}")
    data          = np.load(calib_file)
    camera_matrix = data["mtx"]
    dist_coeffs   = data["dist"]
    print("[INFO] Calibration loaded.")

    class_dims = build_class_dims(mw_width, mw_depth, pkg_width, pkg_depth)

    # ── Stage 0 : Load image once (report separately) ────────────────────────
    with Stopwatch() as sw_load:
        raw_frame = cv2.imread(image_path)
    if raw_frame is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    print(f"[INFO] Image loaded: {image_path}  "
          f"({raw_frame.shape[1]}x{raw_frame.shape[0]})  "
          f"load time = {sw_load.elapsed_ms:.3f} ms\n")

    # ── Warm-up runs (not recorded) ───────────────────────────────────────────
    print(f"[INFO] Warm-up: {warmup} iteration(s)…")
    for _ in range(warmup):
        corrected = correct_ir_cast(raw_frame, strength)
        estimate_pose_timed(corrected, model, camera_matrix, dist_coeffs,
                            class_dims, conf_thresh, yolo_conf, imgsz)
    print("[INFO] Warm-up done.\n")

    # ── Timed iterations ──────────────────────────────────────────────────────
    print(f"[INFO] Running {iters} timed iteration(s)…")

    records = defaultdict(list)
    last_vis = None

    for it in range(iters):
        # ── total wall clock starts here ──────────────────────────────────────
        with Stopwatch() as sw_total:

            # Stage 1 : IR correction
            with Stopwatch() as sw_ir:
                corrected = correct_ir_cast(raw_frame, strength)
            t_ir = sw_ir.elapsed_ms

            # Stages 2-3 (YOLO + PnP + draw)
            vis, timings = estimate_pose_timed(
                corrected, model, camera_matrix, dist_coeffs,
                class_dims, conf_thresh, yolo_conf, imgsz
            )

            # Stage 4 : compose display (no actual imshow in benchmark)
            with Stopwatch() as sw_disp:
                display = vis.copy()   # mirrors the live loop's display = pose_view
            t_disp = sw_disp.elapsed_ms

        t_total = sw_total.elapsed_ms

        records["ir_correction_ms"].append(t_ir)
        records["yolo_inference_ms"].append(timings["yolo_inference_ms"])
        records["pnp_total_ms"].append(timings["pnp_total_ms"])
        records["reprojection_draw_total_ms"].append(timings["reprojection_draw_total_ms"])
        records["hud_draw_total_ms"].append(timings["hud_draw_total_ms"])
        records["display_compose_ms"].append(t_disp)
        records["total_pipeline_ms"].append(t_total)
        records["n_detections"].append(timings["n_detections"])

        last_vis = display

        if (it + 1) % 10 == 0 or it == 0:
            print(f"  iter {it+1:4d}/{iters}  total={t_total:7.3f} ms  "
                  f"yolo={timings['yolo_inference_ms']:7.3f} ms  "
                  f"pnp={timings['pnp_total_ms']:6.3f} ms  "
                  f"dets={timings['n_detections']}")

    # ── Results ───────────────────────────────────────────────────────────────
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  PIPELINE TIMING REPORT  ({iters} iterations, {warmup} warm-up)")
    print(sep)

    order = [
        ("ir_correction_ms",            "Stage 1  IR pink-cast correction"),
        ("yolo_inference_ms",           "Stage 2  YOLO inference"),
        ("pnp_total_ms",                "Stage 3a solvePnP  (sum over dets)"),
        ("reprojection_draw_total_ms",  "Stage 3b Reprojection + axis draw"),
        ("hud_draw_total_ms",           "Stage 3c HUD text rendering"),
        ("display_compose_ms",          "Stage 4  Display composition"),
        ("total_pipeline_ms",           "TOTAL    End-to-end"),
    ]

    for key, label in order:
        s = stats(records[key])
        print_stats_table(label, s)
        # breakdown as % of total for non-total rows
        if key != "total_pipeline_ms":
            pct = s["mean"] / stats(records["total_pipeline_ms"])["mean"] * 100
            print(f"          {'':38} → {pct:5.1f}% of total")

    mean_total = stats(records["total_pipeline_ms"])["mean"]
    eff_fps    = 1000.0 / mean_total if mean_total > 0 else float("inf")
    mean_dets  = np.mean(records["n_detections"])

    print(sep)
    print(f"  Effective throughput (no I/O): {eff_fps:6.1f} FPS  "
          f"(mean total = {mean_total:.3f} ms)")
    print(f"  Mean detections per frame    : {mean_dets:.1f}")
    print(sep)

    # ── Save annotated frame ──────────────────────────────────────────────────
    if save_annotated and last_vis is not None:
        out_path = "benchmark_annotated.png"
        cv2.imwrite(out_path, last_vis)
        print(f"\n[SAVED] Annotated frame → {out_path}")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single-image pipeline timing benchmark"
    )
    parser.add_argument("--image", default="./time-test-image.png",
                    help="Path to test image (JPG / PNG)")
    parser.add_argument("--model",       default="best_26n_mv_pkg.pt")
    parser.add_argument("--calib",       default="camera_params_fisheye_aruko.npz")

    parser.add_argument("--mw-width",    type=float, default=0.44)
    parser.add_argument("--mw-depth",    type=float, default=0.61)
    parser.add_argument("--pkg-width",   type=float, default=0.57)
    parser.add_argument("--pkg-depth",   type=float, default=0.67)

    parser.add_argument("--strength",    type=float, default=1.0)
    parser.add_argument("--conf-thresh", type=float, default=0.30)
    parser.add_argument("--yolo-conf",   type=float, default=0.25)
    parser.add_argument("--imgsz",       type=int,   default=640)

    parser.add_argument("--iters",       type=int,   default=50,
                        help="Number of timed iterations (default 50)")
    parser.add_argument("--warmup",      type=int,   default=3,
                        help="Warm-up iterations before timing (default 3)")
    parser.add_argument("--save-annotated", action="store_true", default=True,
                        help="Save the last annotated frame to benchmark_annotated.png")

    args = parser.parse_args()

    run_benchmark(
        image_path   = args.image,
        model_path   = args.model,
        calib_file   = args.calib,
        mw_width     = args.mw_width,
        mw_depth     = args.mw_depth,
        pkg_width    = args.pkg_width,
        pkg_depth    = args.pkg_depth,
        strength     = args.strength,
        conf_thresh  = args.conf_thresh,
        yolo_conf    = args.yolo_conf,
        imgsz        = args.imgsz,
        iters        = args.iters,
        warmup       = args.warmup,
        save_annotated = args.save_annotated,
    )