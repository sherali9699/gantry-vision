import cv2
import numpy as np
from vision_camera_config import VisionConfig

class VisionPipeline:
    def __init__(
        self,
        min_contour_area=VisionConfig.MIN_CONTOUR_AREA,
        upper_brightness_limit=VisionConfig.BLACK_BRIGHTNESS_LIMIT,
        target_colors=VisionConfig.TARGET_COLORS,
        debug=VisionConfig.DEBUG,
    ):
        self.min_contour_area = min_contour_area
        self.upper_brightness_limit = upper_brightness_limit
        self.target_colors = [c.lower() for c in target_colors]

        self.debug = debug  # >>> NEW: store debug flag

    # --- Homography Calibration Data ---
        # Pixels from your calibration frame
        pts_src = VisionConfig.PTS_SRC
        # Absolute Gantry MM positions
        pts_dst = VisionConfig.PTS_DST

        self.h_matrix, _ = cv2.findHomography(pts_src, pts_dst)

        # Output dimensions for the top-down view (matches your gantry workspace)
        # We will make the image 960x950 to match your mm workspace 1:1
        self.output_w = VisionConfig.OUTPUT_W
        self.output_h = VisionConfig.OUTPUT_H

    # ============================================================
    # >>> NEW: small helper to show images safely (gray or BGR)
    # ============================================================
    def _dbg_show(self, win_name, img, scale=1.0):
        """
        Shows intermediate images. Works for grayscale or BGR.
        """
        if not self.debug:
            return
        if img is None:
            return

        if len(img.shape) == 2:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            vis = img.copy()

        if scale != 1.0:
            vis = cv2.resize(vis, (int(vis.shape[1] * scale), int(vis.shape[0] * scale)))

        cv2.imshow(win_name, vis)
    # ============================================================

    def get_top_down_view(self, img):
        """Warps the image to show the gantry workspace as a flat grid."""
        return cv2.warpPerspective(img, self.h_matrix, (self.output_w, self.output_h))

    def get_real_world_coords(self, px, py, angle_px):
        """Converts pixel data to Gantry MM and Real World Angle"""
        # 1. Transform Center Point
        pt = np.array([[[px, py]]], dtype=float)
        world_pt = cv2.perspectiveTransform(pt, self.h_matrix)
        wx, wy = world_pt[0][0]

        # 2. Transform Angle (handling perspective skew)
        rad = np.radians(angle_px)
        # Vector point 50px away
        v_px, v_py = px + 50 * np.cos(rad), py - 50 * np.sin(rad)
        v_pt = np.array([[[v_px, v_py]]], dtype=float)
        v_world = cv2.perspectiveTransform(v_pt, self.h_matrix)
        vwx, vwy = v_world[0][0]

        # Calculate angle based on the transformed vector
        w_angle = np.degrees(np.arctan2(-(vwy - wy), vwx - wx))
        
        return wx, wy, w_angle

    # ---------- Color masks for each color ----------
    def _get_color_masks(self, hsv_image):
        masks = {}

        if "brown" in self.target_colors:
            # Cardboard-like brown (tuned for your sample image)
            lower_brown = np.array([8,  40,  40], dtype=np.uint8)
            upper_brown = np.array([25, 255, 255], dtype=np.uint8)
            masks["brown"] = cv2.inRange(hsv_image, lower_brown, upper_brown)

        if "black" in self.target_colors:
            lower_black = np.array([0,   0,   0], dtype=np.uint8)
            upper_black = np.array([180, 255, self.upper_brightness_limit], dtype=np.uint8)
            masks["black"] = cv2.inRange(hsv_image, lower_black, upper_black)

        return masks
    # ------------------------------------------------


    def _find_inner_opening_center(self, brown_cleaned_mask, outer_rect, debug_image=None):
        """
        OPTION 1 (fixed):
        - Detect TWO largest "non-brown" blobs INSIDE the box only (thermopols)
        - Union-bbox them -> center

        Returns:
            full_center (cx, cy) in FULL IMAGE coordinates,
            union_box_points (4 pts) axis-aligned for drawing
        """
        H, W = brown_cleaned_mask.shape[:2]
        (ocx, ocy), (ow, oh), _ = outer_rect

        # ============================================================
        # >>> NEW: create a mask that represents the INSIDE of the box
        # ============================================================
        # Make a full-image blank mask
        inside_mask = np.zeros((H, W), dtype=np.uint8)

        # Get rotated rectangle corners of the outer box
        outer_box_pts = cv2.boxPoints(outer_rect)
        outer_box_pts = np.int32(outer_box_pts)

        # Fill the rotated rectangle area => inside of box region (rough)
        cv2.fillConvexPoly(inside_mask, outer_box_pts, 255)

        # Shrink it so we avoid flaps / edges leaking outside
        shrink_k = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))  # tune 25-51
        inside_mask = cv2.erode(inside_mask, shrink_k, iterations=1)

        self._dbg_show("DBG_inside_mask(shrunk box)", inside_mask)

        # ============================================================
        # >>> NEW: compute non-brown ONLY inside the box
        # ============================================================
        non_brown_full = cv2.bitwise_not(brown_cleaned_mask)              # non-brown everywhere
        non_brown_inside = cv2.bitwise_and(non_brown_full, inside_mask)   # keep only inside region

        self._dbg_show("DBG_non_brown_full", non_brown_full)
        self._dbg_show("DBG_non_brown_inside", non_brown_inside)

        # ============================================================
        # >>> NEW: cleanup non-brown-inside mask
        # ============================================================
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        nb_open  = cv2.morphologyEx(non_brown_inside, cv2.MORPH_OPEN,  k, iterations=1)
        nb_clean = cv2.morphologyEx(nb_open,         cv2.MORPH_CLOSE, k, iterations=2)

        self._dbg_show("DBG_nb_open", nb_open)
        self._dbg_show("DBG_nb_clean", nb_clean)

        # ============================================================
        # Contours on cleaned "non-brown-inside" mask
        # ============================================================
        contours, _ = cv2.findContours(nb_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None

        # ============================================================
        # Filter candidates by area (relative to box size)
        # ============================================================
        box_area = max(int(ow * oh), 1)  # approximate box area from outer rect
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 0.01 * box_area:  # tune 0.005-0.03
                continue
            candidates.append((area, cnt))

        if len(candidates) == 0:
            return None, None

        # ============================================================
        # Pick top-2 largest blobs (thermopols)
        # ============================================================
        candidates.sort(key=lambda x: x[0], reverse=True)
        top_contours = [candidates[0][1]]
        if len(candidates) >= 2:
            top_contours.append(candidates[1][1])

        # ============================================================
        # >>> MODIFIED: rotated union rectangle over both thermopols
        # ============================================================
        all_pts = np.vstack(top_contours)          # full-image points (N,1,2)

        union_rect = cv2.minAreaRect(all_pts)      # rotated rectangle (center, (w,h), angle)
        (ucx, ucy), (uw, uh), uang = union_rect

        full_center = (int(ucx), int(ucy))

        union_box_points = cv2.boxPoints(union_rect)
        union_box_points = np.int32(union_box_points)

        # ============================================================
        # >>> NEW: Debug overlay (what was selected)
        # ============================================================
        if self.debug and debug_image is not None:
            dbg = debug_image.copy()

            # Draw shrunk inside mask boundary (optional) -> just draw outer rect for reference
            cv2.polylines(dbg, [outer_box_pts], True, (255, 255, 0), 2)

            # Draw selected thermopol contours (magenta)
            for cnt in top_contours:
                cv2.drawContours(dbg, [cnt], -1, (255, 0, 255), 2)

            # Draw union bbox (green) + final center (red)
            cv2.polylines(dbg, [union_box_points], True, (0, 255, 0), 2)
            cv2.circle(dbg, full_center, 7, (0, 0, 255), -1)

            self._dbg_show("DBG_final_overlay(inside filtered)", dbg)

        return full_center, union_box_points


    def process_vision(self, img):
        color_image = img.copy()
        print("--- Vision Pipeline Execution ---")

        # >>> NEW: show input frame
        self._dbg_show("DBG_0_input", color_image)

        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        color_masks = self._get_color_masks(hsv_image)

        # >>> NEW: show raw masks (before morphology)
        if "brown" in color_masks:
            self._dbg_show("DBG_raw_brown_mask", color_masks["brown"])
        if "black" in color_masks:
            self._dbg_show("DBG_raw_black_mask", color_masks["black"])

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        output_image = color_image.copy()

        detections = []

        for color_name, mask in color_masks.items():
            # Stage 3: Morphological cleaning
            cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel, iterations=1)

            # >>> NEW: show cleaned masks per color
            self._dbg_show(f"DBG_cleaned_{color_name}_mask", cleaned_mask)

            # Stage 4: Contours
            contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                print(f"No contours found for {color_name}")
                continue

            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            print(f"[{color_name}] Largest contour area: {area}")

            # if area < self.min_contour_area:
            if (area/ (1280 * 720)) * 100 < 25: # area threshold as percentage of image area
                print(f"[{color_name}] Contour too small, ignoring.")
                continue

            # Stage 5: minAreaRect on largest contour (outer box or MV)
            rect = cv2.minAreaRect(largest_contour)
            (cx_float, cy_float), (w, h), angle_raw = rect
            cx, cy = int(cx_float), int(cy_float)

            # =====================================================
            # >>> NEW: For brown, replace center with inner union bbox center
            # =====================================================
            inner_box_points = None
            # if color_name == "brown":
            #     inner_center, inner_box_points = self._find_inner_opening_center(
            #         brown_cleaned_mask=cleaned_mask,
            #         outer_rect=rect,
            #         debug_image=output_image  # >>> NEW: used for overlay debug
            #     )
            #     if inner_center is not None:
            #         cx, cy = inner_center
            # =====================================================

            # Angle logic (unchanged)
            angle_raw = abs(angle_raw)
            if w < h:
                dominant_angle = angle_raw + 90
            else:
                dominant_angle = angle_raw

            print(f"[{color_name}] Angle: {dominant_angle:.2f}°, W: {w:.2f}, H: {h:.2f}, X: {cx}, Y: {cy}")

            #Conversion to real-worlld coordinates
            real_x, real_y, real_angle = self.get_real_world_coords(cx, cy, dominant_angle)

            # Draw outer rectangle
            box = cv2.boxPoints(rect)
            box = np.intp(box)

            if color_name == "brown":
                rect_color = (0, 140, 255)
            elif color_name == "black":
                rect_color = (255, 255, 255)
            else:
                rect_color = (0, 0, 255)

            cv2.drawContours(output_image, [box], 0, rect_color, 2)

            # >>> NEW: draw union bbox for thermopols (green) + center (red)
            if color_name == "brown" and inner_box_points is not None:
                cv2.polylines(output_image, [inner_box_points], True, (0, 255, 0), 2)
                cv2.circle(output_image, (cx, cy), 7, (0, 0, 255), -1)

            # Draw orientation line (unchanged, still based on outer rect angle)
            line_length = int(max(w, h) // 2)
            orientation_rad = np.radians(dominant_angle)

            x_end = int(cx + line_length * np.cos(orientation_rad))
            x_start = int(cx - line_length * np.cos(orientation_rad))
            y_end = int(cy - line_length * np.sin(orientation_rad))
            y_start = int(cy + line_length * np.sin(orientation_rad))

            cv2.line(output_image, (x_start, y_start), (x_end, y_end), (0, 255, 255), 1)
            cv2.line(output_image, (cx, cy), (int(cx + line_length), cy), (255, 0, 0), 1, cv2.LINE_AA)
            cv2.circle(output_image, (cx, cy), 5, (0, 255, 0), -1)

            detections.append({
                "color": color_name,
                "angle": dominant_angle,
                "center": (cx, cy),
                "rect": rect,
                "box_points": box,
                "real_world": (real_x, real_y, real_angle)
            })

            print(f"[{color_name}] Real-World -> X: {real_x:.1f} mm, Y: {real_y:.1f} mm, Angle: {real_angle:.2f}°")

        # Labels (unchanged)
        if detections:
            y0 = 30
            dy = 25
            for i, det in enumerate(detections):
                text = f"{det['color'].capitalize()} Box Angle: {det['angle']:.2f} deg"
                cv2.putText(output_image, text, (20, y0 + i * dy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            msg = "Waiting for Black/Brown Boxes..."
            print(msg)
            cv2.putText(output_image, msg, (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # >>> NEW: show final output
        self._dbg_show("DBG_output_final", output_image)

        # >>> NEW: IMPORTANT - allow imshow windows to refresh
        # call cv2.waitKey(1) in your main loop too, but this ensures refresh here
        if self.debug:
            cv2.waitKey(1)

        vis_angle = detections[0]['angle'] if detections else 0.0

        # Generate the Top-Down View
        top_down_img = self.get_top_down_view(img)
        
        # (Optional) Draw a grid on the top-down view to check accuracy
        for x in range(0, self.output_w, 100):
            cv2.line(top_down_img, (x, 0), (x, self.output_h), (50, 50, 50), 1)
        for y in range(0, self.output_h, 100):
            cv2.line(top_down_img, (0, y), (self.output_w, y), (50, 50, 50), 1)

        # return output_image, vis_angle
        return output_image, top_down_img
