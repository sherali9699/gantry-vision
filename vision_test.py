# ------------------------working MinRect version below------------------------

import numpy as np
import cv2

class VisionPipeline:
    def __init__(self, upper_brightness_limit=50, min_contour_area=500):
        self.upper_brightness_limit = upper_brightness_limit
        self.min_contour_area = min_contour_area

    def process_vision(self, img):
        color_image = img.copy() 
        print("--- Vision Pipeline Execution ---")
        
        # --- Stage 1 & 2: Color Conversion and Thresholding ---
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)
        _, _, V = cv2.split(hsv_image)
        dark_mask = cv2.inRange(V, 0, self.upper_brightness_limit)
        
        # cv2.imshow("1 & 2. Dark Mask", dark_mask)
        # cv2.waitKey(500) 

        # --- Stage 3: Morphological Cleaning ---
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # cv2.imshow("3. Cleaned Mask", cleaned_mask)
        # cv2.waitKey(500)
        
        # --- Stage 4: Contour Detection and Initialization ---
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output_image = color_image.copy()
        dominant_angle = None
        microwave_detected = False

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(largest_contour) > self.min_contour_area:
                microwave_detected = True
                # --- Stage 5: Orientation using Minimum Area Rectangle (MAR) ---
                
                # 1. Fit the minimum area rotated rectangle
                rect = cv2.minAreaRect(largest_contour)
                (cx_float, cy_float), (w, h), angle_raw = rect
                
                # Convert center coordinates to integers for drawing functions (Fixes error)
                cx, cy = int(cx_float), int(cy_float)

                # Adjust angle to a more intuitive representation
                angle_raw = abs(90 - angle_raw) 
                print(f"Raw MAR Angle: {angle_raw:.2f}°, Width: {w:.2f}, Height: {h:.2f}")
                
                
                # 2. Normalize the Angle: Adjust OpenCV's angle to the standard 0-180 range
                # OpenCV's minAreaRect angle convention can be tricky. A common normalization:
                if w < h: 
                    # If height (h) is the longest side, angle is measured from the Y-axis. 
                    # Add 90 to reference the X-axis.
                    dominant_angle = angle_raw + 90
                else: 
                    # If width (w) is the longest side, angle is measured from the X-axis.
                    dominant_angle = angle_raw
                
                # print(f"Normalized MAR Angle: {dominant_angle:.2f}°")

                # Ensure angle is in the standard [0, 180) range if desired, or use the raw output.
                # Here, we keep the simplest representation.
                
                
                # 3. Visualization: Draw the Rotated Rectangle and Axis
                
                # Get the 4 corners of the rotated rectangle
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                
                # Draw the rotated rectangle (in RED)
                cv2.drawContours(output_image, [box], 0, (0, 0, 255), 2)
                
                # Draw the orientation line (Major Axis)
                line_length = max(w, h) // 2
                
                # Convert the final angle back to radians for trig functions
                orientation_rad = np.radians(dominant_angle)
                
                # x_end and x_start (Horizontal Axis) remain the same
                x_end = int(cx + line_length * np.cos(orientation_rad))
                x_start = int(cx - line_length * np.cos(orientation_rad))
                
                # y_end and y_start (Vertical Axis) must be negated to flip the drawing direction
                # np.sin is used to flip the y-direction, thus drawing CCW
                y_end = int(cy - line_length * np.sin(orientation_rad)) 
                y_start = int(cy + line_length * np.sin(orientation_rad))
                
                # Draw the Major Axis (in YELLOW)
                cv2.line(output_image, (x_start, y_start), (x_end, y_end), (0, 255, 255), 1)
                
                # Draw the Horizontal Reference Line (in BLUE)
                cv2.line(output_image, (cx, cy), (int(cx + line_length), cy), (255, 0, 0), 1, cv2.LINE_AA)
                
                # Draw Centroid (Green dot)
                cv2.circle(output_image, (cx, cy), 5, (0, 255, 0), -1)
                
                print(f"5. Orientation Detected (MAR): {dominant_angle:.2f}°")

        # --- Stage 6: Final Display ---
        if microwave_detected:
             label = f"Angle: {dominant_angle:.2f} deg (MAR)"
        else:
             label = "Waiting for Microwave..."
             print(label)
        
        cv2.putText(output_image, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # cv2.imshow("6. Final Result (Minimum Area Rectangle)", output_image)
        # cv2.waitKey(0) 
        
        # cv2.destroyAllWindows()
        # return dominant_angle
        return output_image

# # # Usage example:
# # # Ensure you have 'new_image.png' ready!
vision = VisionPipeline()
image = cv2.imread("./new_image.png")
if image is not None:
    cv2.imshow("Processed image",   cv2.cvtColor(vision.process_vision(image),   cv2.COLOR_RGB2BGR))
    # vision.process_vision(image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("🚨 Error: Image was not loaded. Check the file path/integrity.")








# # ------------------------working MinRect version below------------------------

# import numpy as np
# import cv2

# class VisionPipeline:
#     def __init__(self, upper_brightness_limit=50, min_contour_area=500):
#         self.upper_brightness_limit = upper_brightness_limit
#         self.min_contour_area = min_contour_area

#     def process_vision(self, img):
#         color_image = img.copy() 
#         print("--- Vision Pipeline Execution ---")
        
#         # --- Stage 1 & 2: Color Conversion and Thresholding ---
#         hsv_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)
#         _, _, V = cv2.split(hsv_image)
#         dark_mask = cv2.inRange(V, 0, self.upper_brightness_limit)
        
#         cv2.imshow("1 & 2. Dark Mask", dark_mask)
#         cv2.waitKey(500) 

#         # --- Stage 3: Morphological Cleaning ---
#         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#         cleaned_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
#         cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
#         cv2.imshow("3. Cleaned Mask", cleaned_mask)
#         cv2.waitKey(500)
        
#         # --- Stage 4: Contour Detection and Initialization ---
#         contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         output_image = color_image.copy()
#         dominant_angle = None
#         microwave_detected = False

#         if contours:
#             largest_contour = max(contours, key=cv2.contourArea)
            
#             if cv2.contourArea(largest_contour) > self.min_contour_area:
#                 microwave_detected = True
#                 # --- Stage 5: Orientation using Minimum Area Rectangle (MAR) ---
                
#                 # 1. Fit the minimum area rotated rectangle
#                 rect = cv2.minAreaRect(largest_contour)
#                 (cx_float, cy_float), (w, h), angle_raw = rect
                
#                 # Convert center coordinates to integers for drawing functions (Fixes error)
#                 cx, cy = int(cx_float), int(cy_float)

#                 # Adjust angle to a more intuitive representation
#                 angle_raw = abs(90 - angle_raw) 
#                 print(f"Raw MAR Angle: {angle_raw:.2f}°, Width: {w:.2f}, Height: {h:.2f}")
                
                
#                 # 2. Normalize the Angle: Adjust OpenCV's angle to the standard 0-180 range
#                 # OpenCV's minAreaRect angle convention can be tricky. A common normalization:
#                 if w < h: 
#                     # If height (h) is the longest side, angle is measured from the Y-axis. 
#                     # Add 90 to reference the X-axis.
#                     dominant_angle = angle_raw + 90
#                 else: 
#                     # If width (w) is the longest side, angle is measured from the X-axis.
#                     dominant_angle = angle_raw
                
#                 print(f"Normalized MAR Angle: {dominant_angle:.2f}°")

#                 # Ensure angle is in the standard [0, 180) range if desired, or use the raw output.
#                 # Here, we keep the simplest representation.
                
                
#                 # 3. Visualization: Draw the Rotated Rectangle and Axis
                
#                 # Get the 4 corners of the rotated rectangle
#                 box = cv2.boxPoints(rect)
#                 box = np.intp(box)
                
#                 # Draw the rotated rectangle (in RED)
#                 cv2.drawContours(output_image, [box], 0, (0, 0, 255), 2)
                
#                 # Draw the orientation line (Major Axis)
#                 line_length = max(w, h) // 2
                
#                 # Convert the final angle back to radians for trig functions
#                 orientation_rad = np.radians(dominant_angle)
                
#                 # x_end and x_start (Horizontal Axis) remain the same
#                 x_end = int(cx + line_length * np.cos(orientation_rad))
#                 x_start = int(cx - line_length * np.cos(orientation_rad))
                
#                 # y_end and y_start (Vertical Axis) must be negated to flip the drawing direction
#                 # np.sin is used to flip the y-direction, thus drawing CCW
#                 y_end = int(cy - line_length * np.sin(orientation_rad)) 
#                 y_start = int(cy + line_length * np.sin(orientation_rad))
                
#                 # Draw the Major Axis (in YELLOW)
#                 cv2.line(output_image, (x_start, y_start), (x_end, y_end), (0, 255, 255), 1)
                
#                 # Draw the Horizontal Reference Line (in BLUE)
#                 cv2.line(output_image, (cx, cy), (int(cx + line_length), cy), (255, 0, 0), 1, cv2.LINE_AA)
                
#                 # Draw Centroid (Green dot)
#                 cv2.circle(output_image, (cx, cy), 5, (0, 255, 0), -1)
                
#                 print(f"5. Orientation Detected (MAR): {dominant_angle:.2f}°")

#         # --- Stage 6: Final Display ---
#         if microwave_detected:
#              label = f"Angle: {dominant_angle:.2f} deg (MAR)"
#         else:
#              label = "Waiting for Microwave..."
#              print(label)
        
#         cv2.putText(output_image, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
#         cv2.imshow("6. Final Result (Minimum Area Rectangle)", output_image)
#         cv2.waitKey(0) 
        
#         cv2.destroyAllWindows()
#         return dominant_angle
#         # return output_image

# # Usage example:
# # Ensure you have 'new_image.png' ready!
# vision = VisionPipeline()
# image = cv2.imread("./new_image.png")
# if image is not None:
#     vision.process_vision(image)
# else:
#     print("🚨 Error: Image was not loaded. Check the file path/integrity.")