#!/usr/bin/env python3

"""
Generate a single large ArUco marker for camera calibration.

Creates a large printable ArUco marker suitable for printing on A4 paper.
A single large marker is often more practical for calibration than a grid.
"""

import cv2
import numpy as np
import argparse
from cv2 import aruco


def generate_single_aruco_marker(
    marker_id=0,
    marker_size_mm=150,
    border_mm=20,
    dpi=300,
    aruco_dict_name='DICT_5X5_250'
):
    """
    Generate a single large ArUco marker for printing.
    
    Args:
        marker_id: ID of the marker (0-249 for DICT_5X5_250)
        marker_size_mm: Size of marker in mm
        border_mm: Border around marker in mm
        dpi: Print resolution (dots per inch)
        aruco_dict_name: ArUco dictionary name
    
    Returns:
        image: Marker image (numpy array, uint8)
        dimensions_mm: (width_mm, height_mm) total dimensions
    """
    
    # Get ArUco dictionary
    aruco_dict = aruco.getPredefinedDictionary(getattr(aruco, aruco_dict_name))
    
    # Convert mm to pixels
    mm_to_pixel = dpi / 25.4
    
    marker_size_px = int(marker_size_mm * mm_to_pixel)
    border_px = int(border_mm * mm_to_pixel)
    
    # Generate marker image
    marker_image = aruco.generateImageMarker(aruco_dict, marker_id, marker_size_px)
    
    # Create image with border (white background)
    image_width_px = marker_size_px + 2 * border_px
    image_height_px = marker_size_px + 2 * border_px
    image = np.ones((image_height_px, image_width_px), dtype=np.uint8) * 255
    
    # Place marker in center
    image[border_px:border_px + marker_size_px, border_px:border_px + marker_size_px] = marker_image
    
    # Calculate dimensions in mm
    width_mm = image_width_px / mm_to_pixel
    height_mm = image_height_px / mm_to_pixel
    
    return image, (width_mm, height_mm), (image_width_px, image_height_px)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a single large ArUco marker for camera calibration"
    )
    
    parser.add_argument("--output", default="aruco_marker.png",
                       help="Output image filename")
    parser.add_argument("--id", type=int, default=0,
                       help="Marker ID (0-249 for DICT_5X5_250, default: 0)")
    parser.add_argument("--marker-size", type=float, default=150.0,
                       help="Size of marker in mm (default: 150.0)")
    parser.add_argument("--border", type=float, default=20.0,
                       help="Border around marker in mm (default: 20.0)")
    parser.add_argument("--dpi", type=int, default=300,
                       help="Print resolution in DPI (default: 300)")
    parser.add_argument("--dict", default="DICT_5X5_250",
                       help="ArUco dictionary (default: DICT_5X5_250)")
    parser.add_argument("--show", action="store_true",
                       help="Display image before saving")
    
    args = parser.parse_args()
    
    print(f"[INFO] Generating ArUco marker...")
    print(f"       Marker ID: {args.id}")
    print(f"       Marker size: {args.marker_size} mm")
    print(f"       Border: {args.border} mm")
    print(f"       Dictionary: {args.dict}")
    print(f"       DPI: {args.dpi}")
    
    image, dimensions_mm, dimensions_px = generate_single_aruco_marker(
        marker_id=args.id,
        marker_size_mm=args.marker_size,
        border_mm=args.border,
        dpi=args.dpi,
        aruco_dict_name=args.dict
    )
    
    print(f"\n[INFO] Generated marker:")
    print(f"       Marker ID: {args.id}")
    print(f"       Dimensions: {dimensions_mm[0]:.1f} mm × {dimensions_mm[1]:.1f} mm")
    print(f"       Resolution: {dimensions_px[0]} × {dimensions_px[1]} pixels")
    print(f"       Paper size: {dimensions_mm[0]/25.4:.2f}\" × {dimensions_mm[1]/25.4:.2f}\"")
    
    # Check A4 fit
    a4_width = 210  # mm
    a4_height = 297  # mm
    print(f"\n[INFO] A4 paper is {a4_width}×{a4_height} mm")
    if dimensions_mm[0] <= a4_width and dimensions_mm[1] <= a4_height:
        print(f"       ✓ This marker FITS on A4 paper")
    else:
        print(f"       ✗ This marker is TOO LARGE for A4 paper")
        print(f"         Reduce marker size to {min(a4_width, a4_height) - 2*args.border:.0f}mm or less")
    
    # Display if requested
    if args.show:
        print(f"\n[INFO] Displaying marker (close window to continue)...")
        display_image = image.copy()
        if display_image.shape[0] > 1000:
            scale = 1000 / display_image.shape[0]
            display_image = cv2.resize(display_image, None, fx=scale, fy=scale)
        cv2.imshow("ArUco Marker", display_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Save image
    print(f"\n[INFO] Saving to {args.output}...")
    cv2.imwrite(args.output, image)
    print(f"[SAVED] {args.output}")
    
    print(f"\n[NEXT STEPS]")
    print(f"1. Print this image at {args.dpi} DPI on A4 paper")
    print(f"2. Make sure NOT to scale the image (print at 100%)")
    print(f"3. Mount on flat surface")
    print(f"4. Use for camera calibration")
    print(f"\n[MARKER INFORMATION]")
    print(f"   Marker ID: {args.id}")
    print(f"   Dictionary: {args.dict}")
    print(f"   Actual size when printed: {args.marker_size} mm × {args.marker_size} mm")
    print(f"\n[TIPS]")
    print(f"  - Use a rigid surface (foam board, cardboard) to mount the marker")
    print(f"  - Ensure good lighting when capturing images")
    print(f"  - Keep the marker flat and perpendicular to the table")
    print(f"  - For multiple calibration images, rotate the marker slightly")


if __name__ == "__main__":
    main()