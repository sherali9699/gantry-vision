#!/usr/bin/env python3

"""
Generate a printable checkerboard pattern for camera calibration.

Creates an 8×10 checkerboard with customizable square size.
Output is saved as PNG image suitable for printing.
"""

import cv2
import numpy as np
import argparse


def generate_checkerboard(
    width_squares=8,
    height_squares=10,
    square_size_mm=40,
    dpi=300,
    border_mm=10
):
    """
    Generate a checkerboard pattern image.
    
    Args:
        width_squares: Number of squares wide (columns)
        height_squares: Number of squares tall (rows)
        square_size_mm: Size of each square in mm
        dpi: Print resolution (dots per inch)
        border_mm: Border around checkerboard in mm
    
    Returns:
        image: Checkerboard image (numpy array, uint8)
        dimensions_mm: (width_mm, height_mm) total dimensions
    """
    
    # Convert mm to pixels at given DPI
    # 1 inch = 25.4 mm
    mm_to_pixel = dpi / 25.4
    
    square_size_px = int(square_size_mm * mm_to_pixel)
    border_px = int(border_mm * mm_to_pixel)
    
    # Total dimensions
    board_width_px = width_squares * square_size_px
    board_height_px = height_squares * square_size_px
    image_width_px = board_width_px + 2 * border_px
    image_height_px = board_height_px + 2 * border_px
    
    # Create white image
    image = np.ones((image_height_px, image_width_px), dtype=np.uint8) * 255
    
    # Draw checkerboard pattern
    for row in range(height_squares):
        for col in range(width_squares):
            # Alternate black and white
            if (row + col) % 2 == 0:
                color = 0  # Black
            else:
                color = 255  # White (skip, already white)
                continue
            
            # Calculate position
            y_start = border_px + row * square_size_px
            y_end = y_start + square_size_px
            x_start = border_px + col * square_size_px
            x_end = x_start + square_size_px
            
            # Fill square
            image[y_start:y_end, x_start:x_end] = color
    
    # Calculate actual dimensions in mm
    width_mm = image_width_px / mm_to_pixel
    height_mm = image_height_px / mm_to_pixel
    
    return image, (width_mm, height_mm), (image_width_px, image_height_px)


def main():
    parser = argparse.ArgumentParser(
        description="Generate printable checkerboard for camera calibration"
    )
    
    parser.add_argument("--output", default="checkerboard_40mm.png",
                       help="Output image filename")
    parser.add_argument("--width", type=int, default=4,
                       help="Number of squares wide (columns) (default: 4)")
    parser.add_argument("--height", type=int, default=6,
                       help="Number of squares tall (rows) (default: 6)")
    parser.add_argument("--square-size", type=float, default=40.0,
                       help="Size of each square in mm (default: 40.0)")
    parser.add_argument("--dpi", type=int, default=300,
                       help="Print resolution in DPI (default: 300)")
    parser.add_argument("--border", type=float, default=5.0,
                       help="Border around checkerboard in mm (default: 5.0)")
    parser.add_argument("--show", action="store_true",
                       help="Display image before saving")
    
    args = parser.parse_args()
    
    print(f"[INFO] Generating checkerboard pattern...")
    print(f"       Grid: {args.width}×{args.height} squares")
    print(f"       Square size: {args.square_size} mm")
    print(f"       DPI: {args.dpi}")
    print(f"       Border: {args.border} mm")
    
    image, dimensions_mm, dimensions_px = generate_checkerboard(
        width_squares=args.width,
        height_squares=args.height,
        square_size_mm=args.square_size,
        dpi=args.dpi,
        border_mm=args.border
    )
    
    print(f"\n[INFO] Generated checkerboard:")
    print(f"       Dimensions: {dimensions_mm[0]:.1f} mm × {dimensions_mm[1]:.1f} mm")
    print(f"       Resolution: {dimensions_px[0]} × {dimensions_px[1]} pixels")
    print(f"       Paper size: {dimensions_mm[0]/25.4:.1f}\" × {dimensions_mm[1]/25.4:.1f}\"")
    
    # Estimate A4 paper
    a4_width = 210  # mm
    a4_height = 297  # mm
    print(f"\n[INFO] A4 paper is {a4_width}×{a4_height} mm")
    if dimensions_mm[0] <= a4_width and dimensions_mm[1] <= a4_height:
        print(f"       ✓ This checkerboard FITS on A4 paper")
    else:
        print(f"       ✗ This checkerboard is TOO LARGE for A4 paper")
        print(f"         Reduce square size or grid size")
    
    # Display if requested
    if args.show:
        print(f"\n[INFO] Displaying checkerboard (close window to continue)...")
        # Resize for display if too large
        display_image = image.copy()
        if display_image.shape[0] > 1200:
            scale = 1200 / display_image.shape[0]
            display_image = cv2.resize(display_image, None, fx=scale, fy=scale)
        cv2.imshow("Checkerboard Pattern", display_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Save image
    print(f"\n[INFO] Saving to {args.output}...")
    cv2.imwrite(args.output, image)
    print(f"[SAVED] {args.output}")
    
    print(f"\n[NEXT STEPS]")
    print(f"1. Print this image at {args.dpi} DPI on {args.width*args.square_size/10:.1f}cm × {args.height*args.square_size/10:.1f}cm paper")
    print(f"2. Make sure NOT to scale the image (print at 100%)")
    print(f"3. Mount on flat surface")
    print(f"4. Place at known position in gantry workspace")
    print(f"5. Capture image with camera")
    print(f"6. Run calibration:")
    print(f"   python3 calibrate_camera_extrinsics.py \\")
    print(f"     --image <captured_image.png> \\")
    print(f"     --calib camera_params_fisheye_aruko.npz \\")
    print(f"     --square-size {args.square_size}")


if __name__ == "__main__":
    main()