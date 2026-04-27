#!/usr/bin/env python3
"""
Microwave Picking System using PnP with Existing Vision Pipeline
Integrates with VisionPipeline class for detection and PnP for 3D pose estimation
Prints calculated coordinates for picking
"""

import numpy as np
import cv2
import time
from typing import Tuple, Optional, Dict, List
from vision_pipeline import VisionPipeline  # Your existing vision class

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

# Microwave dimensions (from your real-world measurements, in millimeters)
MICROWAVE_WIDTH = 431.8   
MICROWAVE_DEPTH = 609.6   
MICROWAVE_HEIGHT = 355.6  

# Camera calibration parameters (from your calibration script)
CAMERA_MATRIX = np.array([
    [495.38761596, 0, 377.41128188],
    [0, 494.91430573, 185.37079933],
    [0, 0, 1]
], dtype=np.float32)

DISTORTION_COEFFS = np.array([
    -0.34501892, 0.0869633463, -0.00207301019, 0.000315849315, 0.11223285
], dtype=np.float32)

# Camera mounting position in gantry coordinates (YOU MUST CALIBRATE THESE!)
CAMERA_MOUNT_X = 420  # mm - fixed X position (center of workspace)
CAMERA_MOUNT_Z = 1143  # mm - fixed height above workspace
CAMERA_MOUNT_Y =  540   # mm - current Y position (camera moves in Y direction)

# Pick parameters
APPROACH_HEIGHT_OFFSET = 0   # mm - safe distance above microwave

# ============================================================================
# MICROWAVE 3D MODEL - OBJECT COORDINATES
# ============================================================================

def create_microwave_3d_model():
    """
    3D model of mw in mw frame coordinates (mm). Origin is at center of top face of microwave.
    For a down-facing camera:
    - X: right (same as gantry X)
    - Y: forward (maps to gantry Y)
    - Z: UP (maps to gantry Z, but camera sees this as depth)
    """
    w = MICROWAVE_WIDTH / 2
    d = MICROWAVE_DEPTH / 2
    
    # For down-facing camera: 
    # - Top face is closer to camera = smaller Z (or negative depending on convention)
    # Let's put top face at Z = 0, bottom at Z = -h
    model_3d = np.array([
        [-w, -d, 0],   # front-left (top face)
        [ w, -d, 0],   # front-right
        [ w,  d, 0],   # back-right
        [-w,  d, 0],   # back-left
    ], dtype=np.float32)
    
    return model_3d

# ============================================================================
# CORNER ORDERING FOR PNP
# ============================================================================

def order_corners_for_pnp(box_points: np.ndarray) -> np.ndarray:
    """
    Order corners from minAreaRect consistently for PnP algorithm.
    """
    # Ensure input is float32 and correct shape
    box_points = np.asarray(box_points, dtype=np.float32)
    
    # Calculate center of corners
    center = np.mean(box_points, axis=0)
    
    # Calculate angle from center to each corner
    angles = np.arctan2(box_points[:, 1] - center[1], box_points[:, 0] - center[0])
    
    # Sort by angle descending (clockwise order)
    sorted_indices = np.argsort(angles)[::-1]
    ordered_corners = box_points[sorted_indices]
    
    # CRITICAL: Return as (4,2) array
    return ordered_corners.reshape(4, 2)

# ============================================================================
# PNP POSE ESTIMATION
# ============================================================================

def estimate_microwave_pose(
    image_points: np.ndarray,
    object_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray
) -> Optional[Dict]:
    """
    Estimate microwave pose using PnP algorithm.
    """
    # CRITICAL FIX: Ensure correct shape and dtype
    image_points = np.asarray(image_points, dtype=np.float32).reshape(4, 2)
    object_points = np.asarray(object_points, dtype=np.float32).reshape(4, 3)
    
    print(f"   DEBUG: image_points shape={image_points.shape}, dtype={image_points.dtype}")
    print(f"   DEBUG: object_points shape={object_points.shape}, dtype={object_points.dtype}")
    
    # Solve PnP
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        print("PnP solving failed")
        return None
    
    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    
    # Calculate Euler angles (roll, pitch, yaw) in degrees
    sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
    singular = sy < 1e-6
    
    if not singular:
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = 0
    
    euler_angles = np.degrees([roll, pitch, yaw])
    
    return {
        'success': True,
        'rvec': rvec,
        'tvec': tvec,
        'rotation_matrix': rotation_matrix,
        'euler_angles': euler_angles
    }
# ============================================================================
# COORDINATE TRANSFORMATION
# ============================================================================

def transform_to_gantry_coordinates(
    microwave_pose: Dict,
    camera_y_position: float
) -> Optional[Dict]:
    """
    Transform microwave pose from camera coordinates to gantry coordinates.
    
    Camera coordinates (from PnP for DOWNWARD-facing camera):
        X_cam: right (same as gantry X)
        Y_cam: forward (same as gantry Y)
        Z_cam: distance from camera to object (should be ~700mm for floor)
    
    Gantry coordinates:
        X: right
        Y: forward  
        Z: UP (floor = 0)
    """
    if microwave_pose is None:
        return None
    
    # Get microwave position in camera coordinates
    pos_camera = microwave_pose['tvec'].flatten()
    print(f"   Raw camera position: X={pos_camera[0]:.2f}, Y={pos_camera[1]:.2f}, Z={pos_camera[2]:.2f}")
    
    # The camera is at fixed height looking DOWN
    camera_height_mm = CAMERA_MOUNT_Z  # 1000mm above floor
    
    # For a downward-facing camera:
    # - Camera's Z value is the DISTANCE to the object
    # - Object's height in gantry = camera height - distance
    gantry_z = camera_height_mm - pos_camera[2]
    
    # X and Y: camera's X and Y directly map to gantry's X and Y
    gantry_x = CAMERA_MOUNT_X + pos_camera[0]
    gantry_y = camera_y_position + pos_camera[1]
    
    print(f"   Calculated gantry: X={gantry_x:.2f}, Y={gantry_y:.2f}, Z={gantry_z:.2f}")
    
    # Extract yaw angle
    #yaw_degrees = microwave_pose['euler_angles'][2]
    
    return {
        'x': gantry_x,
        'y': gantry_y,
        'z': gantry_z
        #'yaw': yaw_degrees
    }
# ============================================================================
# SAFETY CHECKS
# ============================================================================

def verify_microwave_flat(pose: Dict, tolerance_degrees: float = 5.0) -> bool:
    """
    Verify that microwave is flat (not tilted) before picking.
    
    Args:
        pose: Pose dictionary from estimate_microwave_pose()
        tolerance_degrees: Maximum allowed tilt in degrees
        
    Returns:
        True if microwave is flat enough, False otherwise
    """
    if pose is None:
        return False
    
    roll, pitch, yaw = pose['euler_angles']
    
    if abs(roll) > tolerance_degrees:
        print(f"  ⚠️  Warning: Microwave tilted - Roll = {roll:.1f}° (max {tolerance_degrees}°)")
        return False
    
    if abs(pitch) > tolerance_degrees:
        print(f"  ⚠️  Warning: Microwave tilted - Pitch = {pitch:.1f}° (max {tolerance_degrees}°)")
        return False
    
    return True

def verify_workspace_limits(x: float, y: float, z: float) -> bool:
    """
    Verify that pick position is within workspace limits.
    
    Args:
        x, y, z: Pick position in gantry coordinates (mm)
        
    Returns:
        True if position is safe, False otherwise
    """
    # Define workspace boundaries (adjust based on your actual gantry limits)
    WORKSPACE_X_MIN = 0
    WORKSPACE_X_MAX = 960
    WORKSPACE_Y_MIN = 0
    WORKSPACE_Y_MAX = 950
    WORKSPACE_Z_MIN = 0
    WORKSPACE_Z_MAX = 1500
    
    if x < WORKSPACE_X_MIN or x > WORKSPACE_X_MAX:
        print(f"  ⚠️  Warning: X={x:.1f}mm is outside workspace [{WORKSPACE_X_MIN}, {WORKSPACE_X_MAX}]")
        return False
    
    if y < WORKSPACE_Y_MIN or y > WORKSPACE_Y_MAX:
        print(f"  ⚠️  Warning: Y={y:.1f}mm is outside workspace [{WORKSPACE_Y_MIN}, {WORKSPACE_Y_MAX}]")
        return False
    
    if z < WORKSPACE_Z_MIN or z > WORKSPACE_Z_MAX:
        print(f"  ⚠️  Warning: Z={z:.1f}mm seems unreasonable")
        return False
    
    return True

# ============================================================================
# MAIN PICKING PIPELINE WITH PRINTED COORDINATES
# ============================================================================

def process_microwave_detection(detection: Dict, camera_y_position: float) -> Optional[Dict]:
    """
    Process microwave detection using PnP and print coordinates.
    
    Args:
        detection: Detection dictionary from vision pipeline
        camera_y_position: Current camera Y position in gantry coordinates (mm)
        
    Returns:
        Pick pose dictionary with x, y, z, yaw coordinates
    """
    print("\n" + "="*70)
    print("🔍 PROCESSING MICROWAVE DETECTION WITH PNP")
    print("="*70)
    
    # Get the box points from detection
    if 'box_points' not in detection:
        print("❌ Error: No box points in detection")
        return None
    
    # Print detection info
    print(f"\n📸 Detection Info:")
    print(f"   Color: {detection.get('color', 'unknown')}")
    print(f"   Center (pixels): ({detection.get('center', (0,0))[0]}, {detection.get('center', (0,0))[1]})")
    print(f"   Angle (pixels): {detection.get('angle', 0):.2f}°")
    
    # Order corners for PnP
    ordered_corners = order_corners_for_pnp(detection['box_points'])
    print(f"\n📐 Ordered corners for PnP:")
    for i, corner in enumerate(ordered_corners):
        print(f"   Corner {i}: ({corner[0]:.1f}, {corner[1]:.1f}) pixels")
    
    # Create 3D model
    object_points = create_microwave_3d_model()
    print(f"\n📦 Microwave 3D Model (object coordinates):")
    print(f"   Width: {MICROWAVE_WIDTH} mm")
    print(f"   Depth: {MICROWAVE_DEPTH} mm")
    print(f"   Height: {MICROWAVE_HEIGHT} mm")
    print(f"   Origin: Center of top face")
    
    # Estimate pose using PnP
    print(f"\n🔧 Running PnP Algorithm...")
    pose = estimate_microwave_pose(
        ordered_corners,
        object_points,
        CAMERA_MATRIX,
        DISTORTION_COEFFS
    )
    
    if pose is None:
        print("❌ PnP pose estimation failed")
        return None
    
    # Print PnP results
    print(f"\n✅ PnP Results (Camera Coordinates):")
    print(f"   Position (X, Y, Z): ({pose['tvec'][0][0]:.2f}, {pose['tvec'][1][0]:.2f}, {pose['tvec'][2][0]:.2f}) mm")
    #print(f"   Rotation (Roll, Pitch, Yaw): ({pose['euler_angles'][0]:.2f}°, {pose['euler_angles'][1]:.2f}°, {pose['euler_angles'][2]:.2f}°)")
    
    # Check if microwave is flat
    # print(f"\n🔒 Safety Check:")
    # if not verify_microwave_flat(pose):
    #     print("   ❌ Microwave is not flat - aborting pick")
    #     return None
    # print("   ✅ Microwave is flat - safe to pick")
    
    # Transform to gantry coordinates
    print(f"\n🔄 Transforming to Gantry Coordinates...")
    print(f"   Camera position in gantry: X={CAMERA_MOUNT_X}mm, Y={camera_y_position}mm, Z={CAMERA_MOUNT_Z}mm")
    
    pick_pose = transform_to_gantry_coordinates(pose, camera_y_position)
    
    if pick_pose is None:
        print("❌ Coordinate transformation failed")
        return None
    
    # Verify workspace limits
    if not verify_workspace_limits(pick_pose['x'], pick_pose['y'], pick_pose['z']):
        print("   ❌ Pick position outside workspace limits")
        return None
    print("   ✅ Pick position within workspace limits")
    
    # Print FINAL PICK COORDINATES
    print("\n" + "="*70)
    print("🎯 FINAL PICK COORDINATES (GANTRY COORDINATES)")
    print("="*70)
    print(f"\n   📍 POSITION:")
    print(f"      X: {pick_pose['x']:.2f} mm")
    print(f"      Y: {pick_pose['y']:.2f} mm")
    print(f"      Z: {pick_pose['z']:.2f} mm")
    print(f"\n   🔄 ORIENTATION:")
    #print(f"      Yaw: {pick_pose['yaw']:.2f} degrees")
    #print(f"\n   📏 APPROACH POSITION (safe height):")
    #print(f"      X: {pick_pose['x']:.2f} mm")
    #print(f"      Y: {pick_pose['y']:.2f} mm")
    #print(f"      Z: {pick_pose['z'] + APPROACH_HEIGHT_OFFSET:.2f} mm (Z + {APPROACH_HEIGHT_OFFSET}mm offset)")
    print("\n" + "="*70)
    print("✅ Microwave ready for picking!")
    print("="*70 + "\n")
    
    return pick_pose

# ============================================================================
# MODIFIED CAMERA INTERFACE (INTEGRATES WITH YOUR VISION PIPELINE)
# ============================================================================

class CameraInterface:
    """
    Interface to capture images from camera and process with VisionPipeline.
    """
    
    def __init__(self, camera_id: int = 2):
        self.camera = cv2.VideoCapture(camera_id)
        if not self.camera.isOpened():
            raise RuntimeError("Could not open camera")
        
        # Set camera properties (adjust as needed)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Initialize your vision pipeline
        self.vision = VisionPipeline(debug=False)  # Set debug=False to reduce windows
        
    def capture_and_process(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict]]:
        """
        Capture image and process with vision pipeline.
        
        Returns:
            Tuple of (output_image, top_down_image, detection_info)
            detection_info contains: color, angle, center, real_world, box_points
            Returns (None, None, None) if no detection
        """
        ret, image = self.camera.read()
        if not ret:
            raise RuntimeError("Failed to capture image")
        
        # Process with your vision pipeline
        # NOTE: Your VisionPipeline needs to return detections!
        # You need to modify it to return: return output_image, top_down_image, detections
        output_image, top_down_image, detections = self.vision.process_vision(image)
        
        # Find the black detection (microwave)
        microwave_detection = None
        for det in detections:
            if det['color'] == 'black':
                microwave_detection = det
                break
        
        return output_image, top_down_image, microwave_detection
    
    def release(self):
        """Release camera resources"""
        self.camera.release()

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main function: Run PnP pipeline and print coordinates for picking.
    """
    print("="*70)
    print("🎯 MICROWAVE PICKING SYSTEM - PNP COORDINATE CALCULATOR")
    print("="*70)
    print("\nThis system will:")
    print("  1. Detect microwave using vision pipeline")
    print("  2. Calculate 3D pose using PnP algorithm")
    print("  3. Print pick coordinates in gantry space")
    print("  4. Show visual feedback (press 'q' to quit)")
    print("\n" + "="*70)
    
    # Initialize camera
    print("\n📷 Initializing camera...")
    camera = CameraInterface()
    print("   ✅ Camera ready")
    
    # Current camera Y position (you can update this based on actual position)
    current_camera_y = CAMERA_MOUNT_Y
    print(f"   📍 Camera Y position: {current_camera_y} mm")
    
    try:
        while True:
            # Capture and process image
            output_image, top_down_image, detection = camera.capture_and_process()
            
            if output_image is None: # It could be detection 
                print("❌ Failed to capture/process image")
                continue
            
            # # Show images -- no need to at the moment
            cv2.imshow("Vision Output", output_image)
            # cv2.imshow("Top Down View", top_down_image)
            
            # If microwave detected, process with PnP
            if detection:
                print("\n🔔 Microwave detected!")
                pick_pose = process_microwave_detection(detection, current_camera_y)
                
                if pick_pose:
                    print("\n💡 SUGGESTED PICK SEQUENCE:")
                    print(f"   1. Move to approach: X={pick_pose['x']:.2f}, Y={pick_pose['y']:.2f}, Z={pick_pose['z'] + APPROACH_HEIGHT_OFFSET:.2f}")
                    print(f"   2. Move down to pick: X={pick_pose['x']:.2f}, Y={pick_pose['y']:.2f}, Z={pick_pose['z']:.2f}")
                    print(f"   3. Close gripper")
                    #print(f"   4. Lift to: Z={pick_pose['z'] + LIFT_HEIGHT:.2f}")
                    
                    # Wait a bit before next detection
                    time.sleep(3)
            else:
                print("\r⏳ Waiting for microwave detection...", end="", flush=True)
            
            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\n\n👋 User requested quit")
                break
                
    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        camera.release()
        cv2.destroyAllWindows()
        print("\n✅ System shutdown complete")

# ============================================================================
# TEST FUNCTION (USING SAMPLE IMAGE)
# ============================================================================

def test_with_sample_image(image_path: str):
    """
    Test PnP pipeline with a sample image file.
    
    Args:
        image_path: Path to test image
    """
    print("="*70)
    print("🧪 TEST MODE - Processing Sample Image")
    print("="*70)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image from {image_path}")
        return
    
    # Initialize vision pipeline without debug windows for testing
    vision = VisionPipeline(debug=False)
    
    # Process image
    print("\n📸 Processing image...")
    output_image, top_down_image, detections = vision.process_vision(image)
    
    # Find microwave detection
    microwave_detection = None
    for det in detections:
        if det['color'] == 'black':
            microwave_detection = det
            break
    
    if microwave_detection:
        print("\n✅ Microwave detected in sample image!")
        
        # Process with PnP
        current_camera_y = CAMERA_MOUNT_Y
        pick_pose = process_microwave_detection(microwave_detection, current_camera_y)
        
        if pick_pose:
            print("\n✅ PnP calculation successful!")
        else:
            print("\n❌ PnP calculation failed")
    else:
        print("\n❌ No microwave detected in sample image")
    
    # Show results
    cv2.imshow("Test Output", output_image)
    cv2.imshow("Top Down View", top_down_image)
    print("\n📺 Press any key to close test windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Microwave Picking PnP Calculator')
    parser.add_argument('--test', type=str, metavar='IMAGE_PATH',
                       help='Test with sample image file instead of live camera')
    parser.add_argument('--camera-y', type=float, default=CAMERA_MOUNT_Y,
                       help=f'Camera Y position in mm (default: {CAMERA_MOUNT_Y})')
    parser.add_argument('--camera-x', type=float, default=CAMERA_MOUNT_X,
                       help=f'Camera X position in mm (default: {CAMERA_MOUNT_X})')
    parser.add_argument('--camera-z', type=float, default=CAMERA_MOUNT_Z,
                       help=f'Camera Z position in mm (default: {CAMERA_MOUNT_Z})')
    
    args = parser.parse_args()
    
    # Update camera position if provided
    CAMERA_MOUNT_X = args.camera_x
    CAMERA_MOUNT_Z = args.camera_z
    CAMERA_MOUNT_Y = args.camera_y
    
    if args.test:
        # Test mode with sample image
        test_with_sample_image(args.test)
    else:
        # Normal mode with live camera
        main()