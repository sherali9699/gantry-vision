import cv2
import numpy as np

class StereoVision:
    def __init__(self, min_disparity=0, num_disparities=160, block_size=11):
        """
        Initializes the Stereo Semi-Global Block Matching (SGBM) algorithm.
        
        Args:
            min_disparity: Minimum possible disparity value.
            num_disparities: Maximum disparity minus minimum disparity. Must be divisible by 16.
            block_size: Matched block size. It must be an odd number >=1.
        """
        self.min_disparity = min_disparity
        self.num_disparities = num_disparities
        self.block_size = block_size
        
        # Initialize the Semi-Global Block Matcher
        # These parameters are tuned for general-purpose stereo vision but can be 
        # modified based on the specific scene and lighting conditions.
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=self.min_disparity,
            numDisparities=self.num_disparities,
            blockSize=self.block_size,
            P1=8 * 3 * self.block_size ** 2,
            P2=32 * 3 * self.block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

    def compute_disparity(self, img_left, img_right):
        """
        Computes the disparity map given a left and right rectified stereo image pair.
        
        Args:
            img_left: Rectified left image (grayscale or BGR)
            img_right: Rectified right image (grayscale or BGR)
            
        Returns:
            disparity: The raw disparity map (float32).
            disparity_visual: Normalized uint8 disparity map for visualization.
        """
        # Convert to grayscale if images are colored
        if len(img_left.shape) == 3:
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        else:
            gray_left = img_left
            
        if len(img_right.shape) == 3:
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        else:
            gray_right = img_right
            
        # Compute the raw disparity map
        # Note: cv2.StereoSGBM returns disparity scaled by 16 (16-bit signed single-channel)
        disparity_16S = self.stereo.compute(gray_left, gray_right)
        
        # Convert to float32 and divide by 16 for actual disparity values
        disparity = disparity_16S.astype(np.float32) / 16.0
        
        # Normalize for visualization (0-255 uint8)
        # We ignore negative values (which denote uncomputable/invalid pixels)
        disp_min = disparity.min()
        disp_max = disparity.max()
        disparity_visual = cv2.normalize(disparity, None, alpha=0, beta=255, 
                                         norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                                         
        return disparity, disparity_visual

    def compute_depth(self, disparity, focal_length, baseline):
        """
        Converts a disparity map to a depth map using camera parameters.
        
        Depth (Z) = (focal_length * baseline) / disparity
        
        Args:
            disparity: The float32 disparity map computed by compute_disparity.
            focal_length: Camera focal length in pixels.
            baseline: Distance between the two cameras (in mm or cm).
            
        Returns:
            depth_map: A matrix of real-world depth values (same unit as baseline).
        """
        # Avoid division by zero by setting zero disparity to a small number
        disparity_safe = np.copy(disparity)
        disparity_safe[disparity_safe <= 0.0] = 0.1
        
        depth_map = (focal_length * baseline) / disparity_safe
        
        # Where original disparity was invalid (<= 0), set depth to 0
        depth_map[disparity <= 0.0] = 0.0
        
        return depth_map

if __name__ == "__main__":
    print("Testing StereoVision Initialization...")
    sv = StereoVision()
    print("Stereo SGBM object created successfully.")
    
    # Generate dummy synthetic stereo images to test the pipeline without a camera
    print("Generating synthetic stereo images for pipeline test...")
    h, w = 480, 640
    # Create two gray images
    left_mock = np.zeros((h, w), dtype=np.uint8)
    right_mock = np.zeros((h, w), dtype=np.uint8)
    
    # Draw a mock box shifted by a disparity of 30 pixels in the right image
    cv2.rectangle(left_mock, (200, 150), (400, 350), 255, -1)
    cv2.rectangle(right_mock, (170, 150), (370, 350), 255, -1) # Shifted left by 30px (disparity=30)
    
    print("Computing disparity...")
    disp, disp_vis = sv.compute_disparity(left_mock, right_mock)
    
    print(f"Disparity Map Shape: {disp.shape}")
    print(f"Max Disparity found: {disp.max():.2f}")
    
    print("Computing mock depth (assuming focal_length=800px, baseline=60mm)...")
    depth = sv.compute_depth(disp, focal_length=800, baseline=60)
    
    # Depth at disparity=30 should be (800*60)/30 = 1600mm
    center_depth = depth[250, 250]
    print(f"Calculated depth at mock box center: {center_depth:.2f} mm")
    print("Stereo Vision pipeline test completed successfully!")
