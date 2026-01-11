"""
STEREO DETECTION WITH RECTIFICATION - YELLOW COLOR DETECTION
============================================================
Modified to use HSV color detection instead of YOLO for easier testing.

This script:
1. Loads stereo calibration parameters
2. Rectifies both camera images (aligns epipolar lines horizontally)
3. Detects YELLOW object using HSV color filtering
4. Computes disparity from center positions
5. Converts disparity to real-world depth using triangulation

USAGE:
------
1. First run stereo_calibration.py to create calibration file
2. Run this script
3. Show a YELLOW object (banana, tennis ball, yellow paper) to both cameras
4. Real-time depth measurement displayed
"""

import sys 
import cv2
import numpy as np
import time

# ==============================================================================
# YELLOW OBJECT DETECTION FUNCTION
# ==============================================================================

# Yellow color range in HSV - adjust if needed for your lighting
YELLOW_LOWER = np.array([20, 100, 100])   # [Hue, Saturation, Value]
YELLOW_UPPER = np.array([40, 255, 255])

# Minimum area to consider as valid detection
MIN_AREA = 500

def find_yellow_center(frame, draw=True):
    """
    Detect yellow object in frame and return center coordinates.
    
    Args:
        frame: Input BGR image
        draw: Whether to draw detection on frame
    
    Returns:
        center: (x, y) tuple of center, or None if not detected
    """
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # Create mask for yellow color
    mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    
    # Clean up mask with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    center = None
    
    if len(contours) > 0:
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area > MIN_AREA:
            # Calculate center using moments
            M = cv2.moments(largest_contour)
            
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                center = (cx, cy)
                
                if draw:
                    # Draw bounding circle
                    ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
                    if radius > 10:
                        cv2.circle(frame, (int(x), int(y)), int(radius), 
                                 (255, 255, 0), 2)
                        # Draw center point
                        cv2.circle(frame, center, 7, (0, 0, 255), -1)
                        # Draw crosshair
                        cv2.line(frame, (cx - 10, cy), (cx + 10, cy), 
                               (0, 0, 255), 2)
                        cv2.line(frame, (cx, cy - 10), (cx, cy + 10), 
                               (0, 0, 255), 2)
    
    return center

# ==============================================================================
# LOAD CALIBRATION DATA
# ==============================================================================

print("\n" + "="*70)
print("STEREO YELLOW OBJECT DETECTION - RECTIFIED")
print("="*70)

try:
    print("\n[INFO] Loading calibration data...")
    calib = np.load('stereo_calibration.npz')
    
    # Camera intrinsics
    K1 = calib['K1']  # Left camera matrix
    D1 = calib['D1']  # Left distortion coefficients
    K2 = calib['K2']  # Right camera matrix
    D2 = calib['D2']  # Right distortion coefficients
    
    # Stereo parameters
    R = calib['R']   # Rotation matrix
    T = calib['T']   # Translation vector
    
    # Rectification parameters
    R1 = calib['R1']  # Left rectification rotation
    R2 = calib['R2']  # Right rectification rotation
    P1 = calib['P1']  # Left rectified projection matrix
    P2 = calib['P2']  # Right rectified projection matrix
    Q = calib['Q']    # Disparity-to-depth mapping matrix
    
    img_shape = tuple(calib['img_shape'])
    
    # Extract baseline
    baseline = abs(T[0][0])
    
    print("  ✓ Calibration loaded successfully")
    print(f"  Image size: {img_shape}")
    print(f"  Baseline: {baseline:.2f} cm")
    print(f"  Left focal length: {P1[0,0]:.2f} pixels")
    print(f"  Right focal length: {P2[0,0]:.2f} pixels")
    
except FileNotFoundError:
    print("\n[ERROR] Calibration file not found!")
    print("Please run 'stereo_calibration.py' first to calibrate your cameras.")
    sys.exit(1)

# ==============================================================================
# COMPUTE RECTIFICATION MAPS
# ==============================================================================

print("\n[INFO] Computing rectification maps...")

map1_left, map2_left = cv2.initUndistortRectifyMap(
    K1, D1, R1, P1, img_shape, cv2.CV_32FC1)

map1_right, map2_right = cv2.initUndistortRectifyMap(
    K2, D2, R2, P2, img_shape, cv2.CV_32FC1)

print("  ✓ Rectification maps computed")

# ==============================================================================
# DEPTH CALCULATION FUNCTION
# ==============================================================================

def calculate_depth_rectified(center_left, center_right, P1, P2):
    """
    Calculate depth from rectified image coordinates.
    
    Args:
        center_left: (x, y) in left rectified image
        center_right: (x, y) in right rectified image
        P1: Left projection matrix
        P2: Right projection matrix
    
    Returns:
        depth: Distance in cm, or None if invalid
    """
    x_left, y_left = center_left
    x_right, y_right = center_right
    
    # Calculate disparity
    disparity = x_left - x_right
    
    if disparity <= 0:
        return None
    
    # Calculate depth using calibrated focal length and baseline
    baseline = -P2[0, 3] / P1[0, 0]  # Extract baseline from P2
    focal_length = P1[0, 0]  # Focal length in pixels
    
    depth = (baseline * focal_length) / disparity
    
    return abs(depth)

# ==============================================================================
# CAMERA SETUP
# ==============================================================================

# Camera IDs - change these if your cameras are at different IDs
LEFT_CAMERA_ID = 6
RIGHT_CAMERA_ID = 4

print("\n[INFO] Opening cameras...")
cap_left = cv2.VideoCapture(LEFT_CAMERA_ID, cv2.CAP_V4L2)
cap_right = cv2.VideoCapture(RIGHT_CAMERA_ID, cv2.CAP_V4L2)

if not cap_left.isOpened():
    print(f"[ERROR] Cannot open left camera (ID: {LEFT_CAMERA_ID})")
    print("Available camera IDs to try: 0, 2, 4, 6, 8")
    sys.exit(1)
if not cap_right.isOpened():
    print(f"[ERROR] Cannot open right camera (ID: {RIGHT_CAMERA_ID})")
    print("Available camera IDs to try: 0, 2, 4, 6, 8")
    sys.exit(1)

print("  ✓ Cameras opened successfully")

print("\n" + "="*70)
print("Starting real-time depth measurement...")
print("Show a YELLOW object to both cameras")
print("Press 'q' to quit")
print("="*70 + "\n")

# ==============================================================================
# MAIN LOOP
# ==============================================================================

prev_time = time.time()
frame_count = 0

try:
    while True:
        frame_count += 1
        
        # Capture frames
        ret_left, frame_left_raw = cap_left.read()
        ret_right, frame_right_raw = cap_right.read()
        
        if not ret_left or not ret_right:
            print("[ERROR] Cannot read from cameras")
            break
        
        # Rectify images - KEY STEP for accurate disparity
        frame_left = cv2.remap(frame_left_raw, map1_left, map2_left, 
                               cv2.INTER_LINEAR)
        frame_right = cv2.remap(frame_right_raw, map1_right, map2_right, 
                                cv2.INTER_LINEAR)
        
        # Detect yellow object in both frames
        center_left = find_yellow_center(frame_left, draw=True)
        center_right = find_yellow_center(frame_right, draw=True)
        
        # Calculate depth
        if center_left is None or center_right is None:
            # No detection in one or both cameras
            cv2.putText(frame_left, "TRACKING LOST", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(frame_right, "TRACKING LOST", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(frame_left, "Show YELLOW object", (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame_right, "Show YELLOW object", (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            # Both cameras detected yellow object
            depth = calculate_depth_rectified(center_left, center_right, P1, P2)
            
            if depth is not None and depth > 0:
                # Valid depth measurement
                cv2.putText(frame_left, "TRACKING", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(frame_right, "TRACKING", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                
                # Display depth
                depth_text = f"Depth: {depth:.1f} cm ({depth/100:.2f} m)"
                cv2.putText(frame_left, depth_text, (20, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame_right, depth_text, (20, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Display disparity
                disparity = center_left[0] - center_right[0]
                disp_text = f"Disparity: {disparity:.1f} px"
                cv2.putText(frame_left, disp_text, (20, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Display Y-coordinate difference (should be small for good rectification)
                y_diff = abs(center_left[1] - center_right[1])
                y_text = f"Y-diff: {y_diff} px"
                y_color = (0, 255, 0) if y_diff < 5 else (0, 165, 255)
                cv2.putText(frame_left, y_text, (20, 160), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, y_color, 2)
                
                # Display coordinates
                coord_text = f"L:({center_left[0]},{center_left[1]}) R:({center_right[0]},{center_right[1]})"
                cv2.putText(frame_left, coord_text, (20, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Print to console every 10 frames
                if frame_count % 10 == 0:
                    print(f"Depth: {depth:6.1f} cm | Disparity: {disparity:5.1f} px | "
                          f"Y-diff: {y_diff:2d} px | Left: {center_left} | Right: {center_right}")
            else:
                cv2.putText(frame_left, "INVALID DISPARITY", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
                cv2.putText(frame_right, "INVALID DISPARITY", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
                cv2.putText(frame_left, "Object might be too close", (20, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        # Draw epipolar lines (every 100 pixels) to visualize rectification
        if frame_count % 30 == 0:
            for y in range(100, frame_left.shape[0], 100):
                cv2.line(frame_left, (0, y), (frame_left.shape[1], y), 
                        (255, 0, 0), 1)
                cv2.line(frame_right, (0, y), (frame_right.shape[1], y), 
                        (255, 0, 0), 1)
        
        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        
        cv2.putText(frame_left, f"FPS: {int(fps)}", (20, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.putText(frame_right, f"FPS: {int(fps)}", (20, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Display frames
        cv2.imshow("Left Camera - RECTIFIED", frame_left)
        cv2.imshow("Right Camera - RECTIFIED", frame_right)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user")

# ==============================================================================
# CLEANUP
# ==============================================================================

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()

print("\n[INFO] Program terminated")
print("="*70)