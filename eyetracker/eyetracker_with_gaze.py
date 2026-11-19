import cv2
import time
import numpy as np
from screeninfo import get_monitors
import mediapipe as mp

model_path = 'C:\\Users\\tomme\\Desktop\\ADHD project\\eyetracker\\facetracking_model\\face_landmarker.task'

# 1. Setup MediaPipe FaceLandmarker
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    output_face_blendshapes=True)

landmarker = FaceLandmarker.create_from_options(options)

USE_PRIMARY_MONITOR = True  # Set to True for main screen, False for secondary screen

monitors = get_monitors()
target_monitor = None

if USE_PRIMARY_MONITOR:
    # Target the primary monitor
    for m in monitors:
        if m.is_primary:
            target_monitor = m
            break
else:
    # Target the secondary monitor
    for m in monitors:
        if not m.is_primary:
            target_monitor = m
            break

if target_monitor is None:
    target_monitor = monitors[0]
    print("Secondary monitor not found, using primary.")

screen_w = target_monitor.width
screen_h = target_monitor.height
screen_x = target_monitor.x
screen_y = target_monitor.y

# A 9-point grid for calibration (corners, edges, and center)
calibration_targets_screen = [
    (100, 100), (screen_w // 2, 100), (screen_w - 100, 100),
    (100, screen_h // 2), (screen_w // 2, screen_h // 2), (screen_w - 100, screen_h // 2),
    (100, screen_h - 100), (screen_w // 2, screen_h - 100), (screen_w - 100, screen_h - 100)
]

calibration_iris_points = []
current_calibration_index = 0
homography_matrix = None
is_calibrating = True

# 3. Setup OpenCV Window on the Target Screen
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Change to 1 or 2 if targeting a different physical camera

cv2.namedWindow('Eye Tracker', cv2.WINDOW_NORMAL)
cv2.moveWindow('Eye Tracker', screen_x, screen_y)
cv2.setWindowProperty('Eye Tracker', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

last_timestamp_ms = 0

# Setup for EMA Smoothing
smoothed_x = None
smoothed_y = None
smoothing_factor = 0.2  # Lower = smoother but slower to react. Higher = faster but jittery.

# 4. Main Tracking Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    timestamp_ms = int(time.time() * 1000)
    if timestamp_ms <= last_timestamp_ms:
        timestamp_ms = last_timestamp_ms + 1
    last_timestamp_ms = timestamp_ms
    
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    # --- BLINK DETECTION ---
    is_blinking = False
    if result.face_blendshapes:
        for category in result.face_blendshapes[0]:
            if category.category_name == 'eyeBlinkLeft':
                left_blink = category.score
            elif category.category_name == 'eyeBlinkRight':
                right_blink = category.score
                
        # If either eye is more than 50% closed, flag as blinking
        if left_blink > 0.5 or right_blink > 0.5:
            is_blinking = True

    camera_gaze_x, camera_gaze_y = None, None

    # Extract Iris Coordinates
    if result.face_landmarks:
        face_landmarks = result.face_landmarks[0] # Assume 1 face
        left_iris = face_landmarks[468]
        right_iris = face_landmarks[473]

        # floating-point sub-pixel precision
        cx_left, cy_left = left_iris.x * w, left_iris.y * h
        cx_right, cy_right = right_iris.x * w, right_iris.y * h

        # Average both irises for a more stable single gaze point in the camera frame
        camera_gaze_x = (cx_left + cx_right) / 2.0
        camera_gaze_y = (cy_left + cy_right) / 2.0

    # Create a blank black canvas for the full-screen UI
    ui_canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    key = cv2.waitKey(1) & 0xFF

    if is_calibrating:
        # --- CALIBRATION MODE ---
        target_x, target_y = calibration_targets_screen[current_calibration_index]
        
        # Draw the target dot
        cv2.circle(ui_canvas, (target_x, target_y), 15, (0, 0, 255), -1)
        cv2.putText(ui_canvas, f"Look at the red dot and press SPACE. ({current_calibration_index + 1}/9)", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # If Spacebar is pressed, record the current iris coordinate
        if key == ord(' '):
            if camera_gaze_x is not None:
                calibration_iris_points.append([camera_gaze_x, camera_gaze_y])
                current_calibration_index += 1
                
                # Check if calibration is complete
                if current_calibration_index >= len(calibration_targets_screen):
                    is_calibrating = False
                    
                    # Calculate the Homography Matrix
                    src_pts = np.array(calibration_iris_points, dtype=np.float32)
                    dst_pts = np.array(calibration_targets_screen, dtype=np.float32)
                    homography_matrix, _ = cv2.findHomography(src_pts, dst_pts)
            else:
                print("Face not detected. Keep your face in frame and try again.")
    
    else:
        # --- TRACKING MODE ---
        if is_blinking and smoothed_x is not None and smoothed_y is not None:
            # BLINKING: Freeze the cursor at the last known position
            cv2.circle(ui_canvas, (int(smoothed_x), int(smoothed_y)), 40, (0, 255, 0), 3)
            cv2.putText(ui_canvas, "Blink Detected (Paused)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            
        elif camera_gaze_x is not None and homography_matrix is not None:
            # NOT BLINKING: Calculate new coordinates
            raw_point = np.array([[[camera_gaze_x, camera_gaze_y]]], dtype=np.float32)
            mapped_point = cv2.perspectiveTransform(raw_point, homography_matrix)
            screen_gaze_x = mapped_point[0][0][0]
            screen_gaze_y = mapped_point[0][0][1]

            screen_gaze_x = max(0, min(screen_w, screen_gaze_x))
            screen_gaze_y = max(0, min(screen_h, screen_gaze_y))

            if smoothed_x is None or smoothed_y is None:
                smoothed_x, smoothed_y = screen_gaze_x, screen_gaze_y
            else:
                smoothed_x = (smoothing_factor * screen_gaze_x) + ((1 - smoothing_factor) * smoothed_x)
                smoothed_y = (smoothing_factor * screen_gaze_y) + ((1 - smoothing_factor) * smoothed_y)

            cv2.circle(ui_canvas, (int(smoothed_x), int(smoothed_y)), 40, (0, 255, 0), 3)
            
        cv2.putText(ui_canvas, "Tracking Active. Press 'q' to quit.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Render the UI
    cv2.imshow('Eye Tracker', ui_canvas)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()