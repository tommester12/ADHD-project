import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

model_path = 'C:\\Users\\tomme\\Desktop\\ADHD project\\eyetracker\\facetracking_model\\face_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    output_facial_transformation_matrixes=True)

landmarker = FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

last_timestamp_ms = 0

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

while cap.isOpened():
    # Read a single frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    frame = cv2.flip(frame, 1)
    
    # 3. MediaPipe now requires its own Image object format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # 4. Calculate the timestamp for the video frame (required by the new API)
    #timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    timestamp_ms = int(time.time() * 1000)

    # Ensure the new timestamp is strictly greater than the last one
    if timestamp_ms <= last_timestamp_ms:
        timestamp_ms = last_timestamp_ms + 1
        last_timestamp_ms = timestamp_ms
    
    # 5. Process the frame
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    # 6. Extract iris coordinates if a face is found
    if result.face_landmarks:
        for face_landmarks in result.face_landmarks:
            h, w, _ = frame.shape
            
            # Left iris (468) and Right iris (473)
            left_iris = face_landmarks[468]
            right_iris = face_landmarks[473]

            cx_left = int(left_iris.x * w)
            cy_left = int(left_iris.y * h)
            
            cx_right = int(right_iris.x * w)
            cy_right = int(right_iris.y * h)

            cv2.circle(frame, (cx_left, cy_left), 8, (0, 255, 0), 2)
            cv2.circle(frame, (cx_right, cy_right), 8, (0, 255, 0), 2)

    cv2.imshow('Iris Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera hardware and close all windows
cap.release()
cv2.destroyAllWindows()