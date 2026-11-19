import cv2

def find_cameras():
    print("Scanning for cameras...")
    for i in range(10):
        # cv2.CAP_DSHOW prevents a long timeout delay on Windows
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"Camera found at index {i}")
            cap.release()
        
find_cameras()