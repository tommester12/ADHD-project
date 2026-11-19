import asyncio
import functools
import http.server
import json
import os
import platform
import sys
import threading
import time
import webbrowser
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import websockets

try:
    from cv2_enumerate_cameras import enumerate_cameras
    HAS_ENUMERATOR = True
except ImportError:
    HAS_ENUMERATOR = False

# --- CONFIGURATION ---
WS_HOST = 'localhost'
WS_PORT = 8765
HTTP_PORT = 8080
SMOOTHING_FACTOR = 0.2
BLINK_THRESHOLD = 0.5
SHOW_CAMERA_PREVIEW = False


def resource_path(relative_path: str) -> str:
    """Resolves asset paths for both dev mode and PyInstaller portable builds."""
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


# Checks relative portable path first, then falls back to your local dev path
MODEL_PATH = resource_path(os.path.join('facetracking_model', 'face_landmarker.task'))
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = r'C:\Users\tomme\Desktop\ADHD project\eyetracker\facetracking_model\face_landmarker.task'


def get_camera_backend() -> int:
    """Returns the optimal OpenCV capture backend for the current OS."""
    system = platform.system()
    if system == 'Windows':
        return cv2.CAP_DSHOW
    elif system == 'Darwin':  # macOS
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_V4L2


def open_camera_capture(index: int) -> cv2.VideoCapture:
    """
    Opens a camera index cleanly.
    cv2-enumerate-cameras encodes the backend inside `index` (e.g. 700 for DSHOW, 1200 for AVFoundation).
    """
    if index >= 100:
        return cv2.VideoCapture(index)
    return cv2.VideoCapture(index, get_camera_backend())


def discover_cameras(active_index=None, active_name=None):
    """
    Lists all registered cameras (including virtual cameras like OBS) across Windows & macOS.
    """
    backend = get_camera_backend()
    cameras = []

    if HAS_ENUMERATOR:
        try:
            for cam in enumerate_cameras(backend):
                cameras.append({
                    'index': int(cam.index),
                    'name': cam.name or f'Camera {cam.index}',
                })
            return cameras
        except Exception as exc:
            print(f'[Camera] Enumerator error, falling back to probing: {exc}')

    # Fallback: probe indices 0..7 without disturbing the currently active camera
    for idx in range(8):
        if active_index is not None and (idx == active_index or idx == active_index % 100):
            cameras.append({'index': active_index, 'name': active_name or f'Camera {idx}'})
            continue

        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            cameras.append({'index': idx, 'name': f'Camera {idx}'})
            cap.release()

    return cameras


class EyeTrackerState:
    """Thread-safe shared state between the camera loop and WebSocket server."""

    def __init__(self):
        self.lock = threading.Lock()
        self.running = True

        # Camera management
        self.available_cameras = []
        self.active_camera_index = None
        self.requested_camera_index = None
        self.camera_switch_requested = False
        self.camera_status = 'initializing'  # initializing | active | switching | no_camera | error
        self.camera_error = None
        self.status_version = 0

        # Tracking & calibration
        self.iris_buffer = deque(maxlen=5)
        self.is_blinking = False
        self.calibration_iris_points = []
        self.calibration_target_points = []
        self.homography_matrix = None

        self.smoothed_x = None
        self.smoothed_y = None
        self.frame_id = 0

    def refresh_cameras(self, auto_select_if_none=True):
        with self.lock:
            curr_idx = self.active_camera_index
            curr_name = next(
                (c['name'] for c in self.available_cameras if c['index'] == curr_idx),
                None,
            )

        found = discover_cameras(active_index=curr_idx, active_name=curr_name)

        with self.lock:
            self.available_cameras = found
            if not found:
                self.active_camera_index = None
                self.camera_status = 'no_camera'
                self.camera_error = 'No cameras detected. Connect a webcam and click Refresh.'
            elif auto_select_if_none and self.active_camera_index is None:
                self.requested_camera_index = found[0]['index']
                self.camera_switch_requested = True
                self.camera_status = 'switching'
                self.camera_error = None

            self.status_version += 1
            print(f'[Camera] Found {len(found)} device(s): {[c["name"] for c in found]}')

    def request_camera_switch(self, new_index: int):
        with self.lock:
            self.requested_camera_index = int(new_index)
            self.camera_switch_requested = True
            self.camera_status = 'switching'
            self.camera_error = None

            # Reset calibration because a different camera has a different lens & angle
            self.homography_matrix = None
            self.calibration_iris_points.clear()
            self.calibration_target_points.clear()
            self.iris_buffer.clear()
            self.smoothed_x = None
            self.smoothed_y = None
            self.status_version += 1

    def consume_camera_switch_request(self):
        with self.lock:
            if not self.camera_switch_requested:
                return None
            self.camera_switch_requested = False
            return self.requested_camera_index

    def set_camera_result(self, index, status: str, error: str = None):
        with self.lock:
            self.active_camera_index = index
            self.camera_status = status
            self.camera_error = error
            self.status_version += 1

    def get_camera_state_payload(self):
        with self.lock:
            return self.status_version, {
                'type': 'camera_state',
                'cameras': list(self.available_cameras),
                'activeCamera': self.active_camera_index,
                'status': self.camera_status,
                'error': self.camera_error,
                'isCalibrated': self.homography_matrix is not None,
                'platform': platform.system(),
            }

    def reset_calibration(self):
        with self.lock:
            self.calibration_iris_points.clear()
            self.calibration_target_points.clear()
            self.homography_matrix = None
            self.smoothed_x = None
            self.smoothed_y = None
            self.status_version += 1
            print('[Calibration] Reset started.')

    def record_calibration_point(self, target_x: float, target_y: float) -> bool:
        with self.lock:
            if not self.iris_buffer:
                print('[Calibration] Warning: No face/iris detected for this point.')
                return False

            avg_iris = np.mean(self.iris_buffer, axis=0)
            self.calibration_iris_points.append([float(avg_iris[0]), float(avg_iris[1])])
            self.calibration_target_points.append([float(target_x), float(target_y)])
            return True

    def finish_calibration(self) -> bool:
        with self.lock:
            if len(self.calibration_iris_points) < 4:
                print('[Calibration] Failed: Need at least 4 valid points.')
                return False

            src_pts = np.array(self.calibration_iris_points, dtype=np.float32)
            dst_pts = np.array(self.calibration_target_points, dtype=np.float32)

            matrix, _ = cv2.findHomography(src_pts, dst_pts)
            if matrix is None:
                return False

            self.homography_matrix = matrix
            self.smoothed_x = None
            self.smoothed_y = None
            self.status_version += 1
            print(f'[Calibration] Complete using {len(src_pts)} points!')
            return True

    def update_from_frame(self, iris_xy, is_blinking: bool):
        with self.lock:
            self.is_blinking = is_blinking
            self.frame_id += 1

            if iris_xy is not None and not is_blinking:
                self.iris_buffer.append(iris_xy)

            if self.homography_matrix is not None and not is_blinking and iris_xy is not None:
                raw_point = np.array([[[iris_xy[0], iris_xy[1]]]], dtype=np.float32)
                mapped = cv2.perspectiveTransform(raw_point, self.homography_matrix)

                norm_x = float(np.clip(mapped[0][0][0], 0.0, 1.0))
                norm_y = float(np.clip(mapped[0][0][1], 0.0, 1.0))

                if self.smoothed_x is None or self.smoothed_y is None:
                    self.smoothed_x, self.smoothed_y = norm_x, norm_y
                else:
                    self.smoothed_x = (SMOOTHING_FACTOR * norm_x) + ((1 - SMOOTHING_FACTOR) * self.smoothed_x)
                    self.smoothed_y = (SMOOTHING_FACTOR * norm_y) + ((1 - SMOOTHING_FACTOR) * self.smoothed_y)

    def get_gaze_snapshot(self):
        with self.lock:
            return (
                self.frame_id,
                self.smoothed_x,
                self.smoothed_y,
                self.is_blinking,
                self.homography_matrix is not None,
            )


def camera_tracking_loop(state: EyeTrackerState):
    """Runs OpenCV + MediaPipe FaceLandmarker and handles hot-swapping cameras."""
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        output_face_blendshapes=True,
    )
    landmarker = FaceLandmarker.create_from_options(options)

    # Initial camera scan on startup
    state.refresh_cameras(auto_select_if_none=True)

    cap = None
    last_timestamp_ms = 0
    consecutive_failures = 0

    try:
        while state.running:
            # 1. Handle hot-swap camera requests
            target_idx = state.consume_camera_switch_request()
            if target_idx is not None:
                if cap is not None:
                    cap.release()
                    cap = None

                print(f'[Camera] Switching to camera index {target_idx}...')
                new_cap = open_camera_capture(target_idx)
                if new_cap.isOpened():
                    cap = new_cap
                    consecutive_failures = 0
                    state.set_camera_result(target_idx, 'active', None)
                    print(f'[Camera] Opened camera {target_idx}.')
                else:
                    new_cap.release()
                    hint = (
                        ' On macOS, check System Settings → Privacy & Security → Camera.'
                        if platform.system() == 'Darwin'
                        else ' Make sure no other app is using it.'
                    )
                    state.set_camera_result(
                        None,
                        'error',
                        f'Could not open selected camera.{hint}',
                    )

            # 2. If no camera is currently open, idle cleanly while waiting for selection
            if cap is None or not cap.isOpened():
                time.sleep(0.15)
                continue

            # 3. Read frame
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures == 45:
                    state.set_camera_result(
                        state.active_camera_index,
                        'error',
                        'Camera opened, but no frames are arriving (check permissions or virtual camera output).',
                    )
                time.sleep(0.02)
                continue

            if consecutive_failures >= 45:
                state.set_camera_result(state.active_camera_index, 'active', None)
            consecutive_failures = 0

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )

            timestamp_ms = int(time.time() * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # Blink detection
            left_blink = 0.0
            right_blink = 0.0
            if result.face_blendshapes:
                for category in result.face_blendshapes[0]:
                    if category.category_name == 'eyeBlinkLeft':
                        left_blink = category.score
                    elif category.category_name == 'eyeBlinkRight':
                        right_blink = category.score
            is_blinking = (left_blink > BLINK_THRESHOLD) or (right_blink > BLINK_THRESHOLD)

            # Iris coordinates
            iris_xy = None
            if result.face_landmarks:
                face_landmarks = result.face_landmarks[0]
                left_iris = face_landmarks[468]
                right_iris = face_landmarks[473]
                cx_left, cy_left = left_iris.x * w, left_iris.y * h
                cx_right, cy_right = right_iris.x * w, right_iris.y * h
                iris_xy = ((cx_left + cx_right) / 2.0, (cy_left + cy_right) / 2.0)

            state.update_from_frame(iris_xy, is_blinking)

            if SHOW_CAMERA_PREVIEW:
                cv2.imshow('Eye Tracker Preview', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    state.running = False
                    break
    finally:
        if cap is not None:
            cap.release()
        landmarker.close()
        if SHOW_CAMERA_PREVIEW:
            cv2.destroyAllWindows()
        print('[Camera] Released.')


# --- WEBSOCKET SERVER ---
connected_clients = set()
tracker_state = EyeTrackerState()


async def handle_client(websocket, *args):
    connected_clients.add(websocket)
    print(f'[WS] Client connected ({len(connected_clients)} active)')

    # Immediately send current camera list & status on connect
    _, cam_payload = tracker_state.get_camera_state_payload()
    await websocket.send(json.dumps(cam_payload))

    try:
        async for raw_message in websocket:
            try:
                msg = json.loads(raw_message)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get('type')

            if msg_type == 'list_cameras':
                await asyncio.to_thread(tracker_state.refresh_cameras, False)

            elif msg_type == 'select_camera':
                if 'index' in msg and msg['index'] is not None:
                    tracker_state.request_camera_switch(msg['index'])

            elif msg_type == 'calibrate_start':
                tracker_state.reset_calibration()

            elif msg_type == 'calibrate_point':
                ok = tracker_state.record_calibration_point(msg['targetX'], msg['targetY'])
                await websocket.send(json.dumps({
                    'type': 'calibration_point_ack',
                    'index': msg.get('index'),
                    'ok': ok,
                }))

            elif msg_type == 'calibrate_finish':
                ok = tracker_state.finish_calibration()
                await websocket.send(json.dumps({
                    'type': 'calibration_complete',
                    'ok': ok,
                }))
    except websockets.ConnectionClosed:
        pass
    finally:
        connected_clients.discard(websocket)
        print(f'[WS] Client disconnected ({len(connected_clients)} active)')


async def broadcast_loop(state: EyeTrackerState):
    """Broadcasts both live gaze coordinates and camera status changes."""
    last_sent_frame_id = -1
    last_sent_status_version = -1

    while state.running:
        if connected_clients:
            # 1. Push camera/calibration state updates whenever they change
            status_ver, cam_payload = state.get_camera_state_payload()
            if status_ver != last_sent_status_version:
                last_sent_status_version = status_ver
                msg = json.dumps(cam_payload)
                await asyncio.gather(
                    *(c.send(msg) for c in list(connected_clients)),
                    return_exceptions=True,
                )

            # 2. Push gaze frame updates
            frame_id, gx, gy, is_blinking, is_calibrated = state.get_gaze_snapshot()
            if (
                is_calibrated
                and gx is not None
                and gy is not None
                and frame_id != last_sent_frame_id
            ):
                last_sent_frame_id = frame_id
                gaze_msg = json.dumps({
                    'type': 'gaze',
                    'x': gx,
                    'y': gy,
                    'normalized': True,
                    'blinking': is_blinking,
                })
                await asyncio.gather(
                    *(c.send(gaze_msg) for c in list(connected_clients)),
                    return_exceptions=True,
                )

        await asyncio.sleep(1 / 60)


def start_bundled_frontend_server():
    """
    When running as a packaged .exe/.app (or with --serve),
    serves the built React `dist/` folder locally and opens the browser.
    """
    dist_dir = resource_path('dist')
    if not os.path.isdir(dist_dir):
        return

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=dist_dir)
    httpd = http.server.ThreadingHTTPServer(('127.0.0.1', HTTP_PORT), handler)

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{HTTP_PORT}'
    print(f'[UI] Serving React app at {url}')
    webbrowser.open(url)


async def main():
    cam_thread = threading.Thread(target=camera_tracking_loop, args=(tracker_state,), daemon=True)
    cam_thread.start()

    # Auto-launch bundled React UI if running from PyInstaller exe or passed --serve
    if getattr(sys, 'frozen', False) or '--serve' in sys.argv:
        start_bundled_frontend_server()

    print(f'[WS] Eye Tracker server running on ws://{WS_HOST}:{WS_PORT}')
    async with websockets.serve(handle_client, WS_HOST, WS_PORT):
        await broadcast_loop(tracker_state)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('\n[Server] Shutting down...')
        tracker_state.running = False