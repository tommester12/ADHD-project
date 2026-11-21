import asyncio
import functools
import http.server
import json
import math
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
BLINK_THRESHOLD = 0.5
CALIBRATION_BURST_FRAMES = 15  # Frames averaged/used per calibration dot
RIDGE_ALPHA = 1.5              # Higher = smoother/more stable, Lower = tighter fit
ONE_EURO_MIN_CUTOFF = 0.9      # Lower = steadier when fixating on a word
ONE_EURO_BETA = 2.5            # Higher = faster snap when jumping across words
SHOW_CAMERA_PREVIEW = False    # Set True to inspect eye framing & lighting


def resource_path(relative_path: str) -> str:
    """Resolves asset paths for both dev mode and PyInstaller portable builds."""
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


MODEL_PATH = resource_path(os.path.join('facetracking_model', 'face_landmarker.task'))
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = r'C:\Users\tomme\Desktop\ADHD project\eyetracker\facetracking_model\face_landmarker.task'


def get_camera_backend() -> int:
    system = platform.system()
    if system == 'Windows':
        return cv2.CAP_DSHOW
    elif system == 'Darwin':  # macOS
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_V4L2


def open_camera_capture(index: int) -> cv2.VideoCapture:
    if index >= 100:
        cap = cv2.VideoCapture(index)
    else:
        cap = cv2.VideoCapture(index, get_camera_backend())

    if cap.isOpened():
        # Request 720p @ 60fps when supported for crisp eyes & low latency
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 60)
    return cap


def discover_cameras(active_index=None, active_name=None):
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

    for idx in range(8):
        if active_index is not None and (idx == active_index or idx == active_index % 100):
            cameras.append({'index': active_index, 'name': active_name or f'Camera {idx}'})
            continue

        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            cameras.append({'index': idx, 'name': f'Camera {idx}'})
            cap.release()

    return cameras


class OneEuroFilter:
    """Adaptive low-pass filter: steady during fixations, snappy during saccades."""

    def __init__(self, min_cutoff=0.9, beta=2.5, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float, t: float) -> float:
        if self.t_prev is None or self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = t
            return x

        dt = max(t - self.t_prev, 1e-4)
        a_d = self._alpha(self.d_cutoff, dt)
        dx = (x - self.x_prev) / dt
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


class GazeRidgeModel:
    """
    Standardized Ridge Regression mapping eye-relative iris coordinates,
    MediaPipe gaze blendshapes, and 3D head pose -> normalized viewport (x, y).
    """

    def __init__(self, alpha=RIDGE_ALPHA):
        self.alpha = alpha
        self.mean = None
        self.std = None
        self.weights = None  # shape (D + 1, 2)

    @property
    def is_fitted(self) -> bool:
        return self.weights is not None

    def reset(self):
        self.mean = None
        self.std = None
        self.weights = None

    def fit(self, X: np.ndarray, Y: np.ndarray, num_dots: int):
        N, D = X.shape
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-4] = 1.0  # Prevent blow-up for features that stayed constant

        Xn = (X - mean) / std
        Xb = np.hstack([Xn, np.ones((N, 1), dtype=np.float64)])

        # Scale regularization proportionally to frames-per-dot; never penalize intercept
        effective_alpha = self.alpha * max(1.0, N / max(1, num_dots))
        penalty = effective_alpha * np.eye(D + 1, dtype=np.float64)
        penalty[-1, -1] = 0.0

        A = Xb.T @ Xb + penalty
        B = Xb.T @ Y
        weights = np.linalg.solve(A, B)

        self.mean = mean
        self.std = std
        self.weights = weights

        # Report calibration fit accuracy across training samples
        preds = Xb @ weights
        err_x = float(np.mean(np.abs(preds[:, 0] - Y[:, 0])) * 100.0)
        err_y = float(np.mean(np.abs(preds[:, 1] - Y[:, 1])) * 100.0)
        return err_x, err_y

    def predict(self, feature_vec: np.ndarray):
        if self.weights is None:
            return None, None
        xn = (feature_vec - self.mean) / self.std
        xb = np.append(xn, 1.0)
        pred = xb @ self.weights
        return float(np.clip(pred[0], 0.0, 1.0)), float(np.clip(pred[1], 0.0, 1.0))


def compute_eye_relative_iris(p_c1, p_c2, p_iris, frame_w: int, frame_h: int):
    """
    Computes iris offset in the eye's own corner-to-corner local coordinate frame
    (square pixel space), invariant to head translation, distance, roll, and eyelid droop.
    """
    if p_c1.x <= p_c2.x:
        left_c, right_c = p_c1, p_c2
    else:
        left_c, right_c = p_c2, p_c1

    dx = (right_c.x - left_c.x) * frame_w
    dy = (right_c.y - left_c.y) * frame_h
    eye_w = math.hypot(dx, dy)
    if eye_w < 1e-4:
        return 0.0, 0.0

    ux, uy = dx / eye_w, dy / eye_w  # Rightward along eye corners
    vx, vy = -uy, ux                 # Downward perpendicular to eye corners

    mid_x = (left_c.x + right_c.x) * 0.5 * frame_w
    mid_y = (left_c.y + right_c.y) * 0.5 * frame_h

    ox = (p_iris.x * frame_w) - mid_x
    oy = (p_iris.y * frame_h) - mid_y

    rel_x = (ox * ux + oy * uy) / eye_w
    rel_y = (ox * vx + oy * vy) / eye_w
    return rel_x, rel_y


def extract_gaze_features(result, frame_w: int, frame_h: int):
    """
    Extracts a 10-element head-pose-compensated gaze feature vector + blink state.
    Returns: (feature_vec or None, is_blinking: bool, debug_points: dict)
    """
    bs = {}
    if result.face_blendshapes:
        for cat in result.face_blendshapes[0]:
            bs[cat.category_name] = cat.score

    left_blink = bs.get('eyeBlinkLeft', 0.0)
    right_blink = bs.get('eyeBlinkRight', 0.0)
    is_blinking = (left_blink > BLINK_THRESHOLD) or (right_blink > BLINK_THRESHOLD)

    if not result.face_landmarks or len(result.face_landmarks[0]) < 478:
        return None, is_blinking, {}

    lm = result.face_landmarks[0]

    # Eye-relative iris offsets (both eyes)
    rel_x_a, rel_y_a = compute_eye_relative_iris(lm[33], lm[133], lm[468], frame_w, frame_h)
    rel_x_b, rel_y_b = compute_eye_relative_iris(lm[362], lm[263], lm[473], frame_w, frame_h)

    rel_x_avg = 0.5 * (rel_x_a + rel_x_b)
    rel_y_avg = 0.5 * (rel_y_a + rel_y_b)

    # Neural gaze blendshapes (greatly improves vertical Y stability)
    bs_x = 0.5 * (
        (bs.get('eyeLookOutRight', 0.0) - bs.get('eyeLookInRight', 0.0))
        + (bs.get('eyeLookInLeft', 0.0) - bs.get('eyeLookOutLeft', 0.0))
    )
    bs_y = 0.5 * (
        (bs.get('eyeLookDownLeft', 0.0) - bs.get('eyeLookUpLeft', 0.0))
        + (bs.get('eyeLookDownRight', 0.0) - bs.get('eyeLookUpRight', 0.0))
    )

    # 3D Head Pose (yaw & pitch in radians)
    yaw, pitch = 0.0, 0.0
    if result.facial_transformation_matrixes:
        mat = np.array(result.facial_transformation_matrixes[0], dtype=np.float64)
        R = mat[:3, :3]
        yaw = math.atan2(R[0, 2], R[2, 2])
        pitch = math.atan2(-R[1, 2], math.hypot(R[0, 2], R[2, 2]))

    features = np.array(
        [
            rel_x_a,
            rel_y_a,
            rel_x_b,
            rel_y_b,
            bs_x,
            bs_y,
            yaw,
            pitch,
            rel_x_avg * rel_x_avg,
            rel_y_avg * rel_y_avg,
        ],
        dtype=np.float64,
    )

    debug_points = {
        'irises': [
            (int(lm[468].x * frame_w), int(lm[468].y * frame_h)),
            (int(lm[473].x * frame_w), int(lm[473].y * frame_h)),
        ],
        'corners': [
            (int(lm[33].x * frame_w), int(lm[33].y * frame_h)),
            (int(lm[133].x * frame_w), int(lm[133].y * frame_h)),
            (int(lm[362].x * frame_w), int(lm[362].y * frame_h)),
            (int(lm[263].x * frame_w), int(lm[263].y * frame_h)),
        ],
        'rel': (rel_x_avg, rel_y_avg),
        'pose': (math.degrees(yaw), math.degrees(pitch)),
    }

    return features, is_blinking, debug_points


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
        self.camera_status = 'initializing'
        self.camera_error = None
        self.status_version = 0

        # Feature buffer & Regression model
        self.feature_buffer = deque(maxlen=CALIBRATION_BURST_FRAMES)
        self.is_blinking = False
        self.cal_X = []
        self.cal_Y = []
        self.cal_dot_count = 0
        self.model = GazeRidgeModel(alpha=RIDGE_ALPHA)

        # Adaptive One-Euro filters for smooth, snappy cursor
        self.filter_x = OneEuroFilter(min_cutoff=ONE_EURO_MIN_CUTOFF, beta=ONE_EURO_BETA)
        self.filter_y = OneEuroFilter(min_cutoff=ONE_EURO_MIN_CUTOFF, beta=ONE_EURO_BETA)

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

            self.model.reset()
            self.cal_X.clear()
            self.cal_Y.clear()
            self.cal_dot_count = 0
            self.feature_buffer.clear()
            self.filter_x.reset()
            self.filter_y.reset()
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
                'isCalibrated': self.model.is_fitted,
                'platform': platform.system(),
            }

    def reset_calibration(self):
        with self.lock:
            self.cal_X.clear()
            self.cal_Y.clear()
            self.cal_dot_count = 0
            self.model.reset()
            self.filter_x.reset()
            self.filter_y.reset()
            self.smoothed_x = None
            self.smoothed_y = None
            self.status_version += 1
            print('[Calibration] Reset started.')

    def record_calibration_point(self, target_x: float, target_y: float) -> bool:
        with self.lock:
            if len(self.feature_buffer) < 3:
                print('[Calibration] Warning: Not enough stable face frames for this dot.')
                return False

            # Store the recent burst of non-blinking frames so Ridge sees real noise variance
            samples = list(self.feature_buffer)
            for feat in samples:
                self.cal_X.append(feat)
                self.cal_Y.append([float(target_x), float(target_y)])

            self.cal_dot_count += 1
            print(
                f'[Calibration] Dot {self.cal_dot_count}: captured {len(samples)} frames '
                f'-> target=({target_x:.2f}, {target_y:.2f})'
            )
            return True

    def finish_calibration(self) -> bool:
        with self.lock:
            if self.cal_dot_count < 4 or len(self.cal_X) < 12:
                print('[Calibration] Failed: Need at least 4 valid calibration dots.')
                return False

            X = np.array(self.cal_X, dtype=np.float64)
            Y = np.array(self.cal_Y, dtype=np.float64)

            try:
                err_x, err_y = self.model.fit(X, Y, num_dots=self.cal_dot_count)
            except np.linalg.LinAlgError as exc:
                print(f'[Calibration] Matrix solve failed: {exc}')
                return False

            self.filter_x.reset()
            self.filter_y.reset()
            self.smoothed_x = None
            self.smoothed_y = None
            self.status_version += 1
            print(
                f'[Calibration] Complete ({self.cal_dot_count} dots, {len(X)} frames)! '
                f'Mean fit error: X={err_x:.1f}%, Y={err_y:.1f}% of screen'
            )
            return True

    def update_from_frame(self, feature_vec, is_blinking: bool):
        with self.lock:
            self.is_blinking = is_blinking
            self.frame_id += 1

            if feature_vec is not None and not is_blinking:
                self.feature_buffer.append(feature_vec)

                if self.model.is_fitted:
                    raw_x, raw_y = self.model.predict(feature_vec)
                    if raw_x is not None and raw_y is not None:
                        now = time.perf_counter()
                        self.smoothed_x = float(np.clip(self.filter_x.filter(raw_x, now), 0.0, 1.0))
                        self.smoothed_y = float(np.clip(self.filter_y.filter(raw_y, now), 0.0, 1.0))

    def get_gaze_snapshot(self):
        with self.lock:
            return (
                self.frame_id,
                self.smoothed_x,
                self.smoothed_y,
                self.is_blinking,
                self.model.is_fitted,
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
        output_facial_transformation_matrixes=True,
    )
    landmarker = FaceLandmarker.create_from_options(options)

    state.refresh_cameras(auto_select_if_none=True)

    cap = None
    last_timestamp_ms = 0
    consecutive_failures = 0

    try:
        while state.running:
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
                    state.set_camera_result(None, 'error', f'Could not open selected camera.{hint}')

            if cap is None or not cap.isOpened():
                time.sleep(0.15)
                continue

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
            features, is_blinking, debug_pts = extract_gaze_features(result, w, h)
            state.update_from_frame(features, is_blinking)

            if SHOW_CAMERA_PREVIEW:
                for pt in debug_pts.get('corners', []):
                    cv2.circle(frame, pt, 2, (255, 180, 0), -1)
                for pt in debug_pts.get('irises', []):
                    cv2.circle(frame, pt, 3, (0, 255, 0), -1)
                if 'pose' in debug_pts:
                    yaw_d, pitch_d = debug_pts['pose']
                    rx, ry = debug_pts['rel']
                    cv2.putText(
                        frame,
                        f'RelIris: ({rx:+.2f}, {ry:+.2f})  Head: yaw={yaw_d:+.1f} pitch={pitch_d:+.1f}',
                        (16, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2,
                    )
                cv2.imshow('Eye Tracker Debug Preview', frame)
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
    last_sent_frame_id = -1
    last_sent_status_version = -1

    while state.running:
        if connected_clients:
            status_ver, cam_payload = state.get_camera_state_payload()
            if status_ver != last_sent_status_version:
                last_sent_status_version = status_ver
                msg = json.dumps(cam_payload)
                await asyncio.gather(
                    *(c.send(msg) for c in list(connected_clients)),
                    return_exceptions=True,
                )

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