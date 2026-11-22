import asyncio
import base64
import functools
import http.server
import json
import math
import os
import platform
import socket
import struct
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

LIVELINK_UDP_PORT = 11111
LIVELINK_CAMERA_INDEX = -100

ARKIT6DOF_UDP_PORT = 11112
ARKIT6DOF_CAMERA_INDEX = -101

BLINK_THRESHOLD = 0.72
CALIBRATION_BURST_FRAMES = 15
LINEAR_RIDGE_ALPHA = 1.5
RBF_RIDGE_ALPHA = 0.25
ONE_EURO_MIN_CUTOFF = 1.0
ONE_EURO_BETA = 3.0
SHOW_CAMERA_PREVIEW = False


def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return '127.0.0.1'


def resource_path(relative_path: str) -> str:
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


def find_model_path():
    candidates = [
        resource_path(os.path.join('facetracking_model', 'face_landmarker.task')),
        resource_path(os.path.join('eyetracker', 'facetracking_model', 'face_landmarker.task')),
        resource_path('face_landmarker.task'),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    script_dir = os.path.dirname(os.path.abspath(__file__))
    for root, _, files in os.walk(script_dir):
        if 'face_landmarker.task' in files:
            return os.path.join(root, 'face_landmarker.task')
    return None


MODEL_PATH = find_model_path()
if MODEL_PATH:
    print(f'[Model] Using {MODEL_PATH}')
else:
    print('[Model] Warning: face_landmarker.task not found (webcam mode disabled; iPhone UDP modes still work).')


def get_camera_backend() -> int:
    system = platform.system()
    if system == 'Windows':
        return cv2.CAP_DSHOW
    elif system == 'Darwin':
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_V4L2


def open_camera_capture(index: int) -> cv2.VideoCapture:
    if index >= 100:
        backend = (index // 100) * 100
        device_id = index % 100
        cap = cv2.VideoCapture(device_id, backend)
    else:
        cap = cv2.VideoCapture(index, get_camera_backend())

    if cap.isOpened() and platform.system() != 'Darwin':
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
        except Exception as exc:
            print(f'[Camera] Enumerator error, falling back to probing: {exc}')

    if not cameras:
        for idx in range(8):
            if active_index not in (None, LIVELINK_CAMERA_INDEX, ARKIT6DOF_CAMERA_INDEX) and (
                idx == active_index or idx == active_index % 100
            ):
                cameras.append({'index': active_index, 'name': active_name or f'Camera {idx}'})
                continue

            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                cameras.append({'index': idx, 'name': f'Camera {idx}'})
                cap.release()

    cameras.append({
        'index': ARKIT6DOF_CAMERA_INDEX,
        'name': f'iPhone 6-DoF ARKit App (UDP :{ARKIT6DOF_UDP_PORT})',
    })
    cameras.append({
        'index': LIVELINK_CAMERA_INDEX,
        'name': f'iPhone TrueDepth (Live Link Face — UDP :{LIVELINK_UDP_PORT})',
    })
    return cameras


# --- LIVE LINK FACE (UDP :11111) ---

def decode_livelink_packet(data: bytes):
    if len(data) >= 245 and data[-245] == 61:
        return struct.unpack('>61f', data[-244:])
    if len(data) >= 209 and data[-209] == 52:
        vals52 = struct.unpack('>52f', data[-208:])
        return vals52 + (0.0,) * 9
    return None


def extract_livelink_features(vals):
    left_blink = vals[0]
    right_blink = vals[7]
    is_blinking = (left_blink > BLINK_THRESHOLD) or (right_blink > BLINK_THRESHOLD)

    head_yaw, head_pitch, head_roll = vals[52], vals[53], vals[54]
    l_yaw, l_pitch = vals[55], vals[56]
    r_yaw, r_pitch = vals[58], vals[59]

    eye_yaw_avg = 0.5 * (l_yaw + r_yaw)
    eye_pitch_avg = 0.5 * (l_pitch + r_pitch)

    c_r, s_r = math.cos(head_roll), math.sin(head_roll)
    rot_eye_yaw = (eye_yaw_avg * c_r) - (eye_pitch_avg * s_r)
    rot_eye_pitch = (eye_yaw_avg * s_r) + (eye_pitch_avg * c_r)

    total_yaw = max(-1.1, min(1.1, head_yaw + rot_eye_yaw))
    total_pitch = max(-1.1, min(1.1, head_pitch + rot_eye_pitch))
    ray_x = math.tan(total_yaw)
    ray_y = math.tan(total_pitch)

    vergence = l_yaw - r_yaw
    bs_x = 0.5 * ((vals[10] - vals[9]) + (vals[2] - vals[3]))
    bs_y = 0.5 * ((vals[1] - vals[4]) + (vals[8] - vals[11]))

    features = np.array(
        [
            ray_x,
            ray_y,
            l_yaw,
            l_pitch,
            r_yaw,
            r_pitch,
            head_yaw,
            head_pitch,
            head_roll,
            vergence,
            bs_x,
            bs_y,
        ],
        dtype=np.float64,
    )

    telemetry = {
        'l_yaw': l_yaw,
        'l_pitch': l_pitch,
        'r_yaw': r_yaw,
        'r_pitch': r_pitch,
        'head_yaw': head_yaw,
        'head_pitch': head_pitch,
    }
    return features, is_blinking, telemetry


def render_livelink_preview_b64(telemetry, connected: bool, local_ip: str) -> str:
    canvas = np.full((180, 320, 3), (24, 18, 14), dtype=np.uint8)

    if not connected or telemetry is None:
        cv2.putText(canvas, 'Waiting for Live Link Face...', (24, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 200, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, f'Target: {local_ip}:{LIVELINK_UDP_PORT}', (44, 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 220, 140), 1, cv2.LINE_AA)
    else:
        for cx, yaw, pitch, label in [
            (105, telemetry['l_yaw'], telemetry['l_pitch'], 'L'),
            (215, telemetry['r_yaw'], telemetry['r_pitch'], 'R'),
        ]:
            cy = 78
            cv2.ellipse(canvas, (cx, cy), (40, 24), 0, 0, 360, (90, 110, 130), 2)
            ix = int(cx + np.clip(yaw * 60.0, -28, 28))
            iy = int(cy - np.clip(pitch * 60.0, -16, 16))
            cv2.circle(canvas, (ix, iy), 8, (80, 230, 120), -1)
            cv2.circle(canvas, (ix, iy), 3, (15, 40, 20), -1)
            cv2.putText(canvas, label, (cx - 4, cy - 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 175, 195), 1, cv2.LINE_AA)

        hy = math.degrees(telemetry['head_yaw'])
        hp = math.degrees(telemetry['head_pitch'])
        cv2.putText(canvas, f'TrueDepth 60Hz | Head: {hy:+.0f}deg, {hp:+.0f}deg', (16, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (190, 210, 230), 1, cv2.LINE_AA)

    ok_jpg, buf = cv2.imencode('.jpg', canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    return base64.b64encode(buf).decode('ascii') if ok_jpg else None


# --- CUSTOM 6-DoF ARKIT IOS APP (UDP :11112) ---

def extract_arkit6dof_features(pkt: dict):
    ox = float(pkt.get('ox', 0.0))
    oy = float(pkt.get('oy', 0.0))
    oz = float(pkt.get('oz', -0.4))
    dx = float(pkt.get('dx', 0.0))
    dy = float(pkt.get('dy', 0.0))
    dz = float(pkt.get('dz', 1.0))
    lx = float(pkt.get('lx', 0.0))
    ly = float(pkt.get('ly', 0.0))
    bl = float(pkt.get('bl', 0.0))
    br = float(pkt.get('br', 0.0))
    bsx = float(pkt.get('bsx', 0.0))
    bsy = float(pkt.get('bsy', 0.0))

    is_blinking = (bl > BLINK_THRESHOLD) or (br > BLINK_THRESHOLD)

    dz_safe = dz if abs(dz) >= 0.15 else (0.15 if dz >= 0 else -0.15)
    t = -oz / dz_safe
    ix = float(np.clip(ox + t * dx, -1.5, 1.5))
    iy = float(np.clip(oy + t * dy, -1.5, 1.5))

    features = np.array(
        [ix, iy, ox, oy, oz, dx, dy, lx, ly, bsx, bsy],
        dtype=np.float64,
    )

    telemetry = {
        'ix': ix,
        'iy': iy,
        'dist_cm': int(round(abs(oz) * 100.0)),
    }
    return features, is_blinking, telemetry


def render_arkit6dof_preview_b64(telemetry, connected: bool, local_ip: str) -> str:
    canvas = np.full((180, 320, 3), (24, 18, 14), dtype=np.uint8)

    if not connected or telemetry is None:
        cv2.putText(canvas, 'Waiting for FocusGaze iOS App...', (16, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 200, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, f'Target: {local_ip}:{ARKIT6DOF_UDP_PORT}', (44, 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 220, 140), 1, cv2.LINE_AA)
    else:
        cv2.rectangle(canvas, (30, 20), (290, 120), (75, 90, 110), 1)
        px = int(160 + np.clip(telemetry['ix'] * 320.0, -120, 120))
        py = int(70 - np.clip(telemetry['iy'] * 240.0, -45, 45))
        cv2.circle(canvas, (px, py), 7, (80, 230, 120), -1)

        dist = telemetry['dist_cm']
        dist_ok = 28 <= dist <= 52
        dist_color = (80, 230, 120) if dist_ok else (80, 180, 255)
        label = 'OPTIMAL' if dist_ok else ('TOO CLOSE' if dist < 28 else 'MOVE CLOSER')
        cv2.putText(canvas, f'6-DoF TrueDepth | Dist: {dist} cm ({label})', (14, 148),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, dist_color, 1, cv2.LINE_AA)

    ok_jpg, buf = cv2.imencode('.jpg', canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    return base64.b64encode(buf).decode('ascii') if ok_jpg else None


# --- SMOOTHING & REGRESSION MODELS ---

class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=3.0, d_cutoff=1.0):
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


class GazeKernelRidgeModel:
    def __init__(self, linear_alpha=LINEAR_RIDGE_ALPHA, rbf_alpha=RBF_RIDGE_ALPHA):
        self.linear_alpha = linear_alpha
        self.rbf_alpha = rbf_alpha
        self.mean = None
        self.std = None
        self.linear_weights = None
        self.rbf_centers = None
        self.rbf_weights = None
        self.rbf_gamma = 0.1
        self.drift_offset = np.zeros(2, dtype=np.float64)

    @property
    def is_fitted(self) -> bool:
        return self.linear_weights is not None

    def reset(self):
        self.mean = None
        self.std = None
        self.linear_weights = None
        self.rbf_centers = None
        self.rbf_weights = None
        self.rbf_gamma = 0.1
        self.drift_offset = np.zeros(2, dtype=np.float64)

    def fit(self, X: np.ndarray, Y: np.ndarray, dot_ids: np.ndarray):
        N, D = X.shape
        unique_dots = np.unique(dot_ids)
        M = len(unique_dots)

        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-4] = 1.0
        Xn = (X - mean) / std

        Xb = np.hstack([Xn, np.ones((N, 1), dtype=np.float64)])
        eff_alpha = self.linear_alpha * max(1.0, N / max(1, M))
        penalty = eff_alpha * np.eye(D + 1, dtype=np.float64)
        penalty[-1, -1] = 0.0
        linear_weights = np.linalg.solve(Xb.T @ Xb + penalty, Xb.T @ Y)

        lin_preds = Xb @ linear_weights
        lin_err_x = float(np.mean(np.abs(lin_preds[:, 0] - Y[:, 0])) * 100.0)
        lin_err_y = float(np.mean(np.abs(lin_preds[:, 1] - Y[:, 1])) * 100.0)

        centers = []
        y_dots = []
        for d in unique_dots:
            mask = dot_ids == d
            centers.append(Xn[mask].mean(axis=0))
            y_dots.append(Y[mask][0])

        C = np.array(centers, dtype=np.float64)
        Y_dots = np.array(y_dots, dtype=np.float64)
        Cb = np.hstack([C, np.ones((M, 1), dtype=np.float64)])
        residuals = Y_dots - (Cb @ linear_weights)

        diff = C[:, None, :] - C[None, :, :]
        D2 = np.sum(diff * diff, axis=2)
        upper = D2[np.triu_indices(M, k=1)]
        med_d2 = float(np.median(upper)) if len(upper) > 0 else 1.0
        if med_d2 < 1e-4:
            med_d2 = 1.0
        gamma = 1.0 / (2.0 * med_d2)

        K = np.exp(-gamma * D2)
        rbf_weights = np.linalg.solve(K + self.rbf_alpha * np.eye(M, dtype=np.float64), residuals)

        self.mean = mean
        self.std = std
        self.linear_weights = linear_weights
        self.rbf_centers = C
        self.rbf_weights = rbf_weights
        self.rbf_gamma = gamma
        self.drift_offset = np.zeros(2, dtype=np.float64)

        all_d2 = np.sum((Xn[:, None, :] - C[None, :, :]) ** 2, axis=2)
        hybrid_preds = lin_preds + (np.exp(-gamma * all_d2) @ rbf_weights)
        hyb_err_x = float(np.mean(np.abs(hybrid_preds[:, 0] - Y[:, 0])) * 100.0)
        hyb_err_y = float(np.mean(np.abs(hybrid_preds[:, 1] - Y[:, 1])) * 100.0)

        return (lin_err_x, lin_err_y), (hyb_err_x, hyb_err_y)

    def _predict_raw(self, feature_vec: np.ndarray):
        xn = (feature_vec - self.mean) / self.std
        y_lin = np.append(xn, 1.0) @ self.linear_weights
        d2 = np.sum((self.rbf_centers - xn) ** 2, axis=1)
        y_rbf = np.exp(-self.rbf_gamma * d2) @ self.rbf_weights
        return y_lin + y_rbf + self.drift_offset

    def predict(self, feature_vec: np.ndarray):
        if self.linear_weights is None:
            return None, None
        pred = self._predict_raw(feature_vec)
        return float(np.clip(pred[0], 0.0, 1.0)), float(np.clip(pred[1], 0.0, 1.0))

    def apply_drift_correction(self, clean_features: np.ndarray, target_x: float, target_y: float):
        if self.linear_weights is None or len(clean_features) == 0:
            return False, 0.0, 0.0

        preds = np.array([self._predict_raw(f) for f in clean_features], dtype=np.float64)
        delta_x = float(target_x) - float(np.mean(preds[:, 0]))
        delta_y = float(target_y) - float(np.mean(preds[:, 1]))

        self.drift_offset[0] += delta_x
        self.drift_offset[1] += delta_y
        return True, delta_x * 100.0, delta_y * 100.0


# --- MEDIAPIPE WEBCAM FEATURE EXTRACTION ---

def compute_eye_relative_iris(p_c1, p_c2, p_iris, frame_w: int, frame_h: int):
    if p_c1.x <= p_c2.x:
        left_c, right_c = p_c1, p_c2
    else:
        left_c, right_c = p_c2, p_c1

    dx = (right_c.x - left_c.x) * frame_w
    dy = (right_c.y - left_c.y) * frame_h
    eye_w = math.hypot(dx, dy)
    if eye_w < 1e-4:
        return 0.0, 0.0

    ux, uy = dx / eye_w, dy / eye_w
    vx, vy = -uy, ux

    mid_x = (left_c.x + right_c.x) * 0.5 * frame_w
    mid_y = (left_c.y + right_c.y) * 0.5 * frame_h

    ox = (p_iris.x * frame_w) - mid_x
    oy = (p_iris.y * frame_h) - mid_y

    rel_x = (ox * ux + oy * uy) / eye_w
    rel_y = (ox * vx + oy * vy) / eye_w
    return rel_x, rel_y


def extract_gaze_features(result, frame_w: int, frame_h: int):
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

    rel_x_a, rel_y_a = compute_eye_relative_iris(lm[33], lm[133], lm[468], frame_w, frame_h)
    rel_x_b, rel_y_b = compute_eye_relative_iris(lm[362], lm[263], lm[473], frame_w, frame_h)

    rel_x_avg = 0.5 * (rel_x_a + rel_x_b)
    rel_y_avg = 0.5 * (rel_y_a + rel_y_b)

    bs_x = 0.5 * (
        (bs.get('eyeLookOutRight', 0.0) - bs.get('eyeLookInRight', 0.0))
        + (bs.get('eyeLookInLeft', 0.0) - bs.get('eyeLookOutLeft', 0.0))
    )
    bs_y = 0.5 * (
        (bs.get('eyeLookDownLeft', 0.0) - bs.get('eyeLookUpLeft', 0.0))
        + (bs.get('eyeLookDownRight', 0.0) - bs.get('eyeLookUpRight', 0.0))
    )

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
    }

    return features, is_blinking, debug_points


def trim_feature_outliers(samples):
    arr = np.array(samples, dtype=np.float64)
    if len(arr) >= 6:
        med = np.median(arr, axis=0)
        scale = np.std(arr, axis=0)
        scale[scale < 1e-4] = 1.0
        dists = np.linalg.norm((arr - med) / scale, axis=1)
        keep_count = max(4, int(len(arr) * 0.8))
        arr = arr[np.argsort(dists)[:keep_count]]
    return arr


# --- SHARED THREAD-SAFE TRACKER STATE ---

class EyeTrackerState:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True

        self.available_cameras = []
        self.active_camera_index = None
        self.requested_camera_index = None
        self.camera_switch_requested = False
        self.camera_status = 'initializing'
        self.camera_error = None
        self.status_version = 0

        self.preview_enabled = True
        self.latest_preview_b64 = None
        self.preview_frame_id = 0
        self.preview_face_detected = False
        self.preview_blinking = False

        self.feature_buffer = deque(maxlen=CALIBRATION_BURST_FRAMES)
        self.is_sampling_dot = False
        self.active_sample_frames = []
        self.active_sample_target = None

        self.is_blinking = False
        self.cal_X = []
        self.cal_Y = []
        self.cal_dot_ids = []
        self.cal_dot_count = 0
        self.model = GazeKernelRidgeModel(
            linear_alpha=LINEAR_RIDGE_ALPHA,
            rbf_alpha=RBF_RIDGE_ALPHA,
        )

        self.filter_x = OneEuroFilter(min_cutoff=ONE_EURO_MIN_CUTOFF, beta=ONE_EURO_BETA)
        self.filter_y = OneEuroFilter(min_cutoff=ONE_EURO_MIN_CUTOFF, beta=ONE_EURO_BETA)

        self.smoothed_x = None
        self.smoothed_y = None
        self.frame_id = 0

    def set_preview_enabled(self, enabled: bool):
        with self.lock:
            self.preview_enabled = bool(enabled)

    def update_preview(self, b64_jpeg: str, face_detected: bool, is_blinking: bool):
        if b64_jpeg is None:
            return
        with self.lock:
            self.latest_preview_b64 = b64_jpeg
            self.preview_face_detected = face_detected
            self.preview_blinking = is_blinking
            self.preview_frame_id += 1

    def get_preview_snapshot(self):
        with self.lock:
            return (
                self.preview_enabled,
                self.preview_frame_id,
                self.latest_preview_b64,
                self.preview_face_detected,
                self.preview_blinking,
            )

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
            if auto_select_if_none and self.active_camera_index is None and found:
                self.requested_camera_index = found[0]['index']
                self.camera_switch_requested = True
                self.camera_status = 'switching'
                self.camera_error = None

            self.status_version += 1
            print(f'[Camera] Available device(s): {[c["name"] for c in found]}')

    def request_camera_switch(self, new_index: int):
        with self.lock:
            self.requested_camera_index = int(new_index)
            self.camera_switch_requested = True
            self.camera_status = 'switching'
            self.camera_error = None

            self.latest_preview_b64 = None
            self.is_sampling_dot = False
            self.active_sample_frames.clear()
            self.active_sample_target = None
            self.model.reset()
            self.cal_X.clear()
            self.cal_Y.clear()
            self.cal_dot_ids.clear()
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
            self.cal_dot_ids.clear()
            self.cal_dot_count = 0
            self.is_sampling_dot = False
            self.active_sample_frames.clear()
            self.active_sample_target = None
            self.model.reset()
            self.filter_x.reset()
            self.filter_y.reset()
            self.smoothed_x = None
            self.smoothed_y = None
            self.status_version += 1
            print('[Calibration] Reset started (13-point Hybrid RBF capture).')

    def begin_calibration_sample(self, target_x: float, target_y: float):
        with self.lock:
            self.active_sample_frames.clear()
            self.active_sample_target = (float(target_x), float(target_y))
            self.is_sampling_dot = True

    def end_calibration_sample(self):
        with self.lock:
            self.is_sampling_dot = False
            target = self.active_sample_target
            samples = list(self.active_sample_frames)
            self.active_sample_frames.clear()
            self.active_sample_target = None

            if target is None or len(samples) < 3:
                print(
                    f'[Calibration] Warning: Only {len(samples)} valid face frame(s) captured. '
                    'Retry this dot.'
                )
                return False, len(samples)

            arr = trim_feature_outliers(samples)
            dot_id = self.cal_dot_count
            for feat in arr:
                self.cal_X.append(feat)
                self.cal_Y.append([target[0], target[1]])
                self.cal_dot_ids.append(dot_id)

            self.cal_dot_count += 1
            print(
                f'[Calibration] Dot {self.cal_dot_count}: kept {len(arr)}/{len(samples)} clean frames '
                f'-> target=({target[0]:.2f}, {target[1]:.2f})'
            )
            return True, len(arr)

    def end_recalibration_sample(self):
        with self.lock:
            self.is_sampling_dot = False
            target = self.active_sample_target
            samples = list(self.active_sample_frames)
            self.active_sample_frames.clear()
            self.active_sample_target = None

            if not self.model.is_fitted:
                return False, 'Run full calibration first before quick-recalibrating on a word.'

            if target is None or len(samples) < 3:
                return False, 'Face not detected clearly — look steadily at the red dot and press SPACE again.'

            arr = trim_feature_outliers(samples)
            ok, dx_pct, dy_pct = self.model.apply_drift_correction(arr, target[0], target[1])
            if ok:
                self.filter_x.reset()
                self.filter_y.reset()
                print(
                    f'[Recalibration] Drift corrected at ({target[0]:.2f}, {target[1]:.2f}) '
                    f'using {len(arr)} frames -> offset shift: dx={dx_pct:+.1f}%, dy={dy_pct:+.1f}%'
                )
                return True, ''
            return False, 'Could not apply drift correction.'

    def finish_calibration(self) -> bool:
        with self.lock:
            if self.cal_dot_count < 4 or len(self.cal_X) < 12:
                print('[Calibration] Failed: Need at least 4 valid calibration dots.')
                return False

            X = np.array(self.cal_X, dtype=np.float64)
            Y = np.array(self.cal_Y, dtype=np.float64)
            dot_ids = np.array(self.cal_dot_ids, dtype=np.int32)

            try:
                (lx, ly), (hx, hy) = self.model.fit(X, Y, dot_ids)
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
                f'Linear err: X={lx:.1f}%, Y={ly:.1f}%  -->  Hybrid RBF err: X={hx:.1f}%, Y={hy:.1f}%'
            )
            return True

    def update_from_frame(self, feature_vec, is_blinking: bool):
        if feature_vec is not None and not np.all(np.isfinite(feature_vec)):
            return
        with self.lock:
            self.is_blinking = is_blinking
            self.frame_id += 1

            if feature_vec is not None and not is_blinking:
                self.feature_buffer.append(feature_vec)

                if self.is_sampling_dot:
                    self.active_sample_frames.append(feature_vec)

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


# --- MAIN TRACKING THREAD ---

def camera_tracking_loop(state: EyeTrackerState):
    landmarker = None
    if MODEL_PATH:
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

    local_ip = get_local_ip()
    print(f'[6-DoF ARKit] Point FocusGaze iOS App to {local_ip}:{ARKIT6DOF_UDP_PORT}')
    print(f'[LiveLink]    Point Live Link Face    to {local_ip}:{LIVELINK_UDP_PORT}')

    state.refresh_cameras(auto_select_if_none=True)

    cap = None
    udp_sock = None
    using_livelink = False
    using_arkit6dof = False
    last_timestamp_ms = 0
    consecutive_failures = 0
    last_preview_time = 0.0
    last_udp_packet_time = 0.0
    last_udp_telemetry = None
    last_udp_blinking = False

    try:
        while state.running:
            target_idx = state.consume_camera_switch_request()
            if target_idx is not None:
                if cap is not None:
                    cap.release()
                    cap = None
                if udp_sock is not None:
                    udp_sock.close()
                    udp_sock = None
                using_livelink = False
                using_arkit6dof = False
                last_udp_packet_time = 0.0
                last_udp_telemetry = None

                if target_idx == LIVELINK_CAMERA_INDEX:
                    try:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        sock.bind(('0.0.0.0', LIVELINK_UDP_PORT))
                        sock.setblocking(False)
                        udp_sock = sock
                        using_livelink = True
                        state.set_camera_result(LIVELINK_CAMERA_INDEX, 'active', None)
                        print(f'[LiveLink] Listening on {local_ip}:{LIVELINK_UDP_PORT}')
                    except OSError as exc:
                        state.set_camera_result(
                            None, 'error', f'Could not bind UDP port {LIVELINK_UDP_PORT}: {exc}'
                        )

                elif target_idx == ARKIT6DOF_CAMERA_INDEX:
                    try:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        sock.bind(('0.0.0.0', ARKIT6DOF_UDP_PORT))
                        sock.setblocking(False)
                        udp_sock = sock
                        using_arkit6dof = True
                        state.set_camera_result(ARKIT6DOF_CAMERA_INDEX, 'active', None)
                        print(f'[6-DoF ARKit] Listening on {local_ip}:{ARKIT6DOF_UDP_PORT}')
                    except OSError as exc:
                        state.set_camera_result(
                            None, 'error', f'Could not bind UDP port {ARKIT6DOF_UDP_PORT}: {exc}'
                        )

                else:
                    if landmarker is None:
                        state.set_camera_result(
                            None,
                            'error',
                            'face_landmarker.task not found. Select an iPhone UDP mode or add the model file.',
                        )
                    else:
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

            # --- MODE A1: Live Link Face (UDP :11111) ---
            if using_livelink and udp_sock is not None:
                latest_packet = None
                while True:
                    try:
                        packet, _ = udp_sock.recvfrom(4096)
                        latest_packet = packet
                    except (BlockingIOError, OSError):
                        break

                now_perf = time.perf_counter()

                if latest_packet is not None:
                    vals = decode_livelink_packet(latest_packet)
                    if vals is not None:
                        if last_udp_packet_time == 0.0:
                            print('[LiveLink] First UDP packet received!')
                        features, is_blinking, telemetry = extract_livelink_features(vals)
                        state.update_from_frame(features, is_blinking)
                        last_udp_packet_time = now_perf
                        last_udp_telemetry = telemetry
                        last_udp_blinking = is_blinking
                else:
                    time.sleep(0.004)

                if state.preview_enabled and (now_perf - last_preview_time) >= 0.085:
                    last_preview_time = now_perf
                    connected = (now_perf - last_udp_packet_time) < 1.0
                    b64 = render_livelink_preview_b64(last_udp_telemetry, connected, local_ip)
                    state.update_preview(b64, connected, last_udp_blinking if connected else False)
                continue

            # --- MODE A2: Custom iPhone 6-DoF ARKit App (UDP :11112) ---
            if using_arkit6dof and udp_sock is not None:
                latest_packet = None
                while True:
                    try:
                        packet, _ = udp_sock.recvfrom(4096)
                        latest_packet = packet
                    except (BlockingIOError, OSError):
                        break

                now_perf = time.perf_counter()

                if latest_packet is not None:
                    try:
                        pkt = json.loads(latest_packet.decode('utf-8'))
                        if last_udp_packet_time == 0.0:
                            print(
                                f'[6-DoF ARKit] First UDP packet received! '
                                f'oz={pkt.get("oz")}, bl={pkt.get("bl")}, br={pkt.get("br")}'
                            )
                        features, is_blinking, telemetry = extract_arkit6dof_features(pkt)
                        state.update_from_frame(features, is_blinking)
                        last_udp_packet_time = now_perf
                        last_udp_telemetry = telemetry
                        last_udp_blinking = is_blinking
                    except Exception as exc:
                        print(f'[6-DoF ARKit] Parse error: {exc} | raw={latest_packet[:120]}')
                else:
                    time.sleep(0.004)

                if state.preview_enabled and (now_perf - last_preview_time) >= 0.085:
                    last_preview_time = now_perf
                    connected = (now_perf - last_udp_packet_time) < 1.0
                    b64 = render_arkit6dof_preview_b64(last_udp_telemetry, connected, local_ip)
                    state.update_preview(b64, connected, last_udp_blinking if connected else False)
                continue

            # --- MODE B: Regular Webcam via OpenCV + MediaPipe ---
            if cap is None or not cap.isOpened() or landmarker is None:
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

            now_perf = time.perf_counter()
            if state.preview_enabled and (now_perf - last_preview_time) >= 0.085:
                last_preview_time = now_perf
                preview_w = 320
                scale = preview_w / float(w)
                preview_h = max(1, int(h * scale))
                small = cv2.resize(frame, (preview_w, preview_h))

                for pt in debug_pts.get('corners', []):
                    cv2.circle(small, (int(pt[0] * scale), int(pt[1] * scale)), 2, (0, 180, 255), -1)
                for pt in debug_pts.get('irises', []):
                    cv2.circle(small, (int(pt[0] * scale), int(pt[1] * scale)), 3, (0, 255, 0), -1)

                ok_jpg, buf = cv2.imencode('.jpg', small, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
                if ok_jpg:
                    b64 = base64.b64encode(buf).decode('ascii')
                    state.update_preview(b64, features is not None, is_blinking)

            if SHOW_CAMERA_PREVIEW:
                cv2.imshow('Eye Tracker Debug Preview', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    state.running = False
                    break
    finally:
        if cap is not None:
            cap.release()
        if udp_sock is not None:
            udp_sock.close()
        if landmarker is not None:
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

            elif msg_type == 'set_preview':
                tracker_state.set_preview_enabled(bool(msg.get('enabled', True)))

            elif msg_type == 'calibrate_start':
                tracker_state.reset_calibration()

            elif msg_type == 'calibrate_point_begin':
                tracker_state.begin_calibration_sample(msg['targetX'], msg['targetY'])

            elif msg_type == 'calibrate_point_end':
                ok, count = tracker_state.end_calibration_sample()
                await websocket.send(json.dumps({
                    'type': 'calibration_point_ack',
                    'index': msg.get('index'),
                    'ok': ok,
                    'frames': count,
                }))

            elif msg_type == 'calibrate_finish':
                ok = tracker_state.finish_calibration()
                await websocket.send(json.dumps({
                    'type': 'calibration_complete',
                    'ok': ok,
                }))

            elif msg_type == 'recalibrate_point_begin':
                tracker_state.begin_calibration_sample(msg['targetX'], msg['targetY'])

            elif msg_type == 'recalibrate_point_end':
                ok, reason = tracker_state.end_recalibration_sample()
                await websocket.send(json.dumps({
                    'type': 'recalibration_ack',
                    'ok': ok,
                    'reason': reason,
                }))
    except websockets.ConnectionClosed:
        pass
    finally:
        connected_clients.discard(websocket)
        print(f'[WS] Client disconnected ({len(connected_clients)} active)')


async def broadcast_loop(state: EyeTrackerState):
    last_sent_frame_id = -1
    last_sent_status_version = -1
    last_sent_preview_id = -1

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

            prev_on, prev_id, prev_b64, face_ok, prev_blink = state.get_preview_snapshot()
            if prev_on and prev_b64 is not None and prev_id != last_sent_preview_id:
                last_sent_preview_id = prev_id
                prev_msg = json.dumps({
                    'type': 'preview_frame',
                    'image': prev_b64,
                    'faceDetected': face_ok,
                    'blinking': prev_blink,
                })
                await asyncio.gather(
                    *(c.send(prev_msg) for c in list(connected_clients)),
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