import { useEffect, useRef, useState, useCallback } from 'react';

export function useGazeTracker({ wsUrl = 'ws://localhost:8765' } = {}) {
  const [inputMode, setInputMode] = useState('mouse');
  const [connectionStatus, setConnectionStatus] = useState('disconnected');

  const [cameras, setCameras] = useState([]);
  const [activeCamera, setActiveCamera] = useState(null);
  const [cameraStatus, setCameraStatus] = useState('initializing');
  const [cameraError, setCameraError] = useState(null);
  const [isCalibrated, setIsCalibrated] = useState(false);

  const wsRef = useRef(null);
  const listenersRef = useRef(new Set());

  const subscribeGaze = useCallback((callback) => {
    listenersRef.current.add(callback);
    return () => {
      listenersRef.current.delete(callback);
    };
  }, []);

  const emitGaze = useCallback((gazeData) => {
    listenersRef.current.forEach((cb) => cb(gazeData));
  }, []);

  // Mouse simulation mode
  useEffect(() => {
    if (inputMode !== 'mouse') return;

    const handleMouseMove = (e) => {
      emitGaze({
        x: e.clientX,
        y: e.clientY,
        blinking: false,
        source: 'mouse',
      });
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, [inputMode, emitGaze]);

  // Python WebSocket mode
  useEffect(() => {
    if (inputMode !== 'websocket') {
      setConnectionStatus('disconnected');
      return;
    }

    let isUnmounted = false;
    let reconnectTimer = null;

    const connect = () => {
      if (isUnmounted) return;
      setConnectionStatus('connecting');

      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!isUnmounted) setConnectionStatus('connected');
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === 'gaze') {
            const x = data.normalized ? data.x * window.innerWidth : data.x;
            const y = data.normalized ? data.y * window.innerHeight : data.y;
            emitGaze({
              x,
              y,
              blinking: Boolean(data.blinking),
              source: 'websocket',
            });
          } else if (data.type === 'camera_state') {
            setCameras(data.cameras || []);
            setActiveCamera(data.activeCamera ?? null);
            setCameraStatus(data.status || 'initializing');
            setCameraError(data.error || null);
            setIsCalibrated(Boolean(data.isCalibrated));
          } else if (data.type === 'calibration_complete') {
            setIsCalibrated(Boolean(data.ok));
          }
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err);
        }
      };

      ws.onerror = () => ws.close();

      ws.onclose = () => {
        if (!isUnmounted) {
          setConnectionStatus('disconnected');
          reconnectTimer = setTimeout(connect, 2000);
        }
      };
    };

    connect();

    return () => {
      isUnmounted = true;
      clearTimeout(reconnectTimer);
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [inputMode, wsUrl, emitGaze]);

  const sendCommand = useCallback((payload) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(payload));
      return true;
    }
    return false;
  }, []);

  const selectCamera = useCallback(
    (index) => sendCommand({ type: 'select_camera', index: Number(index) }),
    [sendCommand]
  );

  const refreshCameras = useCallback(
    () => sendCommand({ type: 'list_cameras' }),
    [sendCommand]
  );

  return {
    inputMode,
    setInputMode,
    connectionStatus,
    cameras,
    activeCamera,
    cameraStatus,
    cameraError,
    isCalibrated,
    selectCamera,
    refreshCameras,
    subscribeGaze,
    sendCommand,
  };
}