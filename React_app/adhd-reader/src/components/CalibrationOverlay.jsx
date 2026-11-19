import React, { useEffect, useState, useCallback } from 'react';

const CALIBRATION_POINTS = [
  { x: 0.08, y: 0.08 }, { x: 0.5, y: 0.08 }, { x: 0.92, y: 0.08 },
  { x: 0.08, y: 0.50 }, { x: 0.5, y: 0.50 }, { x: 0.92, y: 0.50 },
  { x: 0.08, y: 0.92 }, { x: 0.5, y: 0.92 }, { x: 0.92, y: 0.92 },
];

export default function CalibrationOverlay({ onComplete, onCancel, sendCommand }) {
  const [step, setStep] = useState(0);

  useEffect(() => {
    sendCommand({ type: 'calibrate_start' });
  }, [sendCommand]);

  const recordCurrentPoint = useCallback(() => {
    const pt = CALIBRATION_POINTS[step];
    sendCommand({
      type: 'calibrate_point',
      index: step,
      targetX: pt.x,
      targetY: pt.y,
    });

    if (step + 1 >= CALIBRATION_POINTS.length) {
      sendCommand({ type: 'calibrate_finish' });
      onComplete();
    } else {
      setStep((prev) => prev + 1);
    }
  }, [step, sendCommand, onComplete]);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.code === 'Space') {
        e.preventDefault();
        recordCurrentPoint();
      } else if (e.code === 'Escape') {
        onCancel();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [recordCurrentPoint, onCancel]);

  const currentPoint = CALIBRATION_POINTS[step];

  return (
    <div className="calibration-overlay">
      <div className="calibration-instructions">
        <h2>Eye Tracker Calibration ({step + 1} / {CALIBRATION_POINTS.length})</h2>
        <p>
          Look directly at the red dot and press <strong>SPACE</strong> (or click the dot).
        </p>
        <button className="btn-secondary" onClick={onCancel}>
          Cancel (Esc)
        </button>
      </div>

      <button
        className="calibration-dot"
        style={{
          left: `${currentPoint.x * 100}%`,
          top: `${currentPoint.y * 100}%`,
        }}
        onClick={recordCurrentPoint}
        aria-label={`Calibration target ${step + 1}`}
      />
    </div>
  );
}