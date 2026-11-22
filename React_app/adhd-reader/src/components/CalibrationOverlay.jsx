import React, { useEffect, useState, useCallback, useRef } from 'react';

const HOLD_CAPTURE_MS = 650;

const IDEAL_READER_GRID = [
  { rx: 0.10, ry: 0.10 }, { rx: 0.50, ry: 0.10 }, { rx: 0.90, ry: 0.10 },
  { rx: 0.30, ry: 0.30 }, { rx: 0.70, ry: 0.30 },
  { rx: 0.10, ry: 0.50 }, { rx: 0.50, ry: 0.50 }, { rx: 0.90, ry: 0.50 },
  { rx: 0.30, ry: 0.70 }, { rx: 0.70, ry: 0.70 },
  { rx: 0.10, ry: 0.90 }, { rx: 0.50, ry: 0.90 }, { rx: 0.90, ry: 0.90 },
];

function buildWordAnchoredTargets() {
  const pane = document.querySelector('.reader-pane');
  const paneRect = pane
    ? pane.getBoundingClientRect()
    : {
        left: 40,
        top: 40,
        width: window.innerWidth * 0.68,
        height: window.innerHeight - 80,
        right: window.innerWidth * 0.7,
        bottom: window.innerHeight - 40,
      };

  const visibleWords = [];
  if (pane) {
    const nodes = pane.querySelectorAll('[data-word-index]');
    for (let i = 0; i < nodes.length; i++) {
      const el = nodes[i];
      const r = el.getBoundingClientRect();
      if (
        r.width > 0 &&
        r.height > 0 &&
        r.top >= paneRect.top + 16 &&
        r.bottom <= paneRect.bottom - 16 &&
        r.left >= paneRect.left + 12 &&
        r.right <= paneRect.right - 12
      ) {
        visibleWords.push({
          index: Number(el.dataset.wordIndex),
          word: el.textContent?.trim() || '',
          cx: (r.left + r.right) / 2,
          cy: (r.top + r.bottom) / 2,
        });
      }
    }
  }

  const usedIndices = new Set();
  const maxSnapDist = Math.min(paneRect.width, paneRect.height) * 0.24;

  return IDEAL_READER_GRID.map(({ rx, ry }) => {
    const idealPx = paneRect.left + paneRect.width * rx;
    const idealPy = paneRect.top + paneRect.height * ry;

    let bestWord = null;
    let bestDist = Infinity;

    for (let i = 0; i < visibleWords.length; i++) {
      const w = visibleWords[i];
      if (usedIndices.has(w.index)) continue;
      const dist = Math.hypot(w.cx - idealPx, w.cy - idealPy);
      if (dist < bestDist && dist <= maxSnapDist) {
        bestDist = dist;
        bestWord = w;
      }
    }

    if (bestWord) {
      usedIndices.add(bestWord.index);
      return {
        px: bestWord.cx,
        py: bestWord.cy,
        x: bestWord.cx / window.innerWidth,
        y: bestWord.cy / window.innerHeight,
        word: bestWord.word,
      };
    }

    return {
      px: idealPx,
      py: idealPy,
      x: idealPx / window.innerWidth,
      y: idealPy / window.innerHeight,
      word: null,
    };
  });
}

export default function CalibrationOverlay({
  onComplete,
  onCancel,
  sendCommand,
  subscribeCalibrationAck,
  inputMode,
}) {
  const [targets] = useState(() => buildWordAnchoredTargets());
  const [step, setStep] = useState(0);
  const [isSampling, setIsSampling] = useState(false);
  const [retryMessage, setRetryMessage] = useState('');

  const samplingTimerRef = useRef(null);
  const fallbackTimerRef = useRef(null);
  const stepRef = useRef(0);
  stepRef.current = step;

  useEffect(() => {
    sendCommand({ type: 'calibrate_start' });
    return () => {
      clearTimeout(samplingTimerRef.current);
      clearTimeout(fallbackTimerRef.current);
    };
  }, [sendCommand]);

  const advanceOrFinish = useCallback(() => {
    setIsSampling(false);
    setRetryMessage('');

    if (stepRef.current + 1 >= targets.length) {
      sendCommand({ type: 'calibrate_finish' });
      onComplete();
    } else {
      setStep((prev) => prev + 1);
    }
  }, [targets.length, sendCommand, onComplete]);

  useEffect(() => {
    if (!subscribeCalibrationAck) return;

    return subscribeCalibrationAck((ack) => {
      if (ack.type !== 'calibration_point_ack') return;
      if (ack.index !== stepRef.current) return;
      clearTimeout(fallbackTimerRef.current);

      if (ack.ok) {
        advanceOrFinish();
      } else {
        setIsSampling(false);
        setRetryMessage(
          'Face not detected clearly — look steadily at the red dot and press SPACE again.'
        );
      }
    });
  }, [subscribeCalibrationAck, advanceOrFinish]);

  const triggerSample = useCallback(() => {
    if (isSampling) return;

    const pt = targets[stepRef.current];
    if (!pt) return;

    setIsSampling(true);
    setRetryMessage('');

    sendCommand({
      type: 'calibrate_point_begin',
      index: stepRef.current,
      targetX: pt.x,
      targetY: pt.y,
    });

    samplingTimerRef.current = setTimeout(() => {
      const sent = sendCommand({
        type: 'calibrate_point_end',
        index: stepRef.current,
      });

      if (inputMode !== 'websocket' || !sent) {
        advanceOrFinish();
      } else {
        fallbackTimerRef.current = setTimeout(() => {
          advanceOrFinish();
        }, 1000);
      }
    }, HOLD_CAPTURE_MS);
  }, [isSampling, targets, sendCommand, inputMode, advanceOrFinish]);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.code === 'Space') {
        e.preventDefault();
        triggerSample();
      } else if (e.code === 'Escape') {
        onCancel();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [triggerSample, onCancel]);

  const currentPoint = targets[step];
  if (!currentPoint) return null;

  return (
    <div className="calibration-overlay in-page">
      <div className="calibration-side-panel">
        <h2>
          Calibration ({step + 1} / {targets.length})
        </h2>
        <p>
          {isSampling
            ? 'Capturing — keep looking steadily at the dot...'
            : 'Look at the red dot over the text and press SPACE when you are looking at it.'}
        </p>

        {currentPoint.word && (
          <div className="calibration-word-pill">
            Target word: <strong>&ldquo;{currentPoint.word}&rdquo;</strong>
          </div>
        )}

        {retryMessage && <p className="calibration-error">{retryMessage}</p>}

        <button type="button" className="btn-secondary full-width" onClick={onCancel}>
          Cancel (Esc)
        </button>
      </div>

      <button
        type="button"
        className={`calibration-dot word-dot ${isSampling ? 'sampling' : ''}`}
        style={{
          left: `${currentPoint.px}px`,
          top: `${currentPoint.py}px`,
          '--capture-ms': `${HOLD_CAPTURE_MS}ms`,
        }}
        onClick={triggerSample}
        aria-label={`Calibration target ${step + 1}`}
      />
    </div>
  );
}