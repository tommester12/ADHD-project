import React, { useEffect, useRef, useState } from 'react';
import { playStrayChime } from '../utils/soundAlert';

const INDIVIDUAL_EFFECTS = [
  {
    key: 'zoom',
    label: 'Zoom (Enlarge)',
    description: 'Anchors active word & parts neighbors smoothly without line wrapping.',
    hasColor: false,
  },
  {
    key: 'bold',
    label: 'Bold',
    description: 'Subtle zero-shift bold emphasis on the active word.',
    hasColor: false,
  },
  {
    key: 'textColor',
    label: 'Text Color',
    description: 'Changes word color (graded shades on next words).',
    hasColor: true,
  },
  {
    key: 'bgHighlight',
    label: 'Background Highlight',
    description: 'Soft highlighter fill behind active words.',
    hasColor: true,
  },
  {
    key: 'underline',
    label: 'Underline',
    description: 'Crisp bottom line under active words.',
    hasColor: true,
  },
  {
    key: 'dimSurroundings',
    label: 'Dim Surroundings',
    description: 'Softly fades non-active words to reduce clutter.',
    hasColor: false,
  },
];

export default function ControlPanel({
  effects = {},
  onToggleEffect,
  zoomScale = 1.15,
  setZoomScale,
  dimPreviousOnly = false,
  setDimPreviousOnly,
  effectColors = {},
  onChangeEffectColor,
  previewNextWords = true,
  setPreviewNextWords,
  fontSize,
  setFontSize,
  fontFamilyId,
  setFontFamilyId,
  fontFamilies = [],
  fontTypeId,
  setFontTypeId,
  fontTypes = [],
  gazeTolerance,
  setGazeTolerance,
  showGazeDot,
  setShowGazeDot,
  guidedMode = false,
  onToggleGuidedMode,
  dynamicGuidance = false,
  setDynamicGuidance,
  dynamicJumpSeconds = 3,
  setDynamicJumpSeconds,
  onResetGuided,
  guidedProgress = { current: 0, total: 0 },
  readFade = 78,
  setReadFade,
  autoPace = false,
  setAutoPace,
  autoPacePaused = false,
  setAutoPacePaused,
  autoPaceWpm = 180,
  setAutoPaceWpm,
  strayAlertEnabled = false,
  setStrayAlertEnabled,
  strayDelayMs = 800,
  setStrayDelayMs,
  strayVolume = 0.5,
  setStrayVolume,
  inputMode,
  setInputMode,
  connectionStatus,
  cameras = [],
  activeCamera,
  cameraStatus,
  cameraError,
  isCalibrated,
  onSelectCamera,
  onRefreshCameras,
  subscribePreview,
  setPreviewEnabled,
  currentWordInfo = { index: null, word: null },
  onLoadText,
  onStartCalibration,
}) {
  const fileInputRef = useRef(null);
  const previewImgRef = useRef(null);

  const [showPasteBox, setShowPasteBox] = useState(false);
  const [pasteValue, setPasteValue] = useState('');
  const [showPreview, setShowPreview] = useState(true);
  const [previewInfo, setPreviewInfo] = useState({
    hasFrame: false,
    faceDetected: false,
    blinking: false,
  });

  // Notify Python when toggling live preview on/off
  useEffect(() => {
    if (inputMode === 'websocket' && connectionStatus === 'connected' && setPreviewEnabled) {
      setPreviewEnabled(showPreview);
    }
  }, [inputMode, connectionStatus, showPreview, setPreviewEnabled]);

  // Clear preview placeholder while switching cameras or disconnected
  useEffect(() => {
    if (connectionStatus !== 'connected' || cameraStatus === 'switching') {
      setPreviewInfo({ hasFrame: false, faceDetected: false, blinking: false });
    }
  }, [connectionStatus, cameraStatus]);

  // Update <img> src directly via DOM ref (~12 FPS, zero parent re-renders)
  useEffect(() => {
    if (!subscribePreview || !showPreview || inputMode !== 'websocket') return;

    return subscribePreview(({ image, faceDetected, blinking }) => {
      if (previewImgRef.current && image) {
        previewImgRef.current.src = `data:image/jpeg;base64,${image}`;
      }
      setPreviewInfo((prev) => {
        if (
          prev.hasFrame &&
          prev.faceDetected === Boolean(faceDetected) &&
          prev.blinking === Boolean(blinking)
        ) {
          return prev;
        }
        return {
          hasFrame: true,
          faceDetected: Boolean(faceDetected),
          blinking: Boolean(blinking),
        };
      });
    });
  }, [subscribePreview, showPreview, inputMode]);

  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      if (typeof event.target?.result === 'string') {
        onLoadText(event.target.result);
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  };

  const handleApplyPaste = () => {
    if (pasteValue.trim()) {
      onLoadText(pasteValue);
      setPasteValue('');
      setShowPasteBox(false);
    }
  };

  const progressPercent =
    guidedProgress.total > 0
      ? Math.round((guidedProgress.current / guidedProgress.total) * 100)
      : 0;

  return (
    <aside className="control-pane">
      <header className="panel-header">
        <h1>FocusGaze Reader</h1>
        <p className="subtitle">ADHD Gaze-Guided Reading Assistant</p>
      </header>

      {/* Tracking Source & Camera Selector */}
      <section className="panel-section">
        <h2>Tracking Source</h2>
        <div className="segmented-control">
          <button
            type="button"
            className={inputMode === 'mouse' ? 'active' : ''}
            onClick={() => setInputMode('mouse')}
          >
            Mouse Test
          </button>
          <button
            type="button"
            className={inputMode === 'websocket' ? 'active' : ''}
            onClick={() => setInputMode('websocket')}
          >
            Eye Tracker
          </button>
        </div>

        {inputMode === 'websocket' && (
          <div className={`status-badge status-${connectionStatus}`}>
            <span className="status-dot" />
            Python Bridge: {connectionStatus}
          </div>
        )}

        {inputMode === 'websocket' && connectionStatus === 'connected' && (
          <div className="camera-picker">
            <div className="camera-picker-header">
              <label htmlFor="camera-select">Camera Device</label>
              <button type="button" className="btn-link" onClick={onRefreshCameras}>
                Refresh
              </button>
            </div>

            {cameras.length > 0 ? (
              <select
                id="camera-select"
                className="camera-select"
                value={activeCamera ?? ''}
                disabled={cameraStatus === 'switching'}
                onChange={(e) => onSelectCamera(e.target.value)}
              >
                {activeCamera === null && <option value="">Select a camera...</option>}
                {cameras.map((cam) => (
                  <option key={cam.index} value={cam.index}>
                    {cam.name}
                  </option>
                ))}
              </select>
            ) : (
              <div className="camera-alert warning">
                No cameras detected. Plug in a webcam and click Refresh.
              </div>
            )}

            {/* Live Camera Preview */}
            {cameras.length > 0 && (
              <div className="camera-preview-section">
                <div className="camera-picker-header">
                  <span>Camera Preview</span>
                  <button
                    type="button"
                    className="btn-link"
                    onClick={() => setShowPreview((v) => !v)}
                  >
                    {showPreview ? 'Hide' : 'Show'}
                  </button>
                </div>

                {showPreview && (
                  <div className="camera-preview-box">
                    <img
                      ref={previewImgRef}
                      alt="Selected camera preview"
                      className="camera-preview-img"
                      style={{ display: previewInfo.hasFrame ? 'block' : 'none' }}
                    />

                    {!previewInfo.hasFrame && (
                      <div className="camera-preview-placeholder">
                        Waiting for camera video...
                      </div>
                    )}

                    {previewInfo.hasFrame && (
                      <span
                        className={`preview-status-pill ${
                          !previewInfo.faceDetected
                            ? 'bad'
                            : previewInfo.blinking
                            ? 'warn'
                            : 'good'
                        }`}
                      >
                        {!previewInfo.faceDetected
                          ? 'No Face Detected'
                          : previewInfo.blinking
                          ? 'Blinking'
                          : 'Face & Eyes Locked'}
                      </span>
                    )}
                  </div>
                )}
              </div>
            )}

            {cameraStatus === 'switching' && (
              <div className="camera-alert info">Switching camera...</div>
            )}

            {cameraError && <div className="camera-alert error">{cameraError}</div>}

            {cameraStatus === 'active' && !isCalibrated && (
              <div className="camera-alert info">
                Camera ready — run calibration to start gaze tracking.
              </div>
            )}
          </div>
        )}

        <button
          type="button"
          className="btn-primary full-width"
          onClick={onStartCalibration}
        >
          Calibrate Eye Tracking (9-Point)
        </button>
      </section>

      {/* Load Text */}
      <section className="panel-section">
        <h2>Load Text</h2>
        <div className="button-row">
          <button
            type="button"
            className="btn-secondary"
            onClick={() => fileInputRef.current?.click()}
          >
            Upload .txt File
          </button>
          <button
            type="button"
            className="btn-secondary"
            onClick={() => setShowPasteBox((v) => !v)}
          >
            {showPasteBox ? 'Close Editor' : 'Paste Text'}
          </button>
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept=".txt,.md"
          style={{ display: 'none' }}
          onChange={handleFileUpload}
        />

        {showPasteBox && (
          <div className="paste-container">
            <textarea
              rows={5}
              placeholder="Paste your article or study notes here..."
              value={pasteValue}
              onChange={(e) => setPasteValue(e.target.value)}
            />
            <button
              type="button"
              className="btn-primary full-width"
              onClick={handleApplyPaste}
            >
              Load Into Reader
            </button>
          </div>
        )}
      </section>

      {/* Guided Reading Mode */}
      <section className="panel-section">
        <h2>Guided Reading</h2>

        <label className="checkbox-row">
          <input
            type="checkbox"
            checked={guidedMode}
            onChange={(e) => onToggleGuidedMode(e.target.checked)}
          />
          <span>Step-by-step guided reading</span>
        </label>

        {guidedMode && (
          <div className="guided-controls">
            <div className="slider-label">
              <span>
                {guidedProgress.total > 0 && guidedProgress.current >= guidedProgress.total
                  ? 'Finished reading!'
                  : `Progress: ${guidedProgress.current} / ${guidedProgress.total} words`}
              </span>
              <span>{progressPercent}%</span>
            </div>

            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${progressPercent}%` }} />
            </div>

            <div className="slider-group">
              <div className="slider-label">
                <span>Read Word Fade</span>
                <span>{readFade}%</span>
              </div>
              <input
                type="range"
                min={0}
                max={100}
                value={readFade}
                onChange={(e) => setReadFade(Number(e.target.value))}
              />
            </div>

            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={dynamicGuidance}
                onChange={(e) => {
                  setDynamicGuidance(e.target.checked);
                  if (e.target.checked) playStrayChime(strayVolume);
                }}
              />
              <span>
                Dynamic guidance (1.5s back alert, {dynamicJumpSeconds}s jump)
              </span>
            </label>

            {dynamicGuidance && (
              <div className="slider-group">
                <div className="slider-label">
                  <span>Jump Hold Time</span>
                  <span>{dynamicJumpSeconds}s</span>
                </div>
                <input
                  type="range"
                  min={1}
                  max={6}
                  step={0.5}
                  value={dynamicJumpSeconds}
                  onChange={(e) => setDynamicJumpSeconds(Number(e.target.value))}
                />
              </div>
            )}

            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={autoPace}
                onChange={(e) => {
                  setAutoPace(e.target.checked);
                  setAutoPacePaused(false);
                }}
              />
              <span>Auto-move highlight at fixed speed</span>
            </label>

            {autoPace && (
              <div className="slider-group">
                <div className="slider-label">
                  <span>Auto-Pace Speed</span>
                  <span>{autoPaceWpm} WPM</span>
                </div>
                <input
                  type="range"
                  min={60}
                  max={450}
                  step={10}
                  value={autoPaceWpm}
                  onChange={(e) => setAutoPaceWpm(Number(e.target.value))}
                />
              </div>
            )}

            <div className="button-row">
              {autoPace && (
                <button
                  type="button"
                  className="btn-primary"
                  onClick={() => setAutoPacePaused((p) => !p)}
                >
                  {autoPacePaused ? 'Resume' : 'Pause'}
                </button>
              )}
              <button
                type="button"
                className="btn-secondary"
                onClick={onResetGuided}
              >
                Reset
              </button>
            </div>
          </div>
        )}
      </section>

      {/* Composable Reading Effects */}
      <section className="panel-section">
        <h2>Reading Effects (Mix &amp; Match)</h2>

        <div className="effect-toggle-list">
          {INDIVIDUAL_EFFECTS.map((item) => {
            const isChecked = Boolean(effects[item.key]);
            return (
              <div
                key={item.key}
                className={`effect-toggle-card ${isChecked ? 'selected' : ''}`}
              >
                <div className="effect-toggle-row">
                  <label className="effect-toggle-label">
                    <input
                      type="checkbox"
                      checked={isChecked}
                      onChange={() => onToggleEffect(item.key)}
                    />
                    <div>
                      <strong>{item.label}</strong>
                      <span>{item.description}</span>
                    </div>
                  </label>

                  {item.hasColor && (
                    <label
                      className={`inline-color-chip ${!isChecked ? 'muted' : ''}`}
                      title={`Choose ${item.label} color`}
                      style={{ backgroundColor: effectColors[item.key] || '#2563eb' }}
                    >
                      <input
                        type="color"
                        value={effectColors[item.key] || '#2563eb'}
                        onChange={(e) => onChangeEffectColor(item.key, e.target.value)}
                      />
                    </label>
                  )}
                </div>

                {item.key === 'zoom' && isChecked && (
                  <div className="effect-subcontrol">
                    <div className="slider-label">
                      <span>Zoom Strength</span>
                      <span>{Math.round(zoomScale * 100)}%</span>
                    </div>
                    <input
                      type="range"
                      min={1.05}
                      max={1.35}
                      step={0.01}
                      value={zoomScale}
                      onChange={(e) => setZoomScale(Number(e.target.value))}
                    />
                  </div>
                )}

                {item.key === 'dimSurroundings' && isChecked && (
                  <div className="effect-subcontrol">
                    <label className="checkbox-row">
                      <input
                        type="checkbox"
                        checked={dimPreviousOnly}
                        onChange={(e) => setDimPreviousOnly(e.target.checked)}
                      />
                      <span>Only dim previous words</span>
                    </label>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        <label className="checkbox-row">
          <input
            type="checkbox"
            checked={previewNextWords}
            onChange={(e) => setPreviewNextWords(e.target.checked)}
          />
          <span>Apply lesser effect to next 2 words</span>
        </label>
      </section>

      {/* Typography Controls */}
      <section className="panel-section">
        <h2>Typography</h2>

        <div className="control-field">
          <label htmlFor="font-family-select">Font Family</label>
          <select
            id="font-family-select"
            className="camera-select"
            value={fontFamilyId}
            onChange={(e) => setFontFamilyId(e.target.value)}
          >
            {fontFamilies.map((f) => (
              <option key={f.id} value={f.id}>
                {f.label}
              </option>
            ))}
          </select>
        </div>

        <div className="control-field">
          <label htmlFor="font-type-select">Font Type</label>
          <select
            id="font-type-select"
            className="camera-select"
            value={fontTypeId}
            onChange={(e) => setFontTypeId(e.target.value)}
          >
            {fontTypes.map((t) => (
              <option key={t.id} value={t.id}>
                {t.label}
              </option>
            ))}
          </select>
        </div>

        <div className="slider-group">
          <div className="slider-label">
            <span>Font Size</span>
            <div className="font-size-stepper">
              <button
                type="button"
                className="step-btn"
                onClick={() => setFontSize((s) => Math.max(14, s - 1))}
              >
                A-
              </button>
              <span>{fontSize}px</span>
              <button
                type="button"
                className="step-btn"
                onClick={() => setFontSize((s) => Math.min(40, s + 1))}
              >
                A+
              </button>
            </div>
          </div>
          <input
            type="range"
            min={14}
            max={40}
            value={fontSize}
            onChange={(e) => setFontSize(Number(e.target.value))}
          />
        </div>
      </section>

      {/* Precision & Sound Alerts */}
      <section className="panel-section">
        <h2>Gaze Precision &amp; Alerts</h2>

        <div className="slider-group">
          <div className="slider-label">
            <span>Gaze Snap Tolerance</span>
            <span>{gazeTolerance}px</span>
          </div>
          <input
            type="range"
            min={0}
            max={100}
            value={gazeTolerance}
            onChange={(e) => setGazeTolerance(Number(e.target.value))}
          />
        </div>

        <label className="checkbox-row">
          <input
            type="checkbox"
            checked={showGazeDot}
            onChange={(e) => setShowGazeDot(e.target.checked)}
          />
          <span>Show live gaze indicator dot</span>
        </label>

        <label className="checkbox-row">
          <input
            type="checkbox"
            checked={strayAlertEnabled}
            onChange={(e) => {
              setStrayAlertEnabled(e.target.checked);
              if (e.target.checked) playStrayChime(strayVolume);
            }}
          />
          <span>Play sound if gaze strays from current/next word</span>
        </label>

        {strayAlertEnabled && (
          <div className="guided-controls">
            <div className="slider-group">
              <div className="slider-label">
                <span>Grace Delay Before Sound</span>
                <span>{(strayDelayMs / 1000).toFixed(1)}s</span>
              </div>
              <input
                type="range"
                min={300}
                max={2500}
                step={100}
                value={strayDelayMs}
                onChange={(e) => setStrayDelayMs(Number(e.target.value))}
              />
            </div>

            <div className="slider-group">
              <div className="slider-label">
                <span>Alert Volume</span>
                <span>{Math.round(strayVolume * 100)}%</span>
              </div>
              <input
                type="range"
                min={0.1}
                max={1}
                step={0.05}
                value={strayVolume}
                onChange={(e) => setStrayVolume(Number(e.target.value))}
              />
            </div>

            <button
              type="button"
              className="btn-secondary full-width"
              onClick={() => playStrayChime(strayVolume)}
            >
              Test Sound
            </button>
          </div>
        )}
      </section>

      <footer className="panel-footer">
        <div className="debug-pill">
          <span>Active Word:</span>
          <strong>
            {currentWordInfo.word
              ? `"${currentWordInfo.word}" (#${currentWordInfo.index})`
              : 'None'}
          </strong>
        </div>
      </footer>
    </aside>
  );
}