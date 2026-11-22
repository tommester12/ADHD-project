import React, { useState, useCallback } from 'react';
import TextReader from './components/TextReader';
import ControlPanel from './components/ControlPanel';
import CalibrationOverlay from './components/CalibrationOverlay';
import { useGazeTracker } from './hooks/useGazeTracker';
import './App.css';

export const FONT_FAMILIES = [
  {
    id: 'system-sans',
    label: 'System Sans (Clean)',
    css: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
  },
  {
    id: 'verdana',
    label: 'Verdana (Wide & ADHD-Friendly)',
    css: 'Verdana, Geneva, sans-serif',
  },
  {
    id: 'atkinson',
    label: 'Trebuchet / Humanist Sans',
    css: "'Atkinson Hyperlegible', 'Trebuchet MS', Helvetica, sans-serif",
  },
  {
    id: 'georgia',
    label: 'Georgia (Serif)',
    css: "Georgia, 'Times New Roman', serif",
  },
  {
    id: 'dyslexic',
    label: 'OpenDyslexic / Comic Sans',
    css: "'OpenDyslexic', 'Comic Sans MS', cursive, sans-serif",
  },
  {
    id: 'monospace',
    label: 'Monospace (Fixed Width)',
    css: "Consolas, Menlo, 'Courier New', monospace",
  },
];

export const FONT_TYPES = [
  { id: 'normal', label: 'Regular', weight: 400, style: 'normal' },
  { id: 'medium', label: 'Medium', weight: 500, style: 'normal' },
  { id: 'semibold', label: 'Semi-Bold', weight: 600, style: 'normal' },
  { id: 'bold', label: 'Bold', weight: 700, style: 'normal' },
  { id: 'italic', label: 'Italic', weight: 400, style: 'italic' },
  { id: 'bold-italic', label: 'Bold Italic', weight: 700, style: 'italic' },
];

const DEFAULT_TEXT = `Reading with ADHD often feels like trying to listen to a quiet conversation in a crowded room. Even when the topic is interesting, your eyes can jump ahead to the bottom of the paragraph, skip an entire line, or lose track of which sentence you were just processing.

Gaze-guided typography helps anchor visual attention directly to where you are reading. By gently enlarging the word you are currently looking at—and slightly previewing the next two words—your visual system gets a clear, low-friction anchor to follow across the page.

Try moving your mouse across this passage right now in Mouse Test mode, or switch to Eye Tracker mode once your Python WebSocket bridge is running. You can also upload your own study material or paste any article from the panel on the right.`;

export default function App() {
  const [text, setText] = useState(DEFAULT_TEXT);

  // Composable individual effects
  const [effects, setEffects] = useState({
    zoom: true,
    bold: false,
    textColor: false,
    bgHighlight: false,
    underline: false,
    dimSurroundings: false,
  });

  const [zoomScale, setZoomScale] = useState(1.15);
  const [dimPreviousOnly, setDimPreviousOnly] = useState(false);

  const [effectColors, setEffectColors] = useState({
    textColor: '#2563eb',
    bgHighlight: '#fde047',
    underline: '#2563eb',
  });

  const [previewNextWords, setPreviewNextWords] = useState(true);

  // Typography settings
  const [fontSize, setFontSize] = useState(22);
  const [fontFamilyId, setFontFamilyId] = useState('system-sans');
  const [fontTypeId, setFontTypeId] = useState('normal');

  // Guided reading settings
  const [guidedMode, setGuidedMode] = useState(false);
  const [lineLocked, setLineLocked] = useState(false);
  const [dynamicGuidance, setDynamicGuidance] = useState(false);
  const [dynamicJumpSeconds, setDynamicJumpSeconds] = useState(3);
  const [guidedResetKey, setGuidedResetKey] = useState(0);
  const [guidedProgress, setGuidedProgress] = useState({ current: 0, total: 0 });
  const [readFade, setReadFade] = useState(78);
  const [autoPace, setAutoPace] = useState(false);
  const [autoPacePaused, setAutoPacePaused] = useState(false);
  const [autoPaceWpm, setAutoPaceWpm] = useState(180);

  // Gaze precision & stray sound alert settings
  const [gazeTolerance, setGazeTolerance] = useState(32);
  const [showGazeDot, setShowGazeDot] = useState(true);
  const [strayAlertEnabled, setStrayAlertEnabled] = useState(false);
  const [strayDelayMs, setStrayDelayMs] = useState(800);
  const [strayVolume, setStrayVolume] = useState(0.5);

  const [isCalibrating, setIsCalibrating] = useState(false);
  const [currentWordInfo, setCurrentWordInfo] = useState({ index: null, word: null });

  const selectedFontFamily =
    FONT_FAMILIES.find((f) => f.id === fontFamilyId) || FONT_FAMILIES[0];
  const selectedFontType =
    FONT_TYPES.find((t) => t.id === fontTypeId) || FONT_TYPES[0];

  const {
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
    subscribePreview,
    subscribeCalibrationAck,
    setPreviewEnabled,
    sendCommand,
  } = useGazeTracker({ wsUrl: 'ws://localhost:8765' });

  const handleToggleEffect = useCallback((key) => {
    setEffects((prev) => ({ ...prev, [key]: !prev[key] }));
  }, []);

  const handleChangeEffectColor = useCallback((key, hex) => {
    setEffectColors((prev) => ({ ...prev, [key]: hex }));
  }, []);

  const handleToggleGuidedMode = useCallback((enabled) => {
    setGuidedMode(enabled);
    setAutoPacePaused(false);
    if (enabled) {
      setGuidedResetKey((k) => k + 1);
    }
  }, []);

  const handleToggleDynamicGuidance = useCallback((enabled) => {
    setDynamicGuidance(enabled);
    if (enabled) setLineLocked(false);
  }, []);

  const handleToggleLineLocked = useCallback((enabled) => {
    setLineLocked(enabled);
    if (enabled) setDynamicGuidance(false);
  }, []);

  const handleResetGuided = useCallback(() => {
    setAutoPacePaused(false);
    setGuidedResetKey((k) => k + 1);
  }, []);

  const handleGuidedProgress = useCallback((current, total) => {
    setGuidedProgress({ current, total });
  }, []);

  const handleWordChange = useCallback((index, word) => {
    setCurrentWordInfo({ index, word });
  }, []);

  return (
    <div className="app-layout">
      <TextReader
        text={text}
        effects={effects}
        zoomScale={zoomScale}
        dimPreviousOnly={dimPreviousOnly}
        effectColors={effectColors}
        previewNextWords={previewNextWords}
        fontSize={fontSize}
        fontFamily={selectedFontFamily.css}
        fontWeight={selectedFontType.weight}
        fontStyle={selectedFontType.style}
        gazeTolerance={gazeTolerance}
        showGazeDot={showGazeDot}
        guidedMode={guidedMode}
        lineLocked={lineLocked}
        dynamicGuidance={dynamicGuidance}
        dynamicJumpSeconds={dynamicJumpSeconds}
        guidedResetKey={guidedResetKey}
        readFade={readFade}
        autoPace={autoPace}
        autoPacePaused={autoPacePaused}
        autoPaceWpm={autoPaceWpm}
        strayAlertEnabled={strayAlertEnabled}
        strayDelayMs={strayDelayMs}
        strayVolume={strayVolume}
        onGuidedProgress={handleGuidedProgress}
        subscribeGaze={subscribeGaze}
        onWordChange={handleWordChange}
        sendCommand={sendCommand}
        subscribeCalibrationAck={subscribeCalibrationAck}
        inputMode={inputMode}
      />

      <ControlPanel
        effects={effects}
        onToggleEffect={handleToggleEffect}
        zoomScale={zoomScale}
        setZoomScale={setZoomScale}
        dimPreviousOnly={dimPreviousOnly}
        setDimPreviousOnly={setDimPreviousOnly}
        effectColors={effectColors}
        onChangeEffectColor={handleChangeEffectColor}
        previewNextWords={previewNextWords}
        setPreviewNextWords={setPreviewNextWords}
        fontSize={fontSize}
        setFontSize={setFontSize}
        fontFamilyId={fontFamilyId}
        setFontFamilyId={setFontFamilyId}
        fontFamilies={FONT_FAMILIES}
        fontTypeId={fontTypeId}
        setFontTypeId={setFontTypeId}
        fontTypes={FONT_TYPES}
        gazeTolerance={gazeTolerance}
        setGazeTolerance={setGazeTolerance}
        showGazeDot={showGazeDot}
        setShowGazeDot={setShowGazeDot}
        guidedMode={guidedMode}
        onToggleGuidedMode={handleToggleGuidedMode}
        lineLocked={lineLocked}
        setLineLocked={handleToggleLineLocked}
        dynamicGuidance={dynamicGuidance}
        setDynamicGuidance={handleToggleDynamicGuidance}
        dynamicJumpSeconds={dynamicJumpSeconds}
        setDynamicJumpSeconds={setDynamicJumpSeconds}
        onResetGuided={handleResetGuided}
        guidedProgress={guidedProgress}
        readFade={readFade}
        setReadFade={setReadFade}
        autoPace={autoPace}
        setAutoPace={setAutoPace}
        autoPacePaused={autoPacePaused}
        setAutoPacePaused={setAutoPacePaused}
        autoPaceWpm={autoPaceWpm}
        setAutoPaceWpm={setAutoPaceWpm}
        strayAlertEnabled={strayAlertEnabled}
        setStrayAlertEnabled={setStrayAlertEnabled}
        strayDelayMs={strayDelayMs}
        setStrayDelayMs={setStrayDelayMs}
        strayVolume={strayVolume}
        setStrayVolume={setStrayVolume}
        inputMode={inputMode}
        setInputMode={setInputMode}
        connectionStatus={connectionStatus}
        cameras={cameras}
        activeCamera={activeCamera}
        cameraStatus={cameraStatus}
        cameraError={cameraError}
        isCalibrated={isCalibrated}
        onSelectCamera={selectCamera}
        onRefreshCameras={refreshCameras}
        subscribePreview={subscribePreview}
        setPreviewEnabled={setPreviewEnabled}
        currentWordInfo={currentWordInfo}
        onLoadText={setText}
        onStartCalibration={() => setIsCalibrating(true)}
      />

      {isCalibrating && (
        <CalibrationOverlay
          sendCommand={sendCommand}
          subscribeCalibrationAck={subscribeCalibrationAck}
          inputMode={inputMode}
          onComplete={() => setIsCalibrating(false)}
          onCancel={() => setIsCalibrating(false)}
        />
      )}
    </div>
  );
}