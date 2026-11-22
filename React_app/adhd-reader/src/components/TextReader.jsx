import React, { useEffect, useLayoutEffect, useMemo, useRef, useState, useCallback } from 'react';
import { playStrayChime } from '../utils/soundAlert';

const MIN_GUIDED_DWELL_MS = 80;
const STRAY_COOLDOWN_MS = 2000;
const DYNAMIC_BACKWARD_ALERT_MS = 1500;
const QUICK_RECAL_MS = 650;

function hexToRgb(hex) {
  const clean = (hex || '#2563eb').replace('#', '');
  const full =
    clean.length === 3
      ? clean
          .split('')
          .map((c) => c + c)
          .join('')
      : clean;
  const num = parseInt(full, 16);
  if (Number.isNaN(num)) return [37, 99, 235];
  return [(num >> 16) & 255, (num >> 8) & 255, num & 255];
}

function getColorShades(hex) {
  const [r, g, b] = hexToRgb(hex);
  const mix = (t) =>
    `rgb(${Math.round(r + (251 - r) * t)}, ${Math.round(g + (250 - g) * t)}, ${Math.round(
      b + (246 - b) * t
    )})`;

  return {
    s1: `rgb(${r}, ${g}, ${b})`,
    s2: mix(0.36),
    s3: mix(0.60),
  };
}

function getHighlightTints(hex) {
  const [r, g, b] = hexToRgb(hex);
  const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  const a1 = luminance > 0.65 ? 0.68 : 0.26;
  const a2 = luminance > 0.65 ? 0.36 : 0.15;
  const a3 = luminance > 0.65 ? 0.18 : 0.08;

  return {
    bg1: `rgba(${r}, ${g}, ${b}, ${a1})`,
    bg2: `rgba(${r}, ${g}, ${b}, ${a2})`,
    bg3: `rgba(${r}, ${g}, ${b}, ${a3})`,
  };
}

function computeLinePartingShifts(boxes, targetIndex, zoom1, zoom2, zoom3, previewNextWords) {
  const shifts = {};
  if (targetIndex === null || !boxes.length) return shifts;

  const scaleByIndex = new Map();
  scaleByIndex.set(targetIndex, zoom1);
  if (previewNextWords) {
    if (targetIndex + 1 < boxes.length) scaleByIndex.set(targetIndex + 1, zoom2);
    if (targetIndex + 2 < boxes.length) scaleByIndex.set(targetIndex + 2, zoom3);
  }

  const affectedLines = [];
  for (const [idx] of scaleByIndex) {
    const box = boxes[idx];
    if (!box) continue;
    const lineHeight = Math.max(12, box.rawBottom - box.rawTop);
    const alreadyTracked = affectedLines.some(
      (line) => Math.abs(line.cy - box.cy) < lineHeight * 0.6
    );
    if (!alreadyTracked) {
      affectedLines.push({ anchorIdx: idx, cy: box.cy, lineHeight });
    }
  }

  for (const line of affectedLines) {
    const lineWords = [];
    for (let i = 0; i < boxes.length; i++) {
      const b = boxes[i];
      if (Math.abs(b.cy - line.cy) < line.lineHeight * 0.6) {
        lineWords.push(b);
      }
    }

    const anchorPos = lineWords.findIndex((b) => b.index === line.anchorIdx);
    if (anchorPos === -1) continue;

    const getHalfExpansion = (box) => {
      const s = scaleByIndex.get(box.index) || 1;
      if (s <= 1) return 0;
      const width = box.rawRight - box.rawLeft;
      return (width * (s - 1)) / 2 + 2;
    };

    const anchorHalf = getHalfExpansion(lineWords[anchorPos]);

    for (let p = 0; p < anchorPos; p++) {
      shifts[lineWords[p].index] = -Number(anchorHalf.toFixed(1));
    }

    shifts[lineWords[anchorPos].index] = 0;
    let cumRight = anchorHalf;

    for (let p = anchorPos + 1; p < lineWords.length; p++) {
      const b = lineWords[p];
      const half = getHalfExpansion(b);
      shifts[b.index] = Number((cumRight + half).toFixed(1));
      cumRight += 2 * half;
    }
  }

  return shifts;
}

const Word = React.memo(function Word({
  word,
  index,
  focusLevel,
  isRead,
  isBeforeCurrent,
  isJumpCharging,
  shiftX,
}) {
  return (
    <span className="word-wrapper" data-word-index={index}>
      <span
        className={`word-inner focus-${focusLevel} ${isRead ? 'word-read' : ''} ${
          isBeforeCurrent ? 'word-prev' : ''
        }`}
        style={shiftX ? { '--shift-x': `${shiftX}px` } : undefined}
      >
        {word}
        {isJumpCharging && <span className="word-jump-bar" />}
      </span>
    </span>
  );
});

function findWordAtPoint(x, y, boxes, tolerance) {
  let expandedMatchIndex = null;
  let expandedBestCenterDist = Infinity;

  let nearestIndex = null;
  let nearestDist = Infinity;

  for (let i = 0; i < boxes.length; i++) {
    const b = boxes[i];

    if (x >= b.rawLeft && x <= b.rawRight && y >= b.rawTop && y <= b.rawBottom) {
      return b.index;
    }

    if (x >= b.left && x <= b.right && y >= b.top && y <= b.bottom) {
      const centerDist = Math.hypot(x - b.cx, (y - b.cy) * 1.4);
      if (centerDist < expandedBestCenterDist) {
        expandedBestCenterDist = centerDist;
        expandedMatchIndex = b.index;
      }
    }

    const dx = Math.max(b.left - x, 0, x - b.right);
    const dy = Math.max(b.top - y, 0, y - b.bottom);
    const weightedDist = Math.hypot(dx, dy * 1.4);

    if (weightedDist < nearestDist && weightedDist <= tolerance) {
      nearestDist = weightedDist;
      nearestIndex = b.index;
    }
  }

  return expandedMatchIndex !== null ? expandedMatchIndex : nearestIndex;
}

function findWordOnLineByX(x, boxes, targetIndex, tolerance) {
  const targetBox = boxes[targetIndex];
  if (!targetBox) return null;

  const lineHeight = Math.max(12, targetBox.rawBottom - targetBox.rawTop);
  let bestIndex = null;
  let bestDist = Infinity;

  for (let i = 0; i < boxes.length; i++) {
    const b = boxes[i];
    if (Math.abs(b.cy - targetBox.cy) >= lineHeight * 0.6) continue;

    if (x >= b.rawLeft && x <= b.rawRight) return b.index;

    if (x >= b.left && x <= b.right) {
      const centerDist = Math.abs(x - b.cx);
      if (centerDist < bestDist) {
        bestDist = centerDist;
        bestIndex = b.index;
      }
      continue;
    }

    const dx = Math.max(b.left - x, 0, x - b.right);
    if (dx <= tolerance && dx < bestDist) {
      bestDist = dx;
      bestIndex = b.index;
    }
  }

  return bestIndex;
}

export default function TextReader({
  text,
  effects = {},
  zoomScale = 1.15,
  dimPreviousOnly = false,
  effectColors = {},
  previewNextWords = true,
  fontSize,
  fontFamily,
  fontWeight,
  fontStyle,
  gazeTolerance,
  showGazeDot,
  guidedMode,
  lineLocked = false,
  dynamicGuidance,
  dynamicJumpSeconds = 3,
  guidedResetKey,
  readFade = 78,
  autoPace,
  autoPacePaused,
  autoPaceWpm,
  strayAlertEnabled,
  strayDelayMs,
  strayVolume,
  onGuidedProgress,
  subscribeGaze,
  onWordChange,
  sendCommand,
  subscribeCalibrationAck,
  inputMode,
}) {
  const containerRef = useRef(null);
  const gazeDotRef = useRef(null);
  const wordBoxesRef = useRef([]);
  const [boxesVersion, setBoxesVersion] = useState(0);

  // Free gaze mode state
  const activeIndexRef = useRef(null);
  const [activeWordIndex, setActiveWordIndex] = useState(null);

  // Guided reading mode state & dwell trackers
  const guidedIndexRef = useRef(0);
  const [guidedIndex, setGuidedIndex] = useState(0);
  const currentDwellMsRef = useRef(0);
  const hasReadCurrentRef = useRef(false);
  const lastGazeWordRef = useRef(null);
  const lastGazeTimeRef = useRef(null);
  const lineEnteredAtRef = useRef(0);

  // Dynamic guidance trackers
  const backwardStartRef = useRef(null);
  const backwardAlertFiredRef = useRef(false);
  const dynamicTargetRef = useRef(null);
  const dynamicStartRef = useRef(null);
  const pendingJumpIndexRef = useRef(null);
  const [pendingJumpIndex, setPendingJumpIndex] = useState(null);

  // Gaze-stray alert trackers
  const isInsideReaderRef = useRef(false);
  const isBlinkingRef = useRef(false);
  const normalAnchorWordRef = useRef(null);
  const strayStartRef = useRef(null);
  const lastChimeTimeRef = useRef(0);

  // Right-click context menu & single-word quick recalibration
  const [contextMenu, setContextMenu] = useState(null);
  const [quickRecal, setQuickRecal] = useState(null);
  const quickRecalRef = useRef(null);
  quickRecalRef.current = quickRecal;
  const quickRecalTimerRef = useRef(null);
  const quickRecalFallbackRef = useRef(null);

  const updatePendingJump = useCallback((idx) => {
    if (pendingJumpIndexRef.current !== idx) {
      pendingJumpIndexRef.current = idx;
      setPendingJumpIndex(idx);
    }
  }, []);

  useEffect(() => {
    return () => {
      clearTimeout(quickRecalTimerRef.current);
      clearTimeout(quickRecalFallbackRef.current);
    };
  }, []);

  useEffect(() => {
    if (!contextMenu) return;
    const closeMenu = () => setContextMenu(null);
    const onKey = (e) => {
      if (e.code === 'Escape') setContextMenu(null);
    };
    window.addEventListener('click', closeMenu);
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('click', closeMenu);
      window.removeEventListener('keydown', onKey);
    };
  }, [contextMenu]);

  useEffect(() => {
    if (!subscribeCalibrationAck) return;
    return subscribeCalibrationAck((ack) => {
      if (ack.type !== 'recalibration_ack') return;
      if (!quickRecalRef.current) return;
      clearTimeout(quickRecalFallbackRef.current);

      if (ack.ok) {
        setQuickRecal(null);
      } else {
        setQuickRecal((prev) =>
          prev
            ? { ...prev, sampling: false, error: ack.reason || 'Recalibration failed.' }
            : null
        );
      }
    });
  }, [subscribeCalibrationAck]);

  const triggerQuickRecalCapture = useCallback(() => {
    const active = quickRecalRef.current;
    if (!active || active.sampling) return;

    setQuickRecal((prev) => (prev ? { ...prev, sampling: true, error: '' } : null));

    const sentBegin = sendCommand?.({
      type: 'recalibrate_point_begin',
      targetX: active.px / window.innerWidth,
      targetY: active.py / window.innerHeight,
    });

    quickRecalTimerRef.current = setTimeout(() => {
      const sentEnd = sendCommand?.({ type: 'recalibrate_point_end' });
      if (inputMode !== 'websocket' || !sentBegin || !sentEnd) {
        setQuickRecal(null);
      } else {
        quickRecalFallbackRef.current = setTimeout(() => {
          setQuickRecal(null);
        }, 1000);
      }
    }, QUICK_RECAL_MS);
  }, [sendCommand, inputMode]);

  useEffect(() => {
    if (!quickRecal) return;
    const onKeyDown = (e) => {
      if (e.code === 'Space') {
        e.preventDefault();
        triggerQuickRecalCapture();
      } else if (e.code === 'Escape') {
        clearTimeout(quickRecalTimerRef.current);
        clearTimeout(quickRecalFallbackRef.current);
        setQuickRecal(null);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [quickRecal, triggerQuickRecalCapture]);

  const handleReaderContextMenu = useCallback((e) => {
    const wordEl = e.target.closest('[data-word-index]');
    if (!wordEl) return;

    e.preventDefault();
    const r = wordEl.getBoundingClientRect();
    setContextMenu({
      menuX: e.clientX,
      menuY: e.clientY,
      word: wordEl.textContent?.trim() || 'this word',
      px: (r.left + r.right) / 2,
      py: (r.top + r.bottom) / 2,
    });
  }, []);

  const { paragraphs, wordTexts } = useMemo(() => {
    let globalIndex = 0;
    const texts = [];
    const parsedParagraphs = text
      .split(/\n+/)
      .filter((p) => p.trim().length > 0)
      .map((paragraphText) =>
        paragraphText
          .trim()
          .split(/\s+/)
          .map((word) => {
            const index = globalIndex++;
            texts[index] = word;
            return { word, index };
          })
      );

    return { paragraphs: parsedParagraphs, wordTexts: texts };
  }, [text]);

  const totalWords = wordTexts.length;

  const textShades = useMemo(
    () => getColorShades(effectColors.textColor || '#2563eb'),
    [effectColors.textColor]
  );
  const bgTints = useMemo(
    () => getHighlightTints(effectColors.bgHighlight || '#fde047'),
    [effectColors.bgHighlight]
  );
  const underlineShades = useMemo(
    () => getColorShades(effectColors.underline || '#2563eb'),
    [effectColors.underline]
  );

  const zoom1 = zoomScale;
  const zoom2 = Number((1 + (zoomScale - 1) * 0.55).toFixed(3));
  const zoom3 = Number((1 + (zoomScale - 1) * 0.25).toFixed(3));

  const setGuidedStep = useCallback(
    (nextIndex) => {
      guidedIndexRef.current = nextIndex;
      currentDwellMsRef.current = 0;
      hasReadCurrentRef.current = false;
      strayStartRef.current = null;
      backwardStartRef.current = null;
      backwardAlertFiredRef.current = false;
      dynamicTargetRef.current = null;
      dynamicStartRef.current = null;
      updatePendingJump(null);

      setGuidedIndex(nextIndex);

      if (onGuidedProgress) {
        onGuidedProgress(nextIndex, totalWords);
      }
      if (onWordChange) {
        onWordChange(
          nextIndex < totalWords ? nextIndex : null,
          nextIndex < totalWords ? wordTexts[nextIndex] : null
        );
      }
    },
    [totalWords, wordTexts, onGuidedProgress, onWordChange, updatePendingJump]
  );

  const measureWordBoxes = useCallback(() => {
    const container = containerRef.current;
    if (!container) return;

    const containerRect = container.getBoundingClientRect();
    const scrollLeft = container.scrollLeft;
    const scrollTop = container.scrollTop;

    const nodes = container.querySelectorAll('[data-word-index]');
    const boxes = [];

    for (let i = 0; i < nodes.length; i++) {
      const el = nodes[i];
      const index = Number(el.dataset.wordIndex);
      const rect = el.getBoundingClientRect();

      const rawLeft = rect.left - containerRect.left + scrollLeft;
      const rawRight = rect.right - containerRect.left + scrollLeft;
      const rawTop = rect.top - containerRect.top + scrollTop;
      const rawBottom = rect.bottom - containerRect.top + scrollTop;
      const rawWidth = rawRight - rawLeft;

      const wordStr = wordTexts[index] || '';
      const cleanLen = wordStr.replace(/[^a-zA-Z0-9]/g, '').length || wordStr.length;

      let padX = 0;
      let padY = 0;
      if (cleanLen <= 4) {
        const minVirtualWidth = cleanLen <= 2 ? 54 : 48;
        padX = Math.max(8, Math.round((minVirtualWidth - rawWidth) / 2));
        padY = 6;
      }

      boxes.push({
        index,
        rawLeft,
        rawRight,
        rawTop,
        rawBottom,
        left: rawLeft - padX,
        right: rawRight + padX,
        top: rawTop - padY,
        bottom: rawBottom + padY,
        cx: (rawLeft + rawRight) / 2,
        cy: (rawTop + rawBottom) / 2,
      });
    }

    wordBoxesRef.current = boxes;
    setBoxesVersion((v) => v + 1);
  }, [wordTexts]);

  useEffect(() => {
    activeIndexRef.current = null;
    setActiveWordIndex(null);

    guidedIndexRef.current = 0;
    setGuidedIndex(0);
    currentDwellMsRef.current = 0;
    hasReadCurrentRef.current = false;
    lastGazeWordRef.current = null;
    lastGazeTimeRef.current = null;
    normalAnchorWordRef.current = null;
    strayStartRef.current = null;
    backwardStartRef.current = null;
    backwardAlertFiredRef.current = false;
    dynamicTargetRef.current = null;
    dynamicStartRef.current = null;
    updatePendingJump(null);

    if (containerRef.current) {
      containerRef.current.scrollTo({ top: 0, behavior: 'smooth' });
    }

    if (onGuidedProgress) {
      onGuidedProgress(0, totalWords);
    }

    if (onWordChange) {
      if (guidedMode && totalWords > 0) {
        onWordChange(0, wordTexts[0]);
      } else {
        onWordChange(null, null);
      }
    }
  }, [
    paragraphs,
    guidedMode,
    guidedResetKey,
    totalWords,
    wordTexts,
    onGuidedProgress,
    onWordChange,
    updatePendingJump,
  ]);

  useEffect(() => {
    if (!guidedMode || !autoPace || autoPacePaused || totalWords === 0) return;

    const intervalMs = Math.max(80, Math.round(60000 / autoPaceWpm));

    const timer = setInterval(() => {
      const current = guidedIndexRef.current;
      if (current >= totalWords) {
        clearInterval(timer);
        return;
      }
      setGuidedStep(current + 1);
    }, intervalMs);

    return () => clearInterval(timer);
  }, [guidedMode, autoPace, autoPacePaused, autoPaceWpm, totalWords, setGuidedStep]);

  useEffect(() => {
    const needsMonitor = strayAlertEnabled || (guidedMode && dynamicGuidance && !lineLocked);
    if (!needsMonitor) {
      strayStartRef.current = null;
      backwardStartRef.current = null;
      backwardAlertFiredRef.current = false;
      dynamicTargetRef.current = null;
      dynamicStartRef.current = null;
      updatePendingJump(null);
      return;
    }

    const dynamicJumpMs = Math.max(500, Math.round(dynamicJumpSeconds * 1000));

    const timer = setInterval(() => {
      if (!isInsideReaderRef.current || isBlinkingRef.current || quickRecalRef.current) {
        strayStartRef.current = null;
        backwardStartRef.current = null;
        backwardAlertFiredRef.current = false;
        dynamicTargetRef.current = null;
        dynamicStartRef.current = null;
        updatePendingJump(null);
        return;
      }

      const now = performance.now();

      if (guidedMode && dynamicGuidance && !lineLocked) {
        if (
          backwardStartRef.current !== null &&
          !backwardAlertFiredRef.current &&
          now - backwardStartRef.current >= DYNAMIC_BACKWARD_ALERT_MS
        ) {
          playStrayChime(strayVolume);
          lastChimeTimeRef.current = now;
          backwardAlertFiredRef.current = true;
        }

        if (
          dynamicTargetRef.current !== null &&
          dynamicStartRef.current !== null &&
          now - dynamicStartRef.current >= dynamicJumpMs
        ) {
          const jumpTo = dynamicTargetRef.current;
          if (jumpTo >= 0 && jumpTo < totalWords) {
            setGuidedStep(jumpTo);
            return;
          }
        }
      }

      if (!strayAlertEnabled) return;
      if (guidedMode && guidedIndexRef.current >= totalWords) {
        strayStartRef.current = null;
        return;
      }
      if (strayStartRef.current === null) return;

      const elapsedStray = now - strayStartRef.current;
      const sinceLastChime = now - lastChimeTimeRef.current;

      if (elapsedStray >= strayDelayMs && sinceLastChime >= STRAY_COOLDOWN_MS) {
        playStrayChime(strayVolume);
        lastChimeTimeRef.current = now;

        if (!guidedMode && lastGazeWordRef.current !== null) {
          normalAnchorWordRef.current = lastGazeWordRef.current;
          strayStartRef.current = null;
        }
      }
    }, 100);

    return () => clearInterval(timer);
  }, [
    strayAlertEnabled,
    strayDelayMs,
    strayVolume,
    guidedMode,
    dynamicGuidance,
    lineLocked,
    dynamicJumpSeconds,
    totalWords,
    setGuidedStep,
    updatePendingJump,
  ]);

  useEffect(() => {
    if (!guidedMode || guidedIndex >= totalWords) return;

    const container = containerRef.current;
    if (!container) return;

    const targetEl = container.querySelector(`[data-word-index="${guidedIndex}"]`);
    if (!targetEl) return;

    const containerRect = container.getBoundingClientRect();
    const elRect = targetEl.getBoundingClientRect();

    const isNearBottom = elRect.bottom > containerRect.bottom - containerRect.height * 0.2;
    const isAboveTop = elRect.top < containerRect.top + 40;

    if (isNearBottom || isAboveTop) {
      targetEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [guidedMode, guidedIndex, totalWords]);

  useLayoutEffect(() => {
    measureWordBoxes();

    const container = containerRef.current;
    if (!container) return;

    const resizeObserver = new ResizeObserver(() => {
      measureWordBoxes();
    });
    resizeObserver.observe(container);

    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(measureWordBoxes);
    }

    return () => resizeObserver.disconnect();
  }, [paragraphs, fontSize, fontFamily, fontWeight, fontStyle, measureWordBoxes]);

  useEffect(() => {
    return subscribeGaze(({ x, y, blinking }) => {
      isBlinkingRef.current = blinking;

      if (gazeDotRef.current && showGazeDot) {
        gazeDotRef.current.style.transform = `translate3d(${x}px, ${y}px, 0)`;
        gazeDotRef.current.style.opacity = blinking ? '0.35' : '0.85';
      }

      if (blinking || quickRecalRef.current) return;

      const container = containerRef.current;
      if (!container) return;

      const rect = container.getBoundingClientRect();
      const isInsideReader =
        x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom;
      isInsideReaderRef.current = isInsideReader;

      let matchedIndex = null;
      if (isInsideReader) {
        const contentX = x - rect.left + container.scrollLeft;
        const contentY = y - rect.top + container.scrollTop;
        matchedIndex = findWordAtPoint(
          contentX,
          contentY,
          wordBoxesRef.current,
          gazeTolerance
        );
      }

      const now = performance.now();

      // --- LINE-LOCKED GUIDED READING (X-axis only + return sweep) ---
      if (guidedMode && lineLocked) {
        const currentTarget = guidedIndexRef.current;
        const boxes = wordBoxesRef.current;
        const currBox = boxes[currentTarget];

        if (currentTarget >= totalWords || !currBox) {
          strayStartRef.current = null;
          return;
        }

        // Visually snap the green gaze dot onto the active line's Y so line-lock is clear
        if (gazeDotRef.current && showGazeDot && isInsideReader) {
          const lockedScreenY = rect.top + currBox.cy - container.scrollTop;
          gazeDotRef.current.style.transform = `translate3d(${x}px, ${lockedScreenY}px, 0)`;
          gazeDotRef.current.style.opacity = blinking ? '0.35' : '0.85';
        }

        if (!isInsideReader) {
          strayStartRef.current = null;
          return;
        }

        if (autoPace && !autoPacePaused) {
          return;
        }

        const contentX = x - rect.left + container.scrollLeft;
        const lineH = Math.max(12, currBox.rawBottom - currBox.rawTop);

        // Find start & end word indices of the current visual line
        let lineStartIdx = currentTarget;
        while (
          lineStartIdx > 0 &&
          Math.abs(boxes[lineStartIdx - 1].cy - currBox.cy) < lineH * 0.6
        ) {
          lineStartIdx--;
        }
        let lineEndIdx = currentTarget;
        while (
          lineEndIdx + 1 < totalWords &&
          Math.abs(boxes[lineEndIdx + 1].cy - currBox.cy) < lineH * 0.6
        ) {
          lineEndIdx++;
        }

        const lineLeftX = boxes[lineStartIdx].rawLeft;
        const lineRightX = boxes[lineEndIdx].rawRight;
        const lineWidth = Math.max(1, lineRightX - lineLeftX);

        // Match using X coordinate ONLY along the current line
        const xMatch = findWordOnLineByX(
          contentX,
          boxes,
          currentTarget,
          Math.max(gazeTolerance, 40)
        );

        // Stray alert check along this line
        const isNearCurrentOrAhead =
          contentX >= currBox.left - Math.max(gazeTolerance, 50) &&
          contentX <= lineRightX + Math.max(gazeTolerance, 80);
        if (isNearCurrentOrAhead) {
          strayStartRef.current = null;
        } else if (strayStartRef.current === null) {
          strayStartRef.current = now;
        }

        // Brief grace period right after jumping to a new line so momentum doesn't skip word 1
        if (now - lineEnteredAtRef.current < 220) {
          return;
        }

        // 1. Return-sweep detection: near the end of the line + eyes sweep back to left third
        const isNearEndOfLine = currentTarget >= lineEndIdx - 1;
        const hasNextLine = lineEndIdx + 1 < totalWords;
        if (isNearEndOfLine && hasNextLine) {
          const sweptBackLeft = contentX <= lineLeftX + lineWidth * 0.32;
          if (sweptBackLeft) {
            lineEnteredAtRef.current = now;
            setGuidedStep(lineEndIdx + 1);
            return;
          }
        }

        // 2. Final word of the entire text
        if (currentTarget === totalWords - 1) {
          if (xMatch === currentTarget || contentX >= currBox.rawLeft) {
            if (lastGazeWordRef.current === currentTarget && lastGazeTimeRef.current !== null) {
              currentDwellMsRef.current += now - lastGazeTimeRef.current;
            }
            lastGazeWordRef.current = currentTarget;
            lastGazeTimeRef.current = now;
            if (currentDwellMsRef.current >= 350 || contentX > currBox.rawRight + 50) {
              setGuidedStep(totalWords);
            }
          }
          return;
        }

        // 3. Smooth forward-only ratchet across the current line (never freezes on overshoots)
        if (xMatch !== null && xMatch > currentTarget && xMatch <= lineEndIdx) {
          const nextStep = xMatch - currentTarget <= 2 ? xMatch : currentTarget + 1;
          setGuidedStep(nextStep);
        } else if (contentX > currBox.rawRight + 12 && currentTarget < lineEndIdx) {
          setGuidedStep(currentTarget + 1);
        }
        return;
      }

      // --- STANDARD / DYNAMIC GUIDED READING MODE ---
      if (guidedMode) {
        const currentTarget = guidedIndexRef.current;

        if (dynamicGuidance && isInsideReader) {
          const isBackward = matchedIndex !== null && matchedIndex < currentTarget;
          if (isBackward) {
            if (backwardStartRef.current === null) {
              backwardStartRef.current = now;
              backwardAlertFiredRef.current = false;
            }
          } else {
            backwardStartRef.current = null;
            backwardAlertFiredRef.current = false;
          }

          const isDifferentWord =
            matchedIndex !== null &&
            matchedIndex !== currentTarget &&
            matchedIndex !== currentTarget + 1;

          if (isDifferentWord) {
            if (dynamicTargetRef.current !== matchedIndex) {
              dynamicTargetRef.current = matchedIndex;
              dynamicStartRef.current = now;
              updatePendingJump(matchedIndex);
            }
          } else {
            dynamicTargetRef.current = null;
            dynamicStartRef.current = null;
            updatePendingJump(null);
          }
        } else {
          backwardStartRef.current = null;
          backwardAlertFiredRef.current = false;
          dynamicTargetRef.current = null;
          dynamicStartRef.current = null;
          updatePendingJump(null);
        }

        if (currentTarget >= totalWords) {
          strayStartRef.current = null;
          return;
        }

        if (!isInsideReader) {
          strayStartRef.current = null;
        } else {
          const isOnTargetOrNext =
            matchedIndex === currentTarget || matchedIndex === currentTarget + 1;
          if (isOnTargetOrNext) {
            strayStartRef.current = null;
          } else if (strayStartRef.current === null) {
            strayStartRef.current = now;
          }
        }

        if (autoPace && !autoPacePaused) {
          lastGazeWordRef.current = matchedIndex;
          lastGazeTimeRef.current = now;
          return;
        }

        if (lastGazeWordRef.current === currentTarget && lastGazeTimeRef.current !== null) {
          currentDwellMsRef.current += now - lastGazeTimeRef.current;
        }
        lastGazeWordRef.current = matchedIndex;
        lastGazeTimeRef.current = now;

        if (currentDwellMsRef.current >= MIN_GUIDED_DWELL_MS) {
          hasReadCurrentRef.current = true;
        }

        const isLastWord = currentTarget === totalWords - 1;
        const lookedAhead =
          matchedIndex === currentTarget + 1 || matchedIndex === currentTarget + 2;

        const shouldAdvance =
          hasReadCurrentRef.current &&
          (lookedAhead ||
            (isLastWord &&
              (matchedIndex !== currentTarget || currentDwellMsRef.current >= 450)));

        if (shouldAdvance) {
          lastGazeTimeRef.current = now;
          setGuidedStep(currentTarget + 1);
        }
        return;
      }

      // --- FREE GAZE MODE ---
      lastGazeWordRef.current = matchedIndex;

      if (!isInsideReader) {
        strayStartRef.current = null;
      } else {
        const anchor = normalAnchorWordRef.current;
        if (matchedIndex !== null) {
          if (anchor === null || matchedIndex === anchor || matchedIndex === anchor + 1) {
            normalAnchorWordRef.current = matchedIndex;
            strayStartRef.current = null;
          } else if (strayStartRef.current === null) {
            strayStartRef.current = now;
          }
        } else if (anchor !== null && strayStartRef.current === null) {
          strayStartRef.current = now;
        }
      }

      if (matchedIndex !== activeIndexRef.current) {
        activeIndexRef.current = matchedIndex;
        setActiveWordIndex(matchedIndex);
        if (onWordChange) {
          onWordChange(matchedIndex, matchedIndex !== null ? wordTexts[matchedIndex] : null);
        }
      }
    });
  }, [
    subscribeGaze,
    gazeTolerance,
    showGazeDot,
    guidedMode,
    lineLocked,
    dynamicGuidance,
    autoPace,
    autoPacePaused,
    totalWords,
    wordTexts,
    setGuidedStep,
    onWordChange,
    updatePendingJump,
  ]);

  const readOpacity = Number((1 - readFade / 100).toFixed(2));
  const currentTargetIndex = guidedMode
    ? guidedIndex < totalWords
      ? guidedIndex
      : null
    : activeWordIndex;

  const lineShifts = useMemo(() => {
    if (!effects.zoom || currentTargetIndex === null) return {};
    return computeLinePartingShifts(
      wordBoxesRef.current,
      currentTargetIndex,
      zoom1,
      zoom2,
      zoom3,
      previewNextWords
    );
  }, [effects.zoom, currentTargetIndex, zoom1, zoom2, zoom3, previewNextWords, boxesVersion]);

  const effectClasses = [
    'reader-content',
    effects.zoom && 'fx-zoom',
    effects.bold && 'fx-bold',
    effects.textColor && 'fx-text-color',
    effects.bgHighlight && 'fx-bg-highlight',
    effects.underline && 'fx-underline',
    effects.dimSurroundings && (dimPreviousOnly ? 'fx-dim-prev' : 'fx-dim'),
    currentTargetIndex !== null && 'has-active-focus',
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <main className="reader-pane" ref={containerRef}>
      {showGazeDot && <div className="gaze-debug-dot" ref={gazeDotRef} />}

      <article
        className={effectClasses}
        onContextMenu={handleReaderContextMenu}
        style={{
          fontSize: `${fontSize}px`,
          fontFamily,
          fontWeight,
          fontStyle,
          '--read-opacity': readOpacity,
          '--jump-duration': `${dynamicJumpSeconds}s`,
          '--zoom-1': zoom1,
          '--zoom-2': zoom2,
          '--zoom-3': zoom3,
          '--fx-text-1': textShades.s1,
          '--fx-text-2': textShades.s2,
          '--fx-text-3': textShades.s3,
          '--fx-bg-1': bgTints.bg1,
          '--fx-bg-2': bgTints.bg2,
          '--fx-bg-3': bgTints.bg3,
          '--fx-ul-1': underlineShades.s1,
          '--fx-ul-2': underlineShades.s2,
          '--fx-ul-3': underlineShades.s3,
        }}
      >
        {paragraphs.map((words, pIdx) => (
          <p key={pIdx} className="reader-paragraph">
            {words.map(({ word, index }) => {
              let focusLevel = 0;
              const isRead = guidedMode && index < guidedIndex;
              const isBeforeCurrent =
                currentTargetIndex !== null && index < currentTargetIndex;

              if (currentTargetIndex !== null) {
                if (index === currentTargetIndex) {
                  focusLevel = 1;
                } else if (previewNextWords && index === currentTargetIndex + 1) {
                  focusLevel = 2;
                } else if (previewNextWords && index === currentTargetIndex + 2) {
                  focusLevel = 3;
                }
              }

              return (
                <React.Fragment key={index}>
                  <Word
                    word={word}
                    index={index}
                    focusLevel={focusLevel}
                    isRead={isRead}
                    isBeforeCurrent={isBeforeCurrent}
                    isJumpCharging={
                      guidedMode && dynamicGuidance && !lineLocked && pendingJumpIndex === index
                    }
                    shiftX={lineShifts[index] || 0}
                  />{' '}
                </React.Fragment>
              );
            })}
          </p>
        ))}
      </article>

      {/* Right-Click Context Menu */}
      {contextMenu && (
        <div
          className="word-context-menu"
          style={{ left: `${contextMenu.menuX}px`, top: `${contextMenu.menuY}px` }}
          onClick={(e) => e.stopPropagation()}
        >
          <button
            type="button"
            className="context-menu-item"
            onClick={() => {
              const target = contextMenu;
              setContextMenu(null);
              setQuickRecal({
                word: target.word,
                px: target.px,
                py: target.py,
                sampling: false,
                error: '',
              });
            }}
          >
            Recalibrate on &ldquo;{contextMenu.word}&rdquo;
          </button>
        </div>
      )}

      {/* Single-Word Quick Recalibration Dot & Prompt */}
      {quickRecal && (
        <div className="quick-recal-backdrop">
          <div
            className="quick-recal-prompt"
            style={{
              left: `${quickRecal.px}px`,
              top: `${quickRecal.py < 110 ? quickRecal.py + 46 : quickRecal.py - 56}px`,
            }}
          >
            <span>
              {quickRecal.sampling
                ? 'Capturing — keep looking at the dot...'
                : 'Press SPACE when you are looking at it'}
            </span>
            {quickRecal.error && (
              <div className="quick-recal-error">{quickRecal.error}</div>
            )}
            <button
              type="button"
              className="btn-link quick-recal-cancel"
              onClick={() => setQuickRecal(null)}
            >
              Cancel (Esc)
            </button>
          </div>

          <button
            type="button"
            className={`calibration-dot word-dot ${quickRecal.sampling ? 'sampling' : ''}`}
            style={{
              left: `${quickRecal.px}px`,
              top: `${quickRecal.py}px`,
              '--capture-ms': `${QUICK_RECAL_MS}ms`,
            }}
            onClick={triggerQuickRecalCapture}
            aria-label="Quick recalibration target"
          />
        </div>
      )}
    </main>
  );
}