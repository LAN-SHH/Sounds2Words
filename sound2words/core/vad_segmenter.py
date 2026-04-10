from __future__ import annotations

import os
import threading
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Callable

import numpy as np

from sound2words.core.audio_capture import AudioFrame


@dataclass(slots=True)
class SpeechSegment:
    start_ms: int
    end_ms: int
    sample_rate: int
    samples: np.ndarray


@dataclass(slots=True)
class _FrameUnit:
    frame_index: int
    start_ms: int
    end_ms: int
    sample_rate: int
    samples: np.ndarray
    vad_result: bool


class VADSegmenter(threading.Thread):
    """Stable VAD segmenter with fixed-frame state machine."""

    def __init__(
        self,
        frame_queue: Queue[AudioFrame | None],
        segment_queue: Queue[SpeechSegment | None],
        stop_event: threading.Event,
        silence_end_ms: int = 800,
        min_segment_ms: int = 400,
        max_segment_ms: int = 12000,
        pad_pre_ms: int = 220,
        pad_post_ms: int = 320,
        vad_rms_threshold: float = 0.03,
        queue_overflow_logger: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.segment_queue = segment_queue
        self.stop_event = stop_event
        self.silence_end_ms = int(silence_end_ms)
        self.min_segment_ms = int(min_segment_ms)
        self.max_segment_ms = int(max_segment_ms)
        self.pad_pre_ms = int(pad_pre_ms)
        self.pad_post_ms = int(pad_post_ms)
        self.vad_rms_threshold = float(vad_rms_threshold)
        self.queue_overflow_logger = queue_overflow_logger

        self.frame_ms = 20
        self.start_voice_frames = 4
        self.end_grace_ms = 260
        self.min_voiced_ms = 260
        self.min_voiced_ratio = 0.18
        self.edge_fade_ms = 12

        self._sample_rate: int | None = None
        self._fixed_frame_samples = 0
        self._sample_buffer = np.empty((0,), dtype=np.float32)
        self._cursor_ms = 0.0
        self._frame_index = 0

        self._state = "silence"  # silence | speaking
        self._pre_roll: deque[_FrameUnit] = deque()
        self._segment_frames: list[_FrameUnit] = []
        self._voice_run = 0
        self._silence_run = 0
        self._last_voiced_idx = -1
        self._pending_end = False
        self._segment_id = 0
        self._last_emitted_frame_idx = -1
        self._skip_preroll_once = False

        self.trace_enabled = os.environ.get("SOUND2WORDS_VAD_TRACE", "0") == "1"
        self.trace_limit = int(os.environ.get("SOUND2WORDS_VAD_TRACE_LIMIT", "300"))
        self._traced = 0
        self.debug_dir = (Path.cwd() / "data" / "debug_segments" / "vad").resolve()
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.trace_log = self.debug_dir / "vad_trace.log"
        self._recent_window: deque[np.ndarray] = deque(maxlen=25)  # ~500ms

    def run(self) -> None:
        while not self.stop_event.is_set():
            incoming = self.frame_queue.get()
            if incoming is None:
                break
            self._consume_audio_frame(incoming)

        self._flush_segment(force=True)
        self.segment_queue.put(None)

    def _consume_audio_frame(self, incoming: AudioFrame) -> None:
        mono = np.asarray(incoming.samples, dtype=np.float32).copy()
        if mono.ndim > 1:
            mono = mono.mean(axis=1).astype(np.float32, copy=False)
        mono = np.clip(mono, -1.0, 1.0).astype(np.float32, copy=False)
        if mono.size == 0:
            return

        if self._sample_rate is None:
            self._sample_rate = int(incoming.sample_rate)
            self._fixed_frame_samples = max(1, int(self._sample_rate * self.frame_ms / 1000))
            self._cursor_ms = float(incoming.start_ms)

        self._sample_buffer = np.concatenate((self._sample_buffer, mono), dtype=np.float32)
        while self._sample_buffer.shape[0] >= self._fixed_frame_samples:
            chunk = self._sample_buffer[: self._fixed_frame_samples].copy()
            self._sample_buffer = self._sample_buffer[self._fixed_frame_samples :]
            self._frame_index += 1
            start_ms = int(round(self._cursor_ms))
            end_ms = start_ms + self.frame_ms
            self._cursor_ms += self.frame_ms
            self._consume_fixed_frame(chunk, start_ms, end_ms)

    def _consume_fixed_frame(self, samples: np.ndarray, start_ms: int, end_ms: int) -> None:
        vad_result = bool(_rms(samples) >= self.vad_rms_threshold)
        unit = _FrameUnit(
            frame_index=self._frame_index,
            start_ms=start_ms,
            end_ms=end_ms,
            sample_rate=int(self._sample_rate or 16000),
            samples=samples.copy(),
            vad_result=vad_result,
        )
        self._log_frame(unit)
        self._save_debug_frame(unit)

        pre_roll_frames = max(1, int(self.pad_pre_ms / self.frame_ms))
        self._pre_roll.append(unit)
        while len(self._pre_roll) > pre_roll_frames:
            self._pre_roll.popleft()

        end_silence_frames = max(1, int(self.silence_end_ms / self.frame_ms))
        end_grace_frames = max(1, int(self.end_grace_ms / self.frame_ms))
        tail_frames = max(0, int(self.pad_post_ms / self.frame_ms))

        if self._state == "silence":
            if vad_result:
                self._voice_run += 1
                if self._voice_run >= self.start_voice_frames:
                    self._state = "speaking"
                    if self._skip_preroll_once:
                        self._segment_frames = [unit]
                        self._skip_preroll_once = False
                    else:
                        self._segment_frames = list(self._pre_roll)
                    self._last_voiced_idx = self._segment_frames[-1].frame_index
                    self._silence_run = 0
                    self._pending_end = False
            else:
                self._voice_run = 0
            return

        # speaking state
        self._segment_frames.append(unit)
        if vad_result:
            self._last_voiced_idx = unit.frame_index
            self._silence_run = 0
            self._pending_end = False
        else:
            self._silence_run += 1

        if self._segment_frames:
            duration = self._segment_frames[-1].end_ms - self._segment_frames[0].start_ms
            if duration >= self.max_segment_ms:
                self._skip_preroll_once = True
                self._flush_segment(force=True)
                return

        if self._silence_run >= end_silence_frames and not self._pending_end:
            self._pending_end = True
            return

        if self._pending_end and self._silence_run >= (end_silence_frames + end_grace_frames):
            self._flush_segment(force=False, tail_frames=tail_frames)

    def _flush_segment(self, force: bool, tail_frames: int = 0) -> None:
        if not self._segment_frames:
            self._reset_state_after_flush()
            return

        frames = self._segment_frames
        if not force:
            cutoff = self._last_voiced_idx + max(0, tail_frames)
            frames = [f for f in self._segment_frames if f.frame_index <= cutoff]
        # Ensure no overlap between consecutive segments.
        frames = [f for f in frames if f.frame_index > self._last_emitted_frame_idx]
        if not frames:
            self._reset_state_after_flush()
            return

        start_ms = frames[0].start_ms
        end_ms = frames[-1].end_ms
        duration = end_ms - start_ms
        if duration < self.min_segment_ms:
            self._reset_state_after_flush()
            return

        voiced_frames = [f for f in frames if f.vad_result]
        voiced_ms = len(voiced_frames) * self.frame_ms
        voiced_ratio = len(voiced_frames) / max(1, len(frames))
        if voiced_ms < self.min_voiced_ms or voiced_ratio < self.min_voiced_ratio:
            self._append_trace(
                f"[VAD-SKIP] seg={self._segment_id + 1}, voiced_ms={voiced_ms}, "
                f"voiced_ratio={voiced_ratio:.2f}, duration_ms={duration}"
            )
            self._reset_state_after_flush()
            return

        samples = np.concatenate([f.samples.copy() for f in frames], dtype=np.float32)
        samples = np.clip(samples, -1.0, 1.0).astype(np.float32, copy=False)
        samples = self._apply_edge_fade(samples, frames[0].sample_rate)
        segment = SpeechSegment(
            start_ms=start_ms,
            end_ms=end_ms,
            sample_rate=frames[0].sample_rate,
            samples=samples.copy(),  # independent copy
        )
        self._segment_id += 1
        self._last_emitted_frame_idx = frames[-1].frame_index
        self._log_segment(frames, segment)
        self._save_debug_segment(self._segment_id, segment)
        self._put_segment_with_drop_oldest(segment)
        self._reset_state_after_flush()

    def _reset_state_after_flush(self) -> None:
        self._state = "silence"
        self._segment_frames = []
        self._voice_run = 0
        self._silence_run = 0
        self._last_voiced_idx = -1
        self._pending_end = False

    def _put_segment_with_drop_oldest(self, segment: SpeechSegment) -> None:
        try:
            self.segment_queue.put_nowait(segment)
            return
        except Exception:
            pass
        try:
            dropped = self.segment_queue.get_nowait()
            if dropped is not None and self.queue_overflow_logger is not None:
                self.queue_overflow_logger("Transcribe queue full, dropped oldest realtime task.")
            self.segment_queue.put_nowait(segment)
        except Exception:
            if self.queue_overflow_logger is not None:
                self.queue_overflow_logger("Transcribe queue congested, skipped realtime task.")

    def _log_frame(self, frame: _FrameUnit) -> None:
        if not self.trace_enabled:
            return
        if self._traced < self.trace_limit:
            msg = (
                f"[VAD-FRAME] idx={frame.frame_index}, len={frame.samples.shape[0]}, "
                f"dtype={frame.samples.dtype}, frame_samples={self._fixed_frame_samples}, "
                f"frame_duration_ms={self.frame_ms}, sample_rate={frame.sample_rate}, "
                f"vad_result={int(frame.vad_result)}, state={self._state}"
            )
            self._append_trace(msg)
        self._traced += 1

    def _log_segment(self, frames: list[_FrameUnit], segment: SpeechSegment) -> None:
        msg = (
            f"[VAD-SEG] seg={self._segment_id}, start_frame={frames[0].frame_index}, "
            f"end_frame={frames[-1].frame_index}, frame_count={len(frames)}, "
            f"total_samples={segment.samples.shape[0]}, duration_ms={segment.end_ms - segment.start_ms}"
        )
        self._append_trace(msg)

    def _append_trace(self, text: str) -> None:
        try:
            with self.trace_log.open("a", encoding="utf-8") as f:
                f.write(text + os.linesep)
        except Exception:
            pass

    def _save_debug_frame(self, frame: _FrameUnit) -> None:
        if not self.trace_enabled or frame.frame_index > self.trace_limit:
            return
        self._recent_window.append(frame.samples.copy())
        frame_path = self.debug_dir / f"debug_vad_frame_{frame.frame_index:04d}.wav"
        _write_wav_mono(frame_path, frame.samples, frame.sample_rate)
        if self._recent_window:
            window = np.concatenate(list(self._recent_window), dtype=np.float32)
            window_path = self.debug_dir / f"debug_vad_window_{frame.frame_index:04d}.wav"
            _write_wav_mono(window_path, window, frame.sample_rate)

    def _save_debug_segment(self, seg_id: int, segment: SpeechSegment) -> None:
        seg_path = self.debug_dir / f"debug_segment_{seg_id:04d}.wav"
        _write_wav_mono(seg_path, segment.samples, segment.sample_rate)

    def _apply_edge_fade(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        fade = max(0, int(sample_rate * self.edge_fade_ms / 1000))
        if fade <= 0 or samples.size <= fade * 2:
            return samples.astype(np.float32, copy=False)
        out = samples.astype(np.float32, copy=True)
        in_ramp = np.linspace(0.0, 1.0, fade, endpoint=False, dtype=np.float32)
        out_ramp = np.linspace(1.0, 0.0, fade, endpoint=False, dtype=np.float32)
        out[:fade] *= in_ramp
        out[-fade:] *= out_ramp
        return out


class FixedWindowSegmenter(threading.Thread):
    """Debug mode: bypass VAD and segment by fixed window."""

    def __init__(
        self,
        frame_queue: Queue[AudioFrame | None],
        segment_queue: Queue[SpeechSegment | None],
        stop_event: threading.Event,
        window_ms: int = 5000,
        min_segment_ms: int = 300,
    ) -> None:
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.segment_queue = segment_queue
        self.stop_event = stop_event
        self.window_ms = max(1000, int(window_ms))
        self.min_segment_ms = max(100, int(min_segment_ms))
        self._frames: list[AudioFrame] = []
        self._start_ms = 0

    def run(self) -> None:
        while not self.stop_event.is_set():
            frame = self.frame_queue.get()
            if frame is None:
                break
            self._consume(frame)
        self._flush()
        self.segment_queue.put(None)

    def _consume(self, frame: AudioFrame) -> None:
        if not self._frames:
            self._start_ms = frame.start_ms
        self._frames.append(frame)
        if (frame.end_ms - self._start_ms) >= self.window_ms:
            self._flush()

    def _flush(self) -> None:
        if not self._frames:
            return
        start_ms = self._frames[0].start_ms
        end_ms = self._frames[-1].end_ms
        if (end_ms - start_ms) < self.min_segment_ms:
            self._frames = []
            return
        samples = np.concatenate([np.asarray(f.samples, dtype=np.float32).copy() for f in self._frames], dtype=np.float32)
        segment = SpeechSegment(
            start_ms=start_ms,
            end_ms=end_ms,
            sample_rate=self._frames[0].sample_rate,
            samples=samples.copy(),
        )
        try:
            self.segment_queue.put_nowait(segment)
        except Exception:
            try:
                self.segment_queue.get_nowait()
                self.segment_queue.put_nowait(segment)
            except Exception:
                pass
        self._frames = []


def _rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    arr = np.asarray(samples, dtype=np.float32)
    return float(np.sqrt(np.mean(np.square(arr))))


def _write_wav_mono(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16.tobytes())
