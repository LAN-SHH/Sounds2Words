from __future__ import annotations

import os
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Callable

import numpy as np


@dataclass(slots=True)
class AudioFrame:
    start_ms: int
    end_ms: int
    sample_rate: int
    samples: np.ndarray

    @property
    def duration_ms(self) -> int:
        return max(1, self.end_ms - self.start_ms)


class SoundDeviceAudioCapture(threading.Thread):
    """持续采集音频帧，并推送到 frame_queue。

    该线程只做采集，不做任何推理，保证录音优先。
    """

    def __init__(
        self,
        recorder: object,
        frame_queue: Queue[AudioFrame | None],
        stop_event: threading.Event,
        level_callback: Callable[[int], None] | None = None,
        frame_ms: int = 20,
        raw_cache_seconds: int = 180,
    ) -> None:
        super().__init__(daemon=True)
        self.recorder = recorder
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.level_callback = level_callback
        self.frame_ms = max(10, frame_ms)
        self._timeline_ms = 0
        cache_frames = max(1, int(raw_cache_seconds * 1000 / self.frame_ms))
        self.raw_cache: deque[np.ndarray] = deque(maxlen=cache_frames)

    def run(self) -> None:
        sr = int(getattr(self.recorder, "sample_rate", 16000) or 16000)
        frames_per_block = max(160, int(sr * self.frame_ms / 1000))
        while not self.stop_event.is_set():
            block = self.recorder.record(numframes=frames_per_block)
            if block.size == 0:
                self._emit_level(0)
                continue
            mono = _to_mono(block)
            # 原始音频先缓存，转写链路消费副本。
            self.raw_cache.append(mono.copy())
            self._emit_level(_estimate_level(mono))
            start_ms = self._timeline_ms
            end_ms = start_ms + int(round(mono.shape[0] * 1000.0 / sr))
            self._timeline_ms = end_ms
            frame = AudioFrame(
                start_ms=start_ms,
                end_ms=end_ms,
                sample_rate=sr,
                samples=mono.astype(np.float32, copy=True),
            )
            self._put_with_drop_oldest(frame)
        self.frame_queue.put(None)

    def _emit_level(self, value: int) -> None:
        if self.level_callback is None:
            return
        try:
            self.level_callback(max(0, min(100, int(value))))
        except Exception:
            pass

    def _put_with_drop_oldest(self, frame: AudioFrame) -> None:
        try:
            self.frame_queue.put_nowait(frame)
        except Exception:
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame)
            except Exception:
                pass


class ProcessLoopbackAudioCapture(threading.Thread):
    """基于 process-audio-capture 的连续采集适配层。

    DLL 以 start/stop + wav 输出工作，这里用短窗口循环采集并拆成帧。
    """

    def __init__(
        self,
        pid: int,
        mode: int,
        frame_queue: Queue[AudioFrame | None],
        stop_event: threading.Event,
        level_callback: Callable[[int], None] | None = None,
        capture_window_ms: int = 1600,
        raw_cache_seconds: int = 180,
    ) -> None:
        super().__init__(daemon=True)
        self.pid = int(pid)
        self.mode = int(mode)
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.level_callback = level_callback
        self.capture_window_ms = max(200, capture_window_ms)
        self._timeline_ms = 0
        cache_frames = max(1, int(raw_cache_seconds * 1000 / 20))
        self.raw_cache: deque[np.ndarray] = deque(maxlen=cache_frames)
        self._crossfade_ms = 40
        self._fade_ms = 8
        self._tail: np.ndarray | None = None

    def run(self) -> None:
        from process_audio_capture import ProcessAudioCapture

        idx = 0
        while not self.stop_event.is_set():
            idx += 1
            path = Path(tempfile.gettempdir()) / f"s2w_frame_{self.pid}_{os.getpid()}_{idx}.wav"
            with ProcessAudioCapture(pid=self.pid, mode=self.mode, output_path=str(path)) as cap:
                cap.start()
                start = time.time()
                while not self.stop_event.is_set() and (time.time() - start) * 1000 < self.capture_window_ms:
                    try:
                        db = float(cap.level_db)
                        normalized = int(max(0.0, min(100.0, ((db + 60.0) / 60.0) * 100.0)))
                        self._emit_level(normalized)
                    except Exception:
                        pass
                    time.sleep(0.05)
                cap.stop()

            audio, sr = _read_wav_as_float(path)
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
            if audio.size == 0:
                self._emit_level(0)
                continue
            stitched = self._stitch_chunk(audio, sr)
            if stitched.size == 0:
                continue
            self._push_as_frames(stitched, sr)

        self.frame_queue.put(None)

    def _push_as_frames(self, audio: np.ndarray, sample_rate: int) -> None:
        block = max(160, int(sample_rate * 0.02))  # 20ms
        pos = 0
        total = audio.shape[0]
        while pos < total and not self.stop_event.is_set():
            frag = audio[pos : pos + block]
            if frag.size == 0:
                break
            start_ms = self._timeline_ms
            end_ms = start_ms + int(round(frag.shape[0] * 1000.0 / sample_rate))
            self._timeline_ms = end_ms
            frame = AudioFrame(
                start_ms=start_ms,
                end_ms=end_ms,
                sample_rate=sample_rate,
                samples=frag.astype(np.float32, copy=True),
            )
            self.raw_cache.append(frame.samples.copy())
            try:
                self.frame_queue.put_nowait(frame)
            except Exception:
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(frame)
                except Exception:
                    pass
            pos += block

    def _stitch_chunk(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        chunk = np.asarray(audio, dtype=np.float32).copy()
        if chunk.size == 0:
            return chunk

        # 与上一块做短交叉淡化，降低 start/stop 的点击音与“滴滴”伪影。
        overlap = int(sample_rate * self._crossfade_ms / 1000)
        if self._tail is not None and overlap > 0 and chunk.size > overlap and self._tail.size == overlap:
            ramp = np.linspace(0.0, 1.0, overlap, endpoint=False, dtype=np.float32)
            head = self._tail * (1.0 - ramp) + chunk[:overlap] * ramp
            chunk = np.concatenate((head, chunk[overlap:]), dtype=np.float32)

        if overlap > 0 and chunk.size > overlap:
            self._tail = chunk[-overlap:].copy()
        else:
            self._tail = None

        # 每块首尾轻微淡入淡出，进一步抑制硬边界爆音。
        fade = int(sample_rate * self._fade_ms / 1000)
        if fade > 0 and chunk.size > fade * 2:
            in_ramp = np.linspace(0.0, 1.0, fade, endpoint=False, dtype=np.float32)
            out_ramp = np.linspace(1.0, 0.0, fade, endpoint=False, dtype=np.float32)
            chunk[:fade] *= in_ramp
            chunk[-fade:] *= out_ramp
        return chunk.astype(np.float32, copy=False)

    def _emit_level(self, value: int) -> None:
        if self.level_callback is None:
            return
        try:
            self.level_callback(max(0, min(100, int(value))))
        except Exception:
            pass


def _to_mono(block: np.ndarray) -> np.ndarray:
    if block.ndim == 1:
        return block.astype(np.float32, copy=False)
    return block.mean(axis=1).astype(np.float32, copy=False)


def _estimate_level(mono: np.ndarray) -> int:
    if mono.size == 0:
        return 0
    rms = float(np.sqrt(np.mean(np.square(mono))))
    return int(max(0.0, min(100.0, rms * 350.0)))


def _read_wav_as_float(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf

        arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
        audio = np.asarray(arr, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32, copy=False), int(sr)
    except Exception:
        pass

    import wave

    with wave.open(str(path), "rb") as wf:
        sr = int(wf.getframerate())
        channels = int(wf.getnchannels())
        sw = int(wf.getsampwidth())
        frames = wf.readframes(wf.getnframes())

    if sw == 2:
        arr = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        arr = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        return np.empty((0,), dtype=np.float32), sr
    if channels > 1:
        arr = arr.reshape(-1, channels).mean(axis=1)
    return arr.astype(np.float32, copy=False), sr
