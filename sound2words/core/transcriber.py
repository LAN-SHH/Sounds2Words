from __future__ import annotations

import math
import os
import re
import site
import sys
import tempfile
import threading
import time
import urllib.request
from ctypes import WinDLL
from glob import glob
from pathlib import Path
from queue import Empty, Queue
from typing import Callable

import numpy as np
import sounddevice as sd
from PySide6.QtCore import QThread, Signal

from sound2words.core.audio_capture import (
    AudioFrame,
    ProcessLoopbackAudioCapture,
    SoundDeviceAudioCapture,
)
from sound2words.core.models import TranscriptSegment
from sound2words.core.transcriber_worker import TranscriberWorker
from sound2words.core.vad_segmenter import FixedWindowSegmenter, SpeechSegment, VADSegmenter


class _SoundDeviceRecorder:
    def __init__(
        self,
        device_index: int,
        sample_rate: int,
        channels: int,
        loopback: bool,
    ) -> None:
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.channels = channels
        self.loopback = loopback
        self._stream: sd.InputStream | None = None

    def __enter__(self) -> "_SoundDeviceRecorder":
        base_kwargs = {
            "samplerate": self.sample_rate,
            "device": self.device_index,
            "channels": self.channels,
            "dtype": "float32",
        }
        self._stream = self._open_stream_compat(base_kwargs)
        self._stream.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def record(self, numframes: int) -> np.ndarray:
        if self._stream is None:
            return np.empty((0,), dtype=np.float32)
        data, _overflowed = self._stream.read(numframes)
        return np.asarray(data, dtype=np.float32)

    def _open_stream_compat(self, base_kwargs: dict) -> sd.InputStream:
        errors: list[str] = []
        if not self.loopback:
            return sd.InputStream(**base_kwargs)

        # sounddevice / PortAudio 在不同版本下对 WASAPI loopback 的参数位置不同。
        def _candidate_kwargs(case_id: int) -> dict:
            if case_id == 1:
                return {**base_kwargs, "extra_settings": sd.WasapiSettings(loopback=True)}
            if case_id == 2:
                return {**base_kwargs, "extra_settings": sd.WasapiSettings(), "loopback": True}
            if case_id == 3:
                return {**base_kwargs, "loopback": True}
            return {**base_kwargs, "extra_settings": sd.WasapiSettings()}

        for case_id in (1, 2, 3, 4):
            try:
                kwargs = _candidate_kwargs(case_id)
                return sd.InputStream(**kwargs)
            except TypeError as exc:
                errors.append(str(exc))
                continue
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))
                continue

        detail = " | ".join(errors[:4]) if errors else "unknown"
        raise RuntimeError(
            "当前 sounddevice 版本不支持 WASAPI loopback 参数组合。"
            f"请升级 sounddevice 后重试。底层信息: {detail}"
        )


class RealtimeTranscriber(QThread):
    segment_ready = Signal(object)
    status_changed = Signal(str)
    error_occurred = Signal(str)
    level_changed = Signal(int)
    model_loading_changed = Signal(int, str)
    FRAME_QUEUE_MAX = 900
    SEGMENT_QUEUE_MAX = 24
    RESULT_QUEUE_MAX = 128

    def __init__(
        self,
        source_type: str,
        file_path: str = "",
        device_name: str = "",
        target_pid: int = 0,
        target_process_name: str = "",
        target_label: str = "",
        model_download_root: str = "",
        language: str = "zh",
        model_size: str = "small",
        sample_rate: int = 16000,
        chunk_seconds: float = 3.0,
    ) -> None:
        super().__init__()
        self.source_type = source_type
        self.file_path = file_path
        self.device_name = device_name
        self.target_pid = target_pid
        self.target_process_name = target_process_name
        self.target_label = target_label
        self.model_download_root = model_download_root
        self.language = language
        self.model_size = model_size
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self._stop_event = threading.Event()
        self._timeline_ms = 0
        self._force_cpu = os.environ.get("SOUND2WORDS_FORCE_CPU", "0") == "1"
        self._text_context = ""
        self._gpu_probe: dict | None = None

    def stop(self) -> None:
        self._stop_event.set()

    @classmethod
    def list_input_devices(cls, source_type: str) -> list[str]:
        devices = cls._list_wasapi_devices()
        if source_type == "system":
            names = [d["name"] for d in devices if int(d.get("max_output_channels", 0)) > 0]
            return sorted(set(names))
        if source_type == "mic":
            names = [d["name"] for d in devices if int(d.get("max_input_channels", 0)) > 0]
            return sorted(set(names))
        return []

    def run(self) -> None:
        try:
            self.model_loading_changed.emit(5, "准备加载模型")
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
            os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "1200")
            os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")
            self._inject_gpu_dll_paths()

            token = os.environ.get("HF_TOKEN", "")
            if token and not token.isascii():
                self.error_occurred.emit(
                    "识别线程异常: HF_TOKEN 包含非 ASCII 字符。\n"
                    "请将 HF_TOKEN 改为真实英文 token，或先删除该环境变量后重试。"
                )
                return

            if self._need_download_model() and not self._can_reach_huggingface():
                self.error_occurred.emit(
                    "识别线程异常: 无法连接 Hugging Face 下载模型。\n"
                    "请先设置镜像后重试:\n"
                    "set HF_ENDPOINT=https://hf-mirror.com\n"
                    "然后重新运行程序。"
                )
                return

            from faster_whisper import WhisperModel
            import faster_whisper
            import ctranslate2

            self._gpu_probe = self._run_gpu_self_check(
                faster_whisper_version=str(faster_whisper.__version__),
                ctranslate2_version=str(ctranslate2.__version__),
            )
            self._emit_gpu_probe_log(self._gpu_probe)

            self.model_loading_changed.emit(25, "正在初始化模型")
            self.status_changed.emit("状态: 正在加载识别模型")
            model = self._load_model_with_fallback(WhisperModel)
            model = self._ensure_model_runtime(model, WhisperModel)
            self.model_loading_changed.emit(100, "模型加载完成")
            self.status_changed.emit("状态: 识别模型已就绪")

            if self.source_type == "file":
                self._transcribe_file(model)
                self.level_changed.emit(0)
                self.status_changed.emit("状态: 文件转写完成，请点击结束保存")
                return

            self._stream_transcribe_pipeline(model)
        except ModuleNotFoundError as exc:
            if getattr(exc, "name", "") == "faster_whisper":
                self.error_occurred.emit(
                    "识别线程异常: 未找到 faster_whisper\n"
                    f"当前解释器: {sys.executable}\n"
                    "请在该解释器下执行: python -m pip install faster-whisper"
                )
                return
            self.error_occurred.emit(
                f"识别线程异常: {exc}\n"
                f"当前解释器: {sys.executable}\n"
                "请在该解释器下安装缺失模块后重试。"
            )
        except Exception as exc:  # noqa: BLE001
            self.error_occurred.emit(f"识别线程异常: {exc}")

    def _stream_transcribe_pipeline(self, model: object) -> None:
        frame_queue: Queue[AudioFrame | None] = Queue(maxsize=self.FRAME_QUEUE_MAX)
        segment_queue: Queue[SpeechSegment | None] = Queue(maxsize=self.SEGMENT_QUEUE_MAX)
        result_queue: Queue[TranscriptSegment] = Queue(maxsize=self.RESULT_QUEUE_MAX)
        debug_mode = os.environ.get("SOUND2WORDS_DEBUG_FIXED5", "0") == "1"
        trace_audio = os.environ.get("SOUND2WORDS_TRACE_AUDIO", "0") == "1"
        debug_dir = self._debug_output_dir()

        capture_thread, capture_cleanup = self._build_capture_thread(frame_queue)
        if debug_mode:
            segmenter = FixedWindowSegmenter(
                frame_queue=frame_queue,
                segment_queue=segment_queue,
                stop_event=self._stop_event,
                window_ms=5000,
                min_segment_ms=300,
            )
            self.status_changed.emit("状态: 调试模式已启用（固定5秒切段，绕过VAD）")
        else:
            segmenter = VADSegmenter(
                frame_queue=frame_queue,
                segment_queue=segment_queue,
                stop_event=self._stop_event,
                silence_end_ms=800,
                min_segment_ms=400,
                max_segment_ms=25000,
                pad_pre_ms=220,
                pad_post_ms=320,
                vad_rms_threshold=0.03,
                queue_overflow_logger=lambda msg: self.status_changed.emit(f"状态: {msg}"),
            )
        worker = TranscriberWorker(
            model=model,
            segment_queue=segment_queue,
            result_queue=result_queue,
            stop_event=self._stop_event,
            language=self.language,
            on_error=lambda msg: self.error_occurred.emit(f"识别线程异常: {msg}"),
            target_sample_rate=self.sample_rate,
            debug_dir=debug_dir,
            trace_audio=trace_audio,
        )

        capture_thread.start()
        segmenter.start()
        worker.start()
        self.status_changed.emit("状态: 转写中")

        try:
            while True:
                self._drain_result_queue(result_queue)
                if self._stop_event.is_set():
                    break
                # 采集结束且无积压时退出（例如设备异常自动结束）。
                if (not capture_thread.is_alive()) and frame_queue.empty() and segment_queue.empty():
                    break
                time.sleep(0.03)
        finally:
            self._stop_event.set()
            try:
                frame_queue.put_nowait(None)
            except Exception:
                pass
            try:
                segment_queue.put_nowait(None)
            except Exception:
                pass

            capture_thread.join(timeout=2.0)
            segmenter.join(timeout=2.0)
            worker.join(timeout=3.0)
            capture_cleanup()
            self._drain_result_queue(result_queue)
            self.level_changed.emit(0)

    def _debug_output_dir(self) -> str:
        if self.model_download_root:
            root = Path(self.model_download_root).resolve().parent
            out = root / "debug_segments"
        else:
            out = (Path.cwd() / "data" / "debug_segments").resolve()
        out.mkdir(parents=True, exist_ok=True)
        return str(out)

    def _drain_result_queue(self, result_queue: Queue[TranscriptSegment]) -> None:
        while True:
            try:
                item = result_queue.get_nowait()
            except Empty:
                break
            self.segment_ready.emit(item)

    def _build_capture_thread(
        self,
        frame_queue: Queue[AudioFrame | None],
    ) -> tuple[threading.Thread, Callable[[], None]]:
        if self.source_type == "system":
            try:
                from process_audio_capture import PacCaptureMode, ProcessAudioCapture
            except ModuleNotFoundError:
                ProcessAudioCapture = None
            if ProcessAudioCapture is not None:
                target_pid = self.target_pid if self.target_pid > 0 else os.getpid()
                mode = PacCaptureMode.INCLUDE if self.target_pid > 0 else PacCaptureMode.EXCLUDE
                if self.target_pid > 0:
                    resolved = self._resolve_active_target_pid(
                        ProcessAudioCapture,
                        self.target_pid,
                        self.target_process_name,
                        self.target_label,
                    )
                    if resolved != self.target_pid:
                        self.status_changed.emit(f"状态: 自动切换到正在发声的PID ({resolved})")
                    target_pid = resolved
                    self.status_changed.emit(f"状态: PID直采中 ({target_pid})")
                else:
                    self.status_changed.emit("状态: 系统全量采集中")

                return (
                    ProcessLoopbackAudioCapture(
                        pid=target_pid,
                        mode=mode,
                        frame_queue=frame_queue,
                        stop_event=self._stop_event,
                        level_callback=self.level_changed.emit,
                        capture_window_ms=1600,
                    ),
                    (lambda: None),
                )

        recorder = self._create_recorder()
        recorder.__enter__()
        thread = SoundDeviceAudioCapture(
            recorder=recorder,
            frame_queue=frame_queue,
            stop_event=self._stop_event,
            level_callback=self.level_changed.emit,
            frame_ms=20,
        )
        return thread, (lambda: recorder.__exit__(None, None, None))

    def _stream_transcribe_system(self, model: object) -> None:
        try:
            from process_audio_capture import PacCaptureMode, ProcessAudioCapture
        except ModuleNotFoundError as exc:
            # 无 process-audio-capture 时再退回 sounddevice。
            if self.target_pid > 0:
                self.status_changed.emit(f"状态: PID直采中 ({self.target_pid})")
            else:
                self.status_changed.emit("状态: 系统音频采集中")
            with self._create_recorder() as recorder:
                self._stream_transcribe(model, recorder)
            return

        target_pid = self.target_pid if self.target_pid > 0 else os.getpid()
        mode = PacCaptureMode.INCLUDE if self.target_pid > 0 else PacCaptureMode.EXCLUDE
        if self.target_pid > 0:
            resolved = self._resolve_active_target_pid(
                ProcessAudioCapture,
                self.target_pid,
                self.target_process_name,
                self.target_label,
            )
            if resolved != self.target_pid:
                self.status_changed.emit(f"状态: 自动切换到正在发声的PID ({resolved})")
            target_pid = resolved
            self.status_changed.emit(f"状态: PID直采中 ({target_pid})")
        else:
            self.status_changed.emit("状态: 系统全量采集中")

        seg_idx = 0
        capture_span = max(1.0, self.chunk_seconds)
        empty_rounds = 0
        while not self._stop_event.is_set():
            seg_idx += 1
            path = Path(tempfile.gettempdir()) / f"s2w_sys_{target_pid}_{os.getpid()}_{seg_idx}.wav"

            def _on_level(db: float) -> None:
                normalized = int(max(0.0, min(100.0, ((db + 60.0) / 60.0) * 100.0)))
                self.level_changed.emit(normalized)

            with ProcessAudioCapture(
                pid=target_pid,
                mode=mode,
                output_path=str(path),
                level_callback=_on_level,
            ) as cap:
                cap.start()
                start = time.time()
                while not self._stop_event.is_set() and (time.time() - start) < capture_span:
                    try:
                        db = float(cap.level_db)
                        normalized = int(max(0.0, min(100.0, ((db + 60.0) / 60.0) * 100.0)))
                        self.level_changed.emit(normalized)
                    except Exception:
                        pass
                    time.sleep(0.1)
                cap.stop()

            if path.exists() and path.stat().st_size > 44:
                audio, sr = self._read_wav_as_float(path)
                if audio.size > 0:
                    empty_rounds = 0
                    if sr != self.sample_rate:
                        audio = self._resample_mono(audio, sr, self.sample_rate)
                    self._transcribe_chunk(model, audio)
                else:
                    empty_rounds += 1
            else:
                empty_rounds += 1

            if empty_rounds >= 5:
                if self.target_pid > 0:
                    resolved = self._resolve_active_target_pid(
                        ProcessAudioCapture,
                        target_pid,
                        self.target_process_name,
                        self.target_label,
                    )
                    if resolved != target_pid:
                        target_pid = resolved
                        self.status_changed.emit(f"状态: 检测到新的发声PID，已切换 ({target_pid})")
                    else:
                        self.status_changed.emit(f"状态: 目标进程暂未输出音频 (PID {target_pid})")
                else:
                    self.status_changed.emit("状态: 当前未检测到系统音频输出")
                empty_rounds = 0

            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass

        self.level_changed.emit(0)

    @staticmethod
    def _resolve_active_target_pid(
        process_audio_capture_cls: type,
        desired_pid: int,
        desired_process_name: str,
        desired_label: str,
    ) -> int:
        try:
            rows = list(process_audio_capture_cls.enumerate_audio_processes())
        except Exception:
            return desired_pid

        active_pids = {int(r.pid) for r in rows}
        if desired_pid in active_pids:
            return desired_pid

        target_name = (desired_process_name or "").strip().lower()
        target_label = (desired_label or "").strip().lower()
        if target_name:
            for row in rows:
                row_name = str(getattr(row, "name", "")).strip().lower()
                if row_name == target_name:
                    return int(row.pid)
            for row in rows:
                row_name = str(getattr(row, "name", "")).strip().lower()
                if target_name in row_name or row_name in target_name:
                    return int(row.pid)

        hints = RealtimeTranscriber._build_match_hints(target_name, target_label)
        if hints:
            scored: list[tuple[int, int]] = []
            for row in rows:
                row_pid = int(getattr(row, "pid", 0))
                row_name = str(getattr(row, "name", "")).strip().lower()
                row_title = str(getattr(row, "window_title", "")).strip().lower()
                haystack = f"{row_name} {row_title}"
                score = 0
                for hint in hints:
                    if hint and hint in haystack:
                        score += max(2, len(hint))
                if score > 0:
                    scored.append((score, row_pid))
            if scored:
                scored.sort(key=lambda item: item[0], reverse=True)
                return scored[0][1]

        return desired_pid

    @staticmethod
    def _build_match_hints(target_name: str, target_label: str) -> list[str]:
        raw = f"{target_name} {target_label}".lower()
        tokens = re.findall(r"[a-z0-9\u4e00-\u9fff]+", raw)
        blacklist = {
            "pid",
            "helper",
            "assist",
            "assistant",
            "service",
            "aux",
            "auxiliary",
            "进程",
            "辅助",
            "服务",
            "音源",
        }
        hints: list[str] = []
        for token in tokens:
            if len(token) <= 1:
                continue
            if token in blacklist:
                continue
            hints.append(token)
        if "qq音乐" in raw or "qqmusic" in raw:
            hints.extend(["qqmusic", "qq音乐"])
        unique: list[str] = []
        for hint in hints:
            if hint not in unique:
                unique.append(hint)
        return unique

    def _transcribe_file(self, model: object) -> None:
        if not self.file_path:
            raise RuntimeError("文件模式未选择输入文件")

        self.status_changed.emit("状态: 正在处理文件")
        segments, _ = model.transcribe(
            self.file_path,
            language=self.language,
            beam_size=5,
            best_of=5,
            temperature=0.0,
            vad_filter=True,
            condition_on_previous_text=True,
            without_timestamps=False,
            initial_prompt="以下是中文语音转写文本。",
        )

        for segment in segments:
            if self._stop_event.is_set():
                break

            text = segment.text.strip()
            if not text:
                continue

            start_ms = int(segment.start * 1000)
            end_ms = int(segment.end * 1000)
            if end_ms <= start_ms:
                end_ms = start_ms + 300

            confidence = self._score_from_logprob(getattr(segment, "avg_logprob", None))
            self.segment_ready.emit(
                TranscriptSegment(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    text=text,
                    confidence=confidence,
                )
            )
            self._update_context(text)

    def _stream_transcribe(self, model: object, recorder: _SoundDeviceRecorder) -> None:
        capture_rate = int(recorder.sample_rate)
        chunk_frames = int(capture_rate * self.chunk_seconds)
        block_frames = max(2048, chunk_frames // 4)
        buffer = np.empty((0,), dtype=np.float32)

        while not self._stop_event.is_set():
            block = recorder.record(numframes=block_frames)
            if block.size == 0:
                self.level_changed.emit(0)
                continue

            mono = self._to_mono(block)
            self.level_changed.emit(self._estimate_level(mono))
            if capture_rate != self.sample_rate:
                mono = self._resample_mono(mono, capture_rate, self.sample_rate)
            buffer = np.concatenate((buffer, mono))

            model_chunk_frames = int(self.sample_rate * self.chunk_seconds)
            while buffer.shape[0] >= model_chunk_frames:
                chunk = buffer[:model_chunk_frames]
                buffer = buffer[model_chunk_frames:]
                self._transcribe_chunk(model, chunk)

    def _transcribe_chunk(self, model: object, audio_chunk: np.ndarray) -> None:
        segments, _ = model.transcribe(
            audio_chunk,
            language=self.language,
            beam_size=5,
            best_of=5,
            temperature=0.0,
            vad_filter=True,
            condition_on_previous_text=True,
            without_timestamps=False,
            initial_prompt=self._text_context or "以下是中文语音转写文本。",
        )
        chunk_ms = int(self.chunk_seconds * 1000)
        base_ms = self._timeline_ms

        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            start_ms = base_ms + int(segment.start * 1000)
            end_ms = base_ms + int(segment.end * 1000)
            if end_ms <= start_ms:
                end_ms = start_ms + 300

            confidence = self._score_from_logprob(getattr(segment, "avg_logprob", None))
            self.segment_ready.emit(
                TranscriptSegment(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    text=text,
                    confidence=confidence,
                )
            )
            self._update_context(text)

        self._timeline_ms += chunk_ms

    @staticmethod
    def _to_mono(block: np.ndarray) -> np.ndarray:
        if block.ndim == 1:
            mono = block
        else:
            mono = block.mean(axis=1)
        return mono.astype(np.float32, copy=False)

    @staticmethod
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

    @staticmethod
    def _resample_mono(mono: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        if mono.size == 0 or src_rate == dst_rate:
            return mono
        src_len = mono.shape[0]
        dst_len = max(1, int(round(src_len * dst_rate / src_rate)))
        src_x = np.linspace(0.0, 1.0, num=src_len, endpoint=False)
        dst_x = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
        return np.interp(dst_x, src_x, mono).astype(np.float32, copy=False)

    @staticmethod
    def _estimate_level(mono: np.ndarray) -> int:
        if mono.size == 0:
            return 0
        rms = float(np.sqrt(np.mean(np.square(mono))))
        level = int(max(0.0, min(100.0, rms * 350.0)))
        return level

    @staticmethod
    def _score_from_logprob(value: float | None) -> float:
        if value is None:
            return 0.9
        probability = math.exp(value)
        return round(float(max(0.0, min(1.0, probability))), 2)

    def _update_context(self, text: str) -> None:
        combined = (self._text_context + " " + text).strip()
        self._text_context = combined[-160:]

    def _create_recorder(self) -> _SoundDeviceRecorder:
        if self.source_type == "system":
            output_device = self._match_wasapi_device(self.device_name, need_output=True)
            if output_device is None:
                raise RuntimeError("未找到可用扬声器设备，无法采集系统音频")

            # 兼容旧版 sounddevice: 优先使用 WASAPI 的 loopback 输入设备，无需 loopback 参数。
            loopback_input = self._match_wasapi_loopback_input(self.device_name, output_device)
            if loopback_input is not None:
                sample_rate = int(
                    loopback_input.get("default_samplerate")
                    or output_device.get("default_samplerate")
                    or 48000
                )
                channels = max(1, min(2, int(loopback_input.get("max_input_channels") or 2)))
                return _SoundDeviceRecorder(
                    device_index=int(loopback_input["index"]),
                    sample_rate=sample_rate,
                    channels=channels,
                    loopback=False,
                )

            # 没有可用回采输入设备时，再尝试新版 sounddevice 的 loopback 参数模式。
            sample_rate = int(output_device.get("default_samplerate") or 48000)
            channels = max(1, min(2, int(output_device.get("max_output_channels") or 2)))
            try:
                return _SoundDeviceRecorder(
                    device_index=int(output_device["index"]),
                    sample_rate=sample_rate,
                    channels=channels,
                    loopback=True,
                )
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "当前环境未找到可用的系统回采输入设备，且 sounddevice 不支持 WASAPI loopback 参数。\n"
                    "建议改用“选择音源 -> 指定应用(PID)”模式，或安装支持 loopback 的 sounddevice 版本。"
                ) from exc

        device = self._match_wasapi_device(self.device_name, need_output=False)
        if device is None:
            raise RuntimeError("未找到可用麦克风设备")
        sample_rate = int(device.get("default_samplerate") or self.sample_rate)
        channels = max(1, min(2, int(device.get("max_input_channels") or 1)))
        return _SoundDeviceRecorder(
            device_index=int(device["index"]),
            sample_rate=sample_rate,
            channels=channels,
            loopback=False,
        )

    @staticmethod
    def _list_wasapi_devices() -> list[dict]:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        wasapi_ids = {
            i for i, h in enumerate(hostapis) if "wasapi" in str(h.get("name", "")).lower()
        }
        result: list[dict] = []
        for idx, dev in enumerate(devices):
            host_id = int(dev.get("hostapi", -1))
            if host_id not in wasapi_ids:
                continue
            row = dict(dev)
            row["index"] = idx
            result.append(row)
        return result

    @classmethod
    def _match_wasapi_device(cls, name: str, need_output: bool) -> dict | None:
        devices = cls._list_wasapi_devices()
        if not devices:
            return None

        exact = None
        for dev in devices:
            if need_output and int(dev.get("max_output_channels", 0)) <= 0:
                continue
            if (not need_output) and int(dev.get("max_input_channels", 0)) <= 0:
                continue
            if name and dev.get("name") == name:
                exact = dev
                break

        if exact is not None:
            return exact

        for dev in devices:
            if need_output and int(dev.get("max_output_channels", 0)) > 0:
                return dev
            if (not need_output) and int(dev.get("max_input_channels", 0)) > 0:
                return dev
        return None

    @classmethod
    def _match_wasapi_loopback_input(cls, selected_name: str, output_device: dict | None) -> dict | None:
        devices = cls._list_wasapi_devices()
        if not devices:
            return None

        output_name = str((output_device or {}).get("name", "")).strip().lower()
        selected = (selected_name or "").strip().lower()
        tokens = [t for t in [selected, output_name] if t]

        candidates = []
        for dev in devices:
            if int(dev.get("max_input_channels", 0)) <= 0:
                continue
            name = str(dev.get("name", "")).strip()
            low = name.lower()
            candidates.append(dev)

        if not candidates:
            return None

        keyword_bonus = (
            "loopback",
            "stereo mix",
            "what u hear",
            "wave out",
            "monitor",
        )
        scored: list[tuple[int, dict]] = []
        base_words = []
        if output_name:
            base_words = [w for w in output_name.replace("(", " ").replace(")", " ").split() if w]

        for dev in candidates:
            low = str(dev.get("name", "")).lower()
            score = 0
            if any(token and token in low for token in tokens):
                score += 8
            score += sum(1 for word in base_words if word.lower() in low)
            if any(key in low for key in keyword_bonus):
                score += 6
            if "microphone" in low or "麦克风" in low:
                score -= 2
            scored.append((score, dev))

        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_dev = scored[0]
        if best_score <= 0:
            return None
        return best_dev

    def _need_download_model(self) -> bool:
        if not self.model_download_root:
            return True
        root = Path(self.model_download_root)
        marker = root / f"models--Systran--faster-whisper-{self.model_size}"
        return not marker.exists()

    def _load_model_with_fallback(self, whisper_model_cls: type) -> object:
        if self._force_cpu:
            self.status_changed.emit("状态: 使用CPU模式加载模型")
            return whisper_model_cls(
                self.model_size,
                device="cpu",
                compute_type="int8",
                download_root=self.model_download_root or None,
            )

        if self._gpu_probe is not None and not bool(self._gpu_probe.get("runtime_ok", False)):
            missing = ", ".join(self._gpu_probe.get("missing_dlls", [])) or "unknown"
            self.status_changed.emit(f"状态: GPU运行时缺失({missing})，回退CPU")
            return whisper_model_cls(
                self.model_size,
                device="cpu",
                compute_type="int8",
                download_root=self.model_download_root or None,
            )

        if not self._cuda_ready():
            self.status_changed.emit("状态: CUDA环境不可用，回退CPU模式")
            return whisper_model_cls(
                self.model_size,
                device="cpu",
                compute_type="int8",
                download_root=self.model_download_root or None,
            )

        # 默认优先 CUDA；先尝试更稳的 int8_float16，失败后再试 float16。
        try:
            self.status_changed.emit("状态: 使用CUDA模式加载模型 (float16)")
            return whisper_model_cls(
                self.model_size,
                device="cuda",
                compute_type="float16",
                download_root=self.model_download_root or None,
            )
        except Exception:
            try:
                self.status_changed.emit("状态: CUDA降级重试 (float32)")
                return whisper_model_cls(
                    self.model_size,
                    device="cuda",
                    compute_type="float32",
                    download_root=self.model_download_root or None,
                )
            except Exception:
                self.status_changed.emit("状态: CUDA加载失败，回退CPU模式")
                return whisper_model_cls(
                    self.model_size,
                    device="cpu",
                    compute_type="int8",
                    download_root=self.model_download_root or None,
                )

    def _ensure_model_runtime(self, model: object, whisper_model_cls: type) -> object:
        """运行一次极小推理，提前暴露 CUDA 运行时缺失并自动回退 CPU。"""
        try:
            warmup = np.zeros((max(1600, self.sample_rate // 5),), dtype=np.float32)
            segments, _ = model.transcribe(
                warmup,
                language=self.language,
                beam_size=1,
                best_of=1,
                temperature=0.0,
                vad_filter=False,
                condition_on_previous_text=False,
                without_timestamps=True,
            )
            # 消费生成器，真正触发 encode/decode。
            for _ in segments:
                break
            return model
        except Exception as exc:  # noqa: BLE001
            message = str(exc).lower()
            cuda_related = (
                "cublas64_12.dll" in message
                or "cuda" in message
                or "cudnn" in message
            )
            if not cuda_related:
                raise
            missing_dll = self._extract_missing_dll(str(exc))
            if missing_dll:
                self.status_changed.emit(f"状态: CUDA运行时缺失({missing_dll})，自动回退CPU")
            else:
                self.status_changed.emit("状态: CUDA运行时缺失，自动回退CPU")
            return whisper_model_cls(
                self.model_size,
                device="cpu",
                compute_type="int8",
                download_root=self.model_download_root or None,
            )

    @staticmethod
    def _extract_missing_dll(message: str) -> str:
        m = re.search(r"([A-Za-z0-9_]+\.dll)", message)
        return m.group(1) if m else ""

    def _run_gpu_self_check(self, faster_whisper_version: str, ctranslate2_version: str) -> dict:
        self._inject_gpu_dll_paths()
        info: dict = {
            "faster_whisper_version": faster_whisper_version,
            "ctranslate2_version": ctranslate2_version,
            "cuda_device_count": 0,
            "runtime_ok": False,
            "missing_dlls": [],
            "expected_runtime": "",
            "recommendation_a": "",
            "recommendation_b": "",
        }
        if self._force_cpu:
            info["expected_runtime"] = "CPU forced by SOUND2WORDS_FORCE_CPU=1"
            return info

        try:
            import ctranslate2

            info["cuda_device_count"] = int(ctranslate2.get_cuda_device_count())
        except Exception as exc:  # noqa: BLE001
            info["expected_runtime"] = f"ctranslate2 unavailable: {exc}"
            return info

        ct_tuple = self._parse_semver(ctranslate2_version)
        expected_dlls: list[str] = []
        if ct_tuple >= (4, 5, 0):
            info["expected_runtime"] = "CUDA 12 + cuDNN 9"
            expected_dlls = ["cublas64_12.dll", "cudnn64_9.dll"]
        elif ct_tuple >= (4, 0, 0):
            info["expected_runtime"] = "CUDA 12 + cuDNN 8 (recommended ctranslate2==4.4.0)"
            expected_dlls = ["cublas64_12.dll", "cudnn64_8.dll"]
        else:
            info["expected_runtime"] = "CUDA 11 + cuDNN 8 (recommended ctranslate2==3.24.0)"
            expected_dlls = ["cublas64_11.dll", "cudnn64_8.dll"]

        missing = [dll for dll in expected_dlls if not self._dll_available(dll)]
        info["missing_dlls"] = missing
        info["runtime_ok"] = info["cuda_device_count"] > 0 and not missing

        info["recommendation_a"] = (
            "方案A(推荐): 若当前非完整 CUDA12+cuDNN9, 请降级 ctranslate2:\n"
            "- CUDA11+cuDNN8 -> ctranslate2==3.24.0\n"
            "- CUDA12+cuDNN8 -> ctranslate2==4.4.0"
        )
        info["recommendation_b"] = (
            "方案B: 保持 ctranslate2>=4.5, 补齐 CUDA12+cuDNN9 运行时并确保 PATH 可见 DLL。"
        )
        return info

    def _emit_gpu_probe_log(self, info: dict) -> None:
        missing = info.get("missing_dlls", [])
        msg = (
            f"[GPU-CHECK] faster-whisper={info.get('faster_whisper_version')}, "
            f"ctranslate2={info.get('ctranslate2_version')}, "
            f"cuda_device_count={info.get('cuda_device_count')}, "
            f"expected={info.get('expected_runtime')}"
        )
        print(msg, flush=True)
        self.status_changed.emit(f"状态: {msg}")
        if missing:
            missing_msg = f"[GPU-CHECK] missing_dlls={', '.join(missing)}"
            print(missing_msg, flush=True)
            self.status_changed.emit(f"状态: {missing_msg}")
            print(info.get("recommendation_a", ""), flush=True)
            print(info.get("recommendation_b", ""), flush=True)

    @staticmethod
    def _parse_semver(text: str) -> tuple[int, int, int]:
        nums = re.findall(r"\d+", text)
        parts = [int(n) for n in nums[:3]]
        while len(parts) < 3:
            parts.append(0)
        return tuple(parts[:3])

    @staticmethod
    def _dll_available(name: str) -> bool:
        try:
            WinDLL(name)
            return True
        except Exception:
            return False

    @staticmethod
    def _inject_gpu_dll_paths() -> None:
        candidates: list[str] = []
        # Conda env runtime location
        prefix = os.environ.get("CONDA_PREFIX", "")
        if prefix:
            candidates.append(str(Path(prefix) / "Library" / "bin"))
        # Python env runtime location fallback
        env_root = Path(sys.executable).resolve().parent.parent
        candidates.append(str(env_root / "Library" / "bin"))
        # NVIDIA pip wheels location
        for root in site.getsitepackages():
            candidates.extend(glob(str(Path(root) / "nvidia" / "**" / "bin"), recursive=True))

        seen: set[str] = set()
        for path in candidates:
            p = str(Path(path))
            if p in seen:
                continue
            seen.add(p)
            if not Path(p).exists():
                continue
            try:
                os.add_dll_directory(p)
            except Exception:
                pass
            if p not in os.environ.get("PATH", ""):
                os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")

    @staticmethod
    def _cuda_ready() -> bool:
        if os.environ.get("SOUND2WORDS_DISABLE_CUDA", "0") == "1":
            return False
        try:
            import ctranslate2

            count = int(ctranslate2.get_cuda_device_count())
            return count > 0
        except Exception:
            return False

    @staticmethod
    def _can_reach_huggingface() -> bool:
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
        test_url = f"{endpoint}/"
        try:
            with urllib.request.urlopen(test_url, timeout=5):
                return True
        except Exception:
            return False
