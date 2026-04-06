from __future__ import annotations

import math
import os
import sys
import threading
import urllib.request
from contextlib import AbstractContextManager
from pathlib import Path

import numpy as np
import soundcard as sc
from PySide6.QtCore import QThread, Signal

from sound2words.core.models import TranscriptSegment


class RealtimeTranscriber(QThread):
    segment_ready = Signal(object)
    status_changed = Signal(str)
    error_occurred = Signal(str)
    level_changed = Signal(int)
    model_loading_changed = Signal(int, str)

    def __init__(
        self,
        source_type: str,
        file_path: str = "",
        device_name: str = "",
        model_download_root: str = "",
        language: str = "zh",
        model_size: str = "tiny",
        sample_rate: int = 16000,
        chunk_seconds: float = 2.0,
    ) -> None:
        super().__init__()
        self.source_type = source_type
        self.file_path = file_path
        self.device_name = device_name
        self.model_download_root = model_download_root
        self.language = language
        self.model_size = model_size
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self._stop_event = threading.Event()
        self._timeline_ms = 0
        self._force_cpu = True

    def stop(self) -> None:
        self._stop_event.set()

    @staticmethod
    def list_input_devices(source_type: str) -> list[str]:
        if source_type == "system":
            return [speaker.name for speaker in sc.all_speakers()]
        if source_type == "mic":
            return [mic.name for mic in sc.all_microphones(include_loopback=False)]
        return []

    def run(self) -> None:
        try:
            major_version = int(np.__version__.split(".", 1)[0])
            if major_version >= 2 and self.source_type in {"mic", "system"}:
                self.error_occurred.emit(
                    "识别线程异常: 当前 numpy 与 soundcard 不兼容（检测到 numpy 2.x）。\n"
                    "请执行: python -m pip install \"numpy<2\" --upgrade"
                )
                return
            self.model_loading_changed.emit(5, "准备加载模型")
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
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
                    "然后重新运行程序。\n"
                    "如有 Token 也可设置: set HF_TOKEN=你的token"
                )
                return
            from faster_whisper import WhisperModel

            self.model_loading_changed.emit(25, "正在初始化模型")
            self.status_changed.emit("状态: 正在加载识别模型")
            model = self._load_model_with_fallback(WhisperModel)
            self.model_loading_changed.emit(100, "模型加载完成")
            self.status_changed.emit("状态: 识别模型已就绪")

            if self.source_type == "file":
                self._transcribe_file(model)
                self.level_changed.emit(0)
                self.status_changed.emit("状态: 文件转写完成，请点击结束保存")
                return

            with self._create_recorder() as recorder:
                self.status_changed.emit("状态: 转写中")
                self._stream_transcribe(model, recorder)
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
            msg = str(exc)
            if "ascii" in msg.lower() and "encode" in msg.lower():
                self.error_occurred.emit(
                    "识别线程异常: 下载参数中包含非 ASCII 字符。\n"
                    "请重点检查 HF_TOKEN 是否为真实英文 token。"
                )
                return
            if "cublas64_12.dll" in msg.lower():
                self.error_occurred.emit(
                    "识别线程异常: 当前环境仍在尝试加载 CUDA 依赖。\n"
                    "请在当前解释器执行:\n"
                    "python -m pip uninstall -y ctranslate2 faster-whisper\n"
                    "python -m pip install ctranslate2==4.4.0 faster-whisper==1.1.1"
                )
                return
            self.error_occurred.emit(f"识别线程异常: {exc}")

    def _transcribe_file(self, model: object) -> None:
        if not self.file_path:
            raise RuntimeError("文件模式未选择输入文件")

        self.status_changed.emit("状态: 正在处理文件")
        segments, _ = model.transcribe(
            self.file_path,
            language=self.language,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=True,
            without_timestamps=False,
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

    def _stream_transcribe(self, model: object, recorder: object) -> None:
        chunk_frames = int(self.sample_rate * self.chunk_seconds)
        block_frames = max(2048, chunk_frames // 4)
        buffer = np.empty((0,), dtype=np.float32)

        while not self._stop_event.is_set():
            block = recorder.record(numframes=block_frames)
            if block.size == 0:
                self.level_changed.emit(0)
                continue

            mono = self._to_mono(block)
            self.level_changed.emit(self._estimate_level(mono))
            buffer = np.concatenate((buffer, mono))

            while buffer.shape[0] >= chunk_frames:
                chunk = buffer[:chunk_frames]
                buffer = buffer[chunk_frames:]
                self._transcribe_chunk(model, chunk)

    def _transcribe_chunk(self, model: object, audio_chunk: np.ndarray) -> None:
        segments, _ = model.transcribe(
            audio_chunk,
            language=self.language,
            beam_size=1,
            best_of=1,
            vad_filter=False,
            condition_on_previous_text=False,
            without_timestamps=False,
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

        self._timeline_ms += chunk_ms

    @staticmethod
    def _to_mono(block: np.ndarray) -> np.ndarray:
        if block.ndim == 1:
            mono = block
        else:
            mono = block.mean(axis=1)
        return mono.astype(np.float32, copy=False)

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

    def _create_recorder(self) -> AbstractContextManager:
        if self.source_type == "system":
            speaker = self._match_speaker(self.device_name) if self.device_name else sc.default_speaker()
            if speaker is None:
                raise RuntimeError("未找到可用扬声器，无法采集系统音频")
            loopback = sc.get_microphone(id=str(speaker.name), include_loopback=True)
            return loopback.recorder(samplerate=self.sample_rate, channels=1)

        microphone = self._match_microphone(self.device_name) if self.device_name else sc.default_microphone()
        if microphone is None:
            raise RuntimeError("未找到可用麦克风")
        return microphone.recorder(samplerate=self.sample_rate, channels=1)

    @staticmethod
    def _match_microphone(name: str):
        for mic in sc.all_microphones(include_loopback=False):
            if mic.name == name:
                return mic
        return None

    @staticmethod
    def _match_speaker(name: str):
        for speaker in sc.all_speakers():
            if speaker.name == name:
                return speaker
        return None

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

        try:
            return whisper_model_cls(
                self.model_size,
                device="auto",
                compute_type="int8",
                download_root=self.model_download_root or None,
            )
        except Exception as exc:  # noqa: BLE001
            message = str(exc).lower()
            cuda_related = (
                "cublas64" in message
                or "cuda" in message
                or "cudnn" in message
                or "cannot be loaded" in message
            )
            if not cuda_related:
                raise

            self.status_changed.emit("状态: 未检测到可用CUDA，自动切换CPU")
            return whisper_model_cls(
                self.model_size,
                device="cpu",
                compute_type="int8",
                download_root=self.model_download_root or None,
            )

    @staticmethod
    def _can_reach_huggingface() -> bool:
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
        test_url = f"{endpoint}/"
        try:
            with urllib.request.urlopen(test_url, timeout=5):
                return True
        except Exception:
            return False
