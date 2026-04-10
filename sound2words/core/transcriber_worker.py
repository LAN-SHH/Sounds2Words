from __future__ import annotations

import math
import os
import re
import threading
import wave
from queue import Queue
from pathlib import Path
from typing import Callable

import numpy as np

from sound2words.core.models import TranscriptSegment
from sound2words.core.vad_segmenter import SpeechSegment


class TranscriberWorker(threading.Thread):
    """消费语音段并做模型推理。"""

    def __init__(
        self,
        model: object,
        segment_queue: Queue[SpeechSegment | None],
        result_queue: Queue[TranscriptSegment],
        stop_event: threading.Event,
        language: str = "zh",
        on_error: Callable[[str], None] | None = None,
        target_sample_rate: int = 16000,
        debug_dir: str = "",
        trace_audio: bool = True,
    ) -> None:
        super().__init__(daemon=True)
        self.model = model
        self.segment_queue = segment_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.language = language
        self.text_context = ""
        self.on_error = on_error
        self.target_sample_rate = int(target_sample_rate)
        self.trace_audio = bool(trace_audio)
        self._prompt_echo_guard = {
            "以下是中文语音转写文本",
            "中文语音转写文本",
        }
        self._last_norm_text = ""
        self._repeat_count = 0
        self._segment_index = 0
        self.debug_dir = Path(debug_dir) if debug_dir else Path.cwd() / "data" / "debug_segments"
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self.debug_dir / "audio_trace.log"

    def run(self) -> None:
        try:
            while not self.stop_event.is_set():
                segment = self.segment_queue.get()
                if segment is None:
                    break
                self._process_segment(segment)
        except Exception as exc:  # noqa: BLE001
            self.stop_event.set()
            if self.on_error is not None:
                try:
                    self.on_error(str(exc))
                except Exception:
                    pass

    def _process_segment(self, segment: SpeechSegment) -> None:
        if segment.samples.size == 0:
            return
        audio = self._prepare_audio_for_model(segment)
        if audio.size == 0:
            return
        duration_sec = float(audio.shape[0] / max(1, self.target_sample_rate))
        if duration_sec < 0.25:
            return

        self._segment_index += 1
        debug_name = f"debug_seg_{self._segment_index:03d}.wav"
        self._save_debug_wav(self.debug_dir / debug_name, audio, self.target_sample_rate)
        self._trace_audio_meta(audio, self.target_sample_rate, duration_sec, debug_name)

        kwargs = {
            "language": self.language,
            "beam_size": 5,
            "best_of": 5,
            "temperature": 0.0,
            "vad_filter": False,
            "condition_on_previous_text": False,
            "without_timestamps": False,
            "no_speech_threshold": 0.6,
            "log_prob_threshold": -1.0,
            "compression_ratio_threshold": 2.4,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3,
        }
        if self.text_context:
            kwargs["initial_prompt"] = self.text_context
        segments, _ = self.model.transcribe(audio, **kwargs)

        texts: list[str] = []
        conf_values: list[float] = []
        for item in segments:
            text = str(item.text).strip()
            if not text:
                continue
            if self._is_prompt_echo(text):
                continue
            texts.append(text)
            conf_values.append(_score_from_logprob(getattr(item, "avg_logprob", None)))

        if not texts:
            return
        full_text = " ".join(texts).strip()
        full_text = self._collapse_repetitions(full_text)
        if self._is_prompt_echo(full_text) or self._is_watermark_credit(full_text):
            return
        if self._is_duplicate_burst(full_text):
            return
        self.text_context = (self.text_context + " " + full_text).strip()[-180:]
        confidence = round(sum(conf_values) / max(1, len(conf_values)), 2)
        result = TranscriptSegment(
            start_ms=segment.start_ms,
            end_ms=max(segment.end_ms, segment.start_ms + 300),
            text=full_text,
            confidence=confidence,
        )
        self._put_result_with_drop_oldest(result)

    def _put_result_with_drop_oldest(self, item: TranscriptSegment) -> None:
        try:
            self.result_queue.put_nowait(item)
            return
        except Exception:
            pass

    def _is_prompt_echo(self, text: str) -> bool:
        normalized = text.replace("，", "").replace("。", "").replace(",", "").replace(".", "").strip()
        if not normalized:
            return True
        for marker in self._prompt_echo_guard:
            if marker in normalized:
                # 只有提示词骨架，没有其他有效内容时判定为回声。
                remain = normalized.replace(marker, "").strip()
                if len(remain) <= 2:
                    return True
        return False

    def _is_watermark_credit(self, text: str) -> bool:
        low = text.lower().strip()
        patterns = [
            r"字幕\s*by",
            r"subtitle\s*by",
            r"翻译[:：]",
            r"字幕组",
        ]
        return any(re.search(p, low) is not None for p in patterns)

    def _is_duplicate_burst(self, text: str) -> bool:
        norm = re.sub(r"\s+", "", text).lower()
        if not norm:
            return True
        if norm == self._last_norm_text:
            self._repeat_count += 1
        else:
            self._last_norm_text = norm
            self._repeat_count = 0
        # 连续重复同句时，保留第一次，后续抑制。
        return self._repeat_count >= 1

    @staticmethod
    def _collapse_repetitions(text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return ""

        # 把连续重复短语折叠为一次，例如 “字幕byX 字幕byX ...”
        parts = re.split(r"[，。,.!?！？\s]+", cleaned)
        parts = [p for p in parts if p]
        if len(parts) <= 1:
            return cleaned

        deduped: list[str] = []
        for p in parts:
            if deduped and p == deduped[-1]:
                continue
            deduped.append(p)
        compact = " ".join(deduped).strip()
        return compact or cleaned

    def _prepare_audio_for_model(self, segment: SpeechSegment) -> np.ndarray:
        # 1) 独立副本，避免共享 buffer 污染
        audio = np.asarray(segment.samples, dtype=np.float32).copy()
        # 2) 单声道保障（防御式）
        if audio.ndim > 1:
            audio = audio.mean(axis=1).astype(np.float32, copy=False)
        # 3) 振幅归一化到 [-1, 1]，防止异常溢出
        if audio.size > 0:
            max_abs = float(np.max(np.abs(audio)))
            if max_abs > 1.0:
                audio = (audio / max_abs).astype(np.float32, copy=False)
            audio = np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)
        # 4) 采样率统一到模型目标值（faster-whisper ndarray 输入建议 16k）
        if int(segment.sample_rate) != self.target_sample_rate:
            audio = self._resample_mono(audio, int(segment.sample_rate), self.target_sample_rate)
        return audio.astype(np.float32, copy=False)

    @staticmethod
    def _resample_mono(mono: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        if mono.size == 0 or src_rate <= 0 or src_rate == dst_rate:
            return mono.astype(np.float32, copy=False)
        src_len = mono.shape[0]
        dst_len = max(1, int(round(src_len * dst_rate / src_rate)))
        src_x = np.linspace(0.0, 1.0, num=src_len, endpoint=False)
        dst_x = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
        return np.interp(dst_x, src_x, mono).astype(np.float32, copy=False)

    def _trace_audio_meta(self, audio: np.ndarray, sample_rate: int, duration_sec: float, file_name: str) -> None:
        if not self.trace_audio:
            return
        msg = (
            f"[SEG {self._segment_index:03d}] "
            f"audio.shape={audio.shape}, audio.dtype={audio.dtype}, "
            f"audio.min()={float(audio.min()) if audio.size else 0.0:.6f}, "
            f"audio.max()={float(audio.max()) if audio.size else 0.0:.6f}, "
            f"sample_rate={sample_rate}, duration_sec={duration_sec:.3f}, file={file_name}"
        )
        print(msg, flush=True)
        try:
            with self._log_file.open("a", encoding="utf-8") as f:
                f.write(msg + os.linesep)
        except Exception:
            pass

    @staticmethod
    def _save_debug_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        clipped = np.clip(audio.astype(np.float32, copy=False), -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sample_rate))
            wf.writeframes(pcm16.tobytes())
        try:
            self.result_queue.get_nowait()
            self.result_queue.put_nowait(item)
        except Exception:
            pass


def _score_from_logprob(value: float | None) -> float:
    if value is None:
        return 0.9
    probability = math.exp(float(value))
    return round(float(max(0.0, min(1.0, probability))), 2)
