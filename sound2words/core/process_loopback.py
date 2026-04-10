from __future__ import annotations

import ctypes
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from ctypes import wintypes

try:
    import psutil
except ModuleNotFoundError:  # pragma: no cover - optional dependency fallback
    psutil = None


@dataclass(slots=True)
class AudioTargetProcess:
    pid: int
    name: str
    window_title: str

    @property
    def label(self) -> str:
        title = self.window_title.strip() if self.window_title else ""
        if title:
            return f"{self.name} ({self.pid}) - {title}"
        return f"{self.name} ({self.pid})"


@dataclass(slots=True)
class AudioSourceOption:
    display_name: str
    option_kind: str
    pid: int | None = None
    pid_list: list[int] = field(default_factory=list)
    process_meta: list[dict[str, Any]] = field(default_factory=list)

    @property
    def list_label(self) -> str:
        if self.option_kind == "group":
            return f"{self.display_name} ({len(self.pid_list)})"
        return self.display_name


@dataclass(slots=True)
class AudioSourceSelection:
    mode: str
    pid: int | None
    label: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _get_exe_description(exe_path: str) -> str:
    if not exe_path:
        return ""
    version_api = ctypes.windll.version
    size = version_api.GetFileVersionInfoSizeW(exe_path, None)
    if size == 0:
        return ""

    buffer = (ctypes.c_byte * size)()
    ok = version_api.GetFileVersionInfoW(exe_path, 0, size, ctypes.byref(buffer))
    if not ok:
        return ""

    ptr = ctypes.c_void_p()
    size_ptr = ctypes.c_uint()
    ok = version_api.VerQueryValueW(
        ctypes.byref(buffer),
        "\\VarFileInfo\\Translation",
        ctypes.byref(ptr),
        ctypes.byref(size_ptr),
    )
    if not ok or size_ptr.value < 4:
        return ""

    lang_codepage = ctypes.string_at(ptr.value, 4)
    lang = int.from_bytes(lang_codepage[0:2], "little")
    codepage = int.from_bytes(lang_codepage[2:4], "little")
    key = f"\\StringFileInfo\\{lang:04x}{codepage:04x}\\FileDescription"
    ok = version_api.VerQueryValueW(ctypes.byref(buffer), key, ctypes.byref(ptr), ctypes.byref(size_ptr))
    if not ok or size_ptr.value == 0:
        return ""
    return ctypes.wstring_at(ptr.value, size_ptr.value).strip().strip("\x00")


def _friendly_process_name(pid: int, fallback: str) -> str:
    if psutil is None:
        return (fallback or "").strip()
    try:
        process = psutil.Process(pid)
        exe_path = process.exe()
        description = _get_exe_description(exe_path)
        if description:
            return description
        stem = Path(exe_path).stem.strip()
        if stem:
            return stem
    except Exception:
        pass
    return (fallback or "").strip()


def _process_name(pid: int) -> str:
    if psutil is None:
        return ""
    try:
        return psutil.Process(pid).name()
    except Exception:
        return ""


def _visible_window_map() -> dict[int, str]:
    user32 = ctypes.windll.user32
    titles: dict[int, str] = {}

    enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    @enum_proc
    def _callback(hwnd: int, _lparam: int) -> bool:
        try:
            if not user32.IsWindowVisible(hwnd):
                return True
            length = user32.GetWindowTextLengthW(hwnd)
            if length <= 0:
                return True
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            title = buf.value.strip()
            if not title:
                return True
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if pid.value <= 0:
                return True
            if pid.value not in titles or len(title) > len(titles[pid.value]):
                titles[pid.value] = title
        except Exception:
            return True
        return True

    user32.EnumWindows(_callback, 0)
    return titles


def _extra_candidate_processes() -> list[AudioTargetProcess]:
    result: list[AudioTargetProcess] = []
    seen: set[int] = set()

    for pid, title in _visible_window_map().items():
        name = _process_name(pid)
        if not name:
            name = f"pid-{pid}"
        result.append(AudioTargetProcess(pid=pid, name=name, window_title=title))
        seen.add(pid)

    # 兜底补充常见媒体进程（如 QQMusic），即使窗口最小化也能在列表中选择。
    if psutil is not None:
        media_keywords = ("qqmusic", "cloudmusic", "spotify", "vlc", "potplayer")
        for proc in psutil.process_iter(attrs=["pid", "name"]):
            try:
                pid = int(proc.info.get("pid") or 0)
                name = str(proc.info.get("name") or "")
            except Exception:
                continue
            if pid <= 0 or pid in seen:
                continue
            low = name.lower()
            if any(k in low for k in media_keywords):
                result.append(AudioTargetProcess(pid=pid, name=name, window_title=""))
                seen.add(pid)

    return result


def resolve_display_name(process: AudioTargetProcess) -> str:
    title = (process.window_title or "").strip()
    if title:
        return title
    readable = _friendly_process_name(process.pid, process.name)
    if readable:
        return readable
    return (process.name or f"PID {process.pid}").strip()


def build_audio_source_options(raw_processes: list[AudioTargetProcess]) -> list[AudioSourceOption]:
    grouped: dict[str, list[AudioTargetProcess]] = defaultdict(list)
    for process in raw_processes:
        display_name = resolve_display_name(process)
        if not display_name:
            display_name = f"PID {process.pid}"
        grouped[display_name].append(process)

    options: list[AudioSourceOption] = []
    for display_name, processes in grouped.items():
        sorted_items = sorted(processes, key=lambda item: item.pid)
        pid_list = [item.pid for item in sorted_items]
        process_meta = [
            {
                "pid": item.pid,
                "process_name": item.name,
                "window_title": item.window_title,
            }
            for item in sorted_items
        ]
        if len(sorted_items) == 1:
            options.append(
                AudioSourceOption(
                    display_name=display_name,
                    option_kind="pid",
                    pid=sorted_items[0].pid,
                    pid_list=pid_list,
                    process_meta=process_meta,
                )
            )
            continue
        options.append(
            AudioSourceOption(
                display_name=display_name,
                option_kind="group",
                pid=None,
                pid_list=pid_list,
                process_meta=process_meta,
            )
        )

    options.sort(key=lambda item: item.display_name.lower())
    return options


class ProcessLoopback:
    @staticmethod
    def list_audio_processes() -> list[AudioTargetProcess]:
        result_by_pid: dict[int, AudioTargetProcess] = {}

        try:
            from process_audio_capture import ProcessAudioCapture
        except ModuleNotFoundError:
            rows = []
        else:
            try:
                rows = ProcessAudioCapture.enumerate_audio_processes()
            except Exception:
                rows = []

        for row in rows:
            pid = int(row.pid)
            result_by_pid[pid] = AudioTargetProcess(
                pid=pid,
                name=str(row.name),
                window_title=str(row.window_title),
            )

        for item in _extra_candidate_processes():
            if item.pid not in result_by_pid:
                result_by_pid[item.pid] = item

        result = list(result_by_pid.values())
        result.sort(key=lambda item: (resolve_display_name(item).lower(), item.pid))
        return result
