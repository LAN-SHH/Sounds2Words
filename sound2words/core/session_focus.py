from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SessionMuteState:
    key: str
    was_muted: bool


class AudioSessionFocus:
    def __init__(self) -> None:
        self._states: list[SessionMuteState] = []
        self._active = False

    @staticmethod
    def list_active_process_names() -> list[str]:
        try:
            from pycaw.pycaw import AudioUtilities
        except ModuleNotFoundError:
            return []

        names: set[str] = set()
        for session in AudioUtilities.GetAllSessions():
            proc = session.Process
            if proc is None:
                continue
            try:
                name = proc.name()
            except Exception:
                continue
            if name:
                names.add(name)
        return sorted(names)

    def enable_focus(self, target_process_name: str) -> str | None:
        if not target_process_name:
            return None

        try:
            from pycaw.pycaw import AudioUtilities
        except ModuleNotFoundError:
            return "未安装 pycaw，无法启用客户端独占采集。请安装: python -m pip install pycaw"

        if self._active:
            return None

        target = target_process_name.lower()
        self._states.clear()

        for session in AudioUtilities.GetAllSessions():
            proc = session.Process
            if proc is None:
                continue

            try:
                name = proc.name()
                pid = proc.pid
            except Exception:
                continue

            if not name:
                continue

            key = f"{name}:{pid}"
            volume = session.SimpleAudioVolume
            try:
                was_muted = bool(volume.GetMute())
            except Exception:
                continue

            self._states.append(SessionMuteState(key=key, was_muted=was_muted))
            if name.lower() != target:
                try:
                    volume.SetMute(1, None)
                except Exception:
                    continue

        self._active = True
        return None

    def disable_focus(self) -> None:
        if not self._active:
            return

        try:
            from pycaw.pycaw import AudioUtilities
        except ModuleNotFoundError:
            self._states.clear()
            self._active = False
            return

        state_map = {state.key: state for state in self._states}
        for session in AudioUtilities.GetAllSessions():
            proc = session.Process
            if proc is None:
                continue
            try:
                name = proc.name()
                pid = proc.pid
            except Exception:
                continue
            if not name:
                continue

            key = f"{name}:{pid}"
            state = state_map.get(key)
            if state is None:
                continue

            try:
                session.SimpleAudioVolume.SetMute(1 if state.was_muted else 0, None)
            except Exception:
                continue

        self._states.clear()
        self._active = False
