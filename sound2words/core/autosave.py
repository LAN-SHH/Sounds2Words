from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from sound2words.core.models import SessionSnapshot


class AutoSaveManager:
    def __init__(self, project_root: Path) -> None:
        self._autosave_dir = project_root / "data" / "autosave"
        self._history_dir = project_root / "data" / "history"
        self._autosave_dir.mkdir(parents=True, exist_ok=True)
        self._history_dir.mkdir(parents=True, exist_ok=True)
        self._active_file = self._autosave_dir / "current_session.json"

    def save_active(self, snapshot: SessionSnapshot) -> None:
        snapshot.updated_at = datetime.now().isoformat()
        self._active_file.write_text(
            json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_recoverable(self) -> SessionSnapshot | None:
        if not self._active_file.exists():
            return None

        payload = json.loads(self._active_file.read_text(encoding="utf-8"))
        snapshot = SessionSnapshot.from_dict(payload)
        if snapshot.status == "running":
            return snapshot
        return None

    def archive_and_clear(self, snapshot: SessionSnapshot) -> Path:
        snapshot.status = "stopped"
        snapshot.updated_at = datetime.now().isoformat()
        date_folder = datetime.now().strftime("%Y-%m-%d")
        bucket = self._history_dir / date_folder
        bucket.mkdir(parents=True, exist_ok=True)
        archive_file = bucket / f"{snapshot.session_id}.json"
        archive_file.write_text(
            json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if self._active_file.exists():
            self._active_file.unlink()
        return archive_file

    def list_history_sessions(self) -> list[tuple[Path, SessionSnapshot]]:
        records: list[tuple[Path, SessionSnapshot]] = []
        for json_file in self._history_dir.rglob("*.json"):
            try:
                payload = json.loads(json_file.read_text(encoding="utf-8"))
                snapshot = SessionSnapshot.from_dict(payload)
                records.append((json_file, snapshot))
            except Exception:
                continue

        records.sort(key=lambda item: item[1].updated_at, reverse=True)
        return records

    def delete_history(self, file_path: Path) -> bool:
        try:
            file_path.unlink(missing_ok=False)
        except FileNotFoundError:
            return False

        parent = file_path.parent
        if parent != self._history_dir and parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
        return True
