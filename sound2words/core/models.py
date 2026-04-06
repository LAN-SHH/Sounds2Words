from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class TranscriptSegment:
    start_ms: int
    end_ms: int
    text: str
    confidence: float = 1.0


@dataclass(slots=True)
class SessionSnapshot:
    session_id: str
    started_at: str
    updated_at: str
    status: str
    source_type: str = "mic"
    source_name: str = "麦克风"
    file_path: str = ""
    segments: list[TranscriptSegment] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["segments"] = [asdict(segment) for segment in self.segments]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SessionSnapshot":
        raw_segments = payload.get("segments", [])
        segments = [TranscriptSegment(**segment) for segment in raw_segments]
        return cls(
            session_id=payload["session_id"],
            started_at=payload["started_at"],
            updated_at=payload.get("updated_at", datetime.now().isoformat()),
            status=payload.get("status", "running"),
            source_type=payload.get("source_type", "mic"),
            source_name=payload.get("source_name", "麦克风"),
            file_path=payload.get("file_path", ""),
            segments=segments,
        )
