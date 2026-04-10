from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from sound2words.core.process_loopback import AudioSourceOption, AudioSourceSelection


@dataclass(slots=True)
class _ChoiceItem:
    key: str
    label: str
    payload: dict


class _ChoiceRow(QFrame):
    def __init__(self, label: str, key: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.key = key
        self.setObjectName("audioSourceRow")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)
        self.text_label = QLabel(label)
        self.text_label.setWordWrap(True)
        self.radio = QRadioButton()
        self.radio.setCursor(Qt.CursorShape.PointingHandCursor)
        layout.addWidget(self.text_label, 1)
        layout.addWidget(self.radio, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._apply_selected_style(False)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self.radio.setChecked(True)
            event.accept()
            return
        super().mousePressEvent(event)

    def set_selected(self, selected: bool) -> None:
        self._apply_selected_style(selected)

    def _apply_selected_style(self, selected: bool) -> None:
        if selected:
            self.setStyleSheet(
                "QFrame#audioSourceRow { border: 1px solid #4f8cff; border-radius: 6px; background: #1d2a44; }"
            )
            return
        self.setStyleSheet(
            "QFrame#audioSourceRow { border: 1px solid #444; border-radius: 6px; background: #242424; }"
        )


class _SingleChoiceDialog(QDialog):
    def __init__(
        self,
        title: str,
        choices: list[_ChoiceItem],
        current_key: str | None = None,
        empty_message: str = "当前没有可选项",
        top_hint: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(480, 520)
        self._choices = choices
        self._rows: dict[str, _ChoiceRow] = {}
        self._selected_key = ""
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._group.buttonClicked.connect(self._sync_selected_state)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        self.top_hint_label = QLabel(top_hint)
        self.top_hint_label.setWordWrap(True)
        self.top_hint_label.setVisible(bool(top_hint))
        root.addWidget(self.top_hint_label)

        self.empty_label = QLabel(empty_message)
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setVisible(False)
        root.addWidget(self.empty_label)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        self.content_layout = QVBoxLayout(content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(8)
        self.scroll.setWidget(content)
        root.addWidget(self.scroll, 1)

        button_row = QHBoxLayout()
        button_row.addStretch()
        self.cancel_button = QPushButton("取消")
        self.ok_button = QPushButton("确定")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        button_row.addWidget(self.cancel_button)
        button_row.addWidget(self.ok_button)
        root.addLayout(button_row)

        self._populate(current_key=current_key)

    def selected_choice(self) -> _ChoiceItem | None:
        if not self._selected_key:
            return None
        for item in self._choices:
            if item.key == self._selected_key:
                return item
        return None

    def _populate(self, current_key: str | None) -> None:
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self._rows.clear()
        self._selected_key = ""

        if not self._choices:
            self.scroll.setVisible(False)
            self.empty_label.setVisible(True)
            self.ok_button.setEnabled(False)
            return

        self.scroll.setVisible(True)
        self.empty_label.setVisible(False)
        self.ok_button.setEnabled(True)

        default_key = current_key if current_key else self._choices[0].key
        for index, choice in enumerate(self._choices):
            row = _ChoiceRow(choice.label, choice.key)
            self._rows[choice.key] = row
            self._group.addButton(row.radio, index)
            row.radio.toggled.connect(self._sync_selected_state)
            self.content_layout.addWidget(row)
            if choice.key == default_key:
                row.radio.setChecked(True)
                self._selected_key = choice.key
        self.content_layout.addStretch()
        self._sync_selected_state()

    def _sync_selected_state(self) -> None:
        checked = self._group.checkedButton()
        for key, row in self._rows.items():
            row.set_selected(row.radio is checked)
            if row.radio is checked:
                self._selected_key = key


class AudioSourcePickerDialog(QDialog):
    def __init__(
        self,
        options: list[AudioSourceOption],
        current_selection: AudioSourceSelection | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._options = options
        self._current_selection = current_selection

    def pick(self) -> AudioSourceSelection | None:
        current_key = "all"
        if self._current_selection and self._current_selection.mode == "pid" and self._current_selection.pid:
            current_key = f"pid:{self._current_selection.pid}"

        choices = [
            _ChoiceItem(
                key="all",
                label="全部音源",
                payload={"kind": "all"},
            )
        ]
        for option in self._options:
            if option.option_kind == "pid" and option.pid:
                key = f"pid:{option.pid}"
            else:
                key = f"group:{option.display_name}"
            choices.append(
                _ChoiceItem(
                    key=key,
                    label=option.list_label,
                    payload={"kind": option.option_kind, "option": option},
                )
            )

        first = _SingleChoiceDialog(
            title="选择系统音源",
            choices=choices,
            current_key=current_key,
            empty_message="当前没有正在输出声音的应用",
            top_hint="当前没有正在输出声音的应用" if not self._options else "",
            parent=self.parentWidget(),
        )
        if first.exec() != QDialog.DialogCode.Accepted:
            return None

        selected = first.selected_choice()
        if selected is None:
            return AudioSourceSelection(mode="all", pid=None, label="全部音源", metadata={"option_kind": "all"})

        if selected.payload["kind"] == "all":
            return AudioSourceSelection(mode="all", pid=None, label="全部音源", metadata={"option_kind": "all"})

        option: AudioSourceOption = selected.payload["option"]
        if option.option_kind == "pid" and option.pid:
            return AudioSourceSelection(
                mode="pid",
                pid=option.pid,
                label=f"{option.display_name} (PID {option.pid})",
                metadata={"option_kind": "pid", "process_meta": option.process_meta},
            )

        pid_choices: list[_ChoiceItem] = []
        for meta in option.process_meta:
            pid = int(meta["pid"])
            process_name = str(meta.get("process_name", "")).strip()
            window_title = str(meta.get("window_title", "")).strip()
            detail = process_name
            if window_title:
                detail = f"{process_name} - {window_title}" if process_name else window_title
            if detail:
                label = f"PID {pid} | {detail}"
            else:
                label = f"PID {pid}"
            pid_choices.append(_ChoiceItem(key=f"pid:{pid}", label=label, payload={"pid": pid, "meta": meta}))

        second = _SingleChoiceDialog(
            title=f"选择 {option.display_name} 的实例",
            choices=pid_choices,
            current_key=None,
            empty_message="该音源没有可选实例",
            parent=self.parentWidget(),
        )
        if second.exec() != QDialog.DialogCode.Accepted:
            return None
        pid_choice = second.selected_choice()
        if pid_choice is None:
            return None

        pid = int(pid_choice.payload["pid"])
        return AudioSourceSelection(
            mode="pid",
            pid=pid,
            label=f"{option.display_name} (PID {pid})",
            metadata={"option_kind": "group", "group_name": option.display_name, "process_meta": pid_choice.payload["meta"]},
        )
