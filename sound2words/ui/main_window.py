from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    from sound2words import __version__
    from sound2words.core.autosave import AutoSaveManager
    from sound2words.core.exporter import TranscriptExporter
    from sound2words.core.models import SessionSnapshot, TranscriptSegment
    from sound2words.core.transcriber import RealtimeTranscriber
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from sound2words import __version__
    from sound2words.core.autosave import AutoSaveManager
    from sound2words.core.exporter import TranscriptExporter
    from sound2words.core.models import SessionSnapshot, TranscriptSegment
    from sound2words.core.transcriber import RealtimeTranscriber


class MainWindow(QMainWindow):
    AUTOSAVE_INTERVAL_MS = 5000
    MODEL_PROGRESS_INTERVAL_MS = 200

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"Sound2Words {__version__}")
        self.resize(1040, 700)

        root = Path(__file__).resolve().parents[2]
        self.autosave = AutoSaveManager(root)
        self.exporter = TranscriptExporter(root)
        self.model_root = str((root / "data" / "models").resolve())
        self.snapshot: SessionSnapshot | None = None
        self.worker: RealtimeTranscriber | None = None
        self.selected_file_path = ""
        self.selected_device_name = ""
        self.history_records: list[tuple[Path, SessionSnapshot]] = []

        self._build_ui()
        self._setup_timers()
        self._check_recovery()
        self._refresh_history_list()

    def _build_ui(self) -> None:
        container = QWidget(self)
        self.setCentralWidget(container)

        root_layout = QVBoxLayout(container)

        self.status_label = QLabel("状态: 待开始")
        root_layout.addWidget(self.status_label)

        self.tabs = QTabWidget()
        self.transcribe_tab = self._build_transcribe_tab()
        self.history_tab = self._build_history_tab()
        self.tabs.addTab(self.transcribe_tab, "转写")
        self.tabs.addTab(self.history_tab, "历史记录")
        root_layout.addWidget(self.tabs)

    def _build_transcribe_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("输入源:"))
        self.source_combo = QComboBox()
        self.source_combo.addItem("麦克风", "mic")
        self.source_combo.addItem("系统音频", "system")
        self.source_combo.addItem("本地文件", "file")
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        source_row.addWidget(self.source_combo)

        self.device_combo = QComboBox()
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        source_row.addWidget(QLabel("设备:"))
        source_row.addWidget(self.device_combo)

        self.refresh_device_button = QPushButton("刷新设备")
        self.refresh_device_button.clicked.connect(self._reload_device_list)
        source_row.addWidget(self.refresh_device_button)

        self.choose_file_button = QPushButton("选择文件")
        self.choose_file_button.clicked.connect(self._choose_file)
        source_row.addWidget(self.choose_file_button)

        self.file_label = QLabel("未选择文件")
        source_row.addWidget(self.file_label)
        source_row.addStretch()
        layout.addLayout(source_row)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("模型加载:"))
        self.model_progress_bar = QProgressBar()
        self.model_progress_bar.setRange(0, 100)
        self.model_progress_bar.setValue(0)
        model_row.addWidget(self.model_progress_bar)
        self.model_progress_text = QLabel("待开始")
        model_row.addWidget(self.model_progress_text)
        model_row.addStretch()
        layout.addLayout(model_row)

        level_row = QHBoxLayout()
        level_row.addWidget(QLabel("输入电平:"))
        self.level_bar = QProgressBar()
        self.level_bar.setRange(0, 100)
        self.level_bar.setValue(0)
        level_row.addWidget(self.level_bar)
        self.level_text = QLabel("0%")
        level_row.addWidget(self.level_text)
        level_row.addStretch()
        layout.addLayout(level_row)

        self.transcript_view = QTextEdit()
        self.transcript_view.setReadOnly(True)
        self.transcript_view.setPlaceholderText("点击开始后，这里会实时显示转写结果")
        layout.addWidget(self.transcript_view)

        controls = QHBoxLayout()
        self.start_button = QPushButton("开始")
        self.stop_button = QPushButton("结束")
        self.stop_button.setEnabled(False)

        self.export_txt_button = QPushButton("导出 TXT")
        self.export_srt_button = QPushButton("导出 SRT")
        self.export_docx_button = QPushButton("导出 Word")
        self._set_export_enabled(False)

        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        controls.addSpacing(20)
        controls.addWidget(self.export_txt_button)
        controls.addWidget(self.export_srt_button)
        controls.addWidget(self.export_docx_button)
        controls.addStretch()
        layout.addLayout(controls)

        self.start_button.clicked.connect(self.start_session)
        self.stop_button.clicked.connect(self.stop_session)
        self.export_txt_button.clicked.connect(lambda: self._export("txt"))
        self.export_srt_button.clicked.connect(lambda: self._export("srt"))
        self.export_docx_button.clicked.connect(lambda: self._export("docx"))

        self._on_source_changed()
        return page

    def _build_history_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("检索:"))
        self.history_search = QLineEdit()
        self.history_search.setPlaceholderText("输入日期/来源/文本关键字")
        self.history_search.textChanged.connect(self._refresh_history_list)
        top_row.addWidget(self.history_search)

        self.history_refresh_button = QPushButton("刷新")
        self.history_refresh_button.clicked.connect(self._refresh_history_list)
        top_row.addWidget(self.history_refresh_button)

        self.history_delete_button = QPushButton("删除所选")
        self.history_delete_button.clicked.connect(self._delete_selected_history)
        top_row.addWidget(self.history_delete_button)
        layout.addLayout(top_row)

        mid_row = QHBoxLayout()
        self.history_list = QListWidget()
        self.history_list.itemSelectionChanged.connect(self._render_history_detail)
        mid_row.addWidget(self.history_list, 1)

        self.history_detail = QTextEdit()
        self.history_detail.setReadOnly(True)
        self.history_detail.setPlaceholderText("选择左侧历史记录后，这里会显示转写详情")
        mid_row.addWidget(self.history_detail, 2)
        layout.addLayout(mid_row)

        return page

    def _setup_timers(self) -> None:
        self.autosave_timer = QTimer(self)
        self.autosave_timer.timeout.connect(self._autosave)
        self.model_progress_timer = QTimer(self)
        self.model_progress_timer.timeout.connect(self._tick_model_progress)

    def _check_recovery(self) -> None:
        recoverable = self.autosave.load_recoverable()
        if recoverable is None:
            return

        answer = QMessageBox.question(
            self,
            "恢复会话",
            "检测到上次未正常结束的转写会话，是否恢复并继续转写？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.snapshot = recoverable
            self._restore_source(recoverable)
            self._render_all_segments()
            self.start_session(reuse_existing=True)
        else:
            recoverable.status = "stopped"
            self.autosave.archive_and_clear(recoverable)

    def _restore_source(self, snapshot: SessionSnapshot) -> None:
        index = self.source_combo.findData(snapshot.source_type)
        if index >= 0:
            self.source_combo.setCurrentIndex(index)

        self.selected_file_path = snapshot.file_path
        self.file_label.setText(Path(snapshot.file_path).name if snapshot.file_path else "未选择文件")
        self.selected_device_name = snapshot.source_name

    def _on_source_changed(self) -> None:
        source_type = self.source_combo.currentData()
        is_file_mode = source_type == "file"
        self.choose_file_button.setVisible(is_file_mode)
        self.file_label.setVisible(is_file_mode)

        self.device_combo.setVisible(not is_file_mode)
        self.refresh_device_button.setVisible(not is_file_mode)

        self.level_bar.setEnabled(not is_file_mode)
        self.level_bar.setValue(0)
        self.level_text.setText("0%")

        self._reload_device_list()

    def _reload_device_list(self) -> None:
        source_type = self.source_combo.currentData()
        self.device_combo.clear()
        if source_type == "file":
            self.selected_device_name = ""
            return

        devices = RealtimeTranscriber.list_input_devices(source_type)
        if not devices:
            self.device_combo.addItem("未找到设备", "")
            self.selected_device_name = ""
            return

        for name in devices:
            self.device_combo.addItem(name, name)

        if self.selected_device_name:
            idx = self.device_combo.findData(self.selected_device_name)
            if idx >= 0:
                self.device_combo.setCurrentIndex(idx)

        self.selected_device_name = self.device_combo.currentData() or ""

    def _on_device_changed(self) -> None:
        data = self.device_combo.currentData()
        self.selected_device_name = data or ""

    def _choose_file(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "选择音视频文件",
            "",
            "Media Files (*.mp3 *.wav *.m4a *.flac *.mp4 *.mkv *.mov *.avi);;All Files (*.*)",
        )
        if not selected:
            return

        self.selected_file_path = selected
        self.file_label.setText(Path(selected).name)

    def start_session(self, reuse_existing: bool = False) -> None:
        if self.worker is not None and self.worker.isRunning():
            return

        source_type = self.source_combo.currentData()
        if source_type == "file" and not self.selected_file_path:
            QMessageBox.warning(self, "缺少文件", "文件模式需要先选择一个音视频文件。")
            return

        if source_type != "file" and not self.selected_device_name:
            QMessageBox.warning(self, "缺少设备", "请先选择输入设备。")
            return

        if not reuse_existing or self.snapshot is None:
            now = datetime.now().isoformat()
            source_name = self.source_combo.currentText() if source_type == "file" else self.selected_device_name
            self.snapshot = SessionSnapshot(
                session_id=str(uuid4()),
                started_at=now,
                updated_at=now,
                status="running",
                source_type=source_type,
                source_name=source_name,
                file_path=self.selected_file_path if source_type == "file" else "",
            )
            self.transcript_view.clear()
        else:
            self.snapshot.status = "running"

        self.worker = RealtimeTranscriber(
            source_type=source_type,
            file_path=self.selected_file_path if source_type == "file" else "",
            device_name=self.selected_device_name,
            model_download_root=self.model_root,
        )
        self.worker.segment_ready.connect(self._handle_segment)
        self.worker.status_changed.connect(self.status_label.setText)
        self.worker.error_occurred.connect(self._handle_worker_error)
        self.worker.level_changed.connect(self._update_level)
        self.worker.model_loading_changed.connect(self._set_model_loading_progress)
        self.worker.finished.connect(self._handle_worker_finished)

        self._set_running(True)
        self._begin_model_loading_progress()
        self.worker.start()
        self._autosave()

    def stop_session(self) -> None:
        if self.snapshot is None:
            return

        self._stop_worker()
        self._set_running(False)
        self.status_label.setText("状态: 已结束")
        archive_file = self.autosave.archive_and_clear(self.snapshot)
        self._set_export_enabled(len(self.snapshot.segments) > 0)
        self._refresh_history_list()
        QMessageBox.information(self, "会话已保存", f"已归档到: {archive_file}")

    def _stop_worker(self) -> None:
        if self.worker is None:
            return

        self.worker.stop()
        self.worker.wait(4000)
        self.worker = None

    def _set_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.source_combo.setEnabled(not running)
        self.device_combo.setEnabled(not running)
        self.refresh_device_button.setEnabled(not running)
        self.choose_file_button.setEnabled(not running)

        if running:
            self._set_export_enabled(False)
            self.autosave_timer.start(self.AUTOSAVE_INTERVAL_MS)
        else:
            self.autosave_timer.stop()
            self.model_progress_timer.stop()
            self.level_bar.setValue(0)
            self.level_text.setText("0%")

    def _set_export_enabled(self, enabled: bool) -> None:
        self.export_txt_button.setEnabled(enabled)
        self.export_srt_button.setEnabled(enabled)
        docx_enabled = enabled and self.snapshot is not None and self.snapshot.source_type == "file"
        self.export_docx_button.setEnabled(docx_enabled)

    def _handle_segment(self, segment: TranscriptSegment) -> None:
        if self.snapshot is None:
            return

        self.snapshot.segments.append(segment)
        self.transcript_view.append(
            f"[{self._ms_to_hhmmss(segment.start_ms)}-{self._ms_to_hhmmss(segment.end_ms)}] {segment.text}"
        )
        self._set_export_enabled(True)

    def _update_level(self, level: int) -> None:
        level = max(0, min(100, level))
        self.level_bar.setValue(level)
        self.level_text.setText(f"{level}%")

    def _begin_model_loading_progress(self) -> None:
        self.model_progress_bar.setValue(0)
        self.model_progress_text.setText("准备中")
        self.model_progress_timer.start(self.MODEL_PROGRESS_INTERVAL_MS)

    def _tick_model_progress(self) -> None:
        current = self.model_progress_bar.value()
        if current >= 95:
            return
        self.model_progress_bar.setValue(current + 1)
        self.model_progress_text.setText("加载中...")

    def _set_model_loading_progress(self, value: int, text: str) -> None:
        progress = max(0, min(100, value))
        if progress > self.model_progress_bar.value():
            self.model_progress_bar.setValue(progress)
        self.model_progress_text.setText(text)
        if progress >= 100:
            self.model_progress_timer.stop()

    def _handle_worker_error(self, message: str) -> None:
        self.status_label.setText("状态: 识别异常")
        self.model_progress_timer.stop()
        QMessageBox.warning(self, "识别异常", message)

    def _handle_worker_finished(self) -> None:
        self._set_model_loading_progress(100, "模型就绪")
        self.model_progress_timer.stop()
        if self.snapshot is None:
            return
        if self.stop_button.isEnabled():
            if self.snapshot.source_type == "file":
                self.status_label.setText("状态: 文件转写完成，请点击结束保存")
            else:
                self.status_label.setText("状态: 识别线程已停止，请点击结束保存")

    def _autosave(self) -> None:
        if self.snapshot is None:
            return

        self.snapshot.status = "running"
        self.autosave.save_active(self.snapshot)

    def _render_all_segments(self) -> None:
        if self.snapshot is None:
            return

        self.transcript_view.clear()
        for segment in self.snapshot.segments:
            self.transcript_view.append(
                f"[{self._ms_to_hhmmss(segment.start_ms)}-{self._ms_to_hhmmss(segment.end_ms)}] {segment.text}"
            )
        self._set_export_enabled(len(self.snapshot.segments) > 0)

    def _refresh_history_list(self) -> None:
        query = self.history_search.text().strip().lower()
        self.history_list.clear()
        self.history_detail.clear()

        all_records = self.autosave.list_history_sessions()
        self.history_records = []
        for path, snapshot in all_records:
            if query and not self._history_match(snapshot, query):
                continue
            self.history_records.append((path, snapshot))
            title = (
                f"{snapshot.updated_at[:19].replace('T', ' ')} | "
                f"{snapshot.source_name} | "
                f"{len(snapshot.segments)} 条"
            )
            item = QListWidgetItem(title)
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            self.history_list.addItem(item)

    @staticmethod
    def _history_match(snapshot: SessionSnapshot, query: str) -> bool:
        if query in snapshot.updated_at.lower():
            return True
        if query in snapshot.source_name.lower():
            return True
        combined = " ".join(seg.text for seg in snapshot.segments).lower()
        return query in combined

    def _render_history_detail(self) -> None:
        row = self.history_list.currentRow()
        if row < 0 or row >= len(self.history_records):
            self.history_detail.clear()
            return

        _, snapshot = self.history_records[row]
        lines = [
            f"会话ID: {snapshot.session_id}",
            f"来源: {snapshot.source_name}",
            f"开始时间: {snapshot.started_at}",
            f"更新时间: {snapshot.updated_at}",
            f"片段数: {len(snapshot.segments)}",
            "",
            "转写内容:",
        ]
        for segment in snapshot.segments:
            lines.append(
                f"[{self._ms_to_hhmmss(segment.start_ms)}-{self._ms_to_hhmmss(segment.end_ms)}] {segment.text}"
            )
        self.history_detail.setPlainText("\n".join(lines))

    def _delete_selected_history(self) -> None:
        row = self.history_list.currentRow()
        if row < 0 or row >= len(self.history_records):
            QMessageBox.information(self, "未选择记录", "请先在左侧选择一条历史记录。")
            return

        path, snapshot = self.history_records[row]
        answer = QMessageBox.question(
            self,
            "确认删除",
            f"确定删除这条历史记录吗？\n来源: {snapshot.source_name}\n时间: {snapshot.updated_at}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        deleted = self.autosave.delete_history(path)
        if not deleted:
            QMessageBox.warning(self, "删除失败", "该记录文件不存在或已被删除。")
            self._refresh_history_list()
            return

        self._refresh_history_list()
        QMessageBox.information(self, "删除成功", "历史记录已删除。")

    def _export(self, fmt: str) -> None:
        if self.snapshot is None or not self.snapshot.segments:
            QMessageBox.information(self, "无可导出内容", "当前没有可导出的转写文本。")
            return

        if fmt == "docx" and self.snapshot.source_type != "file":
            QMessageBox.information(self, "导出限制", "Word 导出仅在文件模式可用。")
            return

        try:
            if fmt == "txt":
                output = self.exporter.export_txt(self.snapshot)
            elif fmt == "srt":
                output = self.exporter.export_srt(self.snapshot)
            else:
                output = self.exporter.export_docx(self.snapshot)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "导出失败", str(exc))
            return

        QMessageBox.information(self, "导出成功", f"文件已生成: {output}")

    @staticmethod
    def _ms_to_hhmmss(value: int) -> str:
        total_seconds = value // 1000
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.stop_button.isEnabled():
            answer = QMessageBox.question(
                self,
                "会话进行中",
                "当前仍在转写，是否结束并保存后退出？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer == QMessageBox.StandardButton.No:
                event.ignore()
                return
            self.stop_session()

        event.accept()


def run() -> None:
    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.windowssystem.warning=false")
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    run()
