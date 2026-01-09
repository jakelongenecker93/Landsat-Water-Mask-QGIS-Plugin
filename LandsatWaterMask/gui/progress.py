from qgis.PyQt.QtCore import Qt, pyqtSignal, QTimer
from qgis.PyQt.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QTextEdit,
    QPushButton,
)

import time


class ProcessingProgressDialog(QDialog):
    """Progress + log popup."""

    cancelRequested = pyqtSignal()
    hideRequested = pyqtSignal()

    def __init__(self, parent=None, *, modal: bool = True, allow_hide: bool = False):
        super().__init__(parent)
        self._finished = False
        self._allow_hide = bool(allow_hide)

        # Heartbeat timer (prints 'Still Processing' messages periodically)
        self._hb_timer = None
        self._hb_start = None
        self._hb_interval_s = 120

        # Ensure the window actually goes away when the user closes it.
        # (Particularly important for modeless dialogs created with .show().)
        try:
            self.setAttribute(Qt.WA_DeleteOnClose, True)
        except Exception:
            pass

        self.setWindowTitle("Landsat Water Mask — Processing")
        self.setModal(bool(modal))
        if not modal:
            self.setWindowModality(Qt.NonModal)

        layout = QVBoxLayout(self)

        # Keep the progress bar at the top (no widgets above it).
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        layout.addWidget(self.bar)

        self.lbl_status = QLabel("Starting…")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(220)
        layout.addWidget(self.log)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)

        self.btn_hide = None
        if self._allow_hide:
            self.btn_hide = QPushButton("Hide")
            self.btn_hide.setToolTip("Hide this window (processing continues in the background).")
            self.btn_hide.clicked.connect(self._on_hide)
            btn_row.addWidget(self.btn_hide)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self._on_cancel)
        btn_row.addWidget(self.btn_cancel)

        layout.addLayout(btn_row)

        self.resize(700, 520)
    def start_heartbeat(self, interval_minutes: int = 2):
        """Append a reassuring log line every N minutes while processing."""
        try:
            interval_minutes = int(interval_minutes)
        except Exception:
            interval_minutes = 2
        interval_minutes = max(1, interval_minutes)
        self._hb_interval_s = interval_minutes * 60
        if self._hb_start is None:
            self._hb_start = time.monotonic()
        if self._hb_timer is None:
            self._hb_timer = QTimer(self)
            self._hb_timer.timeout.connect(self._emit_heartbeat)
        try:
            self._hb_timer.setInterval(self._hb_interval_s * 1000)
            self._hb_timer.start()
        except Exception:
            pass

    def stop_heartbeat(self):
        try:
            if self._hb_timer is not None:
                self._hb_timer.stop()
        except Exception:
            pass

    def _emit_heartbeat(self):
        if self._finished:
            self.stop_heartbeat()
            return
        try:
            if self._hb_start is None:
                self._hb_start = time.monotonic()
            elapsed_min = int((time.monotonic() - self._hb_start + 30) / 60)
            self.append_log(f"Still Processing! Elapsed time = {elapsed_min} minutes")
        except Exception:
            pass

    def _on_cancel(self):
        if self._finished:
            # Prefer an explicit close for modeless windows.
            self.close()
            return
        self.cancelRequested.emit()

    def _on_hide(self):
        self.hideRequested.emit()
        self.hide()

    def set_status(self, text: str):
        self.lbl_status.setText(str(text))

    def set_progress(self, value: int):
        try:
            v = int(value)
        except Exception:
            v = 0
        v = max(0, min(100, v))
        self.bar.setValue(v)

    def append_log(self, text: str):
        self.log.append(str(text))

    def set_finished(self, ok: bool, message: str = "Finished."):
        self._finished = True
        try:
            self.stop_heartbeat()
        except Exception:
            pass
        self.set_progress(100 if ok else self.bar.value())
        self.set_status(message)
        self.btn_cancel.setText("Close")
        if self.btn_hide is not None:
            self.btn_hide.setEnabled(False)

    def closeEvent(self, event):
        # In background mode, treat window close as "hide" while still running.
        if self._allow_hide and not self._finished:
            event.ignore()
            self._on_hide()
            return
        super().closeEvent(event)