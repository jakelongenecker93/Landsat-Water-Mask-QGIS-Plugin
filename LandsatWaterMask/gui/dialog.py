# -*- coding: utf-8 -*-
"""Guided UI for Landsat Water Mask.

This dialog is intentionally "clean": it exposes the common workflow with
clear defaults, while keeping power-user options accessible.

It wraps the processing algorithm:
  landsat_watermask:landsat_water_mask

Compatibility: QGIS 3.10+ (Qt5) via qgis.PyQt.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple

from qgis.PyQt.QtCore import Qt, QCoreApplication, pyqtSignal
from qgis.PyQt.QtGui import QFont, QIcon
from qgis.PyQt.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QGroupBox,
    QRadioButton,
    QButtonGroup,
    QListWidget,
    QListWidgetItem,
    QCheckBox,
    QFrame,
    QSpinBox,
    QDoubleSpinBox,
    QLineEdit,
    QMessageBox,
    QWidget,
    QFormLayout,
    QScrollArea,
    QApplication,
)

from qgis.core import (
    QgsProject,
    QgsRasterLayer,
    QgsProcessingFeedback,
    QgsVectorLayer,
    QgsMapLayer,
    QgsApplication,
    QgsProcessingContext,
)

from .progress import ProcessingProgressDialog


# -----------------------------------------------------------------------------
# Background task keepalive
# -----------------------------------------------------------------------------
# NOTE:
# The guided dialog is modal and typically gets destroyed immediately after
# closing (accept/reject). If we run a background QgsTask from the dialog, we
# must keep *Python* references to the task, context, feedback, and any modeless
# progress window for as long as the task runs; otherwise they can be garbage-
# collected and QGIS may crash when the task completes.
_BG_KEEPALIVE = []


try:
    from qgis.core import QgsProcessingAlgRunnerTask  # QGIS 3.x
    HAS_BG_TASK = True
except Exception:  # pragma: no cover
    QgsProcessingAlgRunnerTask = None
    HAS_BG_TASK = False


class _GuiFeedback(QgsProcessingFeedback):
    """Processing feedback that updates a progress popup and keeps Qt responsive."""

    def __init__(self, dlg: ProcessingProgressDialog):
        super().__init__()
        self._dlg = dlg
        self._errors: List[str] = []

    @property
    def errors(self) -> List[str]:
        return list(self._errors)

    def setProgress(self, progress):  # noqa: N802  (QGIS API)
        try:
            super().setProgress(progress)
        except Exception:
            pass
        try:
            self._dlg.set_progress(int(float(progress)))
        except Exception:
            self._dlg.set_progress(0)
        QCoreApplication.processEvents()

    def pushInfo(self, info):  # noqa: N802
        try:
            super().pushInfo(info)
        except Exception:
            pass
        try:
            self._dlg.append_log(str(info))
        except Exception:
            pass
        QCoreApplication.processEvents()

    def pushWarning(self, warning):  # noqa: N802
        try:
            super().pushWarning(warning)
        except Exception:
            pass
        try:
            self._dlg.append_log(f"[WARNING] {warning}")
        except Exception:
            pass
        QCoreApplication.processEvents()

    def reportError(self, error, fatalError=False):  # noqa: N802
        try:
            super().reportError(error, fatalError)
        except Exception:
            pass
        try:
            self._errors.append(str(error))
            self._dlg.append_log(f"[ERROR] {error}")
        except Exception:
            pass
        QCoreApplication.processEvents()


ALG_ID = "landsat_watermask:landsat_water_mask"


def _hr() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setFrameShadow(QFrame.Sunken)
    return f


def _is_refl_layer(layer: QgsRasterLayer) -> bool:
    try:
        src = (layer.source() or "").split("|")[0].lower()
    except Exception:
        src = ""
    name = (layer.name() or "").lower()
    return ("_refl" in src) or ("_refl" in name)


def _is_qapixel_layer(layer: QgsRasterLayer) -> bool:
    try:
        src = (layer.source() or "").split("|")[0].lower()
    except Exception:
        src = ""
    name = (layer.name() or "").lower()
    return ("qa_pixel" in src) or ("qa_pixel" in name)


def _extract_pathrow_date_from_name(filename: str) -> Tuple[str, str, str]:
    """Best-effort parse for UX display only.

    - Path/Row: _PPPRRR_ (e.g., _022039_)
    - Date: first _YYYYMMDD_ token after the _PPPRRR_ token
    """
    base = os.path.basename(filename)
    m = re.search(r"_(\d{3})(\d{3})_(\d{8})_", base)
    if m:
        return m.group(1), m.group(2), m.group(3)

    # fallback: path/row anywhere + any date
    m2 = re.search(r"_(\d{3})(\d{3})_", base)
    m3 = re.search(r"_(\d{8})_", base)
    return (m2.group(1) if m2 else "???"), (m2.group(2) if m2 else "???"), (m3.group(1) if m3 else "????????")


class _SignalFeedback(QgsProcessingFeedback):
    """Thread-safe feedback for background Tasks.

    Emits Qt signals so the GUI can update a modeless progress popup while the
    processing task runs in QGIS' task manager.
    """

    sigProgress = pyqtSignal(int)
    sigInfo = pyqtSignal(str)
    sigWarning = pyqtSignal(str)
    sigError = pyqtSignal(str)

    def setProgress(self, progress: float):  # noqa: N802
        try:
            p = int(progress)
        except Exception:
            p = 0
        super().setProgress(progress)
        self.sigProgress.emit(max(0, min(100, p)))

    def pushInfo(self, info: str):  # noqa: N802
        super().pushInfo(info)
        self.sigInfo.emit(str(info))

    def pushWarning(self, warning: str):  # noqa: N802
        super().pushWarning(warning)
        self.sigWarning.emit(str(warning))

    def reportError(self, error: str, fatalError: bool = False):  # noqa: N802
        super().reportError(error, fatalError)
        self.sigError.emit(str(error))


class LandsatWaterMaskGuidedDialog(QDialog):
    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self.iface = iface

        icon_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "icon.png")
        self.setWindowIcon(QIcon(icon_path))
        self.setWindowTitle("Landsat Water Mask")
        self.setModal(True)
        self.resize(640, 950)

        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.setLayout(outer)

        # Make the dialog scrollable so Advanced options never go off-screen
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(scroll, 1)

        page = QWidget()
        root = QVBoxLayout()
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)
        page.setLayout(root)
        scroll.setWidget(page)

        # --- Header ---
        title = QLabel("Landsat Water Mask")
        ft = QFont()
        ft.setPointSize(14)
        ft.setBold(True)
        title.setFont(ft)

        subtitle = QLabel(
            "Build per-date <b>water/land rasters</b> & <b>water/land polygons</b> "
            "from Landsat Reflectance (REFL) and QA_PIXEL (PIXEL) layers."
        )
        subtitle.setWordWrap(True)
        subtitle.setTextFormat(Qt.RichText)

        root.addWidget(title)
        root.addWidget(subtitle)
        root.addWidget(_hr())

        # --- Inputs ---
        g_inputs = QGroupBox("1) Inputs")
        v_in = QVBoxLayout(); v_in.setSpacing(6)
        g_inputs.setLayout(v_in)

        self.chk_use_all = QCheckBox("Use all raster layers currently loaded in the project")
        self.chk_use_all.setChecked(True)
        self.chk_use_all.toggled.connect(self._sync_enabled)

        self.layer_list = QListWidget()
        self.layer_list.setMinimumHeight(170)
        self.layer_list.setToolTip(
            "Select specific raster layers to process.\n"
            "Tip: name files with '_refl' for reflectance and 'QA_PIXEL' for pixel QA."
        )

        # Keep mode choices in-sync with layer checks.
        self.layer_list.itemChanged.connect(lambda *_: self._sync_enabled())

        btn_row = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh")
        self.btn_select_all = QPushButton("Select all")
        self.btn_select_none = QPushButton("Select none")
        btn_row.addWidget(self.btn_refresh)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_select_all)
        btn_row.addWidget(self.btn_select_none)

        self.btn_refresh.clicked.connect(self._populate_layers)
        self.btn_select_all.clicked.connect(lambda: self._set_all_checks(True))
        self.btn_select_none.clicked.connect(lambda: self._set_all_checks(False))

        self.lbl_detected = QLabel("")
        self.lbl_detected.setWordWrap(True)

        v_in.addWidget(self.chk_use_all)
        v_in.addWidget(self.layer_list)
        v_in.addLayout(btn_row)
        v_in.addWidget(self.lbl_detected)

        root.addWidget(g_inputs)

        # --- Mode ---
        g_mode = QGroupBox("2) Mode")
        v_m = QVBoxLayout(); v_m.setSpacing(6)
        g_mode.setLayout(v_m)

        self.rb_refl = QRadioButton("REFL (Reflectance RGB thresholds)")
        self.rb_pixel = QRadioButton("PIXEL (QA_PIXEL water code)")
        self.rb_both = QRadioButton("BOTH (REFL + PIXEL; optional SUM outputs)")
        self.rb_both.setChecked(True)

        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.rb_refl, 0)
        self.mode_group.addButton(self.rb_pixel, 1)
        self.mode_group.addButton(self.rb_both, 2)
        self.mode_group.buttonToggled.connect(lambda *_: self._sync_enabled())

        self.chk_do_sum = QCheckBox("In BOTH mode, also create SUM outputs (datewise; dissolution of REFL+PIXEL)")
        self.chk_do_sum.setChecked(True)

        mode_hint = QLabel(
            "<b>Tip:</b> If you’re new, start with <b>PIXEL</b> (most robust), then add <b>REFL</b> later for refinement."
        )
        mode_hint.setWordWrap(True)
        mode_hint.setTextFormat(Qt.RichText)

        v_m.addWidget(self.rb_refl)
        v_m.addWidget(self.rb_pixel)
        v_m.addWidget(self.rb_both)
        v_m.addWidget(self.chk_do_sum)
        v_m.addWidget(mode_hint)

        root.addWidget(g_mode)

        # --- Outputs ---
        g_out = QGroupBox("3) Outputs")
        v_o = QVBoxLayout(); v_o.setSpacing(6)
        g_out.setLayout(v_o)

        self.rb_vec_water = QRadioButton("Water polygons (Water-once)")
        self.rb_vec_land = QRadioButton("Land polygons (Never-water / Always-land)")
        self.rb_vec_both = QRadioButton("Water + Land polygons")
        self.rb_vec_both.setChecked(True)

        self.vec_group = QButtonGroup(self)
        self.vec_group.addButton(self.rb_vec_water, 0)
        self.vec_group.addButton(self.rb_vec_land, 1)
        self.vec_group.addButton(self.rb_vec_both, 2)

        self.chk_write_water_tiffs = QCheckBox("Write Water Classification Count TIFF rasters (datewise count)")
        self.chk_write_water_tiffs.setChecked(False)

        self.chk_write_land_tiffs = QCheckBox("Write Land Classification Count TIFF rasters (datewise count)")
        self.chk_write_land_tiffs.setChecked(False)

        out_hint = QLabel(
            "You can right-click any output layer and choose <b>Export → Save Features As…</b> "
            "or <b>Save As…</b> to write to disk."
        )
        out_hint.setWordWrap(True)
        out_hint.setTextFormat(Qt.RichText)

        v_o.addWidget(self.rb_vec_water)
        v_o.addWidget(self.rb_vec_land)
        v_o.addWidget(self.rb_vec_both)
        v_o.addWidget(self.chk_write_water_tiffs)
        v_o.addWidget(self.chk_write_land_tiffs)
        v_o.addWidget(out_hint)

        root.addWidget(g_out)

        # --- Advanced (collapsible-ish) ---
        g_adv = QGroupBox("Advanced")
        g_adv.setCheckable(True)
        g_adv.setChecked(False)
        v_adv_outer = QVBoxLayout(); v_adv_outer.setContentsMargins(10, 8, 10, 10)
        g_adv.setLayout(v_adv_outer)

        adv_body = QWidget()
        v_adv = QVBoxLayout(); v_adv.setSpacing(10)
        adv_body.setLayout(v_adv)
        v_adv_outer.addWidget(adv_body)
        adv_body.setVisible(False)
        g_adv.toggled.connect(adv_body.setVisible)

        # REFL thresholds
        g_refl = QGroupBox("REFL thresholds")
        f_refl = QFormLayout(); f_refl.setLabelAlignment(Qt.AlignLeft)
        g_refl.setLayout(f_refl)

        self.le_bg_rgb = QLineEdit("0,0,0")
        self.le_bg_rgb.setToolTip("Pixels equal to this RGB triplet are treated as background and excluded.")

        self.chk_default_thresh = QCheckBox("Use default REFL water RGB thresholds")
        self.chk_default_thresh.setChecked(True)
        self.chk_default_thresh.toggled.connect(self._sync_enabled)

        # Compact 6-number input for older-QGIS parity
        self.le_rgb_thresh = QLineEdit("0,60,0,60,11,255")
        self.le_rgb_thresh.setPlaceholderText("Rmin,Rmax,Gmin,Gmax,Bmin,Bmax")
        self.le_rgb_thresh.setToolTip("Values within this range are considered water (Rmin,Rmax,Gmin,Gmax,Bmin,Bmax).")

        f_refl.addRow("Background RGB to exclude", self.le_bg_rgb)
        f_refl.addRow("", self.chk_default_thresh)
        f_refl.addRow("Custom thresholds", self.le_rgb_thresh)
        v_adv.addWidget(g_refl)

        # PIXEL water code
        g_pix = QGroupBox("PIXEL water code")
        f_pix = QFormLayout(); f_pix.setLabelAlignment(Qt.AlignLeft)
        g_pix.setLayout(f_pix)

        self.chk_default_pixelvals = QCheckBox("Use default PIXEL water values")
        self.chk_default_pixelvals.setChecked(True)
        self.chk_default_pixelvals.toggled.connect(self._sync_enabled)

        self.le_pixelvals_457 = QLineEdit("5504")
        self.le_pixelvals_457.setPlaceholderText("e.g., 5504 or 5504,1234")
        self.le_pixelvals_457.setToolTip("Number separated by comma (n,n,n,n)")

        self.le_pixelvals_89 = QLineEdit("21952")
        self.le_pixelvals_89.setPlaceholderText("e.g., 21952 or 21952,1234")
        self.le_pixelvals_89.setToolTip("Number separated by comma (n,n,n,n)")

        note_pix = QLabel("Note: Customize with caution. See Landsat docs for meanings of input variables.")
        note_pix.setWordWrap(True)

        f_pix.addRow("", self.chk_default_pixelvals)
        f_pix.addRow("Landsat 4/5/7", self.le_pixelvals_457)
        f_pix.addRow("Landsat 8/9", self.le_pixelvals_89)
        f_pix.addRow("", note_pix)
        v_adv.addWidget(g_pix)

        # Smoothing
        g_sm = QGroupBox("Smoothing")
        f_sm = QFormLayout(); f_sm.setLabelAlignment(Qt.AlignLeft)
        g_sm.setLayout(f_sm)

        # Fast pixel-based smoothing (majority filter before polygonizing)
        self.chk_pixsmooth = QCheckBox("Pixel smoothing (very fast)")
        self.chk_pixsmooth.setChecked(False)
        self.chk_pixsmooth.setToolTip("Applies a majority filter (N×N) to the binary mask before polygonizing.")
        self.chk_pixsmooth.toggled.connect(self._on_pixsmooth_toggled)

        self.sp_pixsmooth = QSpinBox()
        self.sp_pixsmooth.setRange(1, 25)
        self.sp_pixsmooth.setSingleStep(2)
        self.sp_pixsmooth.setValue(5)
        self.sp_pixsmooth.setToolTip("Kernel size in pixels (odd numbers). 3 & 5 are good defaults.")

        note_pix = QLabel("Note: Pixel smoothing significantly speeds processing of large rasters.")
        note_pix.setWordWrap(True)

        f_sm.addRow("", self.chk_pixsmooth)
        f_sm.addRow("Kernel size (pixels)", self.sp_pixsmooth)
        f_sm.addRow("", note_pix)

        # Smoothify (vector smoothing; more intensive)
        self.chk_smoothify = QCheckBox("Smoothify output polygons (more intensive)")
        self.chk_smoothify.setChecked(False)
        self.chk_smoothify.setToolTip("Uses the Smoothify library for optimized corner-smoothing.")
        self.chk_smoothify.toggled.connect(self._on_smoothify_toggled)

        self.sp_iters = QSpinBox(); self.sp_iters.setRange(1, 8); self.sp_iters.setValue(1)
        self.sp_weight = QDoubleSpinBox(); self.sp_weight.setRange(0.05, 0.49); self.sp_weight.setSingleStep(0.05); self.sp_weight.setValue(0.35)
        self.sp_weight.setToolTip("Chaikin weight; smaller = less smoothing.")

        self.sp_presimplify = QDoubleSpinBox()
        self.sp_presimplify.setRange(0.0, 1000.0)
        self.sp_presimplify.setDecimals(1)
        self.sp_presimplify.setSingleStep(5.0)
        self.sp_presimplify.setValue(30.0)
        self.sp_presimplify.setToolTip(
            "Before Smoothify, simplify boundaries by this tolerance (meters). 30m is standard.\n"
            "This can drastically speed up smoothing on very large coastlines. Set to 0 to disable."
        )

        note_sm = QLabel("Note: Smoothify can take a very long time on large/complex rasters.")
        note_sm.setWordWrap(True)

        f_sm.addRow("", self.chk_smoothify)
        f_sm.addRow("Iterations", self.sp_iters)
        f_sm.addRow("Weight", self.sp_weight)
        f_sm.addRow("Pre-simplify (m)", self.sp_presimplify)
        f_sm.addRow("", note_sm)

        v_adv.addWidget(g_sm)

        # Processing utilities (replaces the old "Open Processing Dialog" button)
        g_utils = QGroupBox("Processing utilities")
        v_utils = QVBoxLayout(); v_utils.setSpacing(6)
        g_utils.setLayout(v_utils)

        r1 = QHBoxLayout(); r1.setSpacing(8)
        self.btn_copy_cmd = QPushButton("Copy as Python command")
        self.btn_batch = QPushButton("Batch processing…")
        r1.addWidget(self.btn_copy_cmd)
        r1.addWidget(self.btn_batch)
        v_utils.addLayout(r1)

        hint = QLabel(
            "These are power-user tools normally found in the Processing dialog. "
            "They live here for ease of access."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666;")
        v_utils.addWidget(hint)

        v_adv.addWidget(g_utils)

        root.addWidget(g_adv)

        # --- Buttons (always visible) ---
        bottom = QWidget()
        bottom_l = QVBoxLayout()
        bottom_l.setContentsMargins(14, 6, 14, 14)
        bottom_l.setSpacing(10)
        bottom.setLayout(bottom_l)
        bottom_l.addWidget(_hr())

        # Execution (always visible)
        self.chk_background = QCheckBox("Run in background (recommended; keep QGIS responsive)")
        self.chk_background.setChecked(True)
        if not HAS_BG_TASK:
            self.chk_background.setEnabled(False)
            self.chk_background.setChecked(False)
            self.chk_background.setToolTip("Background tasks are not available in this QGIS version.")
        else:
            self.chk_background.setToolTip("Runs as a QGIS Task so you can keep using QGIS while it processes.")
        bottom_l.addWidget(self.chk_background)

        exec_note = QLabel("Tip: Background mode is usually best for large datasets. Helps prevent freezing / crashing.")
        exec_note.setWordWrap(True)
        exec_note.setStyleSheet("color: #666;")
        bottom_l.addWidget(exec_note)

        btns = QHBoxLayout()
        self.btn_help = QPushButton("Quick Start Help")
        self.btn_run = QPushButton("Run")
        self.btn_close = QPushButton("Close")

        self.btn_run.setDefault(True)

        btns.addWidget(self.btn_help)
        btns.addStretch(1)
        btns.addWidget(self.btn_close)
        btns.addWidget(self.btn_run)

        bottom_l.addLayout(btns)
        outer.addWidget(bottom)

        self.btn_close.clicked.connect(self.reject)
        self.btn_help.clicked.connect(self._show_help)
        self.btn_run.clicked.connect(self._run)
        self.btn_copy_cmd.clicked.connect(self._copy_python_command)
        self.btn_batch.clicked.connect(self._open_batch_processing)

        # Populate inputs and sync UI
        self._populate_layers()
        self._sync_enabled()

    # -----------------------------
    # UI helpers
    # -----------------------------
    def _populate_layers(self):
        self.layer_list.blockSignals(True)
        self.layer_list.clear()
        rasters = [lyr for lyr in QgsProject.instance().mapLayers().values() if isinstance(lyr, QgsRasterLayer)]
        for lyr in sorted(rasters, key=lambda l: (l.name() or "")):
            item = QListWidgetItem(lyr.name() or "(unnamed raster)")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setData(Qt.UserRole, lyr.id())

            # A short, helpful hint in the item tooltip
            try:
                src = (lyr.source() or "").split("|")[0]
            except Exception:
                src = ""
            pr_p, pr_r, dt = _extract_pathrow_date_from_name(src)
            tags = []
            if _is_refl_layer(lyr):
                tags.append("REFL")
            if _is_qapixel_layer(lyr):
                tags.append("QA_PIXEL")
            tag_s = (" / ".join(tags) + " — ") if tags else ""
            item.setToolTip(f"{tag_s}PR={pr_p}/{pr_r}, DATE={dt}\n{src}")
            self.layer_list.addItem(item)

        self.layer_list.blockSignals(False)
        self._sync_enabled()

    def _set_all_checks(self, checked: bool):
        for i in range(self.layer_list.count()):
            self.layer_list.item(i).setCheckState(Qt.Checked if checked else Qt.Unchecked)
        self._sync_enabled()

    def _selected_layers(self) -> List[QgsRasterLayer]:
        # When "use all" is checked, return empty list -> processing alg auto-collects
        if self.chk_use_all.isChecked():
            return []

        wanted_ids = []
        for i in range(self.layer_list.count()):
            it = self.layer_list.item(i)
            if it.checkState() == Qt.Checked:
                wanted_ids.append(it.data(Qt.UserRole))

        layers = []
        for lid in wanted_ids:
            lyr = QgsProject.instance().mapLayer(lid)
            if isinstance(lyr, QgsRasterLayer):
                layers.append(lyr)
        return layers

    def _current_input_layers(self) -> List[QgsRasterLayer]:
        """Return the raster layers the algorithm will actually use."""
        if self.chk_use_all.isChecked():
            return [
                lyr for lyr in QgsProject.instance().mapLayers().values()
                if isinstance(lyr, QgsRasterLayer)
            ]
        return self._selected_layers()

    def _mode_idx(self) -> int:
        return self.mode_group.checkedId()

    def _vec_idx(self) -> int:
        return self.vec_group.checkedId()

    def _update_detected_label(self):
        layers = self._current_input_layers()
        n_refl = sum(1 for l in layers if _is_refl_layer(l))
        n_qp = sum(1 for l in layers if _is_qapixel_layer(l))
        total = len(layers)
        self.lbl_detected.setText(
            f"Detected: <b>{total}</b> raster(s) — <b>{n_refl}</b> REFL, <b>{n_qp}</b> QA_PIXEL."
        )
        self.lbl_detected.setTextFormat(Qt.RichText)

    def _on_pixsmooth_toggled(self, checked: bool):
        # Pixel smoothing and Smoothify are mutually exclusive
        if checked and self.chk_smoothify.isChecked():
            self.chk_smoothify.blockSignals(True)
            self.chk_smoothify.setChecked(False)
            self.chk_smoothify.blockSignals(False)
        self._sync_enabled()

    def _on_smoothify_toggled(self, checked: bool):
        # Pixel smoothing and Smoothify are mutually exclusive
        if checked and self.chk_pixsmooth.isChecked():
            self.chk_pixsmooth.blockSignals(True)
            self.chk_pixsmooth.setChecked(False)
            self.chk_pixsmooth.blockSignals(False)
        self._sync_enabled()

    def _sync_enabled(self):
        # Smoothing options are mutually exclusive (either/or)
        pix = self.chk_pixsmooth.isChecked()
        sm = self.chk_smoothify.isChecked()
        self.chk_smoothify.setEnabled(not pix)
        self.chk_pixsmooth.setEnabled(not sm)

        # Enable/disable layer list based on use-all
        use_all = self.chk_use_all.isChecked()
        self.layer_list.setEnabled(not use_all)
        self.btn_select_all.setEnabled(not use_all)
        self.btn_select_none.setEnabled(not use_all)

        # Enable/disable mode choices based on what is actually available.
        layers = self._current_input_layers()
        has_any = len(layers) > 0
        has_refl_layers = any(_is_refl_layer(l) for l in layers)
        has_qapixel_layers = any(_is_qapixel_layer(l) for l in layers)

        self.rb_refl.setEnabled(has_refl_layers)
        self.rb_pixel.setEnabled(has_qapixel_layers)
        self.rb_both.setEnabled(has_refl_layers and has_qapixel_layers)

        # Helpful tooltips when disabled
        self.rb_refl.setToolTip("" if has_refl_layers else "Needs REFL rasters (filename/layer name contains '_refl').")
        self.rb_pixel.setToolTip("" if has_qapixel_layers else "Needs QA_PIXEL rasters (filename/layer name contains 'QA_PIXEL').")
        self.rb_both.setToolTip("" if (has_refl_layers and has_qapixel_layers) else "Needs both REFL and QA_PIXEL rasters loaded/selected.")

        # If the current selection becomes invalid (e.g., BOTH selected but only
        # PIXEL layers are loaded), automatically switch to a valid mode.
        cur = self._mode_idx()
        desired = cur
        if cur == 2 and not (has_refl_layers and has_qapixel_layers):
            desired = 0 if has_refl_layers else (1 if has_qapixel_layers else -1)
        elif cur == 0 and not has_refl_layers:
            desired = 1 if has_qapixel_layers else -1
        elif cur == 1 and not has_qapixel_layers:
            desired = 0 if has_refl_layers else -1

        if desired in (0, 1, 2) and desired != cur:
            self.mode_group.blockSignals(True)
            try:
                if desired == 0:
                    self.rb_refl.setChecked(True)
                elif desired == 1:
                    self.rb_pixel.setChecked(True)
                elif desired == 2:
                    self.rb_both.setChecked(True)
            finally:
                self.mode_group.blockSignals(False)

        # Disable Run if nothing is selected/available.
        can_run = has_any and (has_refl_layers or has_qapixel_layers)
        self.btn_run.setEnabled(can_run)
        if not can_run:
            self.btn_run.setToolTip("Load/select REFL and/or QA_PIXEL rasters to enable Run.")
        else:
            self.btn_run.setToolTip("")

        # BOTH: only enable SUM checkbox when BOTH selected
        self.chk_do_sum.setEnabled(self._mode_idx() == 2)

        # REFL-only advanced controls
        has_refl = self._mode_idx() in (0, 2)
        self.le_bg_rgb.setEnabled(has_refl)
        self.chk_default_thresh.setEnabled(has_refl)
        self.le_rgb_thresh.setEnabled(has_refl and (not self.chk_default_thresh.isChecked()))

        # PIXEL-only advanced controls
        has_pixel = self._mode_idx() in (1, 2)
        self.chk_default_pixelvals.setEnabled(has_pixel)
        self.le_pixelvals_457.setEnabled(has_pixel and (not self.chk_default_pixelvals.isChecked()))
        self.le_pixelvals_89.setEnabled(has_pixel and (not self.chk_default_pixelvals.isChecked()))

        # Pixel smoothing controls
        self.sp_pixsmooth.setEnabled(self.chk_pixsmooth.isChecked())

        # Smoothify numeric controls
        self.sp_iters.setEnabled(self.chk_smoothify.isChecked())
        self.sp_weight.setEnabled(self.chk_smoothify.isChecked())
        self.sp_presimplify.setEnabled(self.chk_smoothify.isChecked())

        self._update_detected_label()

    # -----------------------------
    # Actions
    # -----------------------------
    def _show_help(self):
        txt = (
            "<h3>Quick Start</h3>"
            "<ol>"
            "<li>Load Landsat rasters into QGIS (drag/drop or Layer → Add Layer…)</li>"
            "<li>Choose <b>Mode</b>: REFL, PIXEL, or BOTH</li>"
            "<li>Choose outputs (Water/Land polygons; optional TIFFs)</li>"
            "<li>Click <b>Run</b></li>"
            "</ol>"
            "<p><b>How water counts work:</b> rasters are counted per unique acquisition date (YYYYMMDD) "
            "so overlapping same-day scenes only count once.</p>"
            "<p><b>Where files are saved:</b> by default outputs are temporary layers added to the map. "
            "Right-click an output layer to export it to disk.</p>"
            "<p><b>Background mode:</b> enable <b>Run in background</b> (near the bottom) to keep QGIS responsive while it runs.</p>"
            "<p><b>Smoothing:</b> in <b>Advanced</b>, you can enable <b>Pixel smoothing</b> (fast) or <b>Smoothify</b> (slow / more intensive). "
            "Pixel smoothing is recommended for large rasters and replaces the Nearest Neighbor (slow / intensive) polygons.</p>"
            "<p><b>Advanced:</b> By modifying <b>REFL thresholds</b> & <b>PIXEL water codes</b> it's possible to target objects and phenomena other than water/land. White clouds or red buildings, for instance.</p>"

            "<hr>"
            "<h3>Acceptable File Inputs</h3>"
            "<p>Acceptable file inputs can be sourced from <u>Earth Explorer</u>.</p>"

            "<p><b>Landsat Collection 2 Level-2</b><br>"
            "⇒ Landsat 8-9 OLI/TIRS C2 L2<br>"
            "⇒ Landsat 7 ETM+ C2 L2<br>"
            "⇒ Landsat 4-5 TM C2 L2<br>"
            "&nbsp;&nbsp;&nbsp;&nbsp;↪ <code>QA_PIXEL.TIF</code><br>"
            "<br>"
            "<b>Landsat Collection 2 Level-1</b><br>"
            "⇒ Landsat 8-9 OLI/TIRS C2 L1<br>"
            "⇒ Landsat 7 ETM+ C2 L1<br>"
            "⇒ Landsat 4-5 TM C2 L1<br>"
            "&nbsp;&nbsp;&nbsp;&nbsp;↪ <code>Full Resolution Browse (Reflective Color) GeoTIFF</code>"
            "</p>"

            "</ul>"
            "</li>"
            "</ol>"
        )
        QMessageBox.information(self, "Landsat Water Mask — Help", txt)

    def _copy_python_command(self):
        """Copy a runnable Python command (similar to Processing dialog 'Copy as Python command')."""

        # Prefer explicit layer references so the command is reproducible.
        layers = self._current_input_layers()
        if not layers:
            try:
                self.iface.messageBar().pushWarning("Landsat Water Mask", "No input rasters selected.")
            except Exception:
                pass
            return

        layer_exprs = [f"QgsProject.instance().mapLayer('{lyr.id()}')" for lyr in layers]

        mode = self._mode_idx()
        params_lines = [
            "params = {",
            "    'INPUT_LAYERS': [",
        ]
        for expr in layer_exprs:
            params_lines.append(f"        {expr},")
        params_lines.extend([
            "    ],",
            f"    'MODE': {mode},",
            f"    'VEC_WRITE': {self._vec_idx()},",
            f"    'WRITE_TIFFS': {bool(self.chk_write_water_tiffs.isChecked())},",
            f"    'WRITE_LAND_TIFFS': {bool(self.chk_write_land_tiffs.isChecked())},",
            f"    'DO_SUM': {bool(self.chk_do_sum.isChecked())},",
            f"    'BG_RGB': '{(self.le_bg_rgb.text() or '0,0,0')}',",
            f"    'KEEP_DEFAULT': {bool(self.chk_default_thresh.isChecked())},",
            f"    'RGB_THRESHOLDS': '{(self.le_rgb_thresh.text() or '0,60,0,60,11,255')}',",
            f"    'PIXEL_KEEP_DEFAULT_WATER': {bool(self.chk_default_pixelvals.isChecked())},",
            f"    'PIXEL_WATER_VALUES_457': '{(self.le_pixelvals_457.text() or '5504')}',",
            f"    'PIXEL_WATER_VALUES_89': '{(self.le_pixelvals_89.text() or '21952')}',",
            f"    'PIXEL_SMOOTH': {bool(self.chk_pixsmooth.isChecked())},",
            f"    'PIXEL_SMOOTH_SIZE': {int(self.sp_pixsmooth.value())},",
            f"    'SMOOTHIFY': {bool(self.chk_smoothify.isChecked())},",
            f"    'SMOOTHIFY_ITERS': {int(self.sp_iters.value())},",
            f"    'SMOOTHIFY_WEIGHT': {float(self.sp_weight.value())},",
            f"    'SMOOTHIFY_PRESIMPLIFY_M': {float(self.sp_presimplify.value())},",
            "    'OUT_REFL_TIF': 'TEMPORARY_OUTPUT',",
            "    'OUT_REFL_LAND_TIF': 'TEMPORARY_OUTPUT',",
            "    'OUT_REFL_VEC': 'TEMPORARY_OUTPUT',",
            "    'OUT_REFL_LAND_VEC': 'TEMPORARY_OUTPUT',",
            "    'OUT_PIXEL_TIF': 'TEMPORARY_OUTPUT',",
            "    'OUT_PIXEL_LAND_TIF': 'TEMPORARY_OUTPUT',",
            "    'OUT_PIXEL_VEC': 'TEMPORARY_OUTPUT',",
            "    'OUT_PIXEL_LAND_VEC': 'TEMPORARY_OUTPUT',",
            "    'OUT_SUM_TIF': 'TEMPORARY_OUTPUT',",
            "    'OUT_SUM_LAND_TIF': 'TEMPORARY_OUTPUT',",
            "    'OUT_SUM_VEC': 'TEMPORARY_OUTPUT',",
            "    'OUT_SUM_LAND_VEC': 'TEMPORARY_OUTPUT',",
            "}",
        ])

        cmd = "\n".join(
            [
                "import processing",
                "from qgis.core import QgsProject",
                "",
                *params_lines,
                "",
                f"processing.run('{ALG_ID}', params)",
            ]
        )

        QApplication.clipboard().setText(cmd)
        self.iface.messageBar().pushSuccess("Landsat Water Mask", "Python command copied to clipboard.")

    def _open_batch_processing(self):
        """Open the Batch Processing UI for this algorithm (best-effort across QGIS versions)."""
        try:
            # First try the dedicated batch dialog (if available)
            try:
                from qgis.core import QgsApplication
                from processing.gui.BatchAlgorithmDialog import BatchAlgorithmDialog

                alg = QgsApplication.processingRegistry().algorithmById(ALG_ID)
                if alg is None:
                    raise RuntimeError("Algorithm not found in registry")

                dlg = BatchAlgorithmDialog(alg, self)
                dlg.exec()
                return
            except Exception:
                pass

            # Fallback: open the standard algorithm dialog; users can click "Run as batch process".
            import processing
            if hasattr(processing, "execAlgorithmDialog"):
                processing.execAlgorithmDialog(ALG_ID, {})
            else:
                dlg = processing.createAlgorithmDialog(ALG_ID, {})
                dlg.exec()
        except Exception as e:
            QMessageBox.warning(self, "Landsat Water Mask", f"Could not open Batch Processing: {e}")

    def _open_processing_settings(self):
        """Open QGIS options/settings (Processing settings live there)."""
        try:
            if hasattr(self.iface, "showOptionsDialog"):
                self.iface.showOptionsDialog()
            else:
                QMessageBox.information(
                    self,
                    "Landsat Water Mask",
                    "Open QGIS: Settings → Options… → Processing",
                )
        except Exception as e:
            QMessageBox.warning(self, "Landsat Water Mask", f"Could not open settings: {e}")

    def _load_results_to_map(self, results: dict):
        """Load outputs into QGIS (foreground or background runs)."""
        if not isinstance(results, dict) or not results:
            return

        project = QgsProject.instance()

        # Track existing sources to avoid duplicating layers
        existing_sources = set()
        for lyr in project.mapLayers().values():
            try:
                existing_sources.add(lyr.source())
            except Exception:
                pass

        # -------------------------
        # Raster outputs (single + list)
        # -------------------------
        raster_entries = []  # (src, kind, cat)

        def _append_entry(src, kind, cat):
            if not src:
                return
            raster_entries.append((str(src), kind, cat))

        # Single outputs
        _append_entry(results.get('OUT_REFL_TIF'), 'water', 'REFL')
        _append_entry(results.get('OUT_PIXEL_TIF'), 'water', 'QA_PIXEL')
        _append_entry(results.get('OUT_SUM_TIF'), 'water', 'SUM')
        _append_entry(results.get('OUT_REFL_LAND_TIF'), 'land', 'REFL')
        _append_entry(results.get('OUT_PIXEL_LAND_TIF'), 'land', 'QA_PIXEL')
        _append_entry(results.get('OUT_SUM_LAND_TIF'), 'land', 'SUM')

        # List outputs
        list_map = [
            ('OUT_REFL_TIF_LIST', 'water', 'REFL'),
            ('OUT_PIXEL_TIF_LIST', 'water', 'QA_PIXEL'),
            ('OUT_SUM_TIF_LIST', 'water', 'SUM'),
            ('OUT_REFL_LAND_TIF_LIST', 'land', 'REFL'),
            ('OUT_PIXEL_LAND_TIF_LIST', 'land', 'QA_PIXEL'),
            ('OUT_SUM_LAND_TIF_LIST', 'land', 'SUM'),
        ]
        for key, kind, cat in list_map:
            blob = results.get(key)
            if not blob:
                continue
            for line in str(blob).splitlines():
                line = line.strip()
                if line:
                    _append_entry(line, kind, cat)

        # De-dupe preserving order
        seen = set()
        ordered = []
        for src, kind, cat in raster_entries:
            if src in seen:
                continue
            seen.add(src)
            ordered.append((src, kind, cat))

        # Decide whether to show category suffixes
        cats_by_kind = {'water': set(), 'land': set()}
        for _, kind, cat in ordered:
            cats_by_kind.setdefault(kind, set()).add(cat)

        def _parse_pr(src_path: str) -> str | None:
            stem = Path(src_path).stem
            m = re.search(r"_(\d{6})$", stem)
            if not m:
                return None
            tok = m.group(1)
            return f" P{tok[:3]}R{tok[3:]}"

        def _layer_name(kind: str, cat: str, src_path: str) -> str:
            base = 'Water Classification Count' if kind == 'water' else 'Land Classification Count'
            suffix = f" ({cat})" if len(cats_by_kind.get(kind, set())) > 1 else ''
            pr = _parse_pr(src_path) or ''
            return f"{base}{suffix}{pr}"

        for src, kind, cat in ordered:
            if src in existing_sources:
                continue
            name = _layer_name(kind, cat, src)
            rlyr = QgsRasterLayer(src, name)
            if rlyr.isValid():
                project.addMapLayer(rlyr)
                existing_sources.add(src)

        # -------------------------
        # Vector outputs
        # -------------------------
        for key in (
            "OUT_REFL_VEC",
            "OUT_REFL_LAND_VEC",
            "OUT_PIXEL_VEC",
            "OUT_PIXEL_LAND_VEC",
            "OUT_SUM_VEC",
            "OUT_SUM_LAND_VEC",
        ):
            src = results.get(key)
            if not src:
                continue
            src = str(src)
            if src in existing_sources:
                continue
            name = Path(src).stem
            vlyr = QgsVectorLayer(src, name, "ogr")
            if vlyr.isValid():
                project.addMapLayer(vlyr)
                existing_sources.add(src)
    def _run_background(self, params: dict):
        """Run the algorithm as a QGIS Task (background), with a modeless progress window."""

        if not HAS_BG_TASK or QgsProcessingAlgRunnerTask is None:
            QMessageBox.information(
                self,
                "Background processing unavailable",
                "This QGIS version does not support running this algorithm in the background as a Task.\n\n"
                "Uncheck 'Run in background' to run in the foreground.",
            )
            return

        progress_dlg = ProcessingProgressDialog(self.iface.mainWindow(), modal=False, allow_hide=True)
        progress_dlg.set_status("Nom Nom Nom... Processing Bits. They're Delicious!")
        progress_dlg.show()
        try:
            progress_dlg.start_heartbeat(2)
        except Exception:
            pass

        feedback = _SignalFeedback()
        feedback.sigProgress.connect(progress_dlg.set_progress)
        feedback.sigInfo.connect(progress_dlg.append_log)
        feedback.sigWarning.connect(lambda w: progress_dlg.append_log(f"[WARNING] {w}"))
        feedback.sigError.connect(lambda e: progress_dlg.append_log(f"[ERROR] {e}"))

        alg_proto = QgsApplication.processingRegistry().algorithmById(ALG_ID)
        if alg_proto is None:
            progress_dlg.append_log("[ERROR] Could not find algorithm in Processing registry.")
            progress_dlg.set_finished(False, "Failed.")
            QMessageBox.critical(self, "Landsat Water Mask", "Could not find the processing algorithm in the registry.")
            return

        try:
            alg = alg_proto.createInstance()
        except Exception:
            alg = alg_proto

        # IMPORTANT:
        # Do NOT attach QgsProject.instance() to the processing context for background tasks.
        # QgsProject lives on the main thread and referencing it from a Task can crash QGIS.
        context = QgsProcessingContext()

        task = QgsProcessingAlgRunnerTask(alg, params, context, feedback)

        # Keep Python references alive until completion (see QGIS warning).
        keep = {
            "task": task,
            "context": context,
            "feedback": feedback,
            "progress": progress_dlg,
        }
        _BG_KEEPALIVE.append(keep)

        def _on_executed(success, results):
            try:
                # Drop keepalive refs now that the task has finished.
                try:
                    if keep in _BG_KEEPALIVE:
                        _BG_KEEPALIVE.remove(keep)
                except Exception:
                    pass

                if feedback.isCanceled():
                    progress_dlg.append_log("Canceled.")
                    progress_dlg.set_finished(False, "Canceled.")
                    try:
                        self.iface.messageBar().pushWarning("Landsat Water Mask", "Canceled.")
                    except Exception:
                        pass
                    return

                if success:
                    progress_dlg.append_log("Completed.")
                    progress_dlg.set_finished(True, "Completed.")
                    try:
                        from qgis.PyQt.QtCore import QTimer
                        QTimer.singleShot(250, progress_dlg.close)
                    except Exception:
                        try:
                            progress_dlg.close()
                        except Exception:
                            pass

                    if isinstance(results, dict):
                        self._load_results_to_map(results)
                    try:
                        self.iface.messageBar().pushSuccess("Landsat Water Mask", "Completed. Outputs added to the map.")
                    except Exception:
                        pass
                else:
                    progress_dlg.append_log("Failed.")
                    progress_dlg.set_finished(False, "Failed.")
                    try:
                        self.iface.messageBar().pushWarning("Landsat Water Mask", "Failed. See log for details.")
                    except Exception:
                        pass
            except Exception as e:
                try:
                    progress_dlg.append_log(f"[ERROR] {e}")
                    progress_dlg.set_finished(False, "Failed.")
                except Exception:
                    pass

        task.executed.connect(_on_executed)
        progress_dlg.cancelRequested.connect(task.cancel)

        QgsApplication.taskManager().addTask(task)
        try:
            self.iface.messageBar().pushInfo("Landsat Water Mask", "Running in background…")
        except Exception:
            pass

        # Close this (modal) guided dialog so QGIS stays usable
        self.accept()

    def _run(self):
        # Determine the actual input layers + choose the most sensible mode.
        input_layers = self._current_input_layers()
        mode = self._mode_idx()

        if not input_layers:
            QMessageBox.information(
                self,
                "No input rasters selected",
                "Load/select REFL and/or QA_PIXEL raster layers, then try again.",
            )
            return

        n_refl = sum(1 for l in input_layers if _is_refl_layer(l))
        n_qp = sum(1 for l in input_layers if _is_qapixel_layer(l))

        # If the user managed to end up with an invalid mode selection, silently
        # fall back to a mode that matches what is available (no warnings).
        if mode == 2 and (n_refl == 0 or n_qp == 0):
            mode = 0 if n_refl > 0 else (1 if n_qp > 0 else mode)
        elif mode == 0 and n_refl == 0 and n_qp > 0:
            mode = 1
        elif mode == 1 and n_qp == 0 and n_refl > 0:
            mode = 0

        if (mode == 0 and n_refl == 0) or (mode == 1 and n_qp == 0) or (mode == 2 and (n_refl == 0 or n_qp == 0)):
            QMessageBox.information(
                self,
                "No compatible inputs",
                "The selected rasters don't include the required inputs for the chosen mode.\n\n"
                "REFL mode needs '_refl' layers; PIXEL mode needs 'QA_PIXEL' layers.",
            )
            return

        # IMPORTANT:
        # Passing QgsRasterLayer objects directly into a background Task can crash QGIS because
        # those layers are QObjects owned by the main thread. Instead, pass *file paths*.
        # (Processing will load layers safely in the worker thread.)
        input_sources = []
        for lyr in input_layers:
            try:
                src = (lyr.source() or '').split('|')[0]
            except Exception:
                src = ''
            if src:
                input_sources.append(src)

        params = {
            # Inputs
            "INPUT_LAYERS": input_sources,

            # Core options
            "MODE": mode,
            "VEC_WRITE": self._vec_idx(),
            "WRITE_TIFFS": bool(self.chk_write_water_tiffs.isChecked()),
            "WRITE_LAND_TIFFS": bool(self.chk_write_land_tiffs.isChecked()),
            "DO_SUM": bool(self.chk_do_sum.isChecked()),

            # REFL-only
            "BG_RGB": (self.le_bg_rgb.text() or "0,0,0"),
            "KEEP_DEFAULT": bool(self.chk_default_thresh.isChecked()),
            "RGB_THRESHOLDS": (self.le_rgb_thresh.text() or "0,60,0,60,11,255"),

            # PIXEL water code (QA_PIXEL integer values treated as water; comma-separated)
            "PIXEL_KEEP_DEFAULT_WATER": bool(self.chk_default_pixelvals.isChecked()),
            "PIXEL_WATER_VALUES_457": (self.le_pixelvals_457.text() or "5504"),
            "PIXEL_WATER_VALUES_89": (self.le_pixelvals_89.text() or "21952"),

            # Pixel smoothing (fast; majority filter)
            "PIXEL_SMOOTH": bool(self.chk_pixsmooth.isChecked()),
            "PIXEL_SMOOTH_SIZE": int(self.sp_pixsmooth.value()),

            # Smoothify
            "SMOOTHIFY": bool(self.chk_smoothify.isChecked()),
            "SMOOTHIFY_ITERS": int(self.sp_iters.value()),
            "SMOOTHIFY_WEIGHT": float(self.sp_weight.value()),
            "SMOOTHIFY_PRESIMPLIFY_M": float(self.sp_presimplify.value()),

            # Outputs: keep temporary; we load outputs back into the map after completion
            "OUT_REFL_TIF": "TEMPORARY_OUTPUT",
            "OUT_REFL_LAND_TIF": "TEMPORARY_OUTPUT",
            "OUT_REFL_VEC": "TEMPORARY_OUTPUT",
            "OUT_REFL_LAND_VEC": "TEMPORARY_OUTPUT",
            "OUT_PIXEL_TIF": "TEMPORARY_OUTPUT",
            "OUT_PIXEL_LAND_TIF": "TEMPORARY_OUTPUT",
            "OUT_PIXEL_VEC": "TEMPORARY_OUTPUT",
            "OUT_PIXEL_LAND_VEC": "TEMPORARY_OUTPUT",
            "OUT_SUM_TIF": "TEMPORARY_OUTPUT",
            "OUT_SUM_LAND_TIF": "TEMPORARY_OUTPUT",
            "OUT_SUM_VEC": "TEMPORARY_OUTPUT",
            "OUT_SUM_LAND_VEC": "TEMPORARY_OUTPUT",
        }

        progress_dlg = None
        feedback = None
        # Prevent accidental double-runs while processing.
        self.btn_run.setEnabled(False)

        # Optional: run as a background Task
        if hasattr(self, 'chk_background') and self.chk_background.isChecked():
            if HAS_BG_TASK:
                # In background mode, keep the Close button usable so users can dismiss this window.
                self.btn_close.setEnabled(True)
                self._run_background(params)
                return
            else:
                # Fall back to foreground if Tasks are unavailable
                QMessageBox.information(
                    self,
                    "Background processing unavailable",
                    "This QGIS version does not support running this algorithm in the background as a Task.\n\n"
                    "Continuing in the foreground.",
                )

        # Foreground run: disable Close while the modal progress window is active.
        self.btn_close.setEnabled(False)

        try:
            progress_dlg = ProcessingProgressDialog(self, modal=True)
            progress_dlg.set_status("Nom Nom Nom... Processing Bits. They're Delicious!")
            progress_dlg.show()
            try:
                progress_dlg.start_heartbeat(2)
            except Exception:
                pass
            QCoreApplication.processEvents()

            feedback = _GuiFeedback(progress_dlg)
            progress_dlg.cancelRequested.connect(feedback.cancel)

            import processing
            results = processing.run(ALG_ID, params, feedback=feedback)

            if feedback.isCanceled():
                self.iface.messageBar().pushWarning("Landsat Water Mask", "Canceled.")
                return

            # Best-effort layer loading (some QGIS versions already load temp outputs)
            if isinstance(results, dict):
                self._load_results_to_map(results)

            self.iface.messageBar().pushSuccess("Landsat Water Mask", "Completed. Outputs added to the map.")
            self.accept()

        except Exception as e:
            # Treat cancel as a non-error
            if (feedback is not None and feedback.isCanceled()) or ("Canceled" in str(e)):
                self.iface.messageBar().pushWarning("Landsat Water Mask", "Canceled.")
            else:
                QMessageBox.critical(self, "Landsat Water Mask", f"Run failed:\n\n{e}")
        finally:
            try:
                if progress_dlg is not None:
                    progress_dlg.close()
            except Exception:
                pass
            self.btn_run.setEnabled(True)
            self.btn_close.setEnabled(True)
