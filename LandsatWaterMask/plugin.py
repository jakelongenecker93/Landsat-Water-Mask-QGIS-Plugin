# -*- coding: utf-8 -*-
import os

from qgis.core import QgsApplication
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QMenu

from .processing.provider import LandsatWaterMaskProvider


class LandsatWaterMaskPlugin:
    """Qt6-compatible QGIS plugin.

    - Registers a Processing provider (appears in the Processing Toolbox)
    - Adds a toolbar/menu action to open the algorithm dialog
    """

    def __init__(self, iface):
        self.iface = iface
        self.provider = None
        self.action = None

        # Menu bookkeeping (so we can remove cleanly on unload)
        self._conv_menu = None
        self._added_via_iface_raster_menu = False

    def initGui(self):
        # Register processing provider
        self.provider = LandsatWaterMaskProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

        # Toolbar/menu action
        icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
        self.action = QAction(QIcon(icon_path), "Landsat Water Mask", self.iface.mainWindow())
        self.action.setWhatsThis("Open the Landsat Water Mask guided dialog")
        self.action.triggered.connect(self.open_dialog)

        # Place the action under Raster -> Conversion (requested for Raster tools).
        self._add_to_raster_conversion_menu()

        # Optional: keep a toolbar icon (quick access)
        self.iface.addToolBarIcon(self.action)  # shows on the top toolbar (ribbon)

    def unload(self):
        # Remove toolbar/menu action
        try:
            if self.action is not None:
                self.iface.removeToolBarIcon(self.action)
        except Exception:
            pass

        # Remove from Raster -> Conversion (or Raster menu fallback)
        try:
            if self.action is not None:
                if self._conv_menu is not None:
                    self._conv_menu.removeAction(self.action)
                elif self._added_via_iface_raster_menu:
                    # Fallback case where we couldn't locate the Conversion submenu
                    self.iface.removePluginRasterMenu("&Raster", self.action)
        except Exception:
            pass

        self.action = None

        # Unregister processing provider
        if self.provider is not None:
            try:
                QgsApplication.processingRegistry().removeProvider(self.provider)
            except Exception:
                pass
            self.provider = None

    def open_dialog(self):
        """Open a guided (clean) dialog.

        Power users can still open the raw Processing dialog from inside the guided UI.
        """
        try:
            from .gui.dialog import LandsatWaterMaskGuidedDialog
            dlg = LandsatWaterMaskGuidedDialog(self.iface, parent=self.iface.mainWindow())
            dlg.exec()
        except Exception as e:
            # If the guided dialog fails for any reason, fall back to the Processing dialog
            try:
                self._open_processing_dialog(fallback_error=e)
            except Exception:
                try:
                    self.iface.messageBar().pushWarning("Landsat Water Mask", f"Could not open dialog: {e}")
                except Exception:
                    pass

    def _open_processing_dialog(self, fallback_error=None):
        """Open the standard Processing algorithm dialog."""
        alg_id = "landsat_watermask:landsat_water_mask"
        try:
            import processing  # QGIS Processing framework
            if hasattr(processing, "execAlgorithmDialog"):
                processing.execAlgorithmDialog(alg_id, {})
            else:
                dlg = processing.createAlgorithmDialog(alg_id, {})
                dlg.exec()
        except Exception as e:
            msg = f"Could not open algorithm dialog: {e}"
            if fallback_error is not None:
                msg = f"Guided dialog failed ({fallback_error}); {msg}"
            try:
                self.iface.messageBar().pushWarning("Landsat Water Mask", msg)
            except Exception:
                pass

    def _add_to_raster_conversion_menu(self):
        """Attach QAction to Raster -> Conversion when possible.

        In most QGIS builds, the main Raster menu is named 'mRasterMenu' and the
        Conversion submenu is named 'mRasterConversionMenu'. If we can't locate
        the submenu (e.g., unusual UI customization/localization), we fall back
        to adding the action to the top-level Raster menu via iface.
        """
        mw = self.iface.mainWindow()

        raster_menu = None
        conv_menu = None

        try:
            raster_menu = mw.findChild(QMenu, "mRasterMenu")
        except Exception:
            raster_menu = None

        if raster_menu is not None:
            # Preferred: direct lookup by objectName
            try:
                conv_menu = raster_menu.findChild(QMenu, "mRasterConversionMenu")
            except Exception:
                conv_menu = None

            # Fallback: scan Raster menu actions for a submenu labeled 'Conversion'
            if conv_menu is None:
                try:
                    for act in raster_menu.actions():
                        m = act.menu()
                        if not m:
                            continue
                        title = (m.title() or "").replace("&", "").strip().lower()
                        if m.objectName() == "mRasterConversionMenu" or title == "conversion":
                            conv_menu = m
                            break
                except Exception:
                    conv_menu = None

        if conv_menu is not None:
            # Avoid duplicates if plugin gets reloaded
            if self.action not in conv_menu.actions():
                conv_menu.addAction(self.action)
            self._conv_menu = conv_menu
            self._added_via_iface_raster_menu = False
            return

        # If we couldn't find the Conversion submenu, at least put it in Raster
        self.iface.addPluginToRasterMenu("&Raster", self.action)
        self._added_via_iface_raster_menu = True
