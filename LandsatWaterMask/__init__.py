# -*- coding: utf-8 -*-
def classFactory(iface):
    from .plugin import LandsatWaterMaskPlugin
    return LandsatWaterMaskPlugin(iface)
