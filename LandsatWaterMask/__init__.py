# -*- coding: utf-8 -*-
def classFactory(iface):
    from .plugin import SentinelWaterMaskPlugin
    return SentinelWaterMaskPlugin(iface)
