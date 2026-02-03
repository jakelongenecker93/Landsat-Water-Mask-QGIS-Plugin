# -*- coding: utf-8 -*-
from qgis.core import QgsProcessingProvider
from .algorithms.landsat_watermask import SentinelWaterMaskAlgorithm

class SentinelWaterMaskProvider(QgsProcessingProvider):
    def id(self):
        return 'sentinel_watermask'

    def name(self):
        return 'Sentinel Water Mask'

    def longName(self):
        return self.name()

    def loadAlgorithms(self):
        self.addAlgorithm(SentinelWaterMaskAlgorithm())
