# -*- coding: utf-8 -*-
from qgis.core import QgsProcessingProvider
from .algorithms.landsat_watermask import LandsatWaterMaskAlgorithm

class LandsatWaterMaskProvider(QgsProcessingProvider):
    def id(self):
        return 'landsat_watermask'

    def name(self):
        return 'Landsat Water Mask'

    def longName(self):
        return self.name()

    def loadAlgorithms(self):
        self.addAlgorithm(LandsatWaterMaskAlgorithm())
