Landsat Water Mask (QGIS Plugin)
=================================

Adds a Processing Toolbox algorithm:
  Landsat Water Mask ▶ Landsat water mask (REFL / QA_PIXEL)

Acceptable file inputs can be sourced from Earth Explorer.

1. Landsat Collection 2 Level-2
	⇒ Landsat 8-9 OLI/TIRS C2 L2
	⇒ Landsat 7 ETM+ C2 L2
	⇒ Landsat 4-5 TM C2L2
		↪QA_PIXEL.TIF

2. Landsat Collection 2 Level-1
	⇒ Landsat 8-9 OLI/TIRS C2 L1
	⇒ Landsat 7 ETM+ C2 L1
	⇒ Landsat 4-5 TM C2L1
		↪Full Resolution Browse (Reflective Color) GeoTIFF

refl.tif note:
  Water is detected via RBG thresholds.

QA_PIXEL.tif note:
  Water is detected using the QA_PIXEL integer value per Landsat generation #:
    - Landsat 4/5/7: 5504
    - Landsat 8/9:   21952
  
Landsat generation # detected from filename characters 3–4 (04, 05, 07, 08, 09)

Outputs are Processing destinations (TEMPORARY_OUTPUT by default) and will be added back into QGIS.

Install:
  QGIS → Plugins → Manage and Install Plugins… → Install from ZIP
