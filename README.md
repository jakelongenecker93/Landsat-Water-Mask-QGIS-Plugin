Landsat Water Mask (QGIS Plugin)
=================================

A guided, modern dialog (Raster → Conversion → Landsat Water Mask)
for a clean, beginner-friendly workflow.

>>> Acceptable file inputs can be sourced from USGS Earth Explorer.

1. Landsat Collection 2 Level-2
	⇒Landsat 8-9 OLI/TIRS C2 L2
	⇒Landsat 7 ETM+ C2 L2
	⇒Landsat 4-5 TM C2L2
		↪QA_PIXEL.TIF

2. Landsat Collection 2 Level-1
	⇒Landsat 8-9 OLI/TIRS C2 L1
	⇒Landsat 7 ETM+ C2 L1
	⇒Landsat 4-5 TM C2L1
		↪Full Resolution Browse (Reflective Color) GeoTIFF

>>> Acceptable file inputs can be sourced from NASA Earthdata.

3. Sentinel-2
	⇒HLS Sentinel-2 Multi-spectral
	  Instrument Surface-Reflectance
	  Daily Global 30m v2.0
		↪Fmask.tif

4. OPERA
	⇒OPERA Dynamic Surface Water
	  Extent from Harmonized Landsat
	  Sentinel-2 product (Version 1)
		↪BWTR.tif

Useful Notes:

refl.tif note:
  Water is detected via RBG thresholds.

QA_PIXEL.tif note:
  Water is detected using the QA_PIXEL integer value chosen per Landsat generation:
    - Landsat 4/5/7: 5504
    - Landsat 8/9:   21952
  (Detected from filename characters 3–4: 04, 05, 07, 08, 09.)

Fmask.tif note:
  Water is detected using the Fmask integer values:
    - All Water:  32-63,96-127,160-191,224-254
    - Pure Water: 32,96,160,224

BWTR.tif note:
  Water is detected using binary mask:
    - Not modifiable.  

--> User-modifiable variables for REFL,PIXEL,FMASK.
--> All internal logic is dependant on extepected file nomenclatures for each product.
    If you change these outside of spec things will break.

Install from Plugin Store:
QGIS → Plugins → Manage and Install Plugins… → Install Plugin
	Else Install from Zip:
  	QGIS → Plugins → Manage and Install Plugins… → Install from ZIP
	
<img width="3014" height="1704" alt="Screenshot 2026-01-26 210530" src="https://github.com/user-attachments/assets/40721419-1f6a-456e-9ec9-b7c2035ed2aa" />
<img width="3014" height="1704" alt="Screenshot 2026-01-26 210530" src="https://github.com/user-attachments/assets/3684d071-ae22-499f-83b3-db5a0bf95432" />
<img width="922" height="1594" alt="Screenshot 2026-01-09 062647" src="https://github.com/user-attachments/assets/ad79163c-2100-494b-9eda-b84c899deedf" />
<img width="923" height="1594" alt="Screenshot 2026-01-09 062528" src="https://github.com/user-attachments/assets/cf8581a3-75ac-4c64-bd44-8b8d312583a8" />

If this tool is used, please **Cite as**: Longenecker, J. (2026). Landsat Water Mask Plugin v1.4.3. QGIS 3.44 QT5. 
