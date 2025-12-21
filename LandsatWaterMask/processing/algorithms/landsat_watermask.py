# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from osgeo import gdal, ogr, osr

from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterEnum,
    QgsProcessingParameterMultipleLayers,
    QgsProcessingParameterString,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterVectorDestination,
    QgsProcessingOutputString,
    QgsProject,
    QgsRasterLayer,
)


gdal.UseExceptions()

# -----------------------------
# Helpers
# -----------------------------

def _dedupe_paths(paths: list[Path]) -> list[Path]:
    uniq = {}
    for p in paths:
        try:
            key = str(p.resolve()).lower()
        except Exception:
            key = str(p).lower()
        uniq[key] = p
    return sorted(uniq.values())

def _parse_rgb_triplet(s: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError("Background RGB must be 'r,g,b' (e.g., 0,0,0).")
    vals = tuple(int(p) for p in parts)
    for v in vals:
        if v < 0 or v > 255:
            raise ValueError("RGB values must be in 0..255.")
    return vals

def _box_sum(mask: np.ndarray, k: int = 5) -> np.ndarray:
    # Fast kxk moving-window sum using an integral image.
    if k % 2 != 1:
        raise ValueError("k must be odd")
    pad = k // 2
    m = mask.astype(np.uint8)
    p = np.pad(m, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)
    # NOTE: We add an extra top row + left column of zeros to the integral image
    # so the kxk window sum returns the SAME shape as the original (un-padded) mask.
    s = p.cumsum(axis=0).cumsum(axis=1)
    s = np.pad(s, ((1, 0), (1, 0)), mode="constant", constant_values=0)
    out = s[k:, k:] - s[:-k, k:] - s[k:, :-k] + s[:-k, :-k]
    return out.astype(np.uint16)

def _classify_rgb_land_water(
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    bg_rgb: tuple[int, int, int],
    water_ranges: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    # Port of classify_rgb_array_land_water() in Landsat_Vector_v1.2.py
    r = r.astype(np.uint8)
    g = g.astype(np.uint8)
    b = b.astype(np.uint8)

    all_lt_10 = (r < 10) & (g < 10) & (b < 10)

    zero_and_low = np.zeros(r.shape, dtype=bool)
    zero_and_low |= (r == 0) & ((g < 5) | (b < 5))
    zero_and_low |= (g == 0) & ((r < 5) | (b < 5))
    zero_and_low |= (b == 0) & ((r < 5) | (g < 5))

    near_zero_low = _box_sum(zero_and_low, k=5) > 0

    exclude = all_lt_10 | zero_and_low | near_zero_low

    br, bg, bb = bg_rgb
    bg_mask = (r == br) & (g == bg) & (b == bb)

    non_class = exclude | bg_mask

    (rmin, rmax), (gmin, gmax), (bmin, bmax) = water_ranges
    water = (
        (r >= rmin) & (r <= rmax) &
        (g >= gmin) & (g <= gmax) &
        (b >= bmin) & (b <= bmax) &
        (~non_class)
    )

    cls = np.zeros(r.shape, dtype=np.uint8)
    cls[water] = 1
    cls[(~non_class) & (~water)] = 2
    return cls

def _srs_from_wkt(wkt: str) -> osr.SpatialReference:
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    # IMPORTANT (GDAL>=3 / PROJ>=6): enforce traditional GIS axis order (lon,lat)
    # so EPSG:4326 transforms don't end up with swapped axes in some environments.
    try:
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    except Exception:
        pass
    return srs

def _densified_bounds_in_target(ds: gdal.Dataset, target_srs: osr.SpatialReference, densify: int = 21) -> tuple[float, float, float, float]:
    # Approximate rasterio.transform_bounds(..., densify_pts=21) by sampling points along bounds.
    wkt = ds.GetProjection()
    if not wkt:
        raise ValueError(f"{ds.GetDescription()} has no projection/WKT.")
    src_srs = _srs_from_wkt(wkt)
    tx = osr.CoordinateTransformation(src_srs, target_srs)

    gt = ds.GetGeoTransform()
    cols = ds.RasterXSize
    rows = ds.RasterYSize

    def pix2w(px, py):
        x = gt[0] + px * gt[1] + py * gt[2]
        y = gt[3] + px * gt[4] + py * gt[5]
        return x, y

    corners = [pix2w(0,0), pix2w(cols,0), pix2w(cols,rows), pix2w(0,rows)]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    pts = []
    for i in range(densify):
        t = i / (densify - 1) if densify > 1 else 0
        x = minx + t * (maxx - minx)
        pts.append((x, miny))
        pts.append((x, maxy))
        y = miny + t * (maxy - miny)
        pts.append((minx, y))
        pts.append((maxx, y))

    outx, outy = [], []
    for x, y in pts:
        xo, yo, _ = tx.TransformPoint(x, y)
        outx.append(xo)
        outy.append(yo)

    return min(outx), min(outy), max(outx), max(outy)

def _snap_bounds(minx, miny, maxx, maxy, xres, yres):
    left   = math.floor(minx / xres) * xres
    right  = math.ceil (maxx / xres) * xres
    bottom = math.floor(miny / yres) * yres
    top    = math.ceil (maxy / yres) * yres
    return left, bottom, right, top

def _warp_to_grid(
    ds: gdal.Dataset,
    dst_srs_wkt: str,
    bounds: tuple[float, float, float, float],
    xres: float,
    yres: float,
    width: int,
    height: int,
    resample: str = "near",
    dst_alpha: bool = True,
    out_dtype=None,
) -> gdal.Dataset:
    opts = gdal.WarpOptions(
        format="MEM",
        dstSRS=dst_srs_wkt,
        outputBounds=bounds,  # (minX, minY, maxX, maxY)
        xRes=xres,
        yRes=yres,
        width=width,
        height=height,
        resampleAlg=resample,
        dstAlpha=dst_alpha,
        multithread=True,
        outputType=out_dtype,
    )
    return gdal.Warp("", ds, options=opts)

def _write_gtiff(path: Path, arr: np.ndarray, gt, proj_wkt: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(
        str(path),
        int(arr.shape[1]),
        int(arr.shape[0]),
        1,
        gdal.GDT_Byte,
        options=["COMPRESS=LZW", "TILED=YES"]
    )
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj_wkt)
    band = ds.GetRasterBand(1)
    band.WriteArray(arr.astype(np.uint8))
    band.SetNoDataValue(0)
    band.FlushCache()
    ds.FlushCache()
    ds = None

def _delete_shp_family(shp_path: Path):
    base = shp_path.with_suffix("")
    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qpj", ".sbn", ".sbx", ".fix"]:
        p = base.with_suffix(ext)
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass

def _polygonize_water_to_single_feature(binary_tif: Path, out_vec: Path, label: str = "Water", class_id: int = 1) -> bool:
    # Polygonize where raster==1, dissolve to one feature, write EPSG:4326 vector output (SHP/GPKG/GeoJSON).
    binary_tif = Path(binary_tif)
    out_vec = Path(out_vec)

    # Remove existing output
    if out_vec.suffix.lower() == ".shp":
        _delete_shp_family(out_vec)
    else:
        if out_vec.exists():
            try:
                out_vec.unlink()
            except Exception:
                pass

    src = gdal.Open(str(binary_tif))
    if src is None:
        raise RuntimeError(f"Could not open raster: {binary_tif}")

    proj_wkt = src.GetProjection()
    if not proj_wkt:
        raise RuntimeError("Raster has no projection; cannot write EPSG:4326 shapefile.")
    src_srs = _srs_from_wkt(proj_wkt)

    srs4326 = osr.SpatialReference()
    srs4326.ImportFromEPSG(4326)
    try:
        srs4326.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    except Exception:
        pass
    coord_tx = osr.CoordinateTransformation(src_srs, srs4326)

    band = src.GetRasterBand(1)

    mem_driver = ogr.GetDriverByName("Memory")
    mem_ds = mem_driver.CreateDataSource("mem")
    mem_lyr = mem_ds.CreateLayer("poly", src_srs, geom_type=ogr.wkbPolygon)
    mem_lyr.CreateField(ogr.FieldDefn("DN", ogr.OFTInteger))

    gdal.Polygonize(band, None, mem_lyr, 0, options=["8CONNECTED=8"])

    polys = ogr.Geometry(ogr.wkbMultiPolygon)
    mem_lyr.ResetReading()
    for feat in mem_lyr:
        if feat.GetFieldAsInteger("DN") != 1:
            continue
        g = feat.GetGeometryRef()
        if g is None:
            continue
        polys.AddGeometry(g.Clone())

    if polys.GetGeometryCount() == 0:
        src = None
        mem_ds = None
        return False

    dissolved = polys.UnionCascaded()
    dissolved_4326 = dissolved.Clone()
    dissolved_4326.Transform(coord_tx)

    ext = out_vec.suffix.lower()
    if ext == ".shp":
        drv_name = "ESRI Shapefile"
        layer_name = out_vec.stem
    elif ext in (".geojson", ".json"):
        drv_name = "GeoJSON"
        layer_name = out_vec.stem
    else:
        drv_name = "GPKG"
        layer_name = out_vec.stem or "Water"

    out_driver = ogr.GetDriverByName(drv_name)
    if out_driver is None:
        raise RuntimeError(f"OGR driver not available: {drv_name}")

    out_ds = out_driver.CreateDataSource(str(out_vec))
    out_lyr = out_ds.CreateLayer(layer_name, srs4326, geom_type=ogr.wkbMultiPolygon)

    out_lyr.CreateField(ogr.FieldDefn("class_id", ogr.OFTInteger))
    out_lyr.CreateField(ogr.FieldDefn("label", ogr.OFTString))

    out_feat = ogr.Feature(out_lyr.GetLayerDefn())
    out_feat.SetField("class_id", int(class_id))
    out_feat.SetField("label", str(label))
    out_feat.SetGeometry(dissolved_4326)
    out_lyr.CreateFeature(out_feat)

    out_feat = None
    out_ds = None
    mem_ds = None
    src = None
    return True

def _find_refl_tifs(folder: Path) -> list[Path]:
    folder = Path(folder)
    candidates = []
    for ext in ("tif", "tiff", "TIF", "TIFF"):
        candidates.extend(folder.glob(f"*_refl.{ext}"))
    return _dedupe_paths(candidates)

def _find_pixel_tifs(folder: Path) -> list[Path]:
    folder = Path(folder)
    candidates = []
    for ext in ("tif", "tiff", "TIF", "TIFF"):
        candidates.extend(folder.glob(f"*QA_PIXEL*.{ext}"))
        candidates.extend(folder.glob(f"*PIXEL*.{ext}"))
    return _dedupe_paths(candidates)

def _landsat_sensor_code_from_name(name: str) -> str | None:
    """Return Landsat sensor/platform code from the filename.

    Convention used here: the 3rd and 4th characters in the filename are one of
    04, 05, 07, 08, or 09 (e.g., LT04..., LT05..., LE07..., LC08..., LC09...).
    """
    if not name:
        return None
    base = Path(name).name
    if len(base) < 4:
        return None
    code = base[2:4]
    if code in {"04", "05", "07", "08", "09"}:
        return code
    return None

def _pixel_water_value_for_file(name: str) -> int:
    """Map QA_PIXEL 'water' integer to the correct value per Landsat generation.

    - Landsat 4/5/7: water == 5504
    - Landsat 8/9:   water == 21952
    """
    code = _landsat_sensor_code_from_name(name) or ""
    if code in {"04", "05", "07"}:
        return 5504
    # default to Landsat 8/9 behavior
    return 21952

def _process_refl_stack(tifs: list[Path], out_tif: Path, out_vec: Path, bg_rgb, water_ranges, feedback) -> Path:
    tifs = list(tifs)

    ds0 = gdal.Open(str(tifs[0]))
    ref_wkt = ds0.GetProjection()
    if not ref_wkt:
        raise ValueError(f"{tifs[0].name} has no CRS.")
    ref_srs = _srs_from_wkt(ref_wkt)
    gt0 = ds0.GetGeoTransform()
    xres = abs(gt0[1])
    yres = abs(gt0[5])

    minx=miny=None
    maxx=maxy=None
    for tif in tifs:
        ds = gdal.Open(str(tif))
        bxmin, bymin, bxmax, bymax = _densified_bounds_in_target(ds, ref_srs, densify=21)
        minx = bxmin if minx is None else min(minx, bxmin)
        miny = bymin if miny is None else min(miny, bymin)
        maxx = bxmax if maxx is None else max(maxx, bxmax)
        maxy = bymax if maxy is None else max(maxy, bymax)

    left, bottom, right, top = _snap_bounds(minx, miny, maxx, maxy, xres, yres)
    width = int(math.ceil((right - left) / xres))
    height = int(math.ceil((top - bottom) / yres))
    bounds = (left, bottom, right, top)
    gt = (left, xres, 0.0, top, 0.0, -yres)

    feedback.pushInfo(f"[REFL] Target grid: res={xres}x{yres} size={width}x{height}")

    water_union = np.zeros((height, width), dtype=bool)
    land_union = np.zeros((height, width), dtype=bool)

    for i, tif in enumerate(tifs, start=1):
        ds = gdal.Open(str(tif))
        if ds.RasterCount < 3:
            raise ValueError(f"REFL raster must have >=3 bands (R,G,B): {tif.name}")

        warped = _warp_to_grid(ds, ref_wkt, bounds, xres, yres, width, height, dst_alpha=True, out_dtype=gdal.GDT_Byte)
        r = warped.GetRasterBand(1).ReadAsArray()
        g = warped.GetRasterBand(2).ReadAsArray()
        b = warped.GetRasterBand(3).ReadAsArray()
        alpha = warped.GetRasterBand(4).ReadAsArray() if warped.RasterCount >= 4 else None
        valid = (alpha > 0) if alpha is not None else np.ones((height, width), dtype=bool)

        cls = _classify_rgb_land_water(r, g, b, bg_rgb, water_ranges)
        cls[~valid] = 0

        is_water = (cls == 1)
        is_land = (cls == 2)

        water_union |= is_water
        land_union |= is_land

        feedback.pushInfo(f"[REFL] [{i}/{len(tifs)}] {tif.name}: water={int(is_water.sum())} land={int(is_land.sum())}")
        feedback.setProgress(int((i / max(1, len(tifs))) * 60))

        if feedback.isCanceled():
            raise RuntimeError("Canceled.")

    combined = np.zeros((height, width), dtype=np.uint8)
    combined[(~water_union) & land_union] = 2
    combined[water_union] = 1
    water_bin = (combined == 1).astype(np.uint8)

    out_tif = Path(out_tif)
    out_vec = Path(out_vec)
    _write_gtiff(out_tif, water_bin, gt, ref_wkt)

    ok = _polygonize_water_to_single_feature(out_tif, out_vec, label="Water", class_id=1)
    if not ok:
        feedback.pushInfo("[REFL] No water polygons found; shapefile not written.")

    return out_tif

def _process_pixel_stack(tifs: list[Path], out_tif: Path, out_vec: Path, feedback) -> Path:
    tifs = list(tifs)

    LAND_VALS = (21824, 30048)

    ds0 = gdal.Open(str(tifs[0]))
    ref_wkt = ds0.GetProjection()
    if not ref_wkt:
        raise ValueError(f"{tifs[0].name} has no CRS.")
    ref_srs = _srs_from_wkt(ref_wkt)
    gt0 = ds0.GetGeoTransform()
    xres = abs(gt0[1])
    yres = abs(gt0[5])

    minx=miny=None
    maxx=maxy=None
    for tif in tifs:
        ds = gdal.Open(str(tif))
        bxmin, bymin, bxmax, bymax = _densified_bounds_in_target(ds, ref_srs, densify=21)
        minx = bxmin if minx is None else min(minx, bxmin)
        miny = bymin if miny is None else min(miny, bymin)
        maxx = bxmax if maxx is None else max(maxx, bxmax)
        maxy = bymax if maxy is None else max(maxy, bymax)

    left, bottom, right, top = _snap_bounds(minx, miny, maxx, maxy, xres, yres)
    width = int(math.ceil((right - left) / xres))
    height = int(math.ceil((top - bottom) / yres))
    bounds = (left, bottom, right, top)
    gt = (left, xres, 0.0, top, 0.0, -yres)

    feedback.pushInfo(f"[PIXEL] Target grid: res={xres}x{yres} size={width}x{height}")

    water_union = np.zeros((height, width), dtype=bool)
    land_union = np.zeros((height, width), dtype=bool)
    other_union = np.zeros((height, width), dtype=bool)

    for i, tif in enumerate(tifs, start=1):
        ds = gdal.Open(str(tif))
        warped = _warp_to_grid(ds, ref_wkt, bounds, xres, yres, width, height, dst_alpha=True, out_dtype=gdal.GDT_UInt16)

        water_val = _pixel_water_value_for_file(tif.name)

        qa = warped.GetRasterBand(1).ReadAsArray()
        alpha = warped.GetRasterBand(2).ReadAsArray() if warped.RasterCount >= 2 else None
        valid = (alpha > 0) if alpha is not None else np.ones((height, width), dtype=bool)

        is_water = valid & (qa == water_val)
        is_land  = valid & ((qa == LAND_VALS[0]) | (qa == LAND_VALS[1]))
        is_other = valid & (~is_water) & (~is_land)

        water_union |= is_water
        land_union |= is_land
        other_union |= is_other

        sensor = _landsat_sensor_code_from_name(tif.name) or "??"
        feedback.pushInfo(
            f"[PIXEL] [{i}/{len(tifs)}] {tif.name} (L{sensor} water=={water_val}): "
            f"water={int(is_water.sum())} land={int(is_land.sum())} other={int(is_other.sum())}"
        )
        feedback.setProgress(60 + int((i / max(1, len(tifs))) * 30))

        if feedback.isCanceled():
            raise RuntimeError("Canceled.")

    water_mask = water_union

    out_tif = Path(out_tif)
    out_vec = Path(out_vec)
    _write_gtiff(out_tif, water_mask.astype(np.uint8), gt, ref_wkt)

    ok = _polygonize_water_to_single_feature(out_tif, out_vec, label="Water", class_id=1)
    if not ok:
        feedback.pushInfo("[PIXEL] No water polygons found; shapefile not written.")

    return out_tif

def _sum_two_water_binaries(tif_a: Path, tif_b: Path, out_sum_tif: Path, out_sum_shp: Path, feedback):
    a = gdal.Open(str(tif_a))
    b = gdal.Open(str(tif_b))
    if a is None or b is None:
        raise RuntimeError("Could not open inputs for SUM.")

    ref_wkt = a.GetProjection()
    if not ref_wkt:
        raise ValueError("SUM: reference raster has no CRS.")
    ref_srs = _srs_from_wkt(ref_wkt)

    agt = a.GetGeoTransform()
    bgt = b.GetGeoTransform()
    xres = min(abs(agt[1]), abs(bgt[1]))
    yres = min(abs(agt[5]), abs(bgt[5]))

    aminx, aminy, amaxx, amaxy = _densified_bounds_in_target(a, ref_srs, densify=21)
    bminx, bminy, bmaxx, bmaxy = _densified_bounds_in_target(b, ref_srs, densify=21)
    minx = min(aminx, bminx)
    miny = min(aminy, bminy)
    maxx = max(amaxx, bmaxx)
    maxy = max(amaxy, bmaxy)

    left, bottom, right, top = _snap_bounds(minx, miny, maxx, maxy, xres, yres)
    width = int(math.ceil((right - left) / xres))
    height = int(math.ceil((top - bottom) / yres))
    bounds = (left, bottom, right, top)
    gt = (left, xres, 0.0, top, 0.0, -yres)

    wa = _warp_to_grid(a, ref_wkt, bounds, xres, yres, width, height, dst_alpha=False, out_dtype=gdal.GDT_Byte)
    wb = _warp_to_grid(b, ref_wkt, bounds, xres, yres, width, height, dst_alpha=False, out_dtype=gdal.GDT_Byte)
    arr_a = wa.GetRasterBand(1).ReadAsArray().astype(np.uint16)
    arr_b = wb.GetRasterBand(1).ReadAsArray().astype(np.uint16)
    s = np.clip(arr_a + arr_b, 0, 255).astype(np.uint8)

    _write_gtiff(out_sum_tif, s, gt, ref_wkt)

    # polygonize sum>0 by writing a temp binary
    tmp_bin = Path(out_sum_tif).with_name(Path(out_sum_tif).stem + "_tmpbin.tif")
    _write_gtiff(tmp_bin, (s > 0).astype(np.uint8), gt, ref_wkt)
    ok = _polygonize_water_to_single_feature(tmp_bin, out_sum_shp, label="Water", class_id=1)
    try:
        tmp_bin.unlink()
    except Exception:
        pass
    if not ok:
        feedback.pushInfo("[SUM] No water polygons found; shapefile not written.")


class LandsatWaterMaskAlgorithm(QgsProcessingAlgorithm):
    """
    QGIS Processing algorithm.

    - Inputs: raster layers currently loaded in the project (or optionally, a selected subset)
    - Outputs: rasters/vectors returned to QGIS as Processing destinations (TEMPORARY_OUTPUT by default)
    """

    INPUT_LAYERS = "INPUT_LAYERS"
    MODE = "MODE"

    BG_RGB = "BG_RGB"
    KEEP_DEFAULT = "KEEP_DEFAULT"
    R_MIN = "R_MIN"
    R_MAX = "R_MAX"
    G_MIN = "G_MIN"
    G_MAX = "G_MAX"
    B_MIN = "B_MIN"
    B_MAX = "B_MAX"
    DO_SUM = "DO_SUM"

    OUT_REFL_TIF = "OUT_REFL_TIF"
    OUT_REFL_VEC = "OUT_REFL_VEC"
    OUT_PIXEL_TIF = "OUT_PIXEL_TIF"
    OUT_PIXEL_VEC = "OUT_PIXEL_VEC"
    OUT_SUM_TIF = "OUT_SUM_TIF"
    OUT_SUM_VEC = "OUT_SUM_VEC"
    OUT_LOG = "OUT_LOG"

    def name(self):
        return "landsat_water_mask"

    def displayName(self):
        return "Landsat Water Mask (from loaded layers)"

    def group(self):
        return "Landsat Water Mask"

    def groupId(self):
        return "landsat_water_mask"

    def shortHelpString(self):
        return (
            "Builds water binary masks (1=water, 0=not water) and dissolved water polygons from Landsat rasters.\n\n"
            "Acceptable file inputs can be sourced from Earth Explorer.\n"
            "1. Landsat Collection 2 Level-2 ⇒ Landsat 8-9 OLI/TIRS C2 L2 ⇒ Landsat 7 ETM+ C2 L2 ⇒ Landsat 4-5 TM C2L2 ↪ QA_PIXEL.TIF\n"
            "2. Landsat Collection 2 Level-1 ⇒ Landsat 8-9 OLI/TIRS C2 L1 ⇒ Landsat 7 ETM+ C2 L1 ⇒ Landsat 4-5 TM C2L1 ↪ Full Resolution Browse (Reflective Color) GeoTIFF\n"
            "REFL: unions water from one/more *_refl.tif using inclusive RGB ranges (R,G,B).\n"
            "PIXEL: unions water using QA_PIXEL equality, with the water value chosen per Landsat generation.\n"
            "Landsat generation # detected from filename characters 3 – 4 (04, 05, 07, 08, 09):\n"
            "• Landsat 4/5/7: water == 5504\n"
            "• Landsat 8/9:   water == 21952\n"
            "BOTH: runs both file types and can optionally write a water_sum raster and polygons.\n\n"
            "Outputs are Processing destinations (TEMPORARY_OUTPUT by default) and will be added back into QGIS.\n"
            "Important! All files must have the same path/row. I.e. LC09_L2SP_022039_20251211_20251212_02_T1: Path = 022, Row = 039"
        )

    def createInstance(self):
        return LandsatWaterMaskAlgorithm()

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterMultipleLayers(
                self.INPUT_LAYERS,
                "Input raster layers (optional). Leave empty to auto-use ALL raster layers loaded in the current QGIS project.",
                layerType=QgsProcessing.TypeRaster,
                optional=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterEnum(
                self.MODE,
                "Mode",
                options=["REFL", "PIXEL", "BOTH"],
                defaultValue=2
            )
        )

        # --- REFL water ID options ---
        self.addParameter(
            QgsProcessingParameterString(
                self.BG_RGB,
                "REFL background RGB to exclude (r,g,b)",
                defaultValue="0,0,0",
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.KEEP_DEFAULT,
                "REFL: keep default water RGB ranges (R=0..49, G=0..49, B=11..255)",
                defaultValue=True,
            )
        )
        self.addParameter(QgsProcessingParameterNumber(self.R_MIN, "REFL custom R min", QgsProcessingParameterNumber.Integer, 0, minValue=0, maxValue=255))
        self.addParameter(QgsProcessingParameterNumber(self.R_MAX, "REFL custom R max", QgsProcessingParameterNumber.Integer, 60, minValue=0, maxValue=255))
        self.addParameter(QgsProcessingParameterNumber(self.G_MIN, "REFL custom G min", QgsProcessingParameterNumber.Integer, 0, minValue=0, maxValue=255))
        self.addParameter(QgsProcessingParameterNumber(self.G_MAX, "REFL custom G max", QgsProcessingParameterNumber.Integer, 60, minValue=0, maxValue=255))
        self.addParameter(QgsProcessingParameterNumber(self.B_MIN, "REFL custom B min", QgsProcessingParameterNumber.Integer, 11, minValue=0, maxValue=255))
        self.addParameter(QgsProcessingParameterNumber(self.B_MAX, "REFL custom B max", QgsProcessingParameterNumber.Integer, 255, minValue=0, maxValue=255))

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.DO_SUM,
                "BOTH: also write water_sum outputs (requires both REFL and QA_PIXEL layers)",
                defaultValue=True,
            )
        )

        # --- Outputs (default to temporary; QGIS will add them to the map) ---
        self.addParameter(QgsProcessingParameterRasterDestination(self.OUT_REFL_TIF, "REFL water binary raster (1=water)", defaultValue="TEMPORARY_OUTPUT"))
        self.addParameter(QgsProcessingParameterVectorDestination(self.OUT_REFL_VEC, "REFL water polygons (EPSG:4326)", defaultValue="TEMPORARY_OUTPUT"))

        self.addParameter(QgsProcessingParameterRasterDestination(self.OUT_PIXEL_TIF, "QA_PIXEL water binary raster (1=water)", defaultValue="TEMPORARY_OUTPUT"))
        self.addParameter(QgsProcessingParameterVectorDestination(self.OUT_PIXEL_VEC, "QA_PIXEL water polygons (EPSG:4326)", defaultValue="TEMPORARY_OUTPUT"))

        self.addParameter(QgsProcessingParameterRasterDestination(self.OUT_SUM_TIF, "SUM water count raster (REFL+QA)", defaultValue="TEMPORARY_OUTPUT"))
        self.addParameter(QgsProcessingParameterVectorDestination(self.OUT_SUM_VEC, "SUM water polygons (EPSG:4326)", defaultValue="TEMPORARY_OUTPUT"))

        self.addOutput(QgsProcessingOutputString(self.OUT_LOG, "Log"))

    def _collect_layers(self, parameters, context):
        layers = self.parameterAsLayerList(parameters, self.INPUT_LAYERS, context) or []
        if not layers:
            # Auto-pull all raster layers in the current project
            layers = [lyr for lyr in QgsProject.instance().mapLayers().values() if isinstance(lyr, QgsRasterLayer)]
        return layers

    def _categorize(self, layers):
        refl_paths = []
        pixel_paths = []

        for lyr in layers:
            try:
                src = (lyr.source() or "").split("|")[0]
            except Exception:
                src = ""
            if not src:
                continue

            p = Path(src)
            name_l = (lyr.name() or "").lower()
            fname_l = p.name.lower()

            if "_refl" in fname_l or "_refl" in name_l:
                refl_paths.append(p)
            if "qa_pixel" in fname_l or "qa_pixel" in name_l:
                pixel_paths.append(p)

        return _dedupe_paths(refl_paths), _dedupe_paths(pixel_paths)

    def processAlgorithm(self, parameters, context, feedback):
        mode_idx = self.parameterAsEnum(parameters, self.MODE, context)
        mode = ["refl", "pixel", "both"][mode_idx]

        layers = self._collect_layers(parameters, context)
        refl_tifs, pixel_tifs = self._categorize(layers)

        # Sanity checks
        if mode in ("refl", "both") and not refl_tifs:
            raise QgsProcessingException("Mode includes REFL, but no *_refl raster layers were found in the selected/current project layers.")
        if mode in ("pixel", "both") and not pixel_tifs:
            raise QgsProcessingException("Mode includes PIXEL, but no QA_PIXEL raster layers were found in the selected/current project layers.")

        bg_rgb = _parse_rgb_triplet(self.parameterAsString(parameters, self.BG_RGB, context))
        keep_default = self.parameterAsBool(parameters, self.KEEP_DEFAULT, context)

        if keep_default:
            water_ranges = ((0, 49), (0, 49), (11, 255))
        else:
            water_ranges = (
                (int(self.parameterAsInt(parameters, self.R_MIN, context)), int(self.parameterAsInt(parameters, self.R_MAX, context))),
                (int(self.parameterAsInt(parameters, self.G_MIN, context)), int(self.parameterAsInt(parameters, self.G_MAX, context))),
                (int(self.parameterAsInt(parameters, self.B_MIN, context)), int(self.parameterAsInt(parameters, self.B_MAX, context))),
            )

        do_sum = self.parameterAsBool(parameters, self.DO_SUM, context)

        results = {self.OUT_LOG: ""}

        refl_bin_path = None
        pixel_bin_path = None

        # Resolve output destinations
        refl_out_tif = self.parameterAsOutputLayer(parameters, self.OUT_REFL_TIF, context)
        refl_out_vec = self.parameterAsOutputLayer(parameters, self.OUT_REFL_VEC, context)
        pixel_out_tif = self.parameterAsOutputLayer(parameters, self.OUT_PIXEL_TIF, context)
        pixel_out_vec = self.parameterAsOutputLayer(parameters, self.OUT_PIXEL_VEC, context)
        sum_out_tif = self.parameterAsOutputLayer(parameters, self.OUT_SUM_TIF, context)
        sum_out_vec = self.parameterAsOutputLayer(parameters, self.OUT_SUM_VEC, context)

        log_lines = []
        log_lines.append(f"Mode: {mode.upper()}")
        log_lines.append(f"REFL layers: {len(refl_tifs)}")
        log_lines.append(f"QA_PIXEL layers: {len(pixel_tifs)}")

        if mode in ("refl", "both"):
            feedback.pushInfo(f"[REFL] Using {len(refl_tifs)} layer(s).")
            refl_bin_path = _process_refl_stack(
                tifs=refl_tifs,
                out_tif=Path(refl_out_tif),
                out_vec=Path(refl_out_vec),
                bg_rgb=bg_rgb,
                water_ranges=water_ranges,
                feedback=feedback
            )
            results[self.OUT_REFL_TIF] = str(refl_bin_path)
            results[self.OUT_REFL_VEC] = str(Path(refl_out_vec))

        if mode in ("pixel", "both"):
            feedback.pushInfo(f"[PIXEL] Using {len(pixel_tifs)} layer(s).")
            pixel_bin_path = _process_pixel_stack(
                tifs=pixel_tifs,
                out_tif=Path(pixel_out_tif),
                out_vec=Path(pixel_out_vec),
                feedback=feedback
            )
            results[self.OUT_PIXEL_TIF] = str(pixel_bin_path)
            results[self.OUT_PIXEL_VEC] = str(Path(pixel_out_vec))

        if mode == "both" and do_sum and refl_bin_path and pixel_bin_path:
            feedback.pushInfo("[SUM] Writing water_sum outputs.")
            _sum_two_water_binaries(refl_bin_path, pixel_bin_path, Path(sum_out_tif), Path(sum_out_vec), feedback)
            results[self.OUT_SUM_TIF] = str(Path(sum_out_tif))
            results[self.OUT_SUM_VEC] = str(Path(sum_out_vec))

        results[self.OUT_LOG] = "\n".join(log_lines)
        feedback.setProgress(100)
        return results
