"""
SDNHM MaxEnt — Tool 1: Prepare Data
No arcpy dependency. Called from SDNHMMaxent.pyt :: PrepareData.execute().
"""

import os
import math
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds as win_from_bounds
import rasterio.transform as rtransform
import pyproj
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

_NODATA = -9999.0


# ── Public entry point ────────────────────────────────────────────────────────

def run_preprocessing(
    species_name,
    species_csv,
    study_area_shp,
    env_folders,                   # list[str] — any number of raster source folders
    output_dir,
    lat_col               = "Latitude",
    lon_col               = "Longitude",
    species_col           = "Species",
    coordinate_system     = "EPSG:32611",
    raster_res_arcsec     = None,  # None = auto-detect (kept for API parity, not used internally)
    study_area_buffer_deg = 1.0,
    callback              = print,
):
    """
    Run the full data preparation pipeline for Tool 1.

    Parameters
    ----------
    env_folders : list of folder paths.
        Each folder is processed as an independent variable source.
        Output sub-folder is named after the input folder's basename.

    Returns
    -------
    dict
        Paths to key outputs and summary counts.
    """
    species_name = (species_name.strip()
                    .replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-"))

    ref_crs = pyproj.CRS(coordinate_system)
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Study Area ──────────────────────────────────────────────────────────
    callback("1. Study area — loading and reprojecting...")
    study_area = _load_and_reproject_vector(study_area_shp, ref_crs, coordinate_system)
    sa_out = os.path.join(output_dir, "study_area.shp")
    study_area.to_file(sa_out)
    callback(f"   Exported → {sa_out}")

    # ── 2. Reference Grid ──────────────────────────────────────────────────────
    callback("2. Reference grid — reading from first raster...")
    first_tif = _find_first_tif(env_folders)
    if first_tif is None:
        raise FileNotFoundError("No .tif files found in the provided env_folders.")

    ref_transform, grid_w, grid_h = _build_reference_grid(
        first_tif, study_area, ref_crs, study_area_buffer_deg
    )
    thinning_res = abs(ref_transform.a)
    _unit = "deg" if ref_crs.is_geographic else "m"
    bounds = rtransform.array_bounds(grid_h, grid_w, ref_transform)
    raster_extent = [bounds[0], bounds[2], bounds[1], bounds[3]]  # [left, right, bottom, top]

    callback(f"   Grid: {grid_w} × {grid_h} px | pixel size: {thinning_res:.2f} {_unit}")

    # ── 3. Species Occurrences ─────────────────────────────────────────────────
    callback("3. Occurrences — loading, clipping, thinning...")
    occ_raw_gdf, occ_thinned = _process_occurrences(
        species_csv, lat_col, lon_col, coordinate_system,
        study_area, thinning_res, callback,
    )
    spp_dir = os.path.join(output_dir, "species")
    os.makedirs(spp_dir, exist_ok=True)
    occ_csv_out = os.path.join(spp_dir, "occurrences_cleaned.csv")
    occ_shp_out = os.path.join(spp_dir, "occurrences_cleaned.shp")
    export_cols = [c for c in [species_col, lon_col, lat_col] if c in occ_thinned.columns]
    occ_thinned[export_cols].to_csv(occ_csv_out, index=False)
    occ_thinned.to_file(occ_shp_out)
    callback(f"   Exported {len(occ_thinned)} records → {spp_dir}/")

    # ── 4. Rasters ─────────────────────────────────────────────────────────────
    callback("4. Rasters — clipping, reprojecting, masking...")
    raster_profile = {
        "driver"    : "GTiff",
        "dtype"     : "float32",
        "width"     : grid_w,
        "height"    : grid_h,
        "count"     : 1,
        "crs"       : ref_crs,
        "transform" : ref_transform,
        "nodata"    : _NODATA,
        "compress"  : "lzw",
    }

    # processed: {var_key: {"array": ndarray, "out_path": str, "folder_name": str}}
    processed = {}

    for folder in env_folders:
        folder_name = os.path.basename(folder.rstrip("/\\"))
        out_sub = os.path.join(output_dir, folder_name)
        os.makedirs(out_sub, exist_ok=True)

        tifs = sorted(
            f for f in os.listdir(folder)
            if f.lower().endswith(".tif") and not f.endswith(".aux.xml")
        )
        callback(f"   {folder_name}: {len(tifs)} rasters")

        for fname in tqdm(tifs, desc=f"   {folder_name}", leave=False):
            var_key  = os.path.splitext(fname)[0]
            src_path = os.path.join(folder, fname)
            arr      = _clip_reproject_snap(
                src_path, study_area, study_area_buffer_deg,
                ref_transform, ref_crs, grid_w, grid_h,
            )
            processed[var_key] = {
                "array"      : arr,
                "out_path"   : os.path.join(out_sub, fname),
                "folder_name": folder_name,
            }

    # Combined valid mask (all layers valid) + study area polygon mask
    valid_mask = _build_valid_mask(
        processed, study_area, ref_transform, grid_h, grid_w
    )
    valid_pct = valid_mask.sum() / valid_mask.size * 100
    callback(f"   Valid pixels after masking: {valid_mask.sum():,} ({valid_pct:.1f}%)")

    # Write masked rasters to disk
    for entry in processed.values():
        entry["array"][~valid_mask] = _NODATA
        with rasterio.open(entry["out_path"], "w", **raster_profile) as dst:
            dst.write(entry["array"], 1)

    callback(f"   Saved {len(processed)} rasters → {output_dir}/")

    # ── 5. Visualize ───────────────────────────────────────────────────────────
    callback("5. Generating diagnostic plots...")

    _plot_study_area(study_area, species_name, output_dir)
    _plot_thinning(occ_raw_gdf, occ_thinned, study_area, species_name, output_dir)

    for folder in env_folders:
        folder_name = os.path.basename(folder.rstrip("/\\"))
        folder_vars = {k: v for k, v in processed.items()
                       if v["folder_name"] == folder_name}
        if folder_vars:
            _plot_variable_grid(
                folder_vars, folder_name, valid_mask, raster_extent, study_area, output_dir
            )

    callback("Preprocessing complete.")
    return {
        "study_area_shp"  : sa_out,
        "occ_csv"         : occ_csv_out,
        "occ_shp"         : occ_shp_out,
        "output_dir"      : output_dir,
        "n_rasters"       : len(processed),
        "n_occurrences"   : len(occ_thinned),
    }


# ── Step helpers ──────────────────────────────────────────────────────────────

def _load_and_reproject_vector(shp_path, ref_crs, coordinate_system):
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        return gdf.set_crs(coordinate_system)
    if gdf.crs != ref_crs:
        return gdf.to_crs(coordinate_system)
    return gdf


def _find_first_tif(env_folders):
    for folder in env_folders:
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(".tif") and not f.endswith(".aux.xml"):
                return os.path.join(folder, f)
    return None


def _build_reference_grid(first_tif, study_area, ref_crs, buffer_deg):
    """Return (transform, width, height) for the output reference grid."""
    with rasterio.open(first_tif) as src:
        src_crs = src.crs
        sa_src  = study_area.to_crs(src_crs)
        b       = sa_src.total_bounds
        buf     = buffer_deg if src_crs.is_geographic else buffer_deg * 111_000
        win     = win_from_bounds(
            b[0] - buf, b[1] - buf, b[2] + buf, b[3] + buf,
            transform=src.transform,
        ).round_offsets().round_lengths()
        src_tfm = src.window_transform(win)
        win_h   = int(win.height)
        win_w   = int(win.width)

    if ref_crs == src_crs:
        return src_tfm, win_w, win_h

    tfm, w, h = calculate_default_transform(
        src_crs, ref_crs, win_w, win_h,
        *rtransform.array_bounds(win_h, win_w, src_tfm),
    )
    return tfm, int(w), int(h)


def _process_occurrences(species_csv, lat_col, lon_col, coordinate_system,
                         study_area, thinning_res, callback):
    """Load, reproject, clip to study area, and thin occurrences."""
    occ = pd.read_csv(species_csv)
    gdf = gpd.GeoDataFrame(
        occ,
        geometry=gpd.points_from_xy(occ[lon_col], occ[lat_col]),
        crs="EPSG:4326",
    ).to_crs(coordinate_system)

    # Clip to study area
    sa_union   = study_area.union_all()
    gdf_clipped = gdf[gdf.within(sa_union)].reset_index(drop=True)
    callback(f"   {len(occ)} raw → {len(gdf_clipped)} within study area")

    # Spatial thinning: keep one record per reference grid cell
    xmin, ymin = study_area.total_bounds[:2]
    gdf_clipped = gdf_clipped.copy()
    gdf_clipped["_col"] = np.floor(
        (gdf_clipped.geometry.x - xmin) / thinning_res).astype(int)
    gdf_clipped["_row"] = np.floor(
        (gdf_clipped.geometry.y - ymin) / thinning_res).astype(int)
    thinned = (
        gdf_clipped.drop_duplicates(subset=["_col", "_row"])
                   .drop(columns=["_col", "_row"])
                   .reset_index(drop=True)
    )
    callback(f"   {len(gdf_clipped)} → {len(thinned)} after thinning")
    return gdf_clipped, thinned


def _clip_reproject_snap(src_path, study_area, buffer_deg,
                         ref_transform, ref_crs, grid_w, grid_h):
    """Clip a source raster to study area + buffer, reproject, and snap to reference grid."""
    dst = np.full((grid_h, grid_w), _NODATA, dtype=np.float32)
    with rasterio.open(src_path) as src:
        sa_src = study_area.to_crs(src.crs)
        b      = sa_src.total_bounds
        buf    = buffer_deg if src.crs.is_geographic else buffer_deg * 111_000
        win    = win_from_bounds(
            b[0] - buf, b[1] - buf, b[2] + buf, b[3] + buf,
            transform=src.transform,
        ).round_offsets().round_lengths()
        clipped = src.read(1, window=win).astype(np.float32)
        reproject(
            source        = clipped,
            destination   = dst,
            src_transform = src.window_transform(win),
            src_crs       = src.crs,
            dst_transform = ref_transform,
            dst_crs       = ref_crs,
            resampling    = Resampling.bilinear,
            src_nodata    = src.nodata,
            dst_nodata    = _NODATA,
        )
    return dst


def _build_valid_mask(processed, study_area, ref_transform, grid_h, grid_w):
    """Combined mask: valid in all layers AND inside the study area polygon."""
    valid = np.ones((grid_h, grid_w), dtype=bool)
    for entry in processed.values():
        valid &= (entry["array"] != _NODATA)

    poly_mask = geometry_mask(
        [g.__geo_interface__ for g in study_area.geometry],
        transform  = ref_transform,
        invert     = True,
        out_shape  = (grid_h, grid_w),
    )
    return valid & poly_mask


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _plot_study_area(study_area, species_name, output_dir):
    fig, ax = plt.subplots(figsize=(5, 9))
    study_area.plot(ax=ax, color="#dceedd", edgecolor="#2d6a2d", linewidth=1.2)
    ax.set_title(f"Study Area — {species_name.replace('_', ' ')}",
                 fontsize=12, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"01_{species_name}_study_area.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def _plot_thinning(occ_raw, occ_thinned, study_area, species_name, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 8))
    for ax, pts, title, color in [
        (axes[0], occ_raw,     f"Before thinning ({len(occ_raw)})",    "steelblue"),
        (axes[1], occ_thinned, f"After thinning ({len(occ_thinned)})", "crimson"),
    ]:
        study_area.plot(ax=ax, color="#dceedd", edgecolor="#555", linewidth=0.8)
        pts.plot(ax=ax, color=color, markersize=40, zorder=3, alpha=0.85)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, ls="--", alpha=0.3)
    plt.suptitle(f"{species_name.replace('_', ' ')} — Occurrence Thinning",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"02_{species_name}_thinning.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def _plot_variable_grid(folder_vars, folder_name, valid_mask,
                        extent, study_area, output_dir):
    items = list(folder_vars.items())
    ncols = 5
    nrows = math.ceil(len(items) / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.8, nrows * 3.2),
                             squeeze=False,
                             constrained_layout=True)
    ax_flat = axes.flatten()

    for i, (var_key, entry) in enumerate(items):
        ax   = ax_flat[i]
        data = entry["array"].copy().astype(float)
        data[~valid_mask] = np.nan
        im   = ax.imshow(data, extent=extent, origin="upper",
                         cmap="viridis", aspect="equal")
        study_area.plot(ax=ax, color="none", edgecolor="white", linewidth=0.7)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        ax.set_title(var_key, fontsize=9, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(len(items), len(ax_flat)):
        fig.delaxes(ax_flat[j])

    fig.suptitle(f"{folder_name} — processed variables",
                 fontsize=13, fontweight="bold")
    plt.savefig(os.path.join(output_dir, f"03_vars_{folder_name}.png"),
                dpi=120, bbox_inches="tight")
    plt.close()
