"""
SDNHM MaxEnt — Tool 2: Variable Selection
No arcpy dependency. Called from SDNHMMaxent.pyt :: VariableSelection.execute().
"""

import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
from rasterio.features import geometry_mask

warnings.filterwarnings("ignore")

_NODATA = -9999.0


# ── Discovery helpers (also used by .pyt updateParameters) ───────────────────

def discover_raster_subfolders(processed_dir):
    """Return sorted list of subfolder names under processed_dir that contain .tif files."""
    result = []
    for name in sorted(os.listdir(processed_dir)):
        sub = os.path.join(processed_dir, name)
        if not os.path.isdir(sub) or name == "species":
            continue
        tifs = [f for f in os.listdir(sub)
                if f.lower().endswith(".tif") and not f.endswith(".aux.xml")]
        if tifs:
            result.append(name)
    return result


def discover_all_variable_keys(processed_dir):
    """Return sorted list of all .tif filename stems across all raster subfolders."""
    keys = []
    for sf in discover_raster_subfolders(processed_dir):
        sub = os.path.join(processed_dir, sf)
        for f in sorted(os.listdir(sub)):
            if f.lower().endswith(".tif") and not f.endswith(".aux.xml"):
                keys.append(os.path.splitext(f)[0])
    return keys


# ── Public entry point ────────────────────────────────────────────────────────

def run_variable_selection(
    processed_dir,
    variable_sets,
    vars_to_drop          = None,
    correlation_threshold = 0.85,
    callback              = print,
):
    """
    Run variable selection for each requested set.

    Parameters
    ----------
    processed_dir : str
        Output folder from Tool 1.
    variable_sets : dict[str, list[str]]
        {set_name: [subfolder1, subfolder2, ...]}
        Variables from all listed subfolders are merged into one set.
    vars_to_drop : list[str], optional
        Variable keys (filename stems) to exclude before correlation screening.
    correlation_threshold : float
        Pearson |r| threshold for iterative collinearity removal. Default 0.85.

    Returns
    -------
    dict
        {set_name: {"retained": [...], "dropped_manual": [...], "dropped_corr": [...], "csv": path}}
    """
    if vars_to_drop is None:
        vars_to_drop = []

    subfolders = discover_raster_subfolders(processed_dir)
    if not subfolders:
        raise FileNotFoundError(f"No raster subfolders found in: {processed_dir}")

    # Build registry: {var_key: {"path": str, "source": str}}
    registry = {}
    for sf in subfolders:
        sub = os.path.join(processed_dir, sf)
        for f in sorted(os.listdir(sub)):
            if f.lower().endswith(".tif") and not f.endswith(".aux.xml"):
                key = os.path.splitext(f)[0]
                registry[key] = {"path": os.path.join(sub, f), "source": sf}

    callback(
        f"Discovered {len(registry)} variables across {len(subfolders)} source(s): "
        + ", ".join(subfolders)
    )

    # Resolve set definitions: expand each set's subfolders into variable key lists
    set_definitions = {}
    for set_name, selected_subfolders in variable_sets.items():
        keys = []
        for sf in selected_subfolders:
            sub = os.path.join(processed_dir, sf)
            if os.path.isdir(sub):
                keys += [os.path.splitext(f)[0]
                         for f in sorted(os.listdir(sub))
                         if f.lower().endswith(".tif") and not f.endswith(".aux.xml")]
            else:
                callback(f"WARNING: Set '{set_name}' — subfolder '{sf}' not found, skipping.")
        if keys:
            set_definitions[set_name] = keys
        else:
            callback(f"WARNING: Set '{set_name}' — no variables found, skipping.")

    if not set_definitions:
        raise ValueError("No valid variable sets to process.")

    # Load rasters needed by any set (excluding manually dropped vars)
    needed_keys = set()
    for keys in set_definitions.values():
        needed_keys.update(k for k in keys if k not in vars_to_drop)

    callback(f"Loading {len(needed_keys)} rasters into memory...")
    arrays = {}
    grid_h = grid_w = None
    ref_transform = None
    for key in needed_keys:
        if key not in registry:
            callback(f"  WARNING: key '{key}' not in registry — skipping")
            continue
        with rasterio.open(registry[key]["path"]) as src:
            arrays[key] = src.read(1).astype(np.float32)
            if grid_h is None:
                grid_h, grid_w = arrays[key].shape
                ref_transform  = src.transform

    # Build valid pixel mask
    callback("Building valid pixel mask...")
    valid_mask = _build_valid_mask(arrays, processed_dir, ref_transform, grid_h, grid_w)
    callback(f"  Valid pixels: {valid_mask.sum():,}  ({valid_mask.sum()/valid_mask.size*100:.1f}%)")

    # Process each set
    results = {}
    for set_name, all_keys in set_definitions.items():
        available      = [k for k in all_keys if k in arrays]
        dropped_manual = [k for k in all_keys if k in vars_to_drop]
        var_list       = [k for k in available if k not in vars_to_drop]

        callback(f"\n── {set_name}: {len(all_keys)} total → {len(var_list)} after manual exclusion")
        if dropped_manual:
            callback(f"   Manually excluded: {dropped_manual}")

        if len(var_list) < 2:
            callback("   Fewer than 2 variables — skipping correlation screening.")
            csv_path = _export_csv(set_name, var_list, registry, processed_dir)
            callback(f"   Saved → {csv_path}")
            results[set_name] = {
                "retained"       : var_list,
                "dropped_manual" : dropped_manual,
                "dropped_corr"   : [],
                "csv"            : csv_path,
            }
            continue

        # Pearson correlation screening
        corr_full     = _corr_matrix(var_list, arrays, valid_mask)
        dropped_corr, retained = _iterative_drop(corr_full, correlation_threshold)

        callback(
            f"   {len(var_list)} → {len(retained)} after correlation screening "
            f"(dropped {len(dropped_corr)}, threshold={correlation_threshold})"
        )
        if dropped_corr:
            callback(f"   Dropped (collinear): {dropped_corr}")

        # Log high-correlation pairs from the full pre-screening set
        high_pairs = [
            (v1, v2, corr_full.loc[v1, v2])
            for i, v1 in enumerate(var_list)
            for v2 in var_list[i + 1:]
            if abs(corr_full.loc[v1, v2]) > correlation_threshold
        ]
        for v1, v2, r in sorted(high_pairs, key=lambda x: abs(x[2]), reverse=True):
            callback(f"   {v1} × {v2}  |r| = {abs(r):.3f}")

        # Export CSV
        csv_path = _export_csv(set_name, retained, registry, processed_dir)
        callback(f"   Saved → {csv_path}")

        # Export correlation heatmap for retained variables
        corr_retained = _corr_matrix(retained, arrays, valid_mask)
        png_path = os.path.join(processed_dir,
                                f"04_corr_{_safe_filename(set_name)}.png")
        _plot_corr(corr_retained, correlation_threshold,
                   f"{set_name} — {len(retained)} retained variables", png_path)
        callback(f"   Plot  → {png_path}")

        results[set_name] = {
            "retained"       : retained,
            "dropped_manual" : dropped_manual,
            "dropped_corr"   : dropped_corr,
            "csv"            : csv_path,
        }

    callback("\nVariable selection complete.")
    return results


# ── Step helpers ──────────────────────────────────────────────────────────────

def _build_valid_mask(arrays, processed_dir, ref_transform, grid_h, grid_w):
    valid = np.ones((grid_h, grid_w), dtype=bool)
    for arr in arrays.values():
        valid &= (arr != _NODATA)

    sa_shp = os.path.join(processed_dir, "study_area.shp")
    if os.path.isfile(sa_shp):
        study_area = gpd.read_file(sa_shp)
        poly_mask  = geometry_mask(
            [g.__geo_interface__ for g in study_area.geometry],
            transform  = ref_transform,
            invert     = True,
            out_shape  = (grid_h, grid_w),
        )
        valid &= poly_mask
    return valid


def _corr_matrix(var_list, arrays, valid_mask):
    data = np.stack(
        [arrays[v][valid_mask].astype(np.float64) for v in var_list], axis=1
    )
    return pd.DataFrame(np.corrcoef(data.T), index=var_list, columns=var_list)


def _iterative_drop(corr_df, threshold):
    remaining = list(corr_df.columns)
    dropped   = []
    while True:
        sub = corr_df.loc[remaining, remaining].abs()
        np.fill_diagonal(sub.values, 0)
        counts = (sub > threshold).sum(axis=1)
        if counts.max() == 0:
            break
        dropped.append(counts.idxmax())
        remaining.remove(dropped[-1])
    return dropped, remaining


def _export_csv(set_name, var_list, registry, processed_dir):
    rows = [
        {
            "Variable": v,
            "Filename": f"{v}.tif",
            "Source"  : registry[v]["source"] if v in registry else "Unknown",
        }
        for v in var_list
    ]
    path = os.path.join(processed_dir, f"final_vars_{_safe_filename(set_name)}.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _plot_corr(corr_df, threshold, title, out_path):
    n  = len(corr_df)
    sz = min(18, max(8, n * 0.62))
    fig, ax = plt.subplots(figsize=(sz, sz * 0.9))
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    sns.heatmap(
        corr_df, mask=mask, cmap="coolwarm", center=0, vmin=-1, vmax=1,
        annot=True, fmt=".2f", annot_kws={"size": 6},
        square=True, linewidths=0.3, ax=ax,
        cbar_kws={"shrink": 0.7, "label": "Pearson r"},
    )
    for i in range(n):
        for j in range(i):
            if abs(corr_df.iloc[i, j]) > threshold:
                ax.add_patch(plt.Rectangle(
                    (j, i), 1, 1, fill=False, edgecolor="red", lw=2, zorder=3
                ))
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=7)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def _safe_filename(s):
    return s.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-").strip("_")
