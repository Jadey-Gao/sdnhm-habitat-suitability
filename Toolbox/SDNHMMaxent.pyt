import arcpy
import os
import sys

# Make sdnhm/ importable from the same directory as this .pyt file
_TB_DIR = os.path.dirname(os.path.abspath(__file__))
if _TB_DIR not in sys.path:
    sys.path.insert(0, _TB_DIR)


class Toolbox:
    def __init__(self):
        self.label = "SDNHM MaxEnt"
        self.alias  = "sdnhm_maxent"
        self.tools  = [PrepareData, VariableSelection, RunMaxEnt]


# ──────────────────────────────────────────────────────────────────────────────
# Tool 1 — Prepare Data
# ──────────────────────────────────────────────────────────────────────────────

class PrepareData:
    def __init__(self):
        self.label       = "1. Prepare Data"
        self.description = (
            "Reproject study area, thin occurrence records, and clip / align "
            "environmental rasters to a common grid."
        )
        self.canRunInBackground = True

    # ── Parameters ────────────────────────────────────────────────────────────
    def getParameterInfo(self):
        params = []

        # 0 ── Species name
        p = arcpy.Parameter(
            displayName   = "Species Name",
            name          = "species_name",
            datatype      = "GPString",
            parameterType = "Required",
            direction     = "Input",
        )
        params.append(p)

        # 1 ── Occurrence CSV
        p = arcpy.Parameter(
            displayName   = "Species Occurrence CSV",
            name          = "species_csv",
            datatype      = "DEFile",
            parameterType = "Required",
            direction     = "Input",
        )
        p.filter.list = ["csv"]
        params.append(p)

        # 2 ── Study area shapefile
        p = arcpy.Parameter(
            displayName   = "Study Area Shapefile",
            name          = "study_area_shp",
            datatype      = "DEShapefile",
            parameterType = "Required",
            direction     = "Input",
        )
        params.append(p)

        # 3 ── Environmental raster folders (multi-value, any number)
        p = arcpy.Parameter(
            displayName   = "Environmental Raster Folders",
            name          = "env_folders",
            datatype      = "DEFolder",
            parameterType = "Required",
            direction     = "Input",
            multiValue    = True,
        )
        params.append(p)

        # 4 ── Output folder
        p = arcpy.Parameter(
            displayName   = "Output Folder",
            name          = "output_dir",
            datatype      = "DEFolder",
            parameterType = "Required",
            direction     = "Input",
        )
        params.append(p)

        # 5 ── Target CRS  (visible, editable, default from study area)
        p = arcpy.Parameter(
            displayName   = "Target Coordinate System (EPSG code)",
            name          = "target_crs",
            datatype      = "GPString",
            parameterType = "Optional",
            direction     = "Input",
        )
        p.value = "EPSG:32611"
        params.append(p)

        # 6 ── Study area buffer
        p = arcpy.Parameter(
            displayName   = "Study Area Buffer (degrees)",
            name          = "buffer_deg",
            datatype      = "GPDouble",
            parameterType = "Optional",
            direction     = "Input",
        )
        p.value = 1.0
        params.append(p)

        # 7 ── Raster resolution  (visible, auto-detected, editable)
        p = arcpy.Parameter(
            displayName   = "Source Raster Resolution (arcseconds)",
            name          = "raster_res_arcsec",
            datatype      = "GPLong",
            parameterType = "Optional",
            direction     = "Input",
        )
        params.append(p)

        # 8 ── Latitude column  (auto-detected from CSV header, editable)
        p = arcpy.Parameter(
            displayName   = "Latitude Column Name",
            name          = "lat_col",
            datatype      = "GPString",
            parameterType = "Optional",
            direction     = "Input",
        )
        params.append(p)

        # 9 ── Longitude column  (auto-detected, editable)
        p = arcpy.Parameter(
            displayName   = "Longitude Column Name",
            name          = "lon_col",
            datatype      = "GPString",
            parameterType = "Optional",
            direction     = "Input",
        )
        params.append(p)

        # 10 ── Species column  (auto-detected, editable)
        p = arcpy.Parameter(
            displayName   = "Species Column Name",
            name          = "species_col",
            datatype      = "GPString",
            parameterType = "Optional",
            direction     = "Input",
        )
        params.append(p)

        return params

    # ── Auto-detection (runs every time a parameter changes) ──────────────────
    def updateParameters(self, parameters):
        csv_param = parameters[1]
        env_param = parameters[3]
        res_param = parameters[7]
        lat_param = parameters[8]
        lon_param = parameters[9]
        spp_param = parameters[10]

        # Populate column-name dropdowns from CSV header, auto-select best match
        if csv_param.value:
            csv_path = csv_param.valueAsText
            if csv_path and os.path.isfile(csv_path):
                try:
                    import csv as _csv
                    with open(csv_path, newline="", encoding="utf-8-sig") as f:
                        headers = [h.strip() for h in next(_csv.reader(f))]
                    hl = [h.lower() for h in headers]

                    # Set dropdown list for all three params
                    for p in (lat_param, lon_param, spp_param):
                        p.filter.type = "ValueList"
                        p.filter.list = headers

                    # Pre-select a default only when CSV was just changed
                    if csv_param.altered and not csv_param.hasBeenValidated:
                        if not lat_param.altered:
                            for cand in ["latitude", "lat", "y"]:
                                if cand in hl:
                                    lat_param.value = headers[hl.index(cand)]
                                    break

                        if not lon_param.altered:
                            for cand in ["longitude", "lon", "long", "x"]:
                                if cand in hl:
                                    lon_param.value = headers[hl.index(cand)]
                                    break

                        if not spp_param.altered:
                            for cand in ["species", "taxon", "name", "scientificname"]:
                                if cand in hl:
                                    spp_param.value = headers[hl.index(cand)]
                                    break
                except Exception:
                    pass

        # Detect raster resolution from the first folder's first .tif
        if env_param.altered and not env_param.hasBeenValidated:
            folders = _parse_multivalue(env_param.valueAsText)
            if folders and not res_param.altered:
                first_dir = folders[0]
                if os.path.isdir(first_dir):
                    tifs = [f for f in os.listdir(first_dir)
                            if f.lower().endswith(".tif") and not f.endswith(".aux.xml")]
                    if tifs:
                        try:
                            import rasterio
                            with rasterio.open(os.path.join(first_dir, tifs[0])) as src:
                                res_param.value = round(src.res[0] * 3600)
                        except Exception:
                            pass

        return

    def updateMessages(self, parameters):
        return

    # ── Execution ─────────────────────────────────────────────────────────────
    def execute(self, parameters, messages):
        try:
            from sdnhm.preprocessing import run_preprocessing
        except ImportError as e:
            messages.addErrorMessage(
                f"Cannot import sdnhm: {e}\n"
                "Ensure sdnhm/ is next to this .pyt file and all packages are installed."
            )
            return

        species_name = parameters[0].valueAsText
        species_csv  = parameters[1].valueAsText
        study_area   = parameters[2].valueAsText
        env_folders  = _parse_multivalue(parameters[3].valueAsText)
        output_dir   = parameters[4].valueAsText
        target_crs   = parameters[5].valueAsText  or "EPSG:32611"
        buffer_deg   = float(parameters[6].value  or 1.0)
        raster_res   = int(parameters[7].value)    if parameters[7].value else None
        lat_col      = parameters[8].valueAsText   or "Latitude"
        lon_col      = parameters[9].valueAsText   or "Longitude"
        species_col  = parameters[10].valueAsText  or "Species"

        def cb(msg):
            messages.addMessage(str(msg))

        try:
            result = run_preprocessing(
                species_name          = species_name,
                species_csv           = species_csv,
                study_area_shp        = study_area,
                env_folders           = env_folders,
                output_dir            = output_dir,
                lat_col               = lat_col,
                lon_col               = lon_col,
                species_col           = species_col,
                coordinate_system     = target_crs,
                raster_res_arcsec     = raster_res,
                study_area_buffer_deg = buffer_deg,
                callback              = cb,
            )
            messages.addMessage("Preprocessing complete.")
            messages.addMessage(f"  Study area  : {result['study_area_shp']}")
            messages.addMessage(f"  Occurrences : {result['occ_csv']}")
            messages.addMessage(f"  Rasters     : {output_dir}")
        except Exception as e:
            import traceback
            messages.addErrorMessage(f"{e}\n{traceback.format_exc()}")

    def postExecute(self, parameters):
        return


# ──────────────────────────────────────────────────────────────────────────────
# Tool 2 — Variable Selection
# ──────────────────────────────────────────────────────────────────────────────

class VariableSelection:
    def __init__(self):
        self.label       = "2. Variable Selection"
        self.description = (
            "Manual exclusion and Pearson correlation screening to produce "
            "final non-collinear predictor sets."
        )
        self.canRunInBackground = True

    # ── Parameters ────────────────────────────────────────────────────────────
    def getParameterInfo(self):
        params = []

        # 0 ── Processed folder (output from Tool 1)
        p = arcpy.Parameter(
            displayName   = "Processed Folder (output from Tool 1)",
            name          = "processed_dir",
            datatype      = "DEFolder",
            parameterType = "Required",
            direction     = "Input",
        )
        params.append(p)

        # 1 ── Variable sets table  (one row = one Set + Subfolder pair)
        p = arcpy.Parameter(
            displayName   = "Variable Sets  (one row per Set–Subfolder pair)",
            name          = "variable_sets",
            datatype      = "GPValueTable",
            parameterType = "Required",
            direction     = "Input",
        )
        p.columns = [["GPString", "Set Name"], ["GPString", "Subfolder"]]
        p.filters[1].type = "ValueList"
        p.filters[1].list = []
        params.append(p)

        # 2 ── Variables to drop  (dropdown, populated from all .tif stems)
        p = arcpy.Parameter(
            displayName   = "Variables to Drop (manual exclusion)",
            name          = "vars_to_drop",
            datatype      = "GPString",
            parameterType = "Optional",
            direction     = "Input",
            multiValue    = True,
        )
        p.filter.type = "ValueList"
        p.filter.list = []
        params.append(p)

        # 3 ── Pearson correlation threshold
        p = arcpy.Parameter(
            displayName   = "Pearson Correlation Threshold",
            name          = "correlation_threshold",
            datatype      = "GPDouble",
            parameterType = "Optional",
            direction     = "Input",
        )
        p.value = 0.85
        params.append(p)

        return params

    # ── Auto-populate dropdowns when processed folder changes ─────────────────
    def updateParameters(self, parameters):
        proc_param = parameters[0]
        sets_param = parameters[1]
        drop_param = parameters[2]

        if proc_param.value and proc_param.altered and not proc_param.hasBeenValidated:
            proc_dir = proc_param.valueAsText
            if proc_dir and os.path.isdir(proc_dir):
                try:
                    from sdnhm.variable_selection import (
                        discover_raster_subfolders,
                        discover_all_variable_keys,
                    )
                    subfolders = discover_raster_subfolders(proc_dir)
                    sets_param.filters[1].type = "ValueList"
                    sets_param.filters[1].list = subfolders

                    all_keys = discover_all_variable_keys(proc_dir)
                    drop_param.filter.type = "ValueList"
                    drop_param.filter.list = all_keys
                except Exception:
                    pass
        return

    def updateMessages(self, parameters):
        return

    def isLicensed(self):
        return True

    # ── Execution ─────────────────────────────────────────────────────────────
    def execute(self, parameters, messages):
        try:
            from sdnhm.variable_selection import run_variable_selection
        except ImportError as e:
            messages.addErrorMessage(
                f"Cannot import sdnhm: {e}\n"
                "Ensure sdnhm/ is next to this .pyt file and all packages are installed."
            )
            return

        processed_dir         = parameters[0].valueAsText
        variable_sets         = _parse_variable_sets_table(parameters[1].values)
        vars_to_drop          = _parse_multivalue(parameters[2].valueAsText) if parameters[2].value else []
        correlation_threshold = float(parameters[3].value or 0.85)

        def cb(msg):
            messages.addMessage(str(msg))

        try:
            results = run_variable_selection(
                processed_dir         = processed_dir,
                variable_sets         = variable_sets,
                vars_to_drop          = vars_to_drop,
                correlation_threshold = correlation_threshold,
                callback              = cb,
            )
            messages.addMessage("\nSummary:")
            for set_name, info in results.items():
                line = f"  {set_name}: {len(info['retained'])} retained"
                if info["dropped_corr"]:
                    line += f"  |  dropped (collinear): {info['dropped_corr']}"
                messages.addMessage(line)
                if info.get("csv"):
                    messages.addMessage(f"    → {info['csv']}")
        except Exception as e:
            import traceback
            messages.addErrorMessage(f"{e}\n{traceback.format_exc()}")

    def postExecute(self, parameters):
        return


# ──────────────────────────────────────────────────────────────────────────────
# Tool 3 — Run MaxEnt
# ──────────────────────────────────────────────────────────────────────────────

# Feature-type combinations available for selection
_FG_OPTS = [
    "LQ    — linear, quadratic",
    "LQH   — linear, quadratic, hinge",
    "LQHP  — linear, quadratic, hinge, product",
    "LQHPT — linear, quadratic, hinge, product, threshold",
]
_FG_MAP = {
    "LQ"   : ["linear", "quadratic"],
    "LQH"  : ["linear", "quadratic", "hinge"],
    "LQHP" : ["linear", "quadratic", "hinge", "product"],
    "LQHPT": ["linear", "quadratic", "hinge", "product", "threshold"],
}


class RunMaxEnt:
    def __init__(self):
        self.label       = "3. Run MaxEnt"
        self.description = (
            "Background sampling, default-parameter comparison, optional "
            "hyperparameter grid search, final tuned model, and suitability "
            "raster output."
        )
        self.canRunInBackground = True

    # ── Parameters ────────────────────────────────────────────────────────────
    def getParameterInfo(self):
        params = []

        # 0 ── Processed folder (output from Tool 1 / Tool 2)
        p = arcpy.Parameter(
            displayName   = "Processed Folder (output from Tools 1 & 2)",
            name          = "processed_dir",
            datatype      = "DEFolder",
            parameterType = "Required",
            direction     = "Input",
        )
        params.append(p)

        # 1 ── Output folder
        p = arcpy.Parameter(
            displayName   = "Output Folder",
            name          = "output_dir",
            datatype      = "DEFolder",
            parameterType = "Required",
            direction     = "Input",
        )
        params.append(p)

        # 2 ── Target CRS
        p = arcpy.Parameter(
            displayName   = "Target Coordinate System (EPSG code)",
            name          = "target_crs",
            datatype      = "GPString",
            parameterType = "Optional",
            direction     = "Input",
        )
        p.value = "EPSG:32611"
        params.append(p)

        # 3 ── Latitude column (auto-detected)
        p = arcpy.Parameter(
            displayName   = "Latitude Column Name",
            name          = "lat_col",
            datatype      = "GPString",
            parameterType = "Optional",
            direction     = "Input",
        )
        params.append(p)

        # 4 ── Longitude column (auto-detected)
        p = arcpy.Parameter(
            displayName   = "Longitude Column Name",
            name          = "lon_col",
            datatype      = "GPString",
            parameterType = "Optional",
            direction     = "Input",
        )
        params.append(p)

        # 5 ── Species column (auto-detected)
        p = arcpy.Parameter(
            displayName   = "Species Column Name",
            name          = "species_col",
            datatype      = "GPString",
            parameterType = "Optional",
            direction     = "Input",
        )
        params.append(p)

        # 6 ── Background sample size
        p = arcpy.Parameter(
            displayName   = "Background Sample Size",
            name          = "n_background",
            datatype      = "GPLong",
            parameterType = "Optional",
            direction     = "Input",
        )
        p.value = 10000
        params.append(p)

        # 7 ── Feature types for default runs (multi-select)
        p = arcpy.Parameter(
            displayName   = "Feature Types (Default Runs)",
            name          = "feature_types",
            datatype      = "GPString",
            parameterType = "Optional",
            direction     = "Input",
            multiValue    = True,
        )
        p.filter.type = "ValueList"
        p.filter.list = ["linear", "quadratic", "hinge", "product", "threshold"]
        p.value       = "linear;quadratic"
        params.append(p)

        # 8 ── Regularisation multiplier (default runs)
        p = arcpy.Parameter(
            displayName   = "Regularization Multiplier (Default Runs)",
            name          = "beta_multiplier",
            datatype      = "GPDouble",
            parameterType = "Optional",
            direction     = "Input",
        )
        p.value = 1.0
        params.append(p)

        # 9 ── Test fraction
        p = arcpy.Parameter(
            displayName   = "Test Fraction",
            name          = "test_fraction",
            datatype      = "GPDouble",
            parameterType = "Optional",
            direction     = "Input",
        )
        p.value = 0.25
        params.append(p)

        # 10 ── Random seed
        p = arcpy.Parameter(
            displayName   = "Random Seed",
            name          = "random_seed",
            datatype      = "GPLong",
            parameterType = "Optional",
            direction     = "Input",
        )
        p.value = 42
        params.append(p)

        # 11 ── Run grid search toggle
        p = arcpy.Parameter(
            displayName   = "Run Hyperparameter Grid Search",
            name          = "run_gridsearch",
            datatype      = "GPBoolean",
            parameterType = "Optional",
            direction     = "Input",
        )
        p.value = True
        params.append(p)

        # 12 ── Feature-type combinations to test (multi-select dropdown)
        p = arcpy.Parameter(
            displayName   = "Feature-Type Combinations to Test (Grid Search)",
            name          = "feature_grid",
            datatype      = "GPString",
            parameterType = "Optional",
            direction     = "Input",
            multiValue    = True,
        )
        p.filter.type = "ValueList"
        p.filter.list = _FG_OPTS
        p.value       = f"{_FG_OPTS[0]};{_FG_OPTS[2]}"   # LQ and LQHP
        params.append(p)

        # 13 ── Beta grid (multi-value numeric)
        p = arcpy.Parameter(
            displayName   = "Beta Multipliers to Test (Grid Search)",
            name          = "beta_grid",
            datatype      = "GPDouble",
            parameterType = "Optional",
            direction     = "Input",
            multiValue    = True,
        )
        p.value = "0.5;1.0;1.5;2.0;3.0"
        params.append(p)

        # 14 ── Background grid (multi-value numeric)
        p = arcpy.Parameter(
            displayName   = "Background Sizes to Test (Grid Search)",
            name          = "n_bg_grid",
            datatype      = "GPLong",
            parameterType = "Optional",
            direction     = "Input",
            multiValue    = True,
        )
        p.value = "5000;10000;20000"
        params.append(p)

        # 15 ── Variable top-N subsets (multi-value numeric)
        p = arcpy.Parameter(
            displayName   = "Variable Subset Sizes to Test  (Grid Search, values >= n_vars skipped)",
            name          = "var_top_n",
            datatype      = "GPLong",
            parameterType = "Optional",
            direction     = "Input",
            multiValue    = True,
        )
        p.value = "10;18"
        params.append(p)

        return params

    # ── Auto-detect column names; enable/disable grid search params ───────────
    def updateParameters(self, parameters):
        proc_param = parameters[0]
        run_gs     = parameters[11]
        gs_params  = [parameters[12], parameters[13], parameters[14], parameters[15]]

        # Enable / disable grid search parameters based on toggle
        if run_gs.value is not None:
            enabled = bool(run_gs.value)
            for p in gs_params:
                p.enabled = enabled

        # Populate column-name dropdowns from occurrences_cleaned.csv
        if proc_param.value and proc_param.altered and not proc_param.hasBeenValidated:
            proc_dir = proc_param.valueAsText
            if proc_dir and os.path.isdir(proc_dir):
                try:
                    from sdnhm.modeling import discover_occ_columns
                    headers, lat, lon, species = discover_occ_columns(proc_dir)
                    if headers:
                        lat_p, lon_p, spp_p = parameters[3], parameters[4], parameters[5]
                        for p in (lat_p, lon_p, spp_p):
                            p.filter.type = "ValueList"
                            p.filter.list = headers
                        if not lat_p.altered and lat:     lat_p.value = lat
                        if not lon_p.altered and lon:     lon_p.value = lon
                        if not spp_p.altered and species: spp_p.value = species
                except Exception:
                    pass
        return

    def updateMessages(self, parameters):
        return

    def isLicensed(self):
        return True

    # ── Execution ─────────────────────────────────────────────────────────────
    def execute(self, parameters, messages):
        try:
            from sdnhm.modeling import run_maxent_modeling
        except ImportError as e:
            messages.addErrorMessage(
                f"Cannot import sdnhm: {e}\n"
                "Ensure sdnhm/ is next to this .pyt file and all packages are installed."
            )
            return

        processed_dir         = parameters[0].valueAsText
        output_dir            = parameters[1].valueAsText
        target_crs            = parameters[2].valueAsText  or "EPSG:32611"
        lat_col               = parameters[3].valueAsText  or "Latitude"
        lon_col               = parameters[4].valueAsText  or "Longitude"
        species_col           = parameters[5].valueAsText  or "Species"
        n_background          = int(parameters[6].value   or 10000)
        feature_types         = _parse_multivalue(parameters[7].valueAsText) or ["linear", "quadratic"]
        beta_multiplier       = float(parameters[8].value or 1.0)
        test_fraction         = float(parameters[9].value or 0.25)
        random_seed           = int(parameters[10].value  or 42)
        run_gridsearch        = bool(parameters[11].value) if parameters[11].value is not None else True
        feature_grid          = _parse_feature_grid(parameters[12].valueAsText)
        beta_grid             = [float(v) for v in _parse_multivalue(parameters[13].valueAsText)] \
                                    if parameters[13].value else [0.5, 1.0, 1.5, 2.0, 3.0]
        n_bg_grid             = [int(v) for v in _parse_multivalue(parameters[14].valueAsText)] \
                                    if parameters[14].value else [5000, 10000, 20000]
        var_top_n             = [int(v) for v in _parse_multivalue(parameters[15].valueAsText)] \
                                    if parameters[15].value else [10, 18]

        def cb(msg):
            messages.addMessage(str(msg))

        try:
            result = run_maxent_modeling(
                processed_dir         = processed_dir,
                output_dir            = output_dir,
                target_crs            = target_crs,
                lat_col               = lat_col,
                lon_col               = lon_col,
                species_col           = species_col,
                n_background          = n_background,
                feature_types         = feature_types,
                beta_multiplier       = beta_multiplier,
                test_fraction         = test_fraction,
                random_seed           = random_seed,
                run_gridsearch        = run_gridsearch,
                feature_grid          = feature_grid,
                beta_grid             = beta_grid,
                n_bg_grid             = n_bg_grid,
                var_top_n             = var_top_n,
                callback              = cb,
            )
            messages.addMessage(f"  Suitability raster : {result['suit_tif']}")
            messages.addMessage(f"  Summary CSV        : {result['summary_csv']}")
            messages.addMessage(f"  Report             : {result['report_txt']}")
        except Exception as e:
            import traceback
            messages.addErrorMessage(f"{e}\n{traceback.format_exc()}")

    def postExecute(self, parameters):
        return


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_multivalue(raw):
    """Split ArcGIS multi-value string (semicolon-separated, paths may be quoted)."""
    if not raw:
        return []
    return [p.strip().strip("'\"") for p in raw.split(";") if p.strip()]


def _parse_feature_grid(raw):
    """
    Convert multi-value feature-grid selection (display labels) to list[list[str]].
    Each item starts with an abbreviation key (LQ / LQH / LQHP / LQHPT).
    Falls back to [LQ, LQHP] if nothing is parseable.
    """
    result = []
    for item in _parse_multivalue(raw):
        abbr = item.split()[0]
        if abbr in _FG_MAP:
            result.append(_FG_MAP[abbr])
    return result or [["linear", "quadratic"],
                      ["linear", "quadratic", "hinge", "product"]]


def _parse_variable_sets_table(rows):
    """
    Parse GPValueTable rows into {set_name: [subfolder, ...]} dict.
    Each row is [set_name, subfolder]; rows with the same set_name are grouped.
    """
    result = {}
    if not rows:
        return result
    for row in rows:
        set_name = str(row[0]).strip().strip("'\"") if row[0] else ""
        subfolder = str(row[1]).strip().strip("'\"") if row[1] else ""
        if set_name and subfolder:
            result.setdefault(set_name, []).append(subfolder)
    return result
