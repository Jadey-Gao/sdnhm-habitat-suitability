"""
SDNHM MaxEnt — Tool 3: Run MaxEnt
No arcpy dependency. Called from SDNHMMaxent.pyt :: RunMaxEnt.execute().
"""

import os, glob, re, math, warnings, datetime
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns
import rasterio
from pyproj import CRS as _CRS
from scipy.interpolate import make_interp_spline
import elapid
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

_PALETTE = ["steelblue", "darkorange", "seagreen", "purple", "brown"]


# ── Discovery helpers (also used by .pyt updateParameters) ───────────────────

def discover_variable_set_names(processed_dir):
    """Return sorted list of set names inferred from final_vars_*.csv files."""
    csvs = sorted(glob.glob(os.path.join(processed_dir, "final_vars_*.csv")))
    return [re.sub(r"^final_vars_", "", os.path.splitext(os.path.basename(c))[0])
            for c in csvs]


def discover_occ_columns(processed_dir):
    """
    Return (headers, lat_guess, lon_guess, species_guess) from occurrences_cleaned.csv.
    Any guess may be None if no candidate column name is found.
    """
    csv_path = os.path.join(processed_dir, "species", "occurrences_cleaned.csv")
    if not os.path.isfile(csv_path):
        return [], None, None, None
    import csv as _csv
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        headers = [h.strip() for h in next(_csv.reader(f))]
    hl = [h.lower() for h in headers]

    def _guess(candidates):
        for c in candidates:
            if c in hl:
                return headers[hl.index(c)]
        return None

    return (
        headers,
        _guess(["latitude",  "lat",  "y"]),
        _guess(["longitude", "lon",  "long", "x"]),
        _guess(["species",   "taxon", "name", "scientificname"]),
    )


# ── Public entry point ────────────────────────────────────────────────────────

def run_maxent_modeling(
    processed_dir,
    output_dir,
    target_crs            = "EPSG:32611",
    lat_col               = "Latitude",
    lon_col               = "Longitude",
    species_col           = "Species",
    n_background          = 10_000,
    feature_types         = None,
    beta_multiplier       = 1.0,
    test_fraction         = 0.25,
    random_seed           = 42,
    run_gridsearch        = True,
    feature_grid          = None,
    beta_grid             = None,
    n_bg_grid             = None,
    var_top_n             = None,
    callback              = print,
):
    """
    Run the full MaxEnt modeling pipeline for Tool 3.

    Parameters
    ----------
    processed_dir : str
        Output folder from Tool 1 / Tool 2.  Must contain species/,
        study_area.shp, and at least one final_vars_*.csv.
    output_dir : str
        Destination for all outputs (PNGs, TIFs, CSVs, report).
    feature_types : list[str], optional
        Feature types for the default comparison runs.
        Default: ["linear", "quadratic"].
    feature_grid : list[list[str]], optional
        Feature-type combinations to test during grid search.
        Default: [["linear","quadratic"], ["linear","quadratic","hinge","product"]].
    beta_grid : list[float], optional
        Regularisation multipliers for grid search. Default: [0.5,1.0,1.5,2.0,3.0].
    n_bg_grid : list[int], optional
        Background sample sizes for grid search. Default: [5000,10000,20000].
    var_top_n : list[int], optional
        Variable subset sizes to test; values >= n_vars are auto-skipped.
        Default: [10, 18].

    Returns
    -------
    dict  — paths to key output files.
    Note: the continuous suitability raster (cloglog, 0-1) is the primary
    output.  Binary classification is best handled in ArcGIS via
    Classify / Symbology / Reclassify for maximum flexibility.
    """
    if feature_types is None:
        feature_types = ["linear", "quadratic"]
    if feature_grid is None:
        feature_grid = [["linear", "quadratic"],
                        ["linear", "quadratic", "hinge", "product"]]
    if beta_grid is None:
        beta_grid = [0.5, 1.0, 1.5, 2.0, 3.0]
    if n_bg_grid is None:
        n_bg_grid = [5_000, 10_000, 20_000]
    if var_top_n is None:
        var_top_n = [10, 18]

    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Load data ───────────────────────────────────────────────────────────
    callback("1. Loading data...")
    occ_gdf, study_area, species_name = _load_data(
        processed_dir, lat_col, lon_col, species_col, target_crs, callback
    )

    # ── 2. Variable sets ───────────────────────────────────────────────────────
    callback("2. Reading variable sets...")
    variable_sets, var_to_path, var_display = _load_variable_sets(
        processed_dir, callback
    )

    # ── 3. Background sampling ─────────────────────────────────────────────────
    callback(f"3. Sampling {n_background:,} background points...")
    study_area_diss = _dissolve_study_area(study_area)
    np.random.seed(random_seed)
    bg_gdf = _sample_background(study_area_diss, n_background, target_crs)
    _plot_background(study_area, occ_gdf, bg_gdf, species_name, output_dir)
    callback(f"   Background sampled: {len(bg_gdf):,}")

    callback("   Annotating presence and background for all variable sets...")
    annotated_sets = _annotate_all_sets(
        variable_sets, var_to_path, occ_gdf, bg_gdf, callback
    )

    # ── 4. Default-parameter runs ──────────────────────────────────────────────
    callback("4. Default-parameter runs...")
    default_results = {}
    for set_name, (X, y, var_list) in annotated_sets.items():
        callback(f"   Fitting: {set_name}")
        default_results[set_name] = _run_model(
            X, y, var_list, set_name,
            feature_types    = feature_types,
            beta_multiplier  = beta_multiplier,
            test_fraction    = test_fraction,
            random_seed      = random_seed,
        )

    _plot_default_comparison(default_results, output_dir)
    for set_name, result in default_results.items():
        _plot_response_curves(result, var_display, output_dir,
                              prefix=f"04_default_{_safe_filename(set_name)}")
    _plot_permutation_importance(list(default_results.values()), var_display,
                                 output_dir, fname="05_default_permutation_importance.png")
    callback("   Running jackknife for default models (this may take a while)...")
    _plot_jackknife_all(list(default_results.values()), var_display,
                        test_fraction, random_seed, output_dir,
                        fname="06_default_jackknife_importance.png")

    # ── 5. Select best variable set ────────────────────────────────────────────
    callback("5. Selecting best variable set...")
    best_set_name, best_default_result = _select_best(default_results)
    callback(f"   Best: {best_set_name}  "
             f"(Test AUC={best_default_result['test_auc']:.4f}, "
             f"AICc={best_default_result['aicc']:.1f})")

    best_vars         = best_default_result["var_names"]
    best_raster_paths = {v: var_to_path[v] for v in best_vars}

    # ── 6. Grid search ─────────────────────────────────────────────────────────
    gs_csv = os.path.join(output_dir, "gridsearch_results.csv")
    if run_gridsearch:
        callback("6. Grid search...")
        gs_df, best_params = _run_gridsearch(
            study_area_diss, occ_gdf, best_vars, best_raster_paths,
            best_default_result, target_crs,
            feature_grid, beta_grid, n_bg_grid, var_top_n,
            test_fraction, random_seed, output_dir, gs_csv, callback,
        )
    else:
        callback("6. Grid search skipped — using default parameters for final model.")
        gs_df = None
        best_params = {
            "feature_types"  : feature_types,
            "beta_multiplier": beta_multiplier,
            "n_background"   : n_background,
            "var_label"      : f"all {len(best_vars)}",
            "var_list"       : best_vars,
            "test_auc"       : best_default_result["test_auc"],
            "auc_gap"        : round(best_default_result["train_auc"]
                                     - best_default_result["test_auc"], 4),
        }

    final_vars         = best_params["var_list"]
    final_raster_paths = {v: var_to_path[v] for v in final_vars}

    # ── 7. Final model ─────────────────────────────────────────────────────────
    callback("7. Fitting final model...")
    np.random.seed(random_seed)
    bg_final_gdf = _sample_background(
        study_area_diss, best_params["n_background"], target_crs
    )
    raster_list_final = list(final_raster_paths.values())
    pres_final = elapid.annotate(occ_gdf[["geometry"]], raster_list_final,
                                 labels=final_vars, drop_na=True, quiet=True)
    bg_final   = elapid.annotate(bg_final_gdf, raster_list_final,
                                 labels=final_vars, drop_na=True, quiet=True)
    X_final = pd.concat([pres_final[final_vars].astype(float),
                         bg_final  [final_vars].astype(float)], ignore_index=True)
    y_final = np.concatenate([np.ones(len(pres_final)), np.zeros(len(bg_final))])

    r_final = _run_model(
        X_final, y_final, final_vars, "Final Model",
        feature_types    = best_params["feature_types"],
        beta_multiplier  = best_params["beta_multiplier"],
        test_fraction    = test_fraction,
        random_seed      = random_seed,
    )
    callback(f"   Train AUC={r_final['train_auc']:.4f}  "
             f"Test AUC={r_final['test_auc']:.4f}  "
             f"AICc={r_final['aicc']:.1f}")

    _plot_single_roc(r_final, output_dir, fname="08_final_model_roc.png")
    _plot_final_vs_default(best_default_result, r_final, best_params, output_dir)
    _plot_response_curves(r_final, var_display, output_dir, prefix="10_final_model")
    _plot_permutation_importance([r_final], var_display, output_dir,
                                 fname="11_final_model_permutation_importance.png")
    callback("   Running jackknife for final model (this may take a while)...")
    _plot_jackknife_single(r_final, var_display, test_fraction, random_seed,
                           output_dir, fname="12_final_model_jackknife.png")

    # ── 8. Suitability raster ──────────────────────────────────────────────────
    callback("8. Applying final model to rasters...")
    suit_tif = os.path.join(output_dir, "suitability_cloglog.tif")
    elapid.apply_model_to_rasters(
        r_final["model"], raster_list_final,
        output_path=suit_tif, quiet=True,
    )
    with rasterio.open(suit_tif) as src:
        suit_arr = src.read(1).astype(float)
        suit_nd  = src.nodata
        suit_bnd = src.bounds
    suit_arr  = np.where(suit_arr == suit_nd, np.nan, suit_arr)
    n_valid   = int((~np.isnan(suit_arr)).sum())
    callback(f"   Range [{np.nanmin(suit_arr):.3f}, {np.nanmax(suit_arr):.3f}]  "
             f"mean={np.nanmean(suit_arr):.3f}  valid={n_valid:,} px")
    _plot_suitability(suit_arr, suit_bnd, study_area, species_name, output_dir)

    # ── 9. Summary outputs ────────────────────────────────────────────────────
    callback("9. Writing summary...")
    summary_row = {
        "run_date"       : datetime.date.today().isoformat(),
        "species"        : species_name,
        "best_var_set"   : best_set_name,
        "final_var_label": best_params["var_label"],
        "n_vars"         : len(final_vars),
        "feature_types"  : "+".join(best_params["feature_types"]),
        "beta_multiplier": best_params["beta_multiplier"],
        "n_background"   : best_params["n_background"],
        "train_auc"      : round(r_final["train_auc"], 4),
        "test_auc"       : round(r_final["test_auc"],  4),
        "aicc"           : round(r_final["aicc"],      1),
        "auc_gap"        : round(r_final["train_auc"] - r_final["test_auc"], 4),
        "n_valid_px"     : n_valid,
        "suit_raster"    : "suitability_cloglog.tif",
    }
    summary_csv = os.path.join(output_dir, "model_run_summary.csv")
    _append_summary_csv(summary_row, summary_csv)
    callback(f"   Summary CSV → {summary_csv}")

    report_txt = os.path.join(output_dir, "model_report.txt")
    _write_report(summary_row, default_results, best_params, gs_df,
                  run_gridsearch, report_txt, callback)

    callback("\nMaxEnt modeling complete.")
    return {
        "suit_tif"   : suit_tif,
        "summary_csv": summary_csv,
        "report_txt" : report_txt,
    }


# ── Step helpers ──────────────────────────────────────────────────────────────

def _load_data(processed_dir, lat_col, lon_col, species_col, target_crs, callback):
    occ_csv = os.path.join(processed_dir, "species", "occurrences_cleaned.csv")
    occ     = pd.read_csv(occ_csv)
    occ_gdf = gpd.GeoDataFrame(
        occ,
        geometry=gpd.points_from_xy(occ[lon_col], occ[lat_col]),
        crs="EPSG:4326",
    ).to_crs(target_crs)
    species_name = (occ[species_col].iloc[0].strip()
                    .replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-"))

    sa_path        = os.path.join(processed_dir, "study_area.shp")
    study_area_raw = gpd.read_file(sa_path)
    ref_crs        = _CRS(target_crs)
    if study_area_raw.crs is None:
        study_area = study_area_raw.set_crs(target_crs)
    elif study_area_raw.crs != ref_crs:
        study_area = study_area_raw.to_crs(target_crs)
    else:
        study_area = study_area_raw

    callback(f"   Species: {species_name.replace('_', ' ')}  |  Records: {len(occ_gdf)}")
    return occ_gdf, study_area, species_name


def _load_variable_sets(processed_dir, callback):
    """
    Parse final_vars_*.csv files into variable_sets, var_to_path, and var_display.
    Returns ({set_name: [var_keys]}, {var_key: raster_path}, {var_key: display_name}).
    """
    csv_files = sorted(glob.glob(os.path.join(processed_dir, "final_vars_*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No final_vars_*.csv files found in: {processed_dir}")

    raster_dirs = {
        d: os.path.join(processed_dir, d)
        for d in sorted(os.listdir(processed_dir))
        if os.path.isdir(os.path.join(processed_dir, d)) and d != "species"
    }

    variable_sets = {}
    var_to_path   = {}
    var_display   = {}

    for csv_path in csv_files:
        set_name = re.sub(r"^final_vars_", "",
                          os.path.splitext(os.path.basename(csv_path))[0])
        df       = pd.read_csv(csv_path)
        variable_sets[set_name] = df["Variable"].tolist()

        for _, row in df.iterrows():
            key  = row["Variable"]
            full = row.get("Full name") if "Full name" in df.columns else None
            var_display[key] = (str(full) if isinstance(full, str) and full else key)
            src_dir          = raster_dirs.get(str(row.get("Source", "")), "")
            var_to_path[key] = os.path.join(src_dir, row["Filename"]) if src_dir else ""

    callback(f"   Found {len(variable_sets)} variable set(s): "
             + ", ".join(f"{k} ({len(v)} vars)" for k, v in variable_sets.items()))
    return variable_sets, var_to_path, var_display


def _dissolve_study_area(study_area):
    dissolved = study_area.dissolve()
    dissolved["geometry"] = dissolved.geometry.buffer(0)
    return dissolved


def _sample_background(study_area_diss, n, target_crs):
    bg = elapid.sample_geoseries(study_area_diss.geometry, count=n)
    return gpd.GeoDataFrame(geometry=bg, crs=target_crs)


def _annotate_all_sets(variable_sets, var_to_path, occ_gdf, bg_gdf, callback):
    result = {}
    for set_name, var_list in variable_sets.items():
        raster_list = [var_to_path[v] for v in var_list]
        pres = elapid.annotate(occ_gdf[["geometry"]], raster_list,
                               labels=var_list, drop_na=True, quiet=True)
        bg   = elapid.annotate(bg_gdf, raster_list,
                               labels=var_list, drop_na=True, quiet=True)
        X = pd.concat([pres[var_list].astype(float),
                       bg  [var_list].astype(float)], ignore_index=True)
        y = np.concatenate([np.ones(len(pres)), np.zeros(len(bg))])
        result[set_name] = (X, y, var_list)
        callback(f"   {set_name}: presence {len(pres)} | background {len(bg)}")
    return result


def _run_model(X, y, var_names, label, feature_types, beta_multiplier,
               test_fraction, random_seed):
    """Fit MaxEnt, compute AUC / AICc / permutation importance; return result dict."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_fraction, stratify=y, random_state=random_seed,
    )
    model = elapid.MaxentModel(
        feature_types   = feature_types,
        beta_multiplier = beta_multiplier,
        transform       = "cloglog",
    )
    model.fit(X_train, y_train)

    y_prob_train = model.predict(X_train)
    y_prob_test  = model.predict(X_test)
    train_auc    = roc_auc_score(y_train, y_prob_train)
    test_auc     = roc_auc_score(y_test,  y_prob_test)
    fpr, tpr, _  = roc_curve(y_test, y_prob_test)

    pres_idx       = np.where(y_test == 1)[0]
    aicc, n_params = _compute_aicc(model, X_test.iloc[pres_idx])

    def _roc_scorer(est, Xm, ym):
        return roc_auc_score(ym, est.predict(Xm))

    perm = permutation_importance(
        model, X_test, y_test,
        n_repeats=10, scoring=_roc_scorer, random_state=random_seed,
    )
    perm_imp_df = (
        pd.DataFrame({
            "Variable"  : var_names,
            "importance": perm.importances_mean,
            "std"       : perm.importances_std,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "model"          : model,
        "label"          : label,
        "var_names"      : var_names,
        "feature_types"  : feature_types,
        "beta_multiplier": beta_multiplier,
        "X_train"        : X_train,  "X_test"      : X_test,
        "y_train"        : y_train,  "y_test"       : y_test,
        "y_prob_train"   : y_prob_train, "y_prob_test": y_prob_test,
        "train_auc"      : train_auc, "test_auc"    : test_auc,
        "fpr"            : fpr,       "tpr"          : tpr,
        "aicc"           : aicc,      "n_params"     : n_params,
        "perm_imp_df"    : perm_imp_df,
    }


def _compute_aicc(model, X_presence):
    """AICc = -2*logL + 2K + 2K(K+1)/(n-K-1)  (Warren & Seifert 2011)."""
    y_hat   = np.clip(model.predict(X_presence), 1e-10, 1.0)
    log_lik = float(np.sum(np.log(y_hat)))
    try:
        est  = model.estimator
        coef = est[-1].coef_ if hasattr(est, "__getitem__") else est.coef_
        k    = int(np.sum(np.abs(coef) > 1e-8))
    except Exception:
        k = X_presence.shape[1]
    n     = len(X_presence)
    aic   = -2.0 * log_lik + 2.0 * k
    aicc  = aic + (2.0 * k * (k + 1)) / max(n - k - 1, 1)
    return round(aicc, 2), k


def _select_best(default_results):
    """Primary sort: highest test AUC. Tiebreak: lowest AICc."""
    best_name = max(default_results,
                    key=lambda k: (default_results[k]["test_auc"],
                                   -default_results[k]["aicc"]))
    return best_name, default_results[best_name]


def _run_gridsearch(
    study_area_diss, occ_gdf, best_vars, best_raster_paths,
    best_default_result, target_crs,
    feature_grid, beta_grid, n_bg_grid, var_top_n,
    test_fraction, random_seed, output_dir, gs_csv, callback,
):
    imp_ranked = best_default_result["perm_imp_df"]["Variable"].tolist()
    n_best     = len(best_vars)

    var_sets = {}
    for n in sorted(set(var_top_n)):
        if n < n_best:
            var_sets[f"top {n}"] = imp_ranked[:n]
    var_sets[f"all {n_best}"] = best_vars

    raster_list = list(best_raster_paths.values())
    total       = len(n_bg_grid) * len(var_sets) * len(feature_grid) * len(beta_grid)
    callback(f"   {total} combinations  "
             f"({len(n_bg_grid)} bg sizes × {len(var_sets)} var sets × "
             f"{len(feature_grid)} feat combos × {len(beta_grid)} beta values)")

    gs_rows = []
    count   = 0
    for n_bg in n_bg_grid:
        np.random.seed(random_seed)
        bg_gdf = _sample_background(study_area_diss, n_bg, target_crs)

        pres_ann = elapid.annotate(occ_gdf[["geometry"]], raster_list,
                                   labels=best_vars, drop_na=True, quiet=True)
        bg_ann   = elapid.annotate(bg_gdf, raster_list,
                                   labels=best_vars, drop_na=True, quiet=True)
        X_gs = pd.concat([pres_ann[best_vars].astype(float),
                          bg_ann  [best_vars].astype(float)], ignore_index=True)
        y_gs = np.concatenate([np.ones(len(pres_ann)), np.zeros(len(bg_ann))])
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_gs, y_gs, test_size=test_fraction, stratify=y_gs, random_state=random_seed,
        )

        for var_label, var_set in var_sets.items():
            X_tr_sub = X_tr[var_set]
            X_te_sub = X_te[var_set]
            for feat_types in feature_grid:
                feat_label = "+".join(feat_types)
                for beta_val in beta_grid:
                    m = elapid.MaxentModel(
                        feature_types   = feat_types,
                        beta_multiplier = beta_val,
                        transform       = "cloglog",
                    )
                    m.fit(X_tr_sub, y_tr)
                    tr_auc = roc_auc_score(y_tr, m.predict(X_tr_sub))
                    te_auc = roc_auc_score(y_te, m.predict(X_te_sub))
                    gs_rows.append({
                        "var_set"        : var_label,
                        "n_background"   : n_bg,
                        "feature_types"  : feat_label,
                        "beta_multiplier": beta_val,
                        "train_auc"      : round(tr_auc, 4),
                        "test_auc"       : round(te_auc, 4),
                        "auc_gap"        : round(tr_auc - te_auc, 4),
                    })
                    count += 1
                    if count % 10 == 0 or count == total:
                        best_so_far = max(r["test_auc"] for r in gs_rows)
                        callback(f"   [{count}/{total}]  best AUC so far: {best_so_far:.4f}")

    gs_df = (pd.DataFrame(gs_rows)
             .sort_values("test_auc", ascending=False)
             .reset_index(drop=True))
    gs_df.to_csv(gs_csv, index=False)
    callback(f"   Grid search complete. Best test AUC: {gs_df['test_auc'].iloc[0]:.4f}  → {gs_csv}")

    _plot_gridsearch(gs_df, var_sets, output_dir)

    best_row    = gs_df.iloc[0]
    best_params = {
        "feature_types"  : best_row["feature_types"].split("+"),
        "beta_multiplier": float(best_row["beta_multiplier"]),
        "n_background"   : int(best_row["n_background"]),
        "var_label"      : best_row["var_set"],
        "var_list"       : var_sets[best_row["var_set"]],
        "test_auc"       : float(best_row["test_auc"]),
        "auc_gap"        : float(best_row["auc_gap"]),
    }
    return gs_df, best_params


def _jackknife_data(result, test_fraction, random_seed):
    """Return (only_aucs, without_aucs) dicts for all variables in result."""
    var_names = result["var_names"]
    ft        = result["feature_types"]
    beta      = result["beta_multiplier"]
    X_all = pd.concat([result["X_train"], result["X_test"]], ignore_index=True)
    y_all = np.concatenate([result["y_train"], result["y_test"]])

    def _fit_auc(X_sub):
        Xtr, Xte, ytr, yte = train_test_split(
            X_sub, y_all, test_size=test_fraction, stratify=y_all,
            random_state=random_seed,
        )
        m = elapid.MaxentModel(feature_types=ft, beta_multiplier=beta, transform="cloglog")
        m.fit(Xtr, ytr)
        return roc_auc_score(yte, m.predict(Xte))

    only_aucs, without_aucs = {}, {}
    for var in var_names:
        only_aucs[var]    = _fit_auc(X_all[[var]])
        without_aucs[var] = _fit_auc(X_all[[v for v in var_names if v != var]])
    return only_aucs, without_aucs


def _append_summary_csv(row, csv_path):
    df_new = pd.DataFrame([row])
    if os.path.isfile(csv_path):
        pd.concat([pd.read_csv(csv_path), df_new], ignore_index=True).to_csv(
            csv_path, index=False
        )
    else:
        df_new.to_csv(csv_path, index=False)


def _write_report(summary_row, default_results, best_params, gs_df,
                  run_gridsearch, report_txt, callback):
    sep = "=" * 55
    lines = [
        sep,
        "SDNHM MaxEnt Model Report",
        f'Run date  : {summary_row["run_date"]}',
        f'Species   : {summary_row["species"].replace("_", " ")}',
        sep,
        "",
        "[ Variable Sets Evaluated — Default Parameters ]",
    ]
    for set_name, r in default_results.items():
        marker = "  <- best" if set_name == summary_row["best_var_set"] else ""
        lines.append(
            f'  {set_name:<22}: Test AUC {r["test_auc"]:.4f}'
            f'  |  AICc {r["aicc"]:.1f}{marker}'
        )

    lines += ["", "[ Grid Search ]"]
    if run_gridsearch and gs_df is not None:
        lines += [
            f'  Combinations tested : {len(gs_df)}',
            "  Best combination:",
            f'    var_set         : {best_params["var_label"]}'
            f'  ({len(best_params["var_list"])} variables)',
            f'    feature_types   : {"+".join(best_params["feature_types"])}',
            f'    beta_multiplier : {best_params["beta_multiplier"]}',
            f'    n_background    : {best_params["n_background"]:,}',
            f'    test_auc        : {best_params["test_auc"]:.4f}'
            f'  (gap: {best_params["auc_gap"]:+.4f})',
        ]
    else:
        lines.append("  Skipped — default parameters used for final model.")

    lines += [
        "",
        "[ Final Model ]",
        f'  Train AUC : {summary_row["train_auc"]:.4f}',
        f'  Test AUC  : {summary_row["test_auc"]:.4f}',
        f'  AUC gap   : {summary_row["auc_gap"]:+.4f}',
        f'  AICc      : {summary_row["aicc"]:.1f}',
        "",
        "[ Output Files ]",
        f'  {summary_row["suit_raster"]}'
        f'  (valid pixels: {summary_row["n_valid_px"]:,})',
        sep,
    ]

    report = "\n".join(lines)
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write(report)

    for line in lines:
        callback(line)
    callback(f"\nReport saved → {report_txt}")


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _smooth_roc(fpr, tpr, n=300):
    fpr_u, idx = np.unique(fpr, return_index=True)
    tpr_u = tpr[idx]
    if len(fpr_u) < 4:
        return fpr, tpr
    spl   = make_interp_spline(fpr_u, tpr_u, k=3)
    fpr_s = np.linspace(fpr_u[0], fpr_u[-1], n)
    return fpr_s, np.clip(spl(fpr_s), 0, 1)


def _plot_background(study_area, occ_gdf, bg_gdf, species_name, output_dir):
    fig, ax = plt.subplots(figsize=(5, 9))
    study_area.plot(ax=ax, color="#dceedd", edgecolor="#2d6a2d", linewidth=1.2)
    bg_gdf.plot(ax=ax, color="#888888", markersize=3, alpha=0.35, zorder=2,
                label=f"Background ({len(bg_gdf):,})")
    occ_gdf.plot(ax=ax, color="crimson", markersize=40, zorder=3, alpha=0.9,
                 label=f"Presence ({len(occ_gdf)})")
    ax.set_title(f"{species_name.replace('_', ' ')} — Presence & Background",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_background_sampling.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def _plot_default_comparison(default_results, output_dir):
    results_list = list(default_results.values())
    comp_df = pd.DataFrame([{
        "Set"      : r["label"],
        "Train AUC": round(r["train_auc"], 4),
        "Test AUC" : round(r["test_auc"],  4),
        "AICc"     : round(r["aicc"],      1),
    } for r in results_list])
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(results_list))]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    x, w = np.arange(len(comp_df)), 0.35
    ax.bar(x - w / 2, comp_df["Train AUC"], width=w, label="Train AUC",
           color="steelblue", alpha=0.85)
    ax.bar(x + w / 2, comp_df["Test AUC"],  width=w, label="Test AUC",
           color="crimson",   alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(comp_df["Set"], fontsize=9, rotation=15, ha="right")
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUC")
    ax.set_title("Train vs Test AUC", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", ls="--", alpha=0.4)

    ax = axes[1]
    for r, c in zip(results_list, colors):
        fpr_s, tpr_s = _smooth_roc(r["fpr"], r["tpr"])
        ax.plot(fpr_s, tpr_s, color=c, lw=2,
                label=f'{r["label"]}  (AUC={r["test_auc"]:.4f})')
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)

    plt.suptitle("Default-Parameter Model Comparison",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_default_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def _plot_response_curves(result, var_display, output_dir, prefix):
    model     = result["model"]
    var_names = result["var_names"]
    X_train   = result["X_train"]
    imp_df    = result["perm_imp_df"]
    X_mean    = X_train[var_names].mean()

    n_plot = len(var_names)
    ncols  = 4
    nrows  = math.ceil(n_plot / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 5 * nrows), squeeze=False)
    ax_flat = axes.flatten()

    for idx, (_, row) in enumerate(imp_df.iterrows()):
        var  = row["Variable"]
        x_lo = X_train[var].quantile(0.05)
        x_hi = X_train[var].quantile(0.95)
        ax   = ax_flat[idx]
        if x_hi <= x_lo:
            ax.set_visible(False)
            continue
        x_range = np.linspace(x_lo, x_hi, 200)
        X_marg  = pd.DataFrame(np.tile(X_mean.values, (200, 1)), columns=var_names)
        X_marg[var] = x_range
        y_pred = model.predict(X_marg)
        ax.plot(x_range, y_pred, color="steelblue", lw=2)
        ax.fill_between(x_range, y_pred, alpha=0.15, color="steelblue")
        ax.margins(y=0.08)
        ax.set_ylabel("Suitability", fontsize=10)
        ax.set_title(f'#{idx + 1}  {var_display.get(var, var)}\n'
                     f'imp={row["importance"]:.4f}',
                     fontsize=10, fontweight="bold", pad=4)
        ax.tick_params(labelsize=9)
        ax.grid(True, ls="--", alpha=0.3)

    for ax in ax_flat[n_plot:]:
        ax.set_visible(False)

    plt.suptitle(f'Response Curves — {result["label"]}  '
                 f'(sorted by permutation importance)',
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_response_curves.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def _plot_permutation_importance(results_list, var_display, output_dir, fname):
    n        = len(results_list)
    max_vars = max(len(r["var_names"]) for r in results_list)
    colors   = [_PALETTE[i % len(_PALETTE)] for i in range(n)]

    fig, axes = plt.subplots(1, n,
                             figsize=(7 * n, max(6, max_vars * 0.50)),
                             squeeze=False)
    for ax, r, c in zip(axes[0], results_list, colors):
        df = r["perm_imp_df"].copy()
        df["label"] = df["Variable"].map(lambda v: var_display.get(v, v))
        ax.barh(df["label"], df["importance"], color=c, alpha=0.85)
        ax.invert_yaxis()
        ax.set_xlabel("Mean AUC decrease", fontsize=11)
        ax.set_title(r["label"], fontsize=12, fontweight="bold")
        ax.tick_params(labelsize=10)
        ax.grid(axis="x", ls="--", alpha=0.4)

    plt.suptitle("Permutation Variable Importance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()


def _jackknife_axes(ax, only_aucs, without_aucs, var_names, var_display,
                    full_auc, bar_color):
    sorted_vars   = sorted(var_names, key=lambda v: only_aucs[v], reverse=True)
    sorted_labels = [var_display.get(v, v) for v in sorted_vars]
    y_pos = np.arange(len(sorted_vars))
    ax.barh(y_pos - 0.2, [only_aucs[v]    for v in sorted_vars],
            height=0.35, color=bar_color, alpha=0.85)
    ax.barh(y_pos + 0.2, [without_aucs[v] for v in sorted_vars],
            height=0.35, color="#c0392b", alpha=0.85)
    ax.axvline(full_auc, color="black", lw=1.5, ls="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Test AUC", fontsize=11)
    ax.grid(axis="x", ls="--", alpha=0.4)
    ax.legend(handles=[
        Patch(facecolor=bar_color,  alpha=0.85, label="Only this variable"),
        Patch(facecolor="#c0392b",  alpha=0.85, label="Without this variable"),
        Line2D([0], [0], color="black", lw=1.5, ls="--",
               label=f"Full AUC={full_auc:.4f}"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.12),
       ncol=3, fontsize=9, frameon=True)


def _plot_jackknife_all(results_list, var_display, test_fraction, random_seed,
                        output_dir, fname):
    n        = len(results_list)
    max_vars = max(len(r["var_names"]) for r in results_list)
    colors   = [_PALETTE[i % len(_PALETTE)] for i in range(n)]

    fig, axes = plt.subplots(1, n,
                             figsize=(9 * n, max(6, max_vars * 0.52)),
                             squeeze=False)
    for ax, r, c in zip(axes[0], results_list, colors):
        only_aucs, without_aucs = _jackknife_data(r, test_fraction, random_seed)
        _jackknife_axes(ax, only_aucs, without_aucs,
                        r["var_names"], var_display, r["test_auc"], c)
        ax.set_title(r["label"], fontsize=12, fontweight="bold")

    plt.suptitle("Jackknife Variable Importance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_jackknife_single(result, var_display, test_fraction, random_seed,
                           output_dir, fname):
    only_aucs, without_aucs = _jackknife_data(result, test_fraction, random_seed)
    n_vars = len(result["var_names"])
    fig, ax = plt.subplots(figsize=(9, max(6, n_vars * 0.52)))
    _jackknife_axes(ax, only_aucs, without_aucs,
                    result["var_names"], var_display, result["test_auc"], "steelblue")
    ax.set_title(f'Jackknife Variable Importance — {result["label"]}',
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_gridsearch(gs_df, var_sets, output_dir):
    n_bgs         = sorted(gs_df["n_background"].unique())
    var_set_order = list(var_sets.keys())
    feat_order    = sorted(gs_df["feature_types"].unique())
    vmin, vmax    = gs_df["test_auc"].min(), gs_df["test_auc"].max()
    best_val      = gs_df["test_auc"].max()

    nrows = len(n_bgs)
    ncols = len(var_set_order)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 4.5 * nrows), squeeze=False)
    for ri, n_bg in enumerate(n_bgs):
        for ci, var_label in enumerate(var_set_order):
            ax  = axes[ri, ci]
            sub = gs_df[(gs_df["n_background"] == n_bg) &
                        (gs_df["var_set"]       == var_label)]
            if sub.empty:
                ax.set_visible(False)
                continue
            local_feats = [f for f in feat_order if f in sub["feature_types"].values]
            pivot = sub.pivot_table(index="beta_multiplier",
                                    columns="feature_types",
                                    values="test_auc").reindex(columns=local_feats)
            show_cbar = (ci == ncols - 1)
            sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlOrRd",
                        vmin=vmin, vmax=vmax, ax=ax, linewidths=0.5,
                        cbar=show_cbar, annot_kws={"fontsize": 10},
                        cbar_kws={"label": "Test AUC", "shrink": 0.8} if show_cbar else {})
            for _, br in gs_df[(gs_df["n_background"] == n_bg) &
                               (gs_df["var_set"]      == var_label) &
                               (gs_df["test_auc"]     == best_val)].iterrows():
                ft = br["feature_types"]
                if ft in list(pivot.columns):
                    r_i = list(pivot.index).index(br["beta_multiplier"])
                    c_i = list(pivot.columns).index(ft)
                    ax.add_patch(plt.Rectangle(
                        (c_i, r_i), 1, 1, fill=False,
                        edgecolor="black", lw=2.5, zorder=5,
                    ))
            ax.set_title(f"n_bg={n_bg:,}  |  vars: {var_label}",
                         fontsize=12, fontweight="bold")
            ax.set_xlabel("Feature Types", fontsize=10)
            ax.set_ylabel("Beta Multiplier", fontsize=10)
            ax.tick_params(axis="x", rotation=20)

    plt.suptitle("Grid Search: Test AUC  (black border = global best)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "07_gridsearch_heatmap.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def _plot_single_roc(result, output_dir, fname):
    fpr_s, tpr_s = _smooth_roc(result["fpr"], result["tpr"])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr_s, tpr_s, color="steelblue", lw=2,
            label=f'Test AUC = {result["test_auc"]:.4f}')
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.text(0.55, 0.10, f'Train AUC = {result["train_auc"]:.4f}',
            transform=ax.transAxes, fontsize=9, color="grey")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f'ROC — {result["label"]}', fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_final_vs_default(best_default, r_final, best_params, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    pairs = [
        (axes[0], best_default,
         f'Best Default  ({best_default["label"]})'),
        (axes[1], r_final,
         f'Final Tuned  (beta={best_params["beta_multiplier"]}, '
         f'{len(best_params["feature_types"])} feature classes)'),
    ]
    for ax, r, title in pairs:
        fpr_tr, tpr_tr, _ = roc_curve(r["y_train"], r["y_prob_train"])
        ftr_s, ttr_s = _smooth_roc(fpr_tr, tpr_tr)
        fte_s, tte_s = _smooth_roc(r["fpr"],  r["tpr"])
        ax.plot(ftr_s, ttr_s, color="grey",      lw=2, ls="--", alpha=0.7,
                label=f'Train AUC = {r["train_auc"]:.4f}')
        ax.plot(fte_s, tte_s, color="steelblue", lw=2.5,
                label=f'Test AUC  = {r["test_auc"]:.4f}')
        ax.fill_between(fte_s, tte_s, alpha=0.15, color="steelblue")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate",  fontsize=12)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(loc="lower right", fontsize=11)
        ax.tick_params(labelsize=10)
    plt.suptitle("Default vs Tuned Model — ROC Comparison",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "09_default_vs_tuned_roc.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def _plot_suitability(suit_arr, suit_bnd, study_area, species_name, output_dir):
    extent = [suit_bnd.left, suit_bnd.right, suit_bnd.bottom, suit_bnd.top]
    fig, ax = plt.subplots(figsize=(6, 10))
    im = ax.imshow(suit_arr, cmap="YlOrRd", vmin=0, vmax=1,
                   extent=extent, origin="upper", aspect="equal")
    study_area.plot(ax=ax, color="none", edgecolor="#333333", linewidth=1.0, zorder=2)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04,
                 label="Habitat Suitability (cloglog, 0-1)")
    ax.set_xlim(suit_bnd.left, suit_bnd.right)
    ax.set_ylim(suit_bnd.bottom, suit_bnd.top)
    ax.set_xlabel("X", fontsize=11); ax.set_ylabel("Y", fontsize=11)
    ax.set_title(f"{species_name.replace('_', ' ')} — Predicted Habitat Suitability",
                 fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=10)
    ax.grid(True, ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"13_{species_name}_suitability.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


# ── Utility ───────────────────────────────────────────────────────────────────

def _safe_filename(s):
    return (s.replace(" ", "_").replace("(", "").replace(")", "")
             .replace("/", "-").strip("_"))
