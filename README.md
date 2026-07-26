# SDNHM Habitat Suitability

MaxEnt species distribution modeling for the San Diego Natural History Museum, built around
*Bipes biporus* (Baja California worm lizard) on the Baja Peninsula.

Two ways to run the same analysis:

- **Notebooks** — `01_preprocessing.ipynb` and `02_maxent_modeling.ipynb`.
- **ArcGIS Pro toolbox** — `Toolbox/SDNHMMaxent.pyt`, for analysts who don't work in Jupyter.

Both call the same `sdnhm` package, so they produce identical results.

---

## Method

Three variable sets run through the whole pipeline in parallel and are compared on the same
held-out data:

| Set | Predictors |
| --- | --- |
| `BioClim_DEM` | WorldClim bioclimatic variables + USGS elevation |
| `Envirem` | ENVIREM topoclimatic and evapotranspiration surfaces |
| `Combined` | Both of the above, jointly pruned for collinearity |

Pipeline stages:

1. **Preprocessing** — reproject the study area, thin occurrences to one per raster cell, then
   clip and align every raster to a common grid.
   - Thinning matters: museum records cluster near roads and towns. Without it, MaxEnt reads
     sampling effort as habitat quality.
2. **Variable selection** — correlation screening within each set to drop collinear predictors.
3. **Default runs** — fit MaxEnt (`elapid`) per variable set; compare on test AUC and AICc.
4. **Tuning** — grid search over regularization and feature classes for the winning set.
5. **Final model** — ROC, response curves, jackknife, and permutation importance.
6. **Prediction** — continuous cloglog suitability raster plus a thresholded binary range map.

## Repository layout

```
01_preprocessing.ipynb      Notebook 1 — cleaning, thinning, raster alignment, variable selection
02_maxent_modeling.ipynb    Notebook 2 — fitting, comparison, tuning, prediction
Toolbox/
  SDNHMMaxent.pyt           ArcGIS Pro toolbox: Prepare Data / Variable Selection / Run MaxEnt
  sdnhm/                    Shared implementation (pure Python, no arcpy)
    preprocessing.py
    variable_selection.py
    modeling.py
Raw Datasets/Species data/  Occurrence records + Baja Peninsula study-area shapefile
Processed/                  Aligned rasters, thinned occurrences, variable-selection output
Modeled/                    Model diagnostics, comparison figures, prediction rasters
```

## Getting started

```bash
git clone https://github.com/Jadey-Gao/sdnhm-habitat-suitability.git
cd sdnhm-habitat-suitability
pip install -r requirements.txt
jupyter lab
```

- `Processed/` and `Modeled/` are committed, so `02_maxent_modeling.ipynb` runs end to end on a
  fresh clone.
- To re-run `01_preprocessing.ipynb` you need the raw environmental rasters. They are excluded
  from version control (~13 GB) — download them from the sources below.
- For the ArcGIS route, add `Toolbox/SDNHMMaxent.pyt` in the ArcGIS Pro Catalog pane. The three
  tools match the notebook stages and take the same inputs.

## Data sources & citation

| Dataset | Source |
| --- | --- |
| BioClim | [WorldClim v2.1](https://www.worldclim.org/data/worldclim21.html) |
| ENVIREM | [envirem.github.io](https://envirem.github.io/) |
| Elevation | [USGS 3DEP](https://www.usgs.gov/3d-elevation-program) |
| Occurrences | Mahrdt et al. 2022 — see below |

The occurrence points are based on Mahrdt et al. (2022). **Any use of them must credit:**

> Mahrdt, C.R., K.R. Beaman, J.H.V. Villavicencio, and T.J. Papenfuss. 2022.
> *Bipes biporus.* Catalogue of American Amphibians and Reptiles 930:1–39.

## Adapting to another species

The workflow is not hard-coded to *Bipes biporus*. Cells marked `✎` are the ones meant to be
edited:

- **§1.2** — species name and input paths
- **§1.3** — CRS, target resolution, study-area buffer
- **§1.4** — variable sets

`02_maxent_modeling.ipynb` auto-discovers whatever notebook 1 produced, so it needs no edits.
