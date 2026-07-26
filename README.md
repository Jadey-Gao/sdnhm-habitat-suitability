# SDNHM Habitat Suitability

MaxEnt species distribution modeling for the San Diego Natural History Museum, built around
*Bipes biporus* (Baja California worm lizard) on the Baja Peninsula.

The repository ships two things:

- **A two-notebook workflow** (`01_preprocessing.ipynb`, `02_maxent_modeling.ipynb`) that takes
  occurrence records and environmental rasters through cleaning, variable selection, model
  fitting, tuning, and map production.
- **An ArcGIS Pro Python Toolbox** (`Toolbox/SDNHMMaxent.pyt`) that exposes the same three
  stages as GUI tools for analysts who don't work in Jupyter.

Both front-ends call the same `sdnhm` package, so the notebook and the toolbox produce
identical results.

---

## Method

Three candidate variable sets are carried through the whole pipeline in parallel and compared
on the same held-out data:

| Set | Predictors |
| --- | --- |
| `BioClim_DEM` | WorldClim bioclimatic variables + USGS elevation |
| `Envirem` | ENVIREM topoclimatic and evapotranspiration surfaces |
| `Combined` | Both of the above, jointly pruned for collinearity |

Pipeline stages:

1. **Preprocessing** — reproject the study area, spatially thin occurrences to one record per
   raster cell (museum records cluster near roads and towns; without thinning MaxEnt reads
   sampling effort as habitat quality), then clip and align every raster to a common grid.
2. **Variable selection** — correlation screening within each set to drop collinear predictors.
3. **Default runs** — fit MaxEnt (`elapid`) per variable set, compare on test AUC and AICc.
4. **Tuning** — grid search over regularization and feature classes for the winning set.
5. **Final model** — ROC, response curves, jackknife, and permutation importance.
6. **Prediction** — continuous cloglog suitability raster plus a thresholded binary range map.

## Results

Outputs live in [`Modeled/`](Modeled/) — model comparison plots, diagnostics, and the two
prediction rasters (`suitability_cloglog.tif`, `suitability_binary.tif`).

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

`Processed/` and `Modeled/` are committed, so `02_maxent_modeling.ipynb` runs end to end on a
fresh clone. To re-run `01_preprocessing.ipynb` you need the raw environmental rasters, which
are excluded from version control (~13 GB) and downloaded from the sources below.

For the ArcGIS route, add `Toolbox/SDNHMMaxent.pyt` in the ArcGIS Pro Catalog pane. The three
tools map onto the notebook stages and take the same inputs.

## Data sources

| Dataset | Source |
| --- | --- |
| BioClim | [WorldClim v2.1](https://www.worldclim.org/data/worldclim21.html) |
| ENVIREM | [envirem.github.io](https://envirem.github.io/) |
| Elevation | [USGS 3DEP](https://www.usgs.gov/3d-elevation-program) |
| Occurrences | San Diego Natural History Museum collection records |

## Adapting to another species

The workflow is not hard-coded to *Bipes biporus*. In `01_preprocessing.ipynb`, edit §1.2
(species name and input paths) and §1.4 (variable sets); §1.3 holds the CRS, target resolution,
and study-area buffer. `02_maxent_modeling.ipynb` auto-discovers whatever notebook 1 produced.
Cells marked `✎` are the ones intended to be edited.
