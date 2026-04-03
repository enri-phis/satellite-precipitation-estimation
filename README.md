# ML Pipeline for Precipitation Intensity Classification from SEVIRI with IMERG Validation

## 1. Project Title
ML Pipeline for Precipitation Intensity Classification from SEVIRI (MSG) with Final Validation on IMERG.

## 2. Project Overview
This repository implements an end-to-end machine learning pipeline for classifying precipitation intensity from SEVIRI (MSG) satellite observations, with external validation using the IMERG product.  
The project integrates geospatial preprocessing, feature engineering, supervised modeling, and quantitative validation, with a modular and re-executable structure for large-scale experiments.

## 3. Problem Statement
Accurately estimating precipitation intensity from satellite data is a relevant problem for weather monitoring, hydrological risk management, and climate analysis.  
Satellite radiometric data are informative but indirect, noisy, and highly imbalanced (many dry pixels, few intense events).  
This project addresses the problem by building a complete workflow that:

- transforms raw SEVIRI data into multi-class predictions;
- reduces bias due to class imbalance through stratified balancing;
- verifies prediction robustness against an independent reference (IMERG).

## 4. Methodology / Pipeline
The pipeline is organized into modular steps:

1. Satellite Data Preprocessing  
   Loading, conversion, and geographic filtering of SEVIRI observations.

2. Mask Generation  
   Construction of day/night, seasonal, and land/sea masks.

3. Distributional Analysis  
   Diagnostic histograms and exploratory analysis of distributions by channel and context.

4. Multi-class Dataset Balancing  
   Stratified undersampling for 4, 5, 7, and 15-class configurations.

5. Feature Engineering  
   Feature generation from raw channels, differences, local spatial statistics, gradients, entropy, and Laplacian.

6. Model Training  
   Random Forest training for multi-class classification and XGBoost regression for dedicated analyses.

7. Final Validation Against IMERG  
   Prediction-reference comparison with standard metrics (accuracy, precision, recall, F1, HSS) and diagnostic artifacts.

## 5. Tech Stack
Main libraries and tools:

- Python
- NumPy
- Pandas
- SciPy
- scikit-learn
- XGBoost
- h5py
- xarray
- scikit-image
- Matplotlib
- Seaborn
- Cartopy
- PyProj
- Satpy
- ephem

## 6. Repository Structure
Main pipeline scripts (organized in `src/` folder):

- src/1_data_loading_and_geographic_filtering.py  
  Load raw data and apply geographic filtering

- src/2_masks_generation.py  
  Generate day/night, seasonal, and land/sea masks

- src/3_histograms_analysis.py  
  Distributional analysis and diagnostic histograms

- src/4_dataset_balancing.py  
  Dataset balancing for multi-class configurations

- src/5.1_train_random_forest.py  
  Random Forest training for multi-class classification

- src/5.2_train_xgboost_regression.py  
  XGBoost regression for continuous estimation across intensity ranges

- src/5.3_group_class_mappings.py  
  Class aggregation 15 -> 7 -> 5 and evaluation reports

- src/6_ml_feature_engineering.py  
  Advanced feature construction and feature set combinations

- src/7_imerg_download_and_maps.py  
  IMERG download/processing and support maps

- src/8_1_validation_preprocessing.py  
  Preprocessing dedicated to final validation

- src/8_2_validation_features_model.py  
  Features and inference for validation workflow

- src/8_3_validation_imerg.py  
  Final comparison against IMERG and aggregate metrics

## 7. Data Note

⚠️ **Input data is NOT included in this repository** for the following reasons:

- **Size**: Input files (raw SEVIRI, IMERG, global masks) exceed several GB, being high-resolution satellite imagery for an entire year of observations.
- **Seasonal Analysis**: The pipeline is designed to analyze seasonal variability, requiring complete year-round datasets.
- **Simplification Difficulty**: Satellite maps cannot easily be reduced to "mini" versions while maintaining scientific-geographic significance.

To run the pipeline:
1. Obtain raw SEVIRI data (.nat or NetCDF) from authorized providers (e.g., EUMETSAT)
2. Download IMERG from https://gpm.nasa.gov/data
3. Obtain global land/sea masks
4. Place all files in `data/` folder following the structure expected by scripts
5. Execute scripts sequentially from `src/1_*` to `src/8_*`

## 8. How to Run
Basic instructions:

1. Create and activate a Python environment (recommended: virtual environment or Conda).
2. Install main dependencies listed in the Tech Stack section.
3. Configure paths and RUN_... flags in the Configuration section of each script.
4. Execute scripts in sequence.

Recommended order:

1. 1_data_loading_and_geographic_filtering.py
2. 2_masks_generation.py
3. 3_histograms_analysis.py
4. 4_dataset_balancing.py
5. 5.1_train_random_forest.py
6. 5.2_train_xgboost_regression.py
7. 5.3_group_class_mappings.py
8. 6_ml_feature_engineering.py
9. 7_imerg_download_and_maps.py
10. 8_1_validation_preprocessing.py
11. 8_2_validation_features_model.py
12. 8_3_validation_imerg.py

## 9. Results
The pipeline produces:

- multi-class predictions of precipitation intensity;
- confusion matrices and classification reports;
- main metrics: accuracy, precision, recall, F1, HSS;
- graphs and geospatial maps for diagnostic analysis.

Note: specific numerical values are not reported in this README unless associated with reproducible runs documented in outputs.

## 10. Key Features
- Complete pipeline from raw satellite data to external validation.
- Multi-class configuration support (4, 5, 7, 15).
- Feature engineering oriented to spatial/geophysical signals.
- Validation against independent product (IMERG), not just internal training metrics.

