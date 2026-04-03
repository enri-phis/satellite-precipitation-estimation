"""
Dataset balancing script for multiple class configurations.

This module applies stratified undersampling
to create balanced datasets with different precipitation discretization schemes.

Handled configurations:
1. **2 classes**: Simple outlier removal (rain >150mm/h, CH9 >300K)
2. **4 classes**: Intervals [0-0.1, 0.1-1, 1-5, 5-30] mm/h - for generic applications
3. **5 classes**: Intervals [0-0.1, 0.1-1, 1-5, 5-15, 15+] mm/h - light classification
4. **7 classes**: Fine partition for detailed analyses
5. **15 classes**: Maximum granularity for specific studies
6. **5 classes (features)**: same as 5 classes but on feature-engineered dataset

Balancing procedure:
- For each precipitation class, count the number of samples
- Find the class with the minimum number of samples
- Undersample all other classes to the minimum count
- Preserve geographic and temporal masks for later stratification

Output: Balanced datasets in pickle format separated by class,
ready for ML model training.

Configuration:
- DATA_PROCESSED_DIR: filtered data from step 1
- DATA_ML_5CLASS_DIR: feature engineering data from step 6
- Output folders for each class configuration
"""

import os
import pickle
from pathlib import Path

import numpy as np

# =========================
# Configuration
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = DATA_ROOT / "balanced"

DATA_PROCESSED_DIR = DATA_ROOT / "processed"
DATA_ML_5CLASS_DIR = DATA_ROOT / "features"

BALANCING_2CLASS_DIR = OUTPUT_ROOT / "bilanciamento_2classi"
BALANCING_4CLASS_DIR = OUTPUT_ROOT / "bilanciamento_4classi"
BALANCING_5CLASS_DIR = OUTPUT_ROOT / "bilanciamento_5classi"
BALANCING_7CLASS_DIR = OUTPUT_ROOT / "bilanciamento_7classi"
BALANCING_15CLASS_DIR = OUTPUT_ROOT / "bilanciamento_15classi"
BALANCING_5CLASS_IMAGES_DIR = OUTPUT_ROOT / "bilanciamento_5classi_immagini"

# Run switches: enable only the required balancing runs.
RUN_BALANCING_2CLASS = False
RUN_BALANCING_4CLASS = False
RUN_BALANCING_5CLASS = False
RUN_BALANCING_7CLASS = False
RUN_BALANCING_15CLASS = False
RUN_BALANCING_5CLASS_IMAGES = False
# =========================

CHANNEL_FILES = [f"CH_{i}.pickle" for i in range(1, 12)]
BASE_FILE_SET = CHANNEL_FILES + ["DPR.pickle", "maschera_giorno.pickle", "maschera_notte.pickle"]
EXTENDED_FILE_SET = BASE_FILE_SET + ["TERRA_maschera.pickle", "MARE_maschera.pickle", "STAGIONI_maschera.pickle"]
STANDARD_OUTPUT_MAPPING = {
    "CH_1": "CH_1.pickle",
    "CH_2": "CH_2.pickle",
    "CH_3": "CH_3.pickle",
    "CH_4": "CH_4.pickle",
    "CH_5": "CH_5.pickle",
    "CH_6": "CH_6.pickle",
    "CH_7": "CH_7.pickle",
    "CH_8": "CH_8.pickle",
    "CH_9": "CH_9.pickle",
    "CH_10": "CH_10.pickle",
    "CH_11": "CH_11.pickle",
    "DPR": "DPR.pickle",
    "maschera_giorno": "maschera_giorno.pickle",
    "maschera_notte": "maschera_notte.pickle",
    "TERRA_maschera": "TERRA_maschera.pickle",
    "MARE_maschera": "MARE_maschera.pickle",
    "STAGIONI_maschera": "STAGIONI_maschera.pickle",
}


def load_pickle(path):
    """Loads a Python object serialized in pickle format."""
    with open(path, "rb") as file:
        return pickle.load(file)


def save_pickle(path, data):
    """Saves a Python object in pickle format."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(data, file)


def load_dataset(input_dir, file_names):
    """Loads a subset of pickle files from a folder and reports any missing files."""
    data = {}
    missing = []

    for file_name in file_names:
        file_path = os.path.join(input_dir, file_name)
        if not os.path.exists(file_path):
            missing.append(file_name)
            continue

        key = file_name.replace(".pickle", "")
        data[key] = load_pickle(file_path)

    return data, missing


def apply_delete_indices(data_dict, indices):
    """Removes the same indices from all arrays in the dataset."""
    for key in data_dict:
        data_dict[key] = np.delete(data_dict[key], indices)


def validate_balancing_indices(data_dict, indices):
    """Limits balanced indices to the minimum available length across loaded fields."""
    max_size = min(len(v) for v in data_dict.values())
    return indices[indices < max_size]


def apply_balanced_indices(data_dict, balanced_indices):
    """Applies a common set of balanced indices to all dataset variables."""
    for key in data_dict:
        data_dict[key] = data_dict[key][balanced_indices]


def balance_by_intervals(dpr_array, rain_intervals):
    """Performs stratified balancing by precipitation intervals.

    Procedure:
    1. For each interval, count the number of samples
    2. Identify the minimum count (least represented class)
    3. Uniformly undersample all classes to the minimum
    4. Combine indices into a sorted array

    Guarantees a perfectly balanced dataset (same numerosity per class).
    """
    counts = []
    for lower_bound, upper_bound in rain_intervals:
        count_values = np.sum((dpr_array >= lower_bound) & (dpr_array < upper_bound))
        counts.append(count_values)
        print(f"Samples in interval ({lower_bound}, {upper_bound}): {count_values}")

    min_samples = min(counts)
    print(f"Minimum samples per class: {min_samples}")

    balanced_indices_list = []
    for lower_bound, upper_bound in rain_intervals:
        # Identify pixels in this precipitation interval
        indices_in_range = np.where((dpr_array >= lower_bound) & (dpr_array < upper_bound))[0]
        if len(indices_in_range) >= min_samples:
            # Uniformly undersample to the minimum
            selected_indices = np.random.choice(indices_in_range, size=min_samples, replace=False)
            balanced_indices_list.append(selected_indices)

    if not balanced_indices_list:
        return np.array([], dtype=int)

    # Sort indices to keep data consistency
    return np.sort(np.concatenate(balanced_indices_list))


def save_dataset_with_mapping(output_dir, data_dict, mapping):
    """Saves a dataset using an explicit mapping between internal keys and output filenames."""
    os.makedirs(output_dir, exist_ok=True)
    for src_key, output_filename in mapping.items():
        save_pickle(os.path.join(output_dir, output_filename), data_dict[src_key])


# --- 2-class balancing ---

def run_balancing_2class():
    """Filters outliers for the 2-class configuration without explicit undersampling."""
    print("Starting 2-class balancing")
    data, missing = load_dataset(DATA_PROCESSED_DIR, BASE_FILE_SET)

    if missing:
        print(f"Missing files: {missing}")
        return

    dpr = data["DPR"]
    ch9 = data["CH_9"]

    indici_no_dpr = np.where(dpr > 150)[0]
    indici_ch9 = np.where(ch9 > 300)[0]
    indici_eliminati = np.unique(np.concatenate((indici_no_dpr, indici_ch9)))
    print(f"Removed indices: {len(indici_eliminati)}")

    apply_delete_indices(data, indici_eliminati)

    # Keeps original behavior: this block does not perform explicit undersampling.
    output_mapping = {
        "DPR": "DPR_bilanciato.pickle",
        "CH_1": "CH_1_bilanciato.pickle",
        "CH_2": "CH_2_bilanciato.pickle",
        "CH_3": "CH_3_bilanciato.pickle",
        "CH_4": "CH_4_bilanciato.pickle",
        "CH_5": "CH_5_bilanciato.pickle",
        "CH_6": "CH_6_bilanciato.pickle",
        "CH_7": "CH_7_bilanciato.pickle",
        "CH_8": "CH_8_bilanciato.pickle",
        "CH_9": "CH_9_bilanciato.pickle",
        "CH_10": "CH_10_bilanciato.pickle",
        "CH_11": "CH_11_bilanciato.pickle",
        "maschera_giorno": "GIORNO_maschera_bilanciato.pickle",
        "maschera_notte": "NOTTE_maschera_bilanciato.pickle",
    }
    save_dataset_with_mapping(BALANCING_2CLASS_DIR, data, output_mapping)
    print(f"2-class balancing saved in: {BALANCING_2CLASS_DIR}")


# --- 4-class balancing ---

def run_balancing_4class():
    """Balances the dataset according to four precipitation intervals."""
    print("Starting 4-class balancing")
    data, missing = load_dataset(DATA_PROCESSED_DIR, EXTENDED_FILE_SET)

    if missing:
        print(f"Missing files: {missing}")
        return

    indici_eliminati = np.where(data["DPR"] > 30)[0]
    apply_delete_indices(data, indici_eliminati)

    rain_intervals = [(0, 0.1), (0.1, 1), (1, 5), (5, 30)]
    balanced_indices = balance_by_intervals(data["DPR"], rain_intervals)
    balanced_indices = validate_balancing_indices(data, balanced_indices)

    apply_balanced_indices(data, balanced_indices)
    save_dataset_with_mapping(BALANCING_4CLASS_DIR, data, STANDARD_OUTPUT_MAPPING)
    print(f"4-class balancing saved in: {BALANCING_4CLASS_DIR}")


# --- 7-class balancing ---

def run_balancing_7class():
    """Balances the dataset according to seven precipitation intervals."""
    print("Starting 7-class balancing")
    data, missing = load_dataset(DATA_PROCESSED_DIR, EXTENDED_FILE_SET)

    if missing:
        print(f"Missing files: {missing}")
        return

    max_value = np.max(data["DPR"])
    rain_intervals = [
        (0, 0.1),
        (0.1, 1),
        (1, 4),
        (4, 7),
        (7, 13),
        (13, 19),
        (19, max_value),
    ]

    balanced_indices = balance_by_intervals(data["DPR"], rain_intervals)
    balanced_indices = validate_balancing_indices(data, balanced_indices)

    apply_balanced_indices(data, balanced_indices)
    save_dataset_with_mapping(BALANCING_7CLASS_DIR, data, STANDARD_OUTPUT_MAPPING)
    print(f"7-class balancing saved in: {BALANCING_7CLASS_DIR}")


# --- 5-class balancing (pixels) ---

def run_balancing_5class():
    """Balances the dataset according to five precipitation intervals."""
    print("Starting 5-class balancing")
    data, missing = load_dataset(DATA_PROCESSED_DIR, EXTENDED_FILE_SET)

    if missing:
        print(f"Missing files: {missing}")
        return

    max_value = np.max(data["DPR"])
    rain_intervals = [
        (0, 0.1),
        (0.1, 1),
        (1, 5),
        (5, 15),
        (15, max_value),
    ]

    balanced_indices = balance_by_intervals(data["DPR"], rain_intervals)
    balanced_indices = validate_balancing_indices(data, balanced_indices)

    apply_balanced_indices(data, balanced_indices)
    save_dataset_with_mapping(BALANCING_5CLASS_DIR, data, STANDARD_OUTPUT_MAPPING)
    print(f"5-class balancing saved in: {BALANCING_5CLASS_DIR}")


# --- 15-class balancing ---

def run_balancing_15class():
    """Balances the dataset according to the fine intervals used in the 15-class configuration."""
    print("Starting 15-class balancing")
    data, missing = load_dataset(DATA_PROCESSED_DIR, EXTENDED_FILE_SET)

    if missing:
        print(f"Missing files: {missing}")
        return

    max_value = np.max(data["DPR"])
    rain_intervals = [
        (0, 0.1),
        (0.1, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 9),
        (9, 11),
        (11, 14),
        (14, 20),
        (20, max_value),
    ]

    balanced_indices = balance_by_intervals(data["DPR"], rain_intervals)
    balanced_indices = validate_balancing_indices(data, balanced_indices)

    apply_balanced_indices(data, balanced_indices)
    save_dataset_with_mapping(BALANCING_15CLASS_DIR, data, STANDARD_OUTPUT_MAPPING)
    print(f"15-class balancing saved in: {BALANCING_15CLASS_DIR}")


# --- 5-class balancing for images/features ---

def run_balancing_5class_images():
    """5-class balancing for the feature dataset (images/features).

    Loads the dataset already processed with feature engineering and applies the same
    5-class balancing scheme, preserving DataFrame structure
    for compatibility with downstream ML training.
    """
    print("Starting 5-class balancing for images/features")
    os.makedirs(BALANCING_5CLASS_IMAGES_DIR, exist_ok=True)

    # Load dataset and masks from the feature engineering step
    df_features = load_pickle(os.path.join(DATA_ML_5CLASS_DIR, "df_features.pickle"))
    dpr = load_pickle(os.path.join(DATA_ML_5CLASS_DIR, "DPR_ml.pickle"))
    giorno_maschera = load_pickle(os.path.join(DATA_ML_5CLASS_DIR, "maschera_giorno_ml.pickle"))

    data_dict = {
        "df_features": df_features,
        "DPR_ml": dpr,
        "maschera_giorno_ml": giorno_maschera,
    }

    # Define the 5 precipitation intervals
    max_value = np.max(dpr)
    rain_intervals = [
        (0, 0.1),
        (0.1, 1),
        (1, 5),
        (5, 15),
        (15, max_value),
    ]

    # Collect indices for each class and compute the minimum
    class_indices = []
    min_samples = float("inf")

    for low, high in rain_intervals:
        idx = np.where((dpr >= low) & (dpr < high))[0]
        class_indices.append(idx)
        min_samples = min(min_samples, len(idx))

    # Undersample each class to the minimum
    balanced_indices = []
    for idx in class_indices:
        selected = np.random.choice(idx, size=min_samples, replace=False)
        balanced_indices.append(selected)

    balanced_indices = np.sort(np.concatenate(balanced_indices))

    # Apply balancing: preserve type (DataFrame vs array)
    balanced_data = {}
    for key, array in data_dict.items():
        if hasattr(array, "iloc"):
            balanced_data[key] = array.iloc[balanced_indices].reset_index(drop=True)
        else:
            balanced_data[key] = array[balanced_indices]

    # Save balanced datasets
    for key, array in balanced_data.items():
        save_pickle(os.path.join(BALANCING_5CLASS_IMAGES_DIR, f"{key}.pickle"), array)

    print(f"5-class image balancing saved in: {BALANCING_5CLASS_IMAGES_DIR}")


if __name__ == "__main__":
    if RUN_BALANCING_2CLASS:
        run_balancing_2class()

    if RUN_BALANCING_4CLASS:
        run_balancing_4class()

    if RUN_BALANCING_5CLASS:
        run_balancing_5class()

    if RUN_BALANCING_7CLASS:
        run_balancing_7class()

    if RUN_BALANCING_15CLASS:
        run_balancing_15class()

    if RUN_BALANCING_5CLASS_IMAGES:
        run_balancing_5class_images()
