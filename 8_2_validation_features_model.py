"""
Costruzione feature e inferenza del modello per la validazione finale.

Questo modulo isola la parte numericamente più densa della pipeline di validazione:
- pulizia dei canali e sostituzione dei NaN;
- costruzione delle feature locali e geografiche;
- salvataggio in blocchi per contenere l'uso di memoria;
- training/applicazione del modello Random Forest sui blocchi di validazione.
"""

from __future__ import annotations

import glob
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import convolve2d
from sklearn.ensemble import RandomForestClassifier

# =========================
# Configuration
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
MODELS_ROOT = PROJECT_ROOT / "models"

PROCESSED_PICKLE_DIR = DATA_ROOT / "processed_pickle_output"
FEATURES_OUTPUT_DIR = DATA_ROOT / "features_output"
MODEL_OUTPUT_DIR = MODELS_ROOT / "validation"
BALANCED_FEATURES_FILE = DATA_ROOT / "balanced" / "balanced_features.pkl"
BALANCED_DPR_FILE = DATA_ROOT / "balanced" / "DPR_ml.pickle"
BALANCED_DAY_MASK_FILE = DATA_ROOT / "balanced" / "maschera_giorno_ml.pickle"

# Run switches
RUN_FEATURE_ENGINEERING = False
RUN_MODEL = False
RUN_BUILD_VALIDATION_FEATURES = False
RUN_TRAIN_VALIDATION_RF = False
RUN_PREDICT_VALIDATION_BLOCKS = False

PATH_VALIDATION_PICKLES = PROCESSED_PICKLE_DIR
PATH_VALIDATION_FEATURE_BLOCKS = FEATURES_OUTPUT_DIR
PATH_BALANCED_FEATURES_5 = BALANCED_FEATURES_FILE
PATH_BALANCED_DPR_5 = BALANCED_DPR_FILE
PATH_BALANCED_DAY_MASK_5 = BALANCED_DAY_MASK_FILE
PATH_VALIDATION_MODEL_OUT = MODEL_OUTPUT_DIR


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_pickle(path):
    with open(path, "rb") as file_handle:
        return pickle.load(file_handle)


def replace_nans_with_nearest(arr):
    """Sostituisce i NaN con il valore valido più vicino lungo il vettore."""
    arr = arr.copy()
    n = len(arr)
    isnan = np.isnan(arr)
    notnan = ~isnan
    idx = np.arange(n)

    left = np.where(notnan, idx, 0)
    np.maximum.accumulate(left, out=left)

    right = np.where(notnan, idx, n - 1)
    np.minimum.accumulate(right[::-1], out=right[::-1])

    nearest = np.where(isnan, np.where((idx - left) <= (right - idx), left, right), idx)
    arr[isnan] = arr[nearest[isnan]]
    return arr


def replace_nans_with_nearest_blockwise(arr, block_size=1_232_000):
    """Applica la sostituzione dei NaN per blocchi per ridurre il carico in memoria."""
    out = np.empty_like(arr)
    for start in range(0, len(arr), block_size):
        end = min(start + block_size, len(arr))
        out[start:end] = replace_nans_with_nearest(arr[start:end])
    return out


def moving_mean_std_images(vector_1d, width, height, kernel_size=5):
    """Calcola media e deviazione standard locale per ogni immagine dello stack."""
    num_pix = width * height
    num_img = len(vector_1d) // num_pix
    kernel = np.ones((kernel_size, kernel_size), dtype=float) / (kernel_size * kernel_size)
    means = []
    stds = []

    for idx in range(num_img):
        image = vector_1d[idx * num_pix:(idx + 1) * num_pix].reshape(height, width)
        mean = convolve2d(image, kernel, mode="same", boundary="symm")
        mean_sq = convolve2d(image ** 2, kernel, mode="same", boundary="symm")
        std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))
        means.append(mean.flatten())
        stds.append(std.flatten())

    return np.concatenate(means), np.concatenate(stds)


def apply_mask(channel, mask, value=1):
    return np.where(mask == value, channel, 0)


def build_features(processed_dir, output_dir):
    """Costruisce i blocchi di feature usati nella fase di validazione."""
    ensure_dir(output_dir)

    names = [f"CH_{i}.pickle" for i in range(1, 12)] + [
        "Mare_mask_regridded.pickle",
        "Terra_mask_regridded.pickle",
        "stagionalita_maschera_estate_inverno.pickle",
        "giorno_maschera_70.pickle",
    ]

    variables = {}
    for file_name in names:
        path = os.path.join(processed_dir, file_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File mancante: {path}")
        variables[os.path.splitext(file_name)[0]] = load_pickle(path).ravel()

    mare = variables["Mare_mask_regridded"]
    terra = variables["Terra_mask_regridded"]
    stagioni = variables["stagionalita_maschera_estate_inverno"]
    giorno = variables["giorno_maschera_70"]

    channels = [f"CH_{i}" for i in range(1, 12)]
    data = {}
    mean_channels = []
    std_channels = []

    for channel in channels:
        cleaned = replace_nans_with_nearest_blockwise(variables[channel])
        mean, std = moving_mean_std_images(cleaned, width=112, height=55, kernel_size=5)
        mean_channels.append(mean)
        std_channels.append(std)

    for idx, channel in enumerate(channels):
        raw = replace_nans_with_nearest_blockwise(variables[channel].copy())
        data[channel] = raw
        data[f"{channel}_mare"] = replace_nans_with_nearest_blockwise(apply_mask(variables[channel], mare, 1))
        data[f"{channel}_terra"] = replace_nans_with_nearest_blockwise(apply_mask(variables[channel], terra, 1))
        data[f"{channel}_estate"] = replace_nans_with_nearest_blockwise(apply_mask(variables[channel], stagioni, 0))
        data[f"{channel}_inverno"] = replace_nans_with_nearest_blockwise(apply_mask(variables[channel], stagioni, 1))
        data[f"mean_5x5_{channel}"] = replace_nans_with_nearest_blockwise(mean_channels[idx].copy())
        data[f"std_5x5_{channel}"] = replace_nans_with_nearest_blockwise(std_channels[idx].copy())

    df_features = pd.DataFrame(data)
    mask_indices = np.where(giorno == 1)[0]
    df_features = df_features.iloc[mask_indices]

    block_size = 616_000
    num_blocks = len(df_features) // block_size + 1
    for idx in range(num_blocks):
        start = idx * block_size
        end = min((idx + 1) * block_size, len(df_features))
        with open(os.path.join(output_dir, f"df_features_blocco_{idx + 1}.pickle"), "wb") as file_handle:
            pickle.dump(df_features.iloc[start:end], file_handle)

    with open(os.path.join(output_dir, "maschera_terra_ml.pickle"), "wb") as file_handle:
        pickle.dump(terra, file_handle)
    with open(os.path.join(output_dir, "maschera_mare_ml.pickle"), "wb") as file_handle:
        pickle.dump(mare, file_handle)
    with open(os.path.join(output_dir, "maschera_stagioni_ml.pickle"), "wb") as file_handle:
        pickle.dump(stagioni, file_handle)


def map_to_class(value):
    labels = ["Dry", "Light", "Moderate", "Heavy", "Intense"]
    class_ranges = [0, 0.1, 1, 5, 15, np.inf]
    for idx in range(len(class_ranges) - 1):
        if class_ranges[idx] <= value < class_ranges[idx + 1]:
            return labels[idx]
    return labels[-1]


def run_model(features_dir, balanced_features_file, balanced_dpr_file, balanced_day_mask_file, model_out_dir):
    """Addestra il modello RF sui dati bilanciati e genera le predizioni sui blocchi di validazione."""
    ensure_dir(model_out_dir)

    dpr_bal = load_pickle(balanced_dpr_file)
    df_bal = load_pickle(balanced_features_file)
    day_mask_bal = load_pickle(balanced_day_mask_file)

    day_indices = np.where(day_mask_bal == 1)[0]
    df_bal = df_bal.iloc[day_indices]
    dpr_bal = np.asarray(dpr_bal)[day_indices]

    x_min = df_bal.min()
    x_max = df_bal.max()
    x_train = (df_bal - x_min) / (x_max - x_min) * 2 - 1

    classes = ["Dry", "Light", "Moderate", "Heavy", "Intense"]
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    y_train = np.array([class_to_idx[map_to_class(value)] for value in dpr_bal])

    model = RandomForestClassifier(
        random_state=37,
        n_jobs=7,
        n_estimators=150,
        max_depth=50,
        min_samples_leaf=2,
        max_features=0.5,
        max_samples=0.5,
    )
    model.fit(x_train, y_train)

    predictions = []
    for block_file in sorted(glob.glob(os.path.join(features_dir, "df_features_blocco_*.pickle"))):
        df_block = load_pickle(block_file)
        x_block = (df_block - x_min) / (x_max - x_min) * 2 - 1
        predictions.append(model.predict(x_block))

    np.save(os.path.join(model_out_dir, "y_pred.npy"), np.concatenate(predictions))
    with open(os.path.join(model_out_dir, "norm_x_min.pkl"), "wb") as file_handle:
        pickle.dump(x_min, file_handle)
    with open(os.path.join(model_out_dir, "norm_x_max.pkl"), "wb") as file_handle:
        pickle.dump(x_max, file_handle)


def build_validation_feature_blocks(processed_dir, output_dir):
    build_features(processed_dir, output_dir)


def train_validation_rf_5class(balanced_features_file, balanced_dpr_file, balanced_day_mask_file, model_out_dir):
    run_model(PATH_VALIDATION_FEATURE_BLOCKS, balanced_features_file, balanced_dpr_file, balanced_day_mask_file, model_out_dir)


def predict_validation_blocks(features_dir, model_out_dir):
    run_model(features_dir, PATH_BALANCED_FEATURES_5, PATH_BALANCED_DPR_5, PATH_BALANCED_DAY_MASK_5, model_out_dir)


def main() -> None:
    if RUN_FEATURE_ENGINEERING:
        build_features(PROCESSED_PICKLE_DIR, FEATURES_OUTPUT_DIR)

    if RUN_MODEL:
        run_model(FEATURES_OUTPUT_DIR, BALANCED_FEATURES_FILE, BALANCED_DPR_FILE, BALANCED_DAY_MASK_FILE, MODEL_OUTPUT_DIR)

    if RUN_BUILD_VALIDATION_FEATURES:
        build_validation_feature_blocks(PATH_VALIDATION_PICKLES, PATH_VALIDATION_FEATURE_BLOCKS)

    if RUN_TRAIN_VALIDATION_RF:
        train_validation_rf_5class(PATH_BALANCED_FEATURES_5, PATH_BALANCED_DPR_5, PATH_BALANCED_DAY_MASK_5, PATH_VALIDATION_MODEL_OUT)

    if RUN_PREDICT_VALIDATION_BLOCKS:
        predict_validation_blocks(PATH_VALIDATION_FEATURE_BLOCKS, PATH_VALIDATION_MODEL_OUT)


if __name__ == "__main__":
    main()