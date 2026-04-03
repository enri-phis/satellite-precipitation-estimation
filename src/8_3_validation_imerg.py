"""
Final validation of precipitation classes against the IMERG reference.

This module provides the final comparison between model-predicted classes and the IMERG product:
- hourly aggregation of SEVIRI predictions;
- conversion of classes to equivalent precipitation rates;
- comparison on the IMERG grid;
- estimation of global metrics such as accuracy, precision, recall, F1, and HSS.
"""

from __future__ import annotations

import glob
import os
import pickle
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

# =========================
# Configuration
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
MODELS_ROOT = PROJECT_ROOT / "models"

PROCESSED_PICKLE_DIR = DATA_ROOT / "processed_pickle_output"
MODEL_OUTPUT_DIR = MODELS_ROOT / "validation"
IMERG_INPUT_DIR = DATA_ROOT / "imerg" / "input"

# Run switches
RUN_VALIDATION = False
RUN_VALIDATE_AGAINST_IMERG = False

PATH_VALIDATION_PICKLES = PROCESSED_PICKLE_DIR
PATH_VALIDATION_MODEL_OUT = MODEL_OUTPUT_DIR
PATH_IMERG_INPUT = IMERG_INPUT_DIR

RAIN_INTERVALS_5C = [(0, 0.1), (0.1, 1), (1, 5), (5, 15), (15, np.inf)]


def load_pickle(path):
    with open(path, "rb") as file_handle:
        return pickle.load(file_handle)


def classify_precip(precip, intervals=RAIN_INTERVALS_5C):
    """Classify a continuous precipitation map into the five classes used in the project."""
    classified = np.full_like(precip, fill_value=-1, dtype=int)
    for idx, (low, high) in enumerate(intervals):
        classified[(precip >= low) & (precip < high)] = idx
    return classified


def pixel_edges_from_centers(centers):
    centers = np.asarray(centers)
    diff = np.diff(centers)
    first = centers[0] - diff[0] / 2
    last = centers[-1] + diff[-1] / 2
    return np.concatenate([[first], centers[:-1] + diff / 2, [last]])


def heidke_skill_score(cm):
    hits = np.trace(cm)
    total = cm.sum()
    row = cm.sum(axis=1)
    col = cm.sum(axis=0)
    expected = (row * col).sum() / (total if total > 0 else 1.0)
    return (hits - expected) / (total - expected) if (total - expected) != 0 else np.nan


def seviri_frames_for_hour(img_indices, y_pred, img_h, img_w):
    frames = []
    for idx in sorted(img_indices):
        start = idx * (img_h * img_w)
        end = start + (img_h * img_w)
        frames.append(y_pred[start:end].reshape(img_h, img_w))
    return np.stack(frames, axis=0)


def classes_to_hourly_accum_seviri(frames_cls):
    class_to_rate = []
    for low, high in RAIN_INTERVALS_5C:
        class_to_rate.append(low * 1.5 if np.isinf(high) else (low + high) / 2.0)

    accum_1h = np.zeros(frames_cls.shape[1:], dtype=float)
    for frame in frames_cls:
        rates = np.array([class_to_rate[value] if value >= 0 else 0 for value in frame.ravel()]).reshape(frame.shape)
        accum_1h += rates * 0.25
    return accum_1h


def map_seviri_to_imerg(pred_map, lat_f2d, lon_f2d, imerg_classes_2d, lat_i, lon_i, n_classes=5):
    lat_edges = pixel_edges_from_centers(lat_i)
    lon_edges = pixel_edges_from_centers(lon_i)

    lat_v = lat_f2d.ravel()
    lon_v = lon_f2d.ravel()
    pred_v = pred_map.ravel()

    ii = np.digitize(lat_v, lat_edges) - 1
    jj = np.digitize(lon_v, lon_edges) - 1
    valid = (ii >= 0) & (ii < imerg_classes_2d.shape[0]) & (jj >= 0) & (jj < imerg_classes_2d.shape[1])

    ii = ii[valid]
    jj = jj[valid]
    pred_v = pred_v[valid]
    obs_v = imerg_classes_2d[ii, jj]

    cm = np.zeros((n_classes, n_classes), dtype=float)
    for predicted, observed in zip(pred_v, obs_v):
        if predicted >= 0 and observed >= 0:
            cm[int(predicted), int(observed)] += 1.0
    return cm, pred_v, obs_v


def run_validation(model_out_dir, processed_dir, imerg_input_dir):
    """Run the final comparison between SEVIRI predictions and hourly-aggregated IMERG classes."""
    y_pred = np.load(os.path.join(model_out_dir, "y_pred.npy"))
    lat_all = load_pickle(os.path.join(processed_dir, "lat.pickle"))
    lon_all = load_pickle(os.path.join(processed_dir, "lon.pickle"))
    time_all = load_pickle(os.path.join(processed_dir, "TIME.pickle"))
    day_mask = load_pickle(os.path.join(processed_dir, "giorno_maschera_70.pickle"))

    day_indices = np.where(day_mask == 1)[0]
    time_all = np.array(time_all)[day_indices]

    img_h, img_w = 55, 112
    pixels_per_image = img_h * img_w
    num_images = len(y_pred) // pixels_per_image

    hour_groups = defaultdict(list)
    for idx in range(num_images):
        timestamp = pd.to_datetime(time_all[idx * pixels_per_image])
        hour_groups[timestamp.replace(minute=0, second=0, microsecond=0)].append(idx)

    lat_2d = lat_all[:pixels_per_image].reshape(img_h, img_w)
    lon_2d = lon_all[:pixels_per_image].reshape(img_h, img_w)

    hourly_precip = defaultdict(list)
    for file_path in glob.glob(os.path.join(imerg_input_dir, "*.HDF5")):
        filename = os.path.basename(file_path)
        try:
            with h5py.File(file_path, "r") as file_handle:
                lat_i = file_handle["lat"][:]
                lon_i = file_handle["lon"][:]
                precip_all = file_handle["precipitation"][:]
        except Exception:
            continue

        date_str = filename.split("3IMERG.")[1][:8]
        start_part = [part for part in filename.split("-") if part.startswith("S")][0]
        key = (date_str, int(start_part[1:3]))
        for idx in range(precip_all.shape[0]):
            hourly_precip[key].append(precip_all[idx, :, :])

    cm_global = np.zeros((5, 5))
    y_true_all = []
    y_pred_all = []

    for timestamp, img_indices in sorted(hour_groups.items()):
        key = (timestamp.strftime("%Y%m%d"), timestamp.hour)
        arrays = hourly_precip.get(key, [])
        if not arrays:
            continue

        precip_stack = np.stack([np.asarray(arr, dtype=float) for arr in arrays], axis=0)
        classes_imerg = classify_precip((0.5 * precip_stack.sum(axis=0)).T)

        frames = seviri_frames_for_hour(img_indices, y_pred, img_h, img_w)
        cls_hourly_seviri = np.fliplr(classify_precip(classes_to_hourly_accum_seviri(frames)))

        cm, pred_vals, obs_vals = map_seviri_to_imerg(cls_hourly_seviri, lat_2d, lon_2d, classes_imerg, lat_i, lon_i, 5)
        cm_global += cm
        y_true_all.extend(obs_vals.tolist())
        y_pred_all.extend(pred_vals.tolist())

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    acc = accuracy_score(y_true_all, y_pred_all)
    prec = precision_score(y_true_all, y_pred_all, average="weighted", zero_division=0)
    rec = recall_score(y_true_all, y_pred_all, average="weighted", zero_division=0)
    f1 = f1_score(y_true_all, y_pred_all, average="weighted", zero_division=0)
    hss = heidke_skill_score(cm_global)

    print("Global validation results:")
    print(f"Accuracy={acc:.3f}, Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}, HSS={hss:.3f}")
    print(classification_report(y_true_all, y_pred_all, digits=2))


def run_validation_against_imerg(model_out_dir, processed_dir, imerg_input_dir):
    run_validation(model_out_dir, processed_dir, imerg_input_dir)


def main() -> None:
    if RUN_VALIDATION:
        run_validation(MODEL_OUTPUT_DIR, PROCESSED_PICKLE_DIR, IMERG_INPUT_DIR)

    if RUN_VALIDATE_AGAINST_IMERG:
        run_validation_against_imerg(PATH_VALIDATION_MODEL_OUT, PATH_VALIDATION_PICKLES, PATH_IMERG_INPUT)


if __name__ == "__main__":
    main()