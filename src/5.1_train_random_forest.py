"""
Random Forest training for multi-class precipitation classification.

The script:
- loads features and DPR target from ML datasets,
- filters samples under daytime conditions,
- trains a Random Forest classifier for the selected number of classes,
- computes metrics (Accuracy, Precision, Recall, F1, HSS),
- saves models, predicted/true labels, and diagnostic figures (confusion matrix, ROC, feature importance).
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, normalize


# =========================
# Configuration
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"

RUN_RF_4_GIORNO = False
RUN_RF_5_GIORNO = False
RUN_RF_7_GIORNO = False
RUN_RF_15_GIORNO = False

PATH_ML_4 = DATA_ROOT / "ml" / "Dati_ML_plus"
PATH_ML_5 = DATA_ROOT / "ml" / "Dati_ML_plus_5classi"
PATH_ML_7 = DATA_ROOT / "ml" / "Dati_ML_plus_7classi"
PATH_ML_15 = DATA_ROOT / "ml" / "Dati_ML_plus_15classi"


@dataclass
class ClassConfig:
    classes: List[str]
    ranges: List[float]


CLASS_CONFIGS: Dict[str, ClassConfig] = {
    "4": ClassConfig(classes=["Dry", "Light", "Moderate", "Heavy"], ranges=[0, 0.1, 1, 5, 30]),
    "5": ClassConfig(classes=["Dry", "Light", "Moderate", "Heavy", "Intense"], ranges=[0, 0.1, 1, 5, 15, 60]),
    "7": ClassConfig(classes=["Dry", "Very Light", "Light", "Moderate", "Intense", "Very Intense", "Heavy"], ranges=[0, 0.1, 1, 4, 7, 13, 19, 60]),
    "15": ClassConfig(
        classes=["Dry", "Very Light", "Light", "Light-Moderate", "Moderate", "Moderate-Heavy", "Heavy", "Very Heavy", "Extreme I", "Extreme II", "Severe", "Violent", "Catastrophic"],
        ranges=[0, 0.1, 1, 2, 3, 4, 5, 6, 7, 9, 11, 14, 20, 60],
    ),
}


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def map_to_class(value: float, ranges: Sequence[float], classes: Sequence[str]) -> str:
    for i in range(len(ranges) - 1):
        if ranges[i] <= value < ranges[i + 1]:
            return classes[i]
    return classes[-1]


def hss_empirical(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred)
    total = cm.sum()
    correct = np.trace(cm)
    observed = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    expected = np.sum((observed * predicted) / total)
    return (correct - expected) / (total - expected) if (total - expected) != 0 else 0.0


def train_rf_daytime(ml_dir: str, output_dir: str, class_key: str, rf_params: Optional[Dict] = None) -> None:
    """Train Random Forest using only daytime pixels and save metrics/artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, "output_modelli")
    os.makedirs(model_dir, exist_ok=True)

    df_features = load_pickle(os.path.join(ml_dir, "df_features.pickle"))
    dpr = np.asarray(load_pickle(os.path.join(ml_dir, "DPR_ml.pickle")), dtype=np.float32)
    gmask = np.asarray(load_pickle(os.path.join(ml_dir, "maschera_giorno_ml.pickle")), dtype=np.float32)

    idx = np.where(gmask == 1)[0]
    df_features = df_features.iloc[idx].reset_index(drop=True)
    dpr = dpr[idx]

    cfg = CLASS_CONFIGS[class_key]
    max_dpr = float(np.max(dpr)) if len(dpr) else cfg.ranges[-1]
    ranges = list(cfg.ranges)
    ranges[-1] = max_dpr

    mapped = [map_to_class(v, ranges, cfg.classes) for v in dpr]
    df_features["Precipitation_Class"] = mapped

    encoder = LabelEncoder()
    y = encoder.fit_transform(df_features["Precipitation_Class"])
    X = df_features.drop(columns=["Precipitation_Class"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

    params = {
        "random_state": 42,
        "n_jobs": 7,
        "n_estimators": 100,
        "max_depth": 50,
        "min_samples_leaf": 2,
        "max_features": 0.5,
        "max_samples": 0.5,
    }
    if rf_params:
        params.update(rf_params)

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)
    hss_e = hss_empirical(y_test, y_pred)
    hss_ck = cohen_kappa_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=cfg.classes, zero_division=0)

    conf_norm = normalize(cm, axis=1, norm="l1") * 100
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_norm, annot=True, fmt=".2f", cmap="Blues")
    plt.xticks(ticks=np.arange(len(cfg.classes)) + 0.5, labels=cfg.classes, ha="center")
    plt.yticks(ticks=np.arange(len(cfg.classes)) + 0.5, labels=cfg.classes, va="center")
    plt.xlabel("RF predicted classes (mm/h)")
    plt.ylabel("Observed DPR class (mm/h)")
    plt.title(f"Confusion Matrix - RF - daytime ({class_key} classes)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confmatrix_rfclass_giorno_{class_key}classi.png"), dpi=300)
    plt.close()

    y_prob = model.predict_proba(X_test)
    y_test_enc = LabelEncoder().fit_transform(y_test)
    plt.figure(figsize=(8, 6))
    for i, cname in enumerate(cfg.classes):
        fpr, tpr, _ = roc_curve(y_test_enc, y_prob[:, i], pos_label=i)
        plt.plot(fpr, tpr, label=f"{cname}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC - RF - daytime ({class_key} classes)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"roc_rfclass_giorno_{class_key}classi.png"), dpi=300)
    plt.close()

    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:40]
    feat_names = np.array(X.columns)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(order)), importances[order], align="center")
    plt.xticks(range(len(order)), feat_names[order], rotation=60, ha="right", fontsize=7)
    plt.ylabel("Feature importance")
    plt.title(f"RF feature ranking ({class_key} classes)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"importanza_rfclass_giorno_{class_key}classi.png"), dpi=300)
    plt.close()

    metrics = (
        f"Accuracy: {acc:.4f}\n"
        f"Precision: {pre:.4f}\n"
        f"Recall: {rec:.4f}\n"
        f"F1-Score: {f1:.4f}\n"
        f"Empirical HSS: {hss_e:.4f}\n"
        f"HSS CK: {hss_ck:.4f}\n\n"
        f"Confusion Matrix:\n{cm}\n\n"
        f"Classification Report:\n{report}\n"
    )
    with open(os.path.join(output_dir, f"rf_class_giorno_{class_key}classi.txt"), "w", encoding="utf-8") as f:
        f.write(metrics)

    y_true_labels = encoder.inverse_transform(y_test)
    y_pred_labels = encoder.inverse_transform(y_pred)
    save_pickle(os.path.join(output_dir, f"y_true_rf_{class_key}classi_labels.pickle"), y_true_labels)
    save_pickle(os.path.join(output_dir, f"y_pred_rf_{class_key}classi_labels.pickle"), y_pred_labels)
    save_pickle(os.path.join(output_dir, f"y_pred_rf_{class_key}classi.pickle"), y_pred)
    save_pickle(os.path.join(output_dir, f"label_encoder_rf_{class_key}classi.pickle"), encoder)
    joblib.dump(model, os.path.join(model_dir, f"rf_giorno_classifier_{class_key}classi.pkl"))

    print(f"RF training completed ({class_key} classes).")


def main() -> None:
    if RUN_RF_4_GIORNO:
        train_rf_daytime(PATH_ML_4, PATH_ML_4, "4")
    if RUN_RF_5_GIORNO:
        train_rf_daytime(PATH_ML_5, PATH_ML_5, "5")
    if RUN_RF_7_GIORNO:
        train_rf_daytime(PATH_ML_7, PATH_ML_7, "7", rf_params={"min_samples_leaf": 3})
    if RUN_RF_15_GIORNO:
        train_rf_daytime(PATH_ML_15, PATH_ML_15, "15", rf_params={"min_samples_leaf": 3, "max_samples": 0.7})


if __name__ == "__main__":
    main()