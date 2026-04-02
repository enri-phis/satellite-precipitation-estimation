"""
Aggregazione delle classi di precipitazione da 15 a 7 e successivamente a 5 classi.

Il programma:
- carica le etichette reali e predette del modello a 15 classi,
- applica mapping deterministici 15 -> 7 -> 5,
- ricalcola metriche di classificazione sui nuovi gruppi,
- salva etichette aggregate, report testuali e confusion matrix normalizzate.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder, normalize


# =========================
# Configuration
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"

RUN_GROUPING_15_TO_7_TO_5 = False
PATH_ML_15 = DATA_ROOT / "ml" / "Dati_ML_plus_15classi"

GROUP_15_TO_7_MAP = {
    "Dry": "Dry",
    "Very Light": "Light",
    "Light": "Moderate",
    "Light-Moderate": "Moderate",
    "Moderate": "Moderate",
    "Moderate-Heavy": "Extreme",
    "Heavy": "Extreme",
    "Very Heavy": "Extreme",
    "Extreme I": "Severe",
    "Extreme II": "Severe",
    "Severe": "Severe",
    "Violent": "Violent",
    "Catastrophic": "Catastrophic",
}

GROUP_7_TO_5_MAP = {
    "Dry": "Dry",
    "Light": "Light",
    "Moderate": "Moderate",
    "Extreme": "Severe",
    "Severe": "Severe",
    "Violent": "Catastrophic",
    "Catastrophic": "Catastrophic",
}

GROUP_7_CLASSES = ["Dry", "Light", "Moderate", "Extreme", "Severe", "Violent", "Catastrophic"]
GROUP_5_CLASSES = ["Dry", "Light", "Moderate", "Severe", "Catastrophic"]


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def hss_empirical(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred)
    total = cm.sum()
    correct = np.trace(cm)
    observed = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    expected = np.sum((observed * predicted) / total)
    return (correct - expected) / (total - expected) if (total - expected) != 0 else 0.0


def metriche_classificazione(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> Dict[str, object]:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "hss_empirical": hss_empirical(y_true, y_pred),
        "hss_ck": cohen_kappa_score(y_true, y_pred),
        "cm": cm,
        "report": classification_report(y_true, y_pred, target_names=class_names, zero_division=0),
    }


def salva_risultati_grouping(y_true_lbl: np.ndarray, y_pred_lbl: np.ndarray, class_names: List[str], output_dir: str, suffix: str) -> None:
    encoder = LabelEncoder()
    encoder.fit(class_names)
    y_true = encoder.transform(y_true_lbl)
    y_pred = encoder.transform(y_pred_lbl)
    metrics = metriche_classificazione(y_true, y_pred, class_names)

    conf_norm = normalize(metrics["cm"], axis=1, norm="l1") * 100
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_norm, annot=True, fmt=".2f", cmap="Greens")
    plt.xticks(ticks=np.arange(len(class_names)) + 0.5, labels=class_names, ha="center", rotation=30)
    plt.yticks(ticks=np.arange(len(class_names)) + 0.5, labels=class_names, va="center")
    plt.xlabel("Classi predette")
    plt.ylabel("Classi vere")
    plt.title(f"Confusion Matrix grouping ({suffix})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confmatrix_grouping_{suffix}.png"), dpi=300)
    plt.close()

    txt = (
        f"Grouping: {suffix}\n"
        f"Accuracy: {metrics['accuracy']:.4f}\n"
        f"Precision: {metrics['precision']:.4f}\n"
        f"Recall: {metrics['recall']:.4f}\n"
        f"F1-Score: {metrics['f1']:.4f}\n"
        f"HSS empirica: {metrics['hss_empirical']:.4f}\n"
        f"HSS CK: {metrics['hss_ck']:.4f}\n\n"
        f"Confusion Matrix:\n{metrics['cm']}\n\n"
        f"Classification Report:\n{metrics['report']}\n"
    )
    with open(os.path.join(output_dir, f"grouping_{suffix}.txt"), "w", encoding="utf-8") as f:
        f.write(txt)


def esegui_grouping_15_7_5(output_dir_15: str) -> None:
    y_true_15 = np.asarray(load_pickle(os.path.join(output_dir_15, "y_true_rf_15classi_labels.pickle")))
    y_pred_15 = np.asarray(load_pickle(os.path.join(output_dir_15, "y_pred_rf_15classi_labels.pickle")))

    y_true_7 = np.array([GROUP_15_TO_7_MAP[v] for v in y_true_15])
    y_pred_7 = np.array([GROUP_15_TO_7_MAP[v] for v in y_pred_15])
    save_pickle(os.path.join(output_dir_15, "y_true_grouped_7classi_labels.pickle"), y_true_7)
    save_pickle(os.path.join(output_dir_15, "y_pred_grouped_7classi_labels.pickle"), y_pred_7)
    salva_risultati_grouping(y_true_7, y_pred_7, GROUP_7_CLASSES, output_dir_15, "15_to_7")

    y_true_5 = np.array([GROUP_7_TO_5_MAP[v] for v in y_true_7])
    y_pred_5 = np.array([GROUP_7_TO_5_MAP[v] for v in y_pred_7])
    save_pickle(os.path.join(output_dir_15, "y_true_grouped_5classi_labels.pickle"), y_true_5)
    save_pickle(os.path.join(output_dir_15, "y_pred_grouped_5classi_labels.pickle"), y_pred_5)
    salva_risultati_grouping(y_true_5, y_pred_5, GROUP_5_CLASSES, output_dir_15, "7_to_5")

    print("Grouping 15 -> 7 -> 5 completato e salvato.")


def main() -> None:
    if RUN_GROUPING_15_TO_7_TO_5:
        esegui_grouping_15_7_5(PATH_ML_15)


if __name__ == "__main__":
    main()
