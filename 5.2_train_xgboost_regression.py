"""
Addestramento regressori XGBoost per stima continua della precipitazione.

Il programma:
- carica feature e target DPR dai dataset ML,
- seleziona i soli campioni diurni,
- isola i campioni in specifici intervalli di intensita (light/moderate/heavy),
- addestra un modello XGBoost di regressione per ciascun intervallo,
- salva i modelli e le figure di confronto tra valori reali e stimati.
"""

from __future__ import annotations

import importlib
import os
import pickle
from pathlib import Path
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np


# =========================
# Configuration
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"

RUN_XGB_REGRESSIONE = False

PATH_ML_4 = DATA_ROOT / "ml" / "Dati_ML_plus"


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def xgb_regressione_per_range(ml_dir: str, output_dir: str, dpr_range: Tuple[float, float], nome: str) -> None:
    """Addestra un regressore XGBoost su un intervallo DPR in condizioni diurne."""
    try:
        xgb_module = importlib.import_module("xgboost")
        XGBRegressor = xgb_module.XGBRegressor
    except Exception:
        print("xgboost non disponibile: salto regressione XGB.")
        return

    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, "output_modelli")
    os.makedirs(model_dir, exist_ok=True)

    df_features = load_pickle(os.path.join(ml_dir, "df_features.pickle"))
    dpr = np.asarray(load_pickle(os.path.join(ml_dir, "DPR_ml.pickle")), dtype=np.float32)
    gmask = np.asarray(load_pickle(os.path.join(ml_dir, "maschera_giorno_ml.pickle")), dtype=np.float32)

    idx_day = np.where(gmask == 1)[0]
    X = df_features.iloc[idx_day].reset_index(drop=True)
    y = dpr[idx_day]

    idx_range = np.where((y >= dpr_range[0]) & (y < dpr_range[1]))[0]
    X = X.iloc[idx_range].reset_index(drop=True)
    y = y[idx_range]

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]

    model = XGBRegressor(
        random_state=42,
        n_jobs=7,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.5,
        colsample_bytree=0.5,
        reg_alpha=1.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        eval_metric="mae",
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, s=2, alpha=0.4)
    plt.plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], "r--", linewidth=1)
    plt.xlabel("DPR reale (mm/h)")
    plt.ylabel("DPR stimato (mm/h)")
    plt.title(f"XGB regressione - {nome}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"scatter_xgb_giorno_{nome}.png"), dpi=300)
    plt.close()

    joblib.dump(model, os.path.join(model_dir, f"xgb_giorno_{nome}.pkl"))
    print(f"Regressione XGB completata ({nome}).")


def main() -> None:
    if RUN_XGB_REGRESSIONE:
        xgb_regressione_per_range(PATH_ML_4, PATH_ML_4, (0.1, 1.0), "light")
        xgb_regressione_per_range(PATH_ML_4, PATH_ML_4, (1.0, 5.0), "moderate")
        xgb_regressione_per_range(PATH_ML_4, PATH_ML_4, (5.0, 30.0), "heavy")


if __name__ == "__main__":
    main()
