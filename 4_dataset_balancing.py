"""
Script di bilanciamento del dataset per molteplici configurazioni di classi.

Questo modulo applica il sotto-campionamento stratificato (stratified undersampling)
per creare dataset bilanciati con diversi schemi di discretizzazione della precipitazione.

Configurazioni gestite:
1. **2 classi**: Semplice rimozione outlier (pioggia >150mm/h, CH9 >300K)
2. **4 classi**: Intervalli [0-0.1, 0.1-1, 1-5, 5-30] mm/h - per applicazioni generiche
3. **5 classi**: Intervalli [0-0.1, 0.1-1, 1-5, 5-15, 15+] mm/h - classificazione leggera
4. **7 classi**: Partizione fine per analisi dettagliate
5. **15 classi**: Massima granularità per studi specifici
6. **5 classi (features)**: come 5 classi ma sul dataset di feature engineering

Procedura di bilanciamento:
- Per ogni classe di precipitazione, conta il numero di campioni
- Trova la classe con il numero minimo di campioni
- Sotto-campiona tutte le altre classi al numero minimo (undersampling)
- Preserva le maschere geografiche e temporali per stratificazione successiva

Output: Dataset bilanciati in formato pickle separati per classe,
pronti per training di modelli ML.

Configurazione:
- DATA_PROCESSED_DIR: dati filtrati dal passo 1
- DATA_ML_5CLASS_DIR: dati di feature engineering dal passo 6
- Cartelle di output per ogni configurazione di classi
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

# Run switches: attivare solo i bilanciamenti necessari.
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
    """Carica un oggetto Python serializzato in formato pickle."""
    with open(path, "rb") as file:
        return pickle.load(file)


def save_pickle(path, data):
    """Salva un oggetto Python in formato pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(data, file)


def load_dataset(input_dir, file_names):
    """Carica un sottoinsieme di file pickle da una cartella e segnala eventuali file mancanti."""
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
    """Rimuove gli stessi indici da tutti gli array del dataset."""
    for key in data_dict:
        data_dict[key] = np.delete(data_dict[key], indices)


def validate_balancing_indices(data_dict, indices):
    """Limita gli indici bilanciati alla lunghezza minima disponibile tra i campi caricati."""
    max_size = min(len(v) for v in data_dict.values())
    return indices[indices < max_size]


def apply_balanced_indices(data_dict, balanced_indices):
    """Applica un insieme comune di indici bilanciati a tutte le variabili del dataset."""
    for key in data_dict:
        data_dict[key] = data_dict[key][balanced_indices]


def balance_by_intervals(dpr_array, rain_intervals):
    """Esegue il bilanciamento stratificato per intervalli di precipitazione.
    
    Procedura:
    1. Per ogni intervallo, conta il numero di campioni
    2. Identifica il numero minimo (classe meno rappresentata)
    3. Sotto-campiona uniformemente tutte le classi al minimo
    4. Combina gli indici in un array ordinato
    
    Garantisce dataset perfettamente bilanciato (stessa numerosità per classe).
    """
    counts = []
    for lower_bound, upper_bound in rain_intervals:
        count_values = np.sum((dpr_array >= lower_bound) & (dpr_array < upper_bound))
        counts.append(count_values)
        print(f"Campioni intervallo ({lower_bound}, {upper_bound}): {count_values}")

    min_samples = min(counts)
    print(f"Numero minimo di campioni per classe: {min_samples}")

    balanced_indices_list = []
    for lower_bound, upper_bound in rain_intervals:
        # Identifica i pixel in questo intervallo di precipitazione
        indices_in_range = np.where((dpr_array >= lower_bound) & (dpr_array < upper_bound))[0]
        if len(indices_in_range) >= min_samples:
            # Sotto-campiona uniformemente al minimo
            selected_indices = np.random.choice(indices_in_range, size=min_samples, replace=False)
            balanced_indices_list.append(selected_indices)

    if not balanced_indices_list:
        return np.array([], dtype=int)

    # Ordina gli indici per mantenere coerenza nei dati
    return np.sort(np.concatenate(balanced_indices_list))


def save_dataset_with_mapping(output_dir, data_dict, mapping):
    """Salva un dataset usando una mappatura esplicita tra chiavi interne e nomi file output."""
    os.makedirs(output_dir, exist_ok=True)
    for src_key, output_filename in mapping.items():
        save_pickle(os.path.join(output_dir, output_filename), data_dict[src_key])


# --- Bilanciamento 2 classi ---

def run_balancing_2class():
    """Filtra outlier per la configurazione a 2 classi senza undersampling esplicito."""
    print("Avvio bilanciamento 2 classi")
    data, missing = load_dataset(DATA_PROCESSED_DIR, BASE_FILE_SET)

    if missing:
        print(f"File mancanti: {missing}")
        return

    dpr = data["DPR"]
    ch9 = data["CH_9"]

    indici_no_dpr = np.where(dpr > 150)[0]
    indici_ch9 = np.where(ch9 > 300)[0]
    indici_eliminati = np.unique(np.concatenate((indici_no_dpr, indici_ch9)))
    print(f"Indici eliminati: {len(indici_eliminati)}")

    apply_delete_indices(data, indici_eliminati)

    # Mantiene il comportamento originale: in questo blocco non si fa undersampling esplicito.
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
    print(f"Bilanciamento 2 classi salvato in: {BALANCING_2CLASS_DIR}")


# --- Bilanciamento 4 classi ---

def run_balancing_4class():
    """Bilancia il dataset secondo quattro intervalli di precipitazione."""
    print("Avvio bilanciamento 4 classi")
    data, missing = load_dataset(DATA_PROCESSED_DIR, EXTENDED_FILE_SET)

    if missing:
        print(f"File mancanti: {missing}")
        return

    indici_eliminati = np.where(data["DPR"] > 30)[0]
    apply_delete_indices(data, indici_eliminati)

    rain_intervals = [(0, 0.1), (0.1, 1), (1, 5), (5, 30)]
    balanced_indices = balance_by_intervals(data["DPR"], rain_intervals)
    balanced_indices = validate_balancing_indices(data, balanced_indices)

    apply_balanced_indices(data, balanced_indices)
    save_dataset_with_mapping(BALANCING_4CLASS_DIR, data, STANDARD_OUTPUT_MAPPING)
    print(f"Bilanciamento 4 classi salvato in: {BALANCING_4CLASS_DIR}")


# --- Bilanciamento 7 classi ---

def run_balancing_7class():
    """Bilancia il dataset secondo sette intervalli di precipitazione."""
    print("Avvio bilanciamento 7 classi")
    data, missing = load_dataset(DATA_PROCESSED_DIR, EXTENDED_FILE_SET)

    if missing:
        print(f"File mancanti: {missing}")
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
    print(f"Bilanciamento 7 classi salvato in: {BALANCING_7CLASS_DIR}")


# --- Bilanciamento 5 classi (pixel) ---

def run_balancing_5class():
    """Bilancia il dataset secondo cinque intervalli di precipitazione."""
    print("Avvio bilanciamento 5 classi")
    data, missing = load_dataset(DATA_PROCESSED_DIR, EXTENDED_FILE_SET)

    if missing:
        print(f"File mancanti: {missing}")
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
    print(f"Bilanciamento 5 classi salvato in: {BALANCING_5CLASS_DIR}")


# --- Bilanciamento 15 classi ---

def run_balancing_15class():
    """Bilancia il dataset secondo gli intervalli fini usati nella configurazione a 15 classi."""
    print("Avvio bilanciamento 15 classi")
    data, missing = load_dataset(DATA_PROCESSED_DIR, EXTENDED_FILE_SET)

    if missing:
        print(f"File mancanti: {missing}")
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
    print(f"Bilanciamento 15 classi salvato in: {BALANCING_15CLASS_DIR}")


# --- Bilanciamento 5 classi per immagini/features ---

def run_balancing_5class_images():
    """Bilanciamento 5 classi per il dataset di feature (immagini/feature).
    
    Carica il dataset già elaborato con feature engineering e applica lo stesso
    schema di bilanciamento 5 classi, preservando la struttura del DataFrame
    per la compatibilità con training ML downstream.
    """
    print("Avvio bilanciamento 5 classi per immagini/features")
    os.makedirs(BALANCING_5CLASS_IMAGES_DIR, exist_ok=True)

    # Carica dataset e maschere dal passo di feature engineering
    df_features = load_pickle(os.path.join(DATA_ML_5CLASS_DIR, "df_features.pickle"))
    dpr = load_pickle(os.path.join(DATA_ML_5CLASS_DIR, "DPR_ml.pickle"))
    giorno_maschera = load_pickle(os.path.join(DATA_ML_5CLASS_DIR, "maschera_giorno_ml.pickle"))

    data_dict = {
        "df_features": df_features,
        "DPR_ml": dpr,
        "maschera_giorno_ml": giorno_maschera,
    }

    # Definisce i 5 intervalli di precipitazione
    max_value = np.max(dpr)
    rain_intervals = [
        (0, 0.1),
        (0.1, 1),
        (1, 5),
        (5, 15),
        (15, max_value),
    ]

    # Raccoglie indici per ogni classe e calcola il minimo
    class_indices = []
    min_samples = float("inf")

    for low, high in rain_intervals:
        idx = np.where((dpr >= low) & (dpr < high))[0]
        class_indices.append(idx)
        min_samples = min(min_samples, len(idx))

    # Sotto-campiona ogni classe al minimo
    balanced_indices = []
    for idx in class_indices:
        selected = np.random.choice(idx, size=min_samples, replace=False)
        balanced_indices.append(selected)

    balanced_indices = np.sort(np.concatenate(balanced_indices))

    # Applica il bilanciamento: preserva tipo (DataFrame vs array)
    balanced_data = {}
    for key, array in data_dict.items():
        if hasattr(array, "iloc"):
            balanced_data[key] = array.iloc[balanced_indices].reset_index(drop=True)
        else:
            balanced_data[key] = array[balanced_indices]

    # Salva i dataset bilanciati
    for key, array in balanced_data.items():
        save_pickle(os.path.join(BALANCING_5CLASS_IMAGES_DIR, f"{key}.pickle"), array)

    print(f"Bilanciamento 5 classi immagini salvato in: {BALANCING_5CLASS_IMAGES_DIR}")


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
