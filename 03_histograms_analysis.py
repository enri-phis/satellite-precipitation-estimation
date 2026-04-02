"""
Script di analisi distribuzionale e generazione istogrammi diagnostici.

Questo modulo genera una suite completa di istogrammi e visualizzazioni diagnostiche
per esplorare le proprietà statistiche dei dati satellitari e di precipitazione.

Flusso di analisi:
1. **Istogramma DPR globale**: distribuzione delle precipitazioni (scala logaritmica)
2. **Istogrammi per canale**: distribuzione luminanza/riflettanza per le 11 bande MSG
3. **Pioggia vs no-pioggia**: confronto delle distribuzioni tra pixel piovosi e asciutti
4. **Giorno vs notte**: stratificazione temporale degli istogrammi
5. **Estate vs inverno**: stratificazione stagionale
6. **Mare vs terra**: stratificazione geografica
7. **Distribuzioni per classe DPR**: istogrammi per classi di intensità di pioggia
8. **Diagnostica RF (*opzionale*)**: curve ROC e grafici di feature importance

Questo step è importante per:
- Validare la qualità dei dati filtrati
- Esplorare distribuzioni storiche
- Identificare anomalie o artefatti
- Visualizzare bias stagionali/geografici
- Fornire grafici per documentazione e report

Configurazione:
- INPUT_MAT_DIR: cartella con dati filtrati
- MASK_DIR: cartella con le maschere generate
- OUTPUT_ROOT_DIR: cartella base per i grafici (crea sottocartelle automaticamente)
- RF_OUTPUT_DIR: (optional) cartella con predizioni RF per diagnostica
"""

import os
import pickle
import re
from collections import Counter
from typing import Dict, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

# =========================
# CONFIGURAZIONE
# =========================
INPUT_MAT_DIR = "PATH/TO/output_mat_directory"
MASK_DIR = "PATH/TO/masks_directory"
OUTPUT_ROOT_DIR = "PATH/TO/histograms_output"
RF_OUTPUT_DIR = "PATH/TO/rf_output_directory"

RUN_DPR_HIST = False
RUN_CHANNEL_HISTS = False
RUN_RAIN_NORAIN = False
RUN_RAIN_NORAIN_NORMALIZED = False
RUN_DAY_NIGHT = False
RUN_SUMMER_WINTER = False
RUN_SEA_LAND = False
RUN_DPR_CLASS_DISTRIBUTIONS = False
RUN_RF_DIAGNOSTIC_PLOTS = False
RUN_RF_CLASS_BARPLOTS = False

# Cartelle di output interne
OUT_DPR = os.path.join(OUTPUT_ROOT_DIR, "01_dpr")
OUT_CHANNELS = os.path.join(OUTPUT_ROOT_DIR, "02_channels")
OUT_RAIN_NORAIN = os.path.join(OUTPUT_ROOT_DIR, "03_rain_norain")
OUT_RAIN_NORAIN_NORM = os.path.join(OUTPUT_ROOT_DIR, "04_rain_norain_norm")
OUT_DAY_NIGHT = os.path.join(OUTPUT_ROOT_DIR, "05_day_night")
OUT_SUMMER_WINTER = os.path.join(OUTPUT_ROOT_DIR, "06_summer_winter")
OUT_SEA_LAND = os.path.join(OUTPUT_ROOT_DIR, "07_sea_land")
OUT_DPR_CLASSES = os.path.join(OUTPUT_ROOT_DIR, "08_dpr_classes")
OUT_RF_DIAG = os.path.join(OUTPUT_ROOT_DIR, "09_rf_diagnostics")


# =========================
# UTILITA
# =========================
def ensure_dir(path: str) -> None:
    """Crea una directory se non esiste già."""
    os.makedirs(path, exist_ok=True)


def estrai_doy_time(nome_file: str) -> Tuple[int, int]:
    """Estrae giorno dell'anno e tempo dal nome file per l'ordinamento cronologico."""
    match = re.search(r"DOY(\d+)_TIME(\d+)", nome_file)
    if match:
        return int(match.group(1)), int(match.group(2))
    return float("inf"), float("inf")


def list_mat_files_sorted(folder: str):
    """Restituisce i file `.mat` ordinati per data e ora di acquisizione."""
    files = [f for f in os.listdir(folder) if f.endswith(".mat")]
    return sorted(files, key=estrai_doy_time)


def load_all_mat_data(folder: str) -> Dict[str, np.ndarray]:
    """Carica e concatena tutti i file .mat di un folder.
    
    Ignora chiavi MATLAB interne (che iniziano con __).
    Concatena i dati per ogni chiave in array unici.
    """
    data = {}
    for file_name in list_mat_files_sorted(folder):
        mat_data = sio.loadmat(os.path.join(folder, file_name))
        for key, value in mat_data.items():
            if key.startswith("__"):
                continue
            data.setdefault(key, []).append(value)

    for key in data:
        try:
            data[key] = np.concatenate(data[key], axis=None)
        except Exception:
            data[key] = np.array(data[key]).ravel()

    return data


def load_channels_1_11(folder):
    """Carica e concatena i canali da 1 a 11 in un unico dizionario di array 1D."""
    data = {str(i): [] for i in range(1, 12)}
    for file_name in list_mat_files_sorted(folder):
        mat_data = sio.loadmat(os.path.join(folder, file_name))
        for k in data:
            if k in mat_data:
                data[k].append(mat_data[k])

    for k in data:
        data[k] = np.concatenate(data[k], axis=None) if data[k] else np.array([])

    return data


def bins_for_channel(channel_key: str) -> np.ndarray:
    """Restituisce i bin istogramma appropriati per canali visibili o infrarossi."""
    if channel_key in ["1", "2", "3"]:
        return np.arange(0.5, 80, 0.2)
    return np.arange(180, 340, 2)


def channel_axis_label(channel_key: str) -> str:
    """Restituisce l'etichetta asse x coerente con il tipo di canale."""
    return "Reflectance (%)" if channel_key in ["1", "2", "3"] else "Brightness temperature (K)"


def histogram_masked(values, mask, channel_key, density=True):
    """Calcola un istogramma su soli pixel selezionati dalla maschera binaria."""
    valid = values[mask == 1]
    valid = valid[~np.isnan(valid)]
    bins = bins_for_channel(channel_key)
    hist, _ = np.histogram(valid, bins=bins, density=density)
    return hist, bins


def histogram_masked_normalized_total(values, mask, channel_key, total_valid_pixels):
    """Calcola un istogramma normalizzato rispetto al numero totale di pixel validi."""
    valid = values[mask == 1]
    valid = valid[~np.isnan(valid)]
    bins = bins_for_channel(channel_key)
    hist, _ = np.histogram(valid, bins=bins, density=False)
    if total_valid_pixels > 0:
        hist = hist / total_valid_pixels
    return hist, bins


# =========================
# 1) ISTOGRAMMA DPR GLOBALE
# =========================
def run_dpr_histogram():
    """Genera un istogramma delle precipitazioni DPR con scala logaritmica.
    
    Utile per:
    - Visualizzare la distribuzione di frequenza delle piogge
    - Identificare la modalità principale (frequenza max)
    - Rilevare cose insolite/artefatti ai margini
    """
    ensure_dir(OUT_DPR)

    dpr_values = []
    for file_name in list_mat_files_sorted(INPUT_MAT_DIR):
        mat_data = sio.loadmat(os.path.join(INPUT_MAT_DIR, file_name))
        if "dpr" in mat_data:
            dpr = mat_data["dpr"]
        elif "precipitazione" in mat_data:
            dpr = mat_data["precipitazione"]
        else:
            continue

        # Raccoglie tutti i valori DPR non-NaN
        dpr_values.extend(dpr[~np.isnan(dpr)].flatten())

    dpr_values = np.array(dpr_values)
    if dpr_values.size == 0:
        print("Nessun dato DPR trovato")
        return

    # Crea istogramma con scala log per visualizzare meglio la coda rara
    plt.figure(figsize=(10, 6))
    plt.hist(dpr_values, bins=np.linspace(0.1, 200, 100), color="dodgerblue", alpha=0.7, edgecolor="black")
    plt.yscale("log")
    plt.xlabel("Precipitazione (mm/h)")
    plt.ylabel("Frequenza (scala log)")

    out_path = os.path.join(OUT_DPR, "istogramma_dpr.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Istogramma DPR salvato: {out_path}")


# =========================
# 2) ISTOGRAMMI CANALI 1-11
# =========================
def run_channel_histograms():
    """Genera gli istogrammi individuali per ciascuno degli 11 canali satellitari."""
    ensure_dir(OUT_CHANNELS)
    data = load_channels_1_11(INPUT_MAT_DIR)

    for ch_key, ch_values in data.items():
        if ch_values.size == 0:
            continue

        if ch_key in ["1", "2", "3"]:
            num_bins = int((25 - 0.5) / 0.1)
            bins = np.linspace(0.5, 25, num_bins + 1)
        else:
            num_bins = int((340 - 180) / 2)
            bins = np.linspace(180, 340, num_bins + 1)

        valid = ch_values[~np.isnan(ch_values)]
        hist, edges = np.histogram(valid.flatten(), bins=bins, density=False)

        plt.figure(figsize=(10, 6))
        plt.hist(edges[:-1], edges, weights=hist, alpha=0.7, density=True)
        plt.xlabel(channel_axis_label(ch_key))
        plt.ylabel("Densita")

        out_path = os.path.join(OUT_CHANNELS, f"istogramma_canale_{ch_key}.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    print(f"Istogrammi canali salvati in: {OUT_CHANNELS}")


# =========================
# 3) RAIN / NO RAIN
# =========================
def run_rain_norain(normalized_total=False):
    """Confronta le distribuzioni spettrali tra pixel piovosi e asciutti.
    
    Utile per:
    - Visualizzare le firme spettrali della pioggia su ogni canale
    - Identificare quali canali sono più sensibili alla precipitazione
    - Diagnosticare artefatti o anomalie stagionali
    
    Se normalized_total=True, normalizza le istogrammi per il totale di pixel,
    anziché usare densità di probabilità. Facilita il confronto con/senza pioggia.
    """
    out_dir = OUT_RAIN_NORAIN_NORM if normalized_total else OUT_RAIN_NORAIN
    ensure_dir(out_dir)

    # Carica dati e maschere di pioggia
    data = load_all_mat_data(INPUT_MAT_DIR)
    if "maschera_rain" in data or "maschera_norain" in data:
        # fallback raro se maschere fossero nei .mat dati
        mask_rain = data.get("maschera_rain", np.array([])).flatten()
        mask_norain = data.get("maschera_norain", np.array([])).flatten()
    else:
        # Carica le maschere generate dal passo precedente
        mask_rain = sio.loadmat(os.path.join(MASK_DIR, "maschera_rain.mat"))["maschera_rain"].flatten()
        mask_norain = sio.loadmat(os.path.join(MASK_DIR, "maschera_norain.mat"))["maschera_norain"].flatten()

    # Per ogni canale, crea istogrammi confrontati
    for ch in [str(i) for i in range(1, 12)]:
        if ch not in data:
            continue

        values = data[ch].flatten()
        if normalized_total:
            total_valid = np.sum(~np.isnan(values))
            hist_rain, bins = histogram_masked_normalized_total(values, mask_rain, ch, total_valid)
            hist_norain, _ = histogram_masked_normalized_total(values, mask_norain, ch, total_valid)
            ylabel = "Frazione di pixel totali"
        else:
            hist_rain, bins = histogram_masked(values, mask_rain, ch, density=True)
            hist_norain, _ = histogram_masked(values, mask_norain, ch, density=True)
            ylabel = "Densità"

        # Plot confronto rain vs no-rain
        plt.figure(figsize=(16, 12))
        plt.rcParams.update({"font.size": 32})
        plt.bar(bins[:-1], hist_rain, width=np.diff(bins), alpha=0.9, color="#00f2aa", label="Rain")
        plt.bar(bins[:-1], hist_norain, width=np.diff(bins), alpha=0.8, color="#E8AEE0", label="No Rain")
        plt.xlabel(channel_axis_label(ch), fontsize=40)
        plt.ylabel(ylabel, fontsize=40)
        plt.legend(fontsize=28)
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"confronto_rain_norain_canale_{ch}.png")
        plt.savefig(out_path)
        plt.close()

    print(f"Confronto rain/no-rain salvato in: {out_dir}")


# =========================
# 4) GIORNO / NOTTE
# =========================
def run_day_night():
    """Confronta le distribuzioni dei canali tra condizioni diurne e notturne."""
    ensure_dir(OUT_DAY_NIGHT)
    data = load_channels_1_11(INPUT_MAT_DIR)

    mask_day = sio.loadmat(os.path.join(MASK_DIR, "giorno_maschera_3h.mat"))["giorno_maschera"].flatten()
    mask_night = sio.loadmat(os.path.join(MASK_DIR, "notte_maschera_3h.mat"))["notte_maschera"].flatten()

    for ch, values in data.items():
        if values.size == 0:
            continue

        hist_day, bins = histogram_masked(values, mask_day, ch, density=True)
        hist_night, _ = histogram_masked(values, mask_night, ch, density=True)

        plt.figure(figsize=(16, 12))
        plt.rcParams.update({"font.size": 32})
        plt.bar(bins[:-1], hist_day, width=np.diff(bins), alpha=0.7, color="#FFD700", label="Day")
        plt.bar(bins[:-1], hist_night, width=np.diff(bins), alpha=0.5, color="#00BFFF", label="Night")
        plt.xlabel(channel_axis_label(ch), fontsize=40)
        plt.ylabel("Density", fontsize=40)
        plt.legend(fontsize=28)
        plt.tight_layout()

        out_path = os.path.join(OUT_DAY_NIGHT, f"confronto_canale_{ch}_3h.png")
        plt.savefig(out_path)
        plt.close()

    print(f"Confronto giorno/notte salvato in: {OUT_DAY_NIGHT}")


# =========================
# 5) ESTATE / INVERNO
# =========================
def run_summer_winter():
    """Confronta le distribuzioni dei canali tra estate e inverno."""
    ensure_dir(OUT_SUMMER_WINTER)

    mask_path = os.path.join(MASK_DIR, "stagionalita_maschera_estate_inverno.mat")
    mask_data = sio.loadmat(mask_path)
    season_mask = mask_data["stagionalita_maschera"].flatten()

    data = load_channels_1_11(INPUT_MAT_DIR)

    for idx, values in enumerate(data.values(), start=1):
        if values.size == 0:
            continue
        if len(values) != len(season_mask):
            continue

        mask_summer = (season_mask == 0).astype(int)
        mask_winter = (season_mask == 1).astype(int)

        ch = str(idx)
        hist_summer, bins = histogram_masked(values, mask_summer, ch, density=True)
        hist_winter, _ = histogram_masked(values, mask_winter, ch, density=True)

        plt.figure(figsize=(16, 12))
        plt.rcParams.update({"font.size": 32})
        plt.bar(bins[:-1], hist_summer, width=np.diff(bins), color="#FFA500", alpha=0.6, label="Summer")
        plt.bar(bins[:-1], hist_winter, width=np.diff(bins), color="#0055FF", alpha=0.5, label="Winter")
        plt.xlabel(channel_axis_label(ch), fontsize=40)
        plt.ylabel("Density", fontsize=40)
        plt.legend(fontsize=28)
        plt.tight_layout()

        out_path = os.path.join(OUT_SUMMER_WINTER, f"istogramma_estate_inverno_canale_{ch}.png")
        plt.savefig(out_path)
        plt.close()

    print(f"Confronto estate/inverno salvato in: {OUT_SUMMER_WINTER}")


# =========================
# 6) MARE / TERRA
# =========================
def run_sea_land():
    """Confronta le distribuzioni dei canali tra pixel su mare e su terra."""
    ensure_dir(OUT_SEA_LAND)

    mask_sea = sio.loadmat(os.path.join(MASK_DIR, "Mare_mask_regridded.mat"))["mask"].flatten()
    mask_land = sio.loadmat(os.path.join(MASK_DIR, "Terra_mask_regridded.mat"))["mask"].flatten()

    data = load_channels_1_11(INPUT_MAT_DIR)

    for ch, values in data.items():
        if values.size == 0:
            continue

        if len(values) != len(mask_sea):
            continue

        hist_sea, bins = histogram_masked(values, mask_sea, ch, density=True)
        hist_land, _ = histogram_masked(values, mask_land, ch, density=True)

        plt.figure(figsize=(16, 12))
        plt.rcParams.update({"font.size": 32})
        plt.bar(bins[:-1], hist_sea, width=np.diff(bins), alpha=0.8, color="lightblue", label="Sea")
        plt.bar(bins[:-1], hist_land, width=np.diff(bins), alpha=0.5, color="sandybrown", label="Land")
        plt.xlabel(channel_axis_label(ch), fontsize=40)
        plt.ylabel("Density", fontsize=40)
        plt.legend(fontsize=28)
        plt.tight_layout()

        out_path = os.path.join(OUT_SEA_LAND, f"confronto_mare_terra_canale_{ch}.png")
        plt.savefig(out_path)
        plt.close()

    print(f"Confronto mare/terra salvato in: {OUT_SEA_LAND}")


# =========================
# 7) CLASSI DPR
# =========================
def run_dpr_class_distributions():
    """Stima le distribuzioni dei canali stratificate per classe di intensità DPR."""
    ensure_dir(OUT_DPR_CLASSES)

    from tqdm import tqdm
    import seaborn as sns

    data = load_all_mat_data(INPUT_MAT_DIR)
    if "dpr" not in data:
        print("Variabile dpr non trovata")
        return

    dpr = data["dpr"].flatten()
    thresholds = [0, 0.1, 1, 5, 15, np.inf]
    class_names = ["Dry", "Light", "Moderate", "Heavy", "Extreme"]
    colors = ["blue", "green", "yellow", "orange", "red"]

    for ch in tqdm([str(i) for i in range(1, 12)], desc="Canali"):
        if ch not in data:
            continue

        ch_values = data[ch].flatten()
        valid = ~np.isnan(dpr) & ~np.isnan(ch_values)
        dpr_valid = dpr[valid]
        ch_values = ch_values[valid]

        pos = ch_values >= 0
        dpr_valid = dpr_valid[pos]
        ch_values = ch_values[pos]

        plt.figure(figsize=(16, 12))
        plt.rcParams.update({"font.size": 32})

        for i in range(len(thresholds) - 1):
            low, high = thresholds[i], thresholds[i + 1]
            mask = (dpr_valid >= low) & (dpr_valid < high)
            subset = ch_values[mask]
            if len(subset) == 0:
                continue
            if len(subset) > 1_000_000:
                subset = np.random.choice(subset, 500000, replace=False)
            sns.kdeplot(subset, color=colors[i], label=class_names[i], clip=(0, np.inf), linewidth=3.5)

        if ch in ["1", "2", "3"]:
            plt.xlim(0, 60)
            plt.ylim(0, 0.17)

        plt.xlabel(channel_axis_label(ch), fontsize=40)
        plt.ylabel("Density", fontsize=40)
        plt.legend(fontsize=28)
        plt.tight_layout()

        out_path = os.path.join(OUT_DPR_CLASSES, f"distribuzione_classi_canale_{ch}.png")
        plt.savefig(out_path)
        plt.close()

    print(f"Distribuzioni per classi DPR salvate in: {OUT_DPR_CLASSES}")


# =========================
# 8) GRAFICI RF (opzionali)
# =========================
def load_canale9_dpr_concat_for_rf():
    """Carica e concatena canale 9 e DPR per la diagnostica dei risultati RF."""
    canale9_tot = []
    dpr_tot = []

    for file_name in list_mat_files_sorted(INPUT_MAT_DIR):
        mat_data = sio.loadmat(os.path.join(INPUT_MAT_DIR, file_name))
        if "9" in mat_data:
            ch9 = mat_data["9"].flatten()
        elif "channel9" in mat_data:
            ch9 = mat_data["channel9"].flatten()
        else:
            continue

        if "dpr" in mat_data:
            dpr = mat_data["dpr"].flatten()
        elif "DPR" in mat_data:
            dpr = mat_data["DPR"].flatten()
        else:
            continue

        if len(ch9) != len(dpr):
            continue

        canale9_tot.append(ch9)
        dpr_tot.append(dpr)

    if not canale9_tot:
        return np.array([]), np.array([])

    return np.concatenate(canale9_tot), np.concatenate(dpr_tot)


def run_rf_diagnostics():
    """Genera grafici diagnostici che mettono in relazione classi RF, canale 9 e DPR."""
    ensure_dir(OUT_RF_DIAG)

    y_pred_path = os.path.join(RF_OUTPUT_DIR, "y_pred_RF.pickle")
    if not os.path.exists(y_pred_path):
        print("File y_pred_RF.pickle non trovato")
        return

    with open(y_pred_path, "rb") as f:
        y_pred = pickle.load(f)

    canale9_tot, dpr_tot = load_canale9_dpr_concat_for_rf()
    if canale9_tot.size == 0 or dpr_tot.size == 0:
        print("Dati canale9/DPR non disponibili")
        return

    if len(y_pred) != len(canale9_tot):
        print("Dimensioni non allineate tra y_pred e canale9")
        return

    ordered_classes = ["Dry", "Light", "Moderate", "Heavy", "Intense"]
    class_to_index = {label: i for i, label in enumerate(ordered_classes)}

    # Plot 1: boxplot + scatter classe/canale9
    plt.figure(figsize=(14, 6))
    data_ord = [canale9_tot[np.array(y_pred) == label] for label in ordered_classes]

    plt.subplot(1, 2, 1)
    plt.boxplot(data_ord, labels=ordered_classes)
    plt.title("Boxplot Canale 9 per Classe Predetta")
    plt.xlabel("Classe Predetta")
    plt.ylabel("Valori Canale 9")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.subplot(1, 2, 2)
    x_pred = [class_to_index[c] for c in y_pred if c in class_to_index]
    y_vals = [canale9_tot[i] for i, c in enumerate(y_pred) if c in class_to_index]
    colors = ["#cce5ff", "#99ccff", "#66b3ff", "#3399ff", "#0066cc"]
    cvals = [colors[class_to_index[c]] for c in y_pred if c in class_to_index]

    plt.scatter(x_pred, y_vals, c=cvals, alpha=0.4, s=5)
    plt.xticks(range(len(ordered_classes)), ordered_classes)
    plt.title("Scatter: Classe Predetta vs Canale 9")
    plt.xlabel("Classe Predetta")
    plt.ylabel("Valori Canale 9")
    plt.grid(True, linestyle="--", alpha=0.5)

    out1 = os.path.join(OUT_RF_DIAG, "rf_boxplot_scatter_canale9.png")
    plt.tight_layout()
    plt.savefig(out1)
    plt.close()

    # Plot 2: scatter DPR vs canale9
    dpr_bins = [0, 0.1, 1, 5, 15, np.max(dpr_tot)]
    dpr_idx = np.digitize(dpr_tot, dpr_bins) - 1
    shades = ["#cce5ff", "#99ccff", "#66b3ff", "#3399ff", "#0066cc"]
    c_dpr = [shades[i] if 0 <= i < len(shades) else "black" for i in dpr_idx]

    plt.figure(figsize=(10, 6))
    plt.scatter(dpr_tot, canale9_tot, c=c_dpr, alpha=0.4, s=5)
    plt.xscale("log")
    plt.xlabel("DPR (mm/h, scala log)")
    plt.ylabel("Canale 9")
    plt.title("Distribuzione Canale 9 in funzione di DPR")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    legend = [
        mpatches.Patch(color=shades[0], label="0-0.1 mm/h"),
        mpatches.Patch(color=shades[1], label="0.1-1 mm/h"),
        mpatches.Patch(color=shades[2], label="1-5 mm/h"),
        mpatches.Patch(color=shades[3], label="5-15 mm/h"),
        mpatches.Patch(color=shades[4], label=">15 mm/h"),
    ]
    plt.legend(handles=legend, title="Classi DPR", loc="upper right")

    out2 = os.path.join(OUT_RF_DIAG, "rf_scatter_dpr_canale9.png")
    plt.tight_layout()
    plt.savefig(out2)
    plt.close()

    print(f"Grafici diagnostici RF salvati in: {OUT_RF_DIAG}")


def run_rf_class_barplots():
    """Confronta la distribuzione delle classi reali DPR con quella delle predizioni RF."""
    ensure_dir(OUT_RF_DIAG)

    y_pred_path = os.path.join(RF_OUTPUT_DIR, "y_pred_RF.pickle")
    if not os.path.exists(y_pred_path):
        print("File y_pred_RF.pickle non trovato")
        return

    with open(y_pred_path, "rb") as f:
        y_pred = pickle.load(f)

    _, dpr_tot = load_canale9_dpr_concat_for_rf()
    if dpr_tot.size == 0:
        print("DPR non disponibile")
        return

    ordered_classes = ["Dry", "Light", "Moderate", "Heavy", "Intense"]
    dpr_bins = [0, 0.1, 1, 5, 15, np.max(dpr_tot)]
    dpr_cls_idx = np.digitize(dpr_tot, dpr_bins) - 1
    dpr_labels = [ordered_classes[i] if 0 <= i < len(ordered_classes) else "Unknown" for i in dpr_cls_idx]

    cnt_dpr = Counter(dpr_labels)
    cnt_pred = Counter(y_pred)

    freq_dpr = np.array([cnt_dpr.get(label, 0) for label in ordered_classes])
    freq_pred = np.array([cnt_pred.get(label, 0) for label in ordered_classes])

    x = np.arange(len(ordered_classes))
    width = 0.35

    # assoluto
    plt.figure(figsize=(10, 6))
    b1 = plt.bar(x - width / 2, freq_dpr, width=width, label="DPR Reale", color="#3399ff")
    b2 = plt.bar(x + width / 2, freq_pred, width=width, label="Predizione", color="#ff9933")

    for rect in list(b1) + list(b2):
        h = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, h + 1, str(int(h)), ha="center", va="bottom", fontsize=8)

    plt.xticks(x, ordered_classes)
    plt.ylabel("Numero di pixel")
    plt.title("Distribuzione Assoluta: DPR vs Predizione")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    out_abs = os.path.join(OUT_RF_DIAG, "rf_class_distribution_absolute.png")
    plt.tight_layout()
    plt.savefig(out_abs)
    plt.close()

    # normalizzato
    freq_dpr_norm = freq_dpr / max(freq_dpr.sum(), 1) * 100
    freq_pred_norm = freq_pred / max(freq_pred.sum(), 1) * 100

    plt.figure(figsize=(10, 6))
    b1 = plt.bar(x - width / 2, freq_dpr_norm, width=width, label="DPR Reale", color="#3399ff")
    b2 = plt.bar(x + width / 2, freq_pred_norm, width=width, label="Predizione", color="#ff9933")

    for rect in list(b1) + list(b2):
        h = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, h + 0.3, f"{h:.1f}%", ha="center", va="bottom", fontsize=8)

    plt.xticks(x, ordered_classes)
    plt.ylabel("Percentuale di pixel (%)")
    plt.title("Distribuzione Percentuale: DPR vs Predizione")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    out_norm = os.path.join(OUT_RF_DIAG, "rf_class_distribution_percentage.png")
    plt.tight_layout()
    plt.savefig(out_norm)
    plt.close()

    print(f"Grafici classi RF salvati in: {OUT_RF_DIAG}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    ensure_dir(OUTPUT_ROOT_DIR)

    if RUN_DPR_HIST:
        run_dpr_histogram()

    if RUN_CHANNEL_HISTS:
        run_channel_histograms()

    if RUN_RAIN_NORAIN:
        run_rain_norain(normalized_total=False)

    if RUN_RAIN_NORAIN_NORMALIZED:
        run_rain_norain(normalized_total=True)

    if RUN_DAY_NIGHT:
        run_day_night()

    if RUN_SUMMER_WINTER:
        run_summer_winter()

    if RUN_SEA_LAND:
        run_sea_land()

    if RUN_DPR_CLASS_DISTRIBUTIONS:
        run_dpr_class_distributions()

    if RUN_RF_DIAGNOSTIC_PLOTS:
        run_rf_diagnostics()

    if RUN_RF_CLASS_BARPLOTS:
        run_rf_class_barplots()
