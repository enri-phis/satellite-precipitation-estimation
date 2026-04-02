"""
Script di caricamento e filtro geografico dei dati satellitari.

Questo modulo gestisce il caricamento dei file satellitari grezzi in formato HDF5/MATLAB v7.3
e applica un filtro geografico per selezionare solo le immagini che ricadono nell'area
di interesse definita tramite intervalli di latitudine e longitudine.

Flusso operativo:
1. Legge i file .mat grezzi (HDF5 v7.3) dalla cartella di input.
2. Per ogni file, estrae le 11 bande spettrali MSG, la precipitazione DPR,
   le griglie geografiche (lat/lon) e i metadati temporali.
3. Applica il filtro geografico per scartare le immagini fuori dall'area di interesse.
4. Salva le immagini filtrate come singoli file .mat in una cartella di output normalizzata.

Questo step è il primo della pipeline di pre-processing: prepara i dati grezzi
prima del calcolo delle maschere geografiche e dell'estrazione delle feature.

Configurazione:
- RAW_DATA_DIR: cartella contenente i file .mat grezzi (HDF5 v7.3)
- FILTERED_OUTPUT_DIR: cartella dove salvare i file filtrati
- LAT_RANGE, LON_RANGE: intervalli geografici di interesse
- GROUP_SIZE: numero di immagini caricate per volta (per limitare RAM)
"""

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Set, Tuple

import h5py
import numpy as np
from scipy.io import savemat

# =========================
# Configuration
# =========================
# Percorsi di progetto: modificare solo questa sezione per adattare lo script al proprio ambiente.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_ROOT / "raw"
FILTERED_OUTPUT_DIR = DATA_ROOT / "filtered"

LAT_RANGE = (35, 50)
LON_RANGE = (-10, 25)
GROUP_SIZE = 100
# =========================

LATITUDE_KEY = 12
LONGITUDE_KEY = 13


def carica_blocchi_dati_v73(directory: str, nome_file: str, start_idx: int, end_idx: int) -> Dict[str, Dict[Any, np.ndarray]]:
    """
    Carica un blocco di immagini da un file HDF5 v7.3 (.mat).

    Ogni immagine è composta da:
    - 11 bande spettrali MSG (canali infrarossi e visibili)
    - 1 campo di precipitazione DPR
    - 1 griglia di latitudine, 1 di longitudine
    - Metadati temporali (giorno dell'anno, ora UTC)
    
    Args:
        directory (str): cartella di input contenente il file HDF5/MATLAB.
        nome_file (str): nome del file `.mat` da leggere.
        start_idx (int): indice iniziale del blocco di immagini.
        end_idx (int): indice finale del blocco di immagini.

    Returns:
        Dict[str, Dict[Any, np.ndarray]]: dizionario contenente i dati caricati
        per ciascuna immagine del blocco.
    """
    dati = {}
    percorso_file = os.path.join(directory, nome_file)
    print(f"Caricamento {nome_file}: immagini {start_idx + 1}-{end_idx}")

    with h5py.File(percorso_file, "r") as file_hdf5:
        numero_immagini = file_hdf5["msgData"].shape[0]
        end_idx = min(end_idx, numero_immagini)

        for indice in range(start_idx, end_idx):
            nome_dato = f"{nome_file}_img_{indice}"
            dati[nome_dato] = {}

            for canale in range(11):
                dati[nome_dato][canale + 1] = np.array(file_hdf5["msgData"][indice, canale, :, :])

            dati[nome_dato][LATITUDE_KEY] = np.array(file_hdf5["Latitude"][indice, :, :])
            dati[nome_dato][LONGITUDE_KEY] = np.array(file_hdf5["Longitude"][indice, :, :])
            dati[nome_dato]["dpr"] = np.array(file_hdf5["dprTotalPrecip"][indice, :, :])
            dati[nome_dato]["dayOfYear"] = np.array(file_hdf5["dayOfYear"][:, indice])
            dati[nome_dato]["iTmOfDay"] = np.array(file_hdf5["iTmOfDay"][:, indice])

    print(f"Caricate {len(dati)} immagini")
    return dati


def filtro_geografico(
    dati: Dict[str, Dict[Any, np.ndarray]],
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
) -> Dict[str, Dict[Any, np.ndarray]]:
    """
    Filtra le immagini in base all'area geografica di interesse.

    Mantiene solo le immagini il cui bounding box geografico interseca l'area
    definita da lat_range e lon_range. Le immagini fuori dall'area vengono scartate.

    Args:
        dati (Dict[str, Dict[Any, np.ndarray]]): immagini caricate dal file sorgente.
        lat_range (Tuple[float, float]): intervallo di latitudine ammesso.
        lon_range (Tuple[float, float]): intervallo di longitudine ammesso.

    Returns:
        Dict[str, Dict[Any, np.ndarray]]: sole immagini che intersecano
        l'area geografica di interesse.
    """
    print(f"Controllo geografico su {len(dati)} immagini")
    dati_filtrati = {}

    for nome_immagine, dati_immagine in dati.items():
        # Verifica disponibilità griglie geografiche
        if LATITUDE_KEY not in dati_immagine or LONGITUDE_KEY not in dati_immagine:
            print(f"Saltato {nome_immagine}: latitudine o longitudine non disponibili")
            continue

        latitudine = np.array(dati_immagine[LATITUDE_KEY])
        longitudine = np.array(dati_immagine[LONGITUDE_KEY])

        # Verifica consistenza dimensioni
        if latitudine.shape != longitudine.shape:
            print(f"Saltato {nome_immagine}: dimensioni di latitudine e longitudine non compatibili")
            continue

        # Crea maschere booleane per il filtro geografico
        mask_lat = (lat_range[0] <= latitudine) & (latitudine <= lat_range[1])
        mask_lon = (lon_range[0] <= longitudine) & (longitudine <= lon_range[1])
        mask     = mask_lat & mask_lon

        # Mantieni solo immagini con almeno un pixel nell'area
        if not mask.any():
            continue

        dati_filtrati[nome_immagine] = dati_immagine

    print(f"Accettate {len(dati_filtrati)} immagini dopo il filtro geografico")
    return dati_filtrati


def salva_dati_filtrati(
    dati: Dict[Any, np.ndarray],
    output_dir: str,
    nome_file: str,
    gia_visti: Set[Tuple[int, int]],
) -> None:
    """
    Salva un'immagine filtrata in formato `.mat`, evitando duplicati temporali.

    La chiave di unicità è definita dalla coppia `(dayOfYear, iTmOfDay)`.
    Se un'immagine con la stessa coppia è già stata salvata, viene ignorata.

    Args:
        dati (Dict[Any, np.ndarray]): contenuto dell'immagine da salvare.
        output_dir (str): cartella di destinazione.
        nome_file (str): nome del file sorgente, usato per costruire il nome output.
        gia_visti (Set[Tuple[int, int]]): insieme delle coppie temporali già esportate.
    """
    os.makedirs(output_dir, exist_ok=True)

    giorno = int(np.squeeze(dati["dayOfYear"]))
    ora = int(np.squeeze(dati["iTmOfDay"]))
    chiave = (giorno, ora)

    if chiave in gia_visti:
        print(f"Immagine duplicata ignorata: giorno {giorno}, ora {ora}")
        return

    gia_visti.add(chiave)
    nome_base       = f"{os.path.splitext(nome_file)[0]}_DOY{giorno}_TIME{ora}"
    output_file_mat = os.path.join(output_dir, f"{nome_base}.mat")
    dati_mat        = {str(key): array for key, array in dati.items()}

    savemat(output_file_mat, dati_mat)
    print(f"Salvato {output_file_mat}")


def processa_gruppi(
    directory: str,
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
    output_dir: str,
    group_size: int = 100,
) -> Dict[str, Dict[Any, np.ndarray]]:
    """
    Elabora tutti i file del directory in blocchi per gestire la memoria.

    Per ogni file HDF5:
    1. Carica blocchi di immagini (da limitare l'uso di RAM)
    2. Applica il filtro geografico
    3. Salva le immagini filtrate come file .mat separati

    Args:
        directory (str): cartella input con file `.mat` grezzi.
        lat_range (Tuple[float, float]): intervallo di latitudine.
        lon_range (Tuple[float, float]): intervallo di longitudine.
        output_dir (str): cartella di output.
        group_size (int): numero di immagini da elaborare per blocco.

    Returns:
        Dict[str, Dict[Any, np.ndarray]]: immagini filtrate raccolte durante
        l'elaborazione, utile per controlli rapidi o statistiche finali.
    """
    gia_visti = set()

    print(f"Preparazione cartella di output: {output_dir}")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    if not os.path.exists(directory):
        print(f"Cartella non trovata: {directory}")
        return {}

    # Lista file di input
    file_list = sorted(f for f in os.listdir(directory) if f.endswith(".mat"))
    filtered_data = {}
    print(f"Trovati {len(file_list)} file di input")

    # Processa ciascun file in blocchi
    for file_name in file_list:
        percorso_file = os.path.join(directory, file_name)
        with h5py.File(percorso_file, "r") as file_hdf5:
            numero_immagini = file_hdf5["msgData"].shape[0]

        print(f"Elaborazione di {file_name} ({numero_immagini} immagini)")

        # Carica e processa blocchi successivi
        for start_idx in range(0, numero_immagini, group_size):
            end_idx = start_idx + group_size
            dati_blocco = carica_blocchi_dati_v73(directory, file_name, start_idx, end_idx)
            dati_filtrati = filtro_geografico(dati_blocco, lat_range, lon_range)

            # Salva ciascuna immagine filtrata come file separato
            for _, dati in dati_filtrati.items():
                salva_dati_filtrati(dati, output_dir, file_name, gia_visti)

            filtered_data.update(dati_filtrati)

    print(f"Totale immagini filtrate: {len(filtered_data)}")
    return filtered_data


# --- Esecuzione principale ---
if __name__ == "__main__":
    if not Path(RAW_DATA_DIR).exists():
        print(f"Cartella input non trovata: {RAW_DATA_DIR}")
        print("Impostare i percorsi nella sezione Configuration prima di eseguire.")
    else:
        processa_gruppi(RAW_DATA_DIR, LAT_RANGE, LON_RANGE, FILTERED_OUTPUT_DIR, group_size=GROUP_SIZE)


# --- Opzionale: verifica dell'output ---
# Decommentare per ispezionare un file di output dopo l'esecuzione principale.
#
# import scipy.io
# SAMPLE_FILE = FILTERED_OUTPUT_DIR / "example_output_file.mat"
# dati_file = scipy.io.loadmat(SAMPLE_FILE)
# chiavi_da_controllare = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, "dpr", "dayOfYear", "iTmOfDay"]
# print(f"Chiavi disponibili: {list(dati_file.keys())}")
# for chiave in chiavi_da_controllare:
#     chiave_str = str(chiave)
#     if chiave_str in dati_file:
#         print(f"{chiave}: shape={dati_file[chiave_str].shape}")
#     else:
#         print(f"{chiave}: non trovata")


# --- Opzionale: mappa di copertura geografica ---
# Richiede cartopy. Decommentare dopo l'esecuzione principale.
#
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# import matplotlib.pyplot as plt
# import scipy.io
#
# lat_totali, lon_totali = [], []
# for nome_file in os.listdir(FILTERED_OUTPUT_DIR):
#     if not nome_file.endswith(".mat"):
#         continue
#     try:
#         dati = scipy.io.loadmat(os.path.join(FILTERED_OUTPUT_DIR, nome_file))
#         lat, lon = dati.get("12"), dati.get("13")
#         if lat is not None and lon is not None:
#             lat_totali.extend(lat.flatten())
#             lon_totali.extend(lon.flatten())
#     except Exception as e:
#         print(f"Errore nella lettura di {nome_file}: {e}")
#
# lat_arr = np.array(lat_totali)
# lon_arr = np.array(lon_totali)
# validi  = ~np.isnan(lat_arr) & ~np.isnan(lon_arr)
#
# plt.figure(figsize=(10, 6))
# ax = plt.axes(projection=ccrs.PlateCarree())
# ax.add_feature(cfeature.LAND, facecolor="white")
# ax.add_feature(cfeature.OCEAN, facecolor="mintcream")
# ax.coastlines(resolution="10m", linewidth=0.6)
# ax.add_feature(cfeature.BORDERS, linestyle="--", linewidth=0.2)
# gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
# gl.top_labels = False
# gl.right_labels = False
# gl.xlabel_style = {"size": 12}
# gl.ylabel_style = {"size": 12}
# ax.set_extent([-20, 40, 25, 60], crs=ccrs.PlateCarree())
# ax.scatter(lon_arr[validi], lat_arr[validi], s=1, color="#81D8D0", alpha=1,
#            transform=ccrs.PlateCarree(), label="Loaded points")
# plt.legend(fontsize=13)
# plt.show()
