"""
Script di generazione delle maschere geografiche e temporali.

Questo modulo genera le maschere essenziali per il filtraggio dei dati satellitari:

1. **Maschera giorno/notte (day/night mask)**: criteri fisici basati su alba/tramonto
   calcolati con coordinate geografiche precise (ephem). Include isteresi temporale
   (±2 ore dall'alba/tramonto) per evitare instabilità alla transizione luce/ombra.

2. **Maschera stagionale (seasonal mask)**: divide i dati tra stagione estiva
   (primavera/estate, DOY 152-243) e invernale (autunno/inverno, DOY 335-59).
   Consente confronti stagionali delle precipitazioni.

3. **Maschera terra/mare (sea/land mask)**: regridda una mappa globale terra/mare
   (da file NetCDF) alle coordinate dei dati di input via interpolazione nearest-neighbor.

Sono usate per:
- Filtrare dati notturni o poco affidabili (night removal)
- Stratificare analisi per stagione
- Analisi differenziate su oceano vs terra

Output: File .mat MATLAB contenenti le maschere (array di 0/1) con dimensioni
pari ai dati di input.

Configurazione:
- DATA_DIRECTORY: cartella con file .mat filtrati (output del paso 1)
- MASK_DIRECTORY: cartella di output per le maschere
- SEA_LAND_NETCDF_FILE: mappa globale terra/mare (NetCDF)
"""

from datetime import datetime, timedelta
import os
from pathlib import Path
import re
from typing import Tuple

# =========================
# Configuration
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"

DATA_DIRECTORY = DATA_ROOT / "filtered"
MASK_DIRECTORY = DATA_ROOT / "masks"
SEA_LAND_NETCDF_FILE = DATA_ROOT / "support" / "sea_land_mask.nc"

DAY_MASK_FILENAME = "giorno_maschera_3h.mat"
NIGHT_MASK_FILENAME = "notte_maschera_3h.mat"
SEASON_MASK_FILENAME = "stagionalita_maschera_estate_inverno.mat"
SEA_MASK_FILENAME = "Mare_mask_regridded.mat"
LAND_MASK_FILENAME = "Terra_mask_regridded.mat"
# =========================

import ephem
import h5py
import numpy as np
import scipy.io as sio
from scipy.interpolate import griddata
from scipy.io import loadmat, savemat

LATITUDE_KEY = "12"
LONGITUDE_KEY = "13"


def estrai_doy_time(nome_file: str) -> Tuple[int, int]:
    """Estrae giorno dell'anno e ora dal nome file per l'ordinamento cronologico."""
    match = re.search(r"DOY(\d+)_TIME(\d+)", nome_file)
    if match:
        doy = int(match.group(1))
        time = int(match.group(2))
        return doy, time
    return float("inf"), float("inf")


def lista_file_mat_ordinati(data_directory: str):
    """Restituisce i file `.mat` ordinati per giorno dell'anno e tempo di acquisizione."""
    return sorted([f for f in os.listdir(data_directory) if f.endswith(".mat")], key=estrai_doy_time)


# --- Maschera giorno/notte ---

def calcola_fuso_orario(longitudine: float) -> int:
    """Approssima il fuso orario locale a partire dalla longitudine."""
    return round(longitudine / 15)


def get_sunrise_sunset(dt_utc: datetime, latitude: float, longitude: float):
    """Calcola alba e tramonto locali per una data e una coordinata geografica."""
    observer = ephem.Observer()
    observer.lat = str(latitude)
    observer.lon = str(longitude)
    observer.date = dt_utc.date()
    observer.pressure = 0

    sunrise = observer.next_rising(ephem.Sun(), use_center=True)
    sunset = observer.next_setting(ephem.Sun(), use_center=True)
    alba = ephem.localtime(sunrise)
    tramonto = ephem.localtime(sunset)
    return alba, tramonto


def is_daytime(local_datetime, sunrise, sunset):
    """Classifica un istante locale come diurno usando una finestra conservativa tra alba e tramonto."""
    daytime_start = sunrise + timedelta(hours=2)
    daytime_end = sunset - timedelta(hours=2)
    return daytime_start <= local_datetime <= daytime_end


def genera_maschera_giorno_notte(data_directory: str, mask_directory: str) -> None:
    """
    Genera la maschera giorno/notte basata su calcoli astronomici precisi.
    
    Per ogni pixel di ogni immagine:
    - Estrae i metadati temporali (giorno dell'anno, ora UTC)
    - Converte l'ora a zona locale usando la longitudine
    - Calcola alba/tramonto con la libreria ephem (efemeridi astronomiche)
    - Assegna valore 1 (giorno) se nell'intervallo [alba+2h, tramonto-2h]
    - Altrimenti assegna 0 (notte)
    
    L'isteresi di ±2 ore riduce gli effetti dell'illuminazione diffusa al crepuscolo.
    """
    giorno_maschera = []
    notte_maschera = []

    mat_files = lista_file_mat_ordinati(data_directory)
    if not mat_files:
        print("Nessun file .mat trovato per la maschera giorno/notte.")
        return

    print(f"Maschera giorno/notte: trovati {len(mat_files)} file")

    for idx, file_name in enumerate(mat_files, start=1):
        file_path = os.path.join(data_directory, file_name)
        print(f"[{idx}/{len(mat_files)}] Elaborazione {file_name}")

        try:
            dati = loadmat(file_path)

            if "dayOfYear" not in dati or "iTmOfDay" not in dati:
                print(f"Saltato {file_name}: metadati temporali mancanti")
                continue

            day_of_year = int(dati["dayOfYear"].flatten()[0])
            minute_of_day = int(dati["iTmOfDay"].flatten()[0]) * 15
            dt_utc = datetime(2020, 1, 1) + timedelta(days=day_of_year - 1, minutes=minute_of_day)

            if LATITUDE_KEY not in dati or LONGITUDE_KEY not in dati:
                print(f"Saltato {file_name}: latitudine/longitudine mancanti")
                continue

            latitudes = dati[LATITUDE_KEY].flatten()
            longitudes = dati[LONGITUDE_KEY].flatten()

            for lat, lon in zip(latitudes, longitudes):
                # Calcola il fuso orario dalla longitudine e converte l'ora UTC a locale
                fuso = calcola_fuso_orario(lon)
                dt_local = dt_utc + timedelta(hours=fuso)
                # Ottiene alba e tramonto per la posizione e la data specificate
                alba, tramonto = get_sunrise_sunset(dt_utc, lat, lon)

                # Classifica il pixel come giorno o notte
                if is_daytime(dt_local, alba, tramonto):
                    giorno_maschera.append(1)
                    notte_maschera.append(0)
                else:
                    giorno_maschera.append(0)
                    notte_maschera.append(1)

        except Exception as errore:
            print(f"Errore nel file {file_name}: {errore}")

    savemat(os.path.join(mask_directory, DAY_MASK_FILENAME), {"giorno_maschera": np.array(giorno_maschera)})
    savemat(os.path.join(mask_directory, NIGHT_MASK_FILENAME), {"notte_maschera": np.array(notte_maschera)})
    print(f"Maschera giorno/notte salvata. Totale punti: {len(giorno_maschera)}")


# --- Maschera stagionale ---

def get_season_by_day_of_year(day_of_year: int) -> int:
    """Assegna la stagione climatologica usata nel progetto: estate, inverno o periodo escluso."""
    if 335 <= day_of_year <= 365 or 1 <= day_of_year <= 59:
        return 1  # inverno
    if 152 <= day_of_year <= 243:
        return 0  # estate
    return 2  # primavera/autunno (esclusi)


def genera_maschera_stagionale(data_directory: str, mask_directory: str) -> None:
    """
    Genera la maschera stagionale basata sul giorno dell'anno (day-of-year).
    
    - Estate (estate): DOY 152-243 (giugno-agosto)
    - Inverno: DOY 1-59 e 335-365 (gennaio + dicembre)
    - Esclusi: primavera e autunno (mancano dati 2020 sufficienti)
    
    La stagione è calcolata una sola volta per file (usa il DOY del primo pixel).
    """
    stagionalita_maschera = []

    mat_files = lista_file_mat_ordinati(data_directory)
    if not mat_files:
        print("Nessun file .mat trovato per la maschera stagionale.")
        return

    print(f"Maschera stagionale: trovati {len(mat_files)} file")

    for file_name in mat_files:
        file_path = os.path.join(data_directory, file_name)

        try:
            dati = loadmat(file_path)

            # Verifica disponibilità dati geografici
            if LATITUDE_KEY in dati and LONGITUDE_KEY in dati and dati[LATITUDE_KEY].size > 0 and dati[LONGITUDE_KEY].size > 0:
                latitudine = dati[LATITUDE_KEY].flatten()
                day_of_year = int(dati["dayOfYear"][0])

                # Determina la stagione dalla data del file
                for _ in latitudine:
                    stagione = get_season_by_day_of_year(day_of_year)
                    stagionalita_maschera.append(stagione)
            else:
                print(f"File {file_name} senza lat/lon validi")

        except Exception as errore:
            print(f"Errore nel caricamento di {file_name}: {errore}")

    output_path = os.path.join(mask_directory, SEASON_MASK_FILENAME)
    savemat(output_path, {"stagionalita_maschera": np.array(stagionalita_maschera)})
    print(f"Maschera stagionale salvata in {output_path}")


# --- Maschera terra/mare ---

def carica_sealand_mask(netcdf_file: str):
    """Carica latitudine, longitudine e maschera terra/mare da file NetCDF/HDF5."""
    with h5py.File(netcdf_file, "r") as f:
        latitudes = np.array(f["latitude"][:])
        longitudes = np.array(f["longitude"][:])
        sealand_mask = np.array(f["lsm"][0, :, :])
    return latitudes, longitudes, sealand_mask


def regridding(latitudes_dati, longitudes_dati, mask, latitudes_mask, longitudes_mask):
    """Interpola la maschera globale sulle coordinate dei dati usando nearest-neighbor."""
    grid_x, grid_y = np.meshgrid(longitudes_mask, latitudes_mask)
    points = np.array([grid_x.flatten(), grid_y.flatten()]).T
    mask_values = mask.flatten()
    return griddata(points, mask_values, (longitudes_dati, latitudes_dati), method="nearest")


def salva_mascherati(latitudes, longitudes, mare_mask, terra_mask, output_dir):
    """Salva le maschere mare e terra con le coordinate associate."""
    sio.savemat(os.path.join(output_dir, SEA_MASK_FILENAME), {"mask": mare_mask, "latitudes": latitudes, "longitudes": longitudes})
    sio.savemat(os.path.join(output_dir, LAND_MASK_FILENAME), {"mask": terra_mask, "latitudes": latitudes, "longitudes": longitudes})


def genera_maschera_terra_mare(data_directory: str, mask_directory: str, netcdf_file: str) -> None:
    """
    Genera le maschere terra/mare tramite regridding di una mappa globale.
    
    Procedura:
    1. Carica la mappa globale terra/mare dal file NetCDF (lsm)
    2. Estrae tutte le coordinate lat/lon dai dati di input
    3. Esegue interpolazione nearest-neighbor della mappa sui dati
    4. Crea due maschere: una per mare (0→1) e una per terra (1→1)
    """
    file_ordinati = lista_file_mat_ordinati(data_directory)
    if not file_ordinati:
        print("Nessun file .mat trovato per la maschera terra/mare.")
        return

    # Raccoglie tutte le coordinate geografiche dai file di input
    latitudes_list = []
    longitudes_list = []

    for filename in file_ordinati:
        filepath = os.path.join(data_directory, filename)
        dati_mat = sio.loadmat(filepath)
        latitudes_list.append(np.array(dati_mat[LATITUDE_KEY]).flatten())
        longitudes_list.append(np.array(dati_mat[LONGITUDE_KEY]).flatten())

    latitudes_dati = np.concatenate(latitudes_list)
    longitudes_dati = np.concatenate(longitudes_list)

    # Carica la mappa globale terra/mare
    latitudes_mask, longitudes_mask, sealand_mask = carica_sealand_mask(netcdf_file)
    # Interpola la mappa globale alle coordinate dei dati
    regridded_mask = regridding(latitudes_dati, longitudes_dati, sealand_mask, latitudes_mask, longitudes_mask)

    # Crea le due maschere binarie
    mare_mask = (regridded_mask == 0).astype(int)
    terra_mask = (regridded_mask == 1).astype(int)

    salva_mascherati(latitudes_dati, longitudes_dati, mare_mask, terra_mask, mask_directory)
    print("Maschere mare/terra salvate.")


# --- Esecuzione principale ---

if __name__ == "__main__":
    if not Path(DATA_DIRECTORY).exists():
        print(f"Cartella input non trovata: {DATA_DIRECTORY}")
        print("Impostare i percorsi nella sezione Configuration prima di eseguire.")
    elif not Path(SEA_LAND_NETCDF_FILE).exists():
        print(f"File maschera terra/mare non trovato: {SEA_LAND_NETCDF_FILE}")
    else:
        os.makedirs(MASK_DIRECTORY, exist_ok=True)

        genera_maschera_giorno_notte(DATA_DIRECTORY, MASK_DIRECTORY)
        genera_maschera_stagionale(DATA_DIRECTORY, MASK_DIRECTORY)
        genera_maschera_terra_mare(DATA_DIRECTORY, MASK_DIRECTORY, SEA_LAND_NETCDF_FILE)
