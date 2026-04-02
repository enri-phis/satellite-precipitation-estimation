"""
Download e elaborazione dati IMERG per validazione esterna della classificazione precipitazione.

Questo modulo supporta:
1. Download di file HDF5 da URL Earthdata (con autenticazione).
2. Ispezione della struttura HDF5 di un file campione.
3. Generazione di mappe geospaziali di precipitazione grezza per singolo file.
4. Aggregazione oraria di conteggi precipitazione per classe (no/pioggia, intensità).

IMERG fornisce il dato di precipitazione esterno (ground truth) per confrontare 
le predizioni della rete SEVIRI con misure da satellite indipendente.

Nota: credenziali Earthdata caricabili da variabili di ambiente (EARTHDATA_USERNAME, EARTHDATA_PASSWORD).
Tutti i percorsi sono placeholder da impostare prima dell'esecuzione.
"""

import os
import glob
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import h5py
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# =========================
# Configuration
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "imerg"

LINKS_FILE = DATA_ROOT / "imerg" / "links_imerg.txt"
DOWNLOAD_OUTPUT_DIR = DATA_ROOT / "imerg" / "downloads"
WGET_PATH = os.getenv("WGET_PATH", "wget")

# Credenziali Earthdata: usare variabili di ambiente, non hardcodare nel file.
EARTHDATA_USERNAME = os.getenv("EARTHDATA_USERNAME", "")
EARTHDATA_PASSWORD = os.getenv("EARTHDATA_PASSWORD", "")

IMERG_INPUT_DIR = DATA_ROOT / "imerg" / "input"
RAW_MAPS_OUTPUT_DIR = OUTPUT_ROOT / "raw_maps"
HOURLY_CLASS_MAPS_OUTPUT_DIR = OUTPUT_ROOT / "hourly_class_maps"

# Run switches
RUN_DOWNLOAD = False
RUN_HDF5_INSPECTION = False
RUN_RAW_MAPS = False
RUN_HOURLY_CLASS_MAPS = False

# File da ispezionare (solo se RUN_HDF5_INSPECTION=True)
SAMPLE_HDF5_FILE = IMERG_INPUT_DIR / "example.HDF5"
# =========================


def estrai_nome_file_da_url(url):
    """Estrae il nome file finale da un URL IMERG, anche quando è passato come query string."""
    parsed_url = urlparse(url)
    query = parse_qs(parsed_url.query)
    if "FILENAME" in query:
        filename = os.path.basename(query["FILENAME"][0])
    else:
        filename = os.path.basename(parsed_url.path)
    return unquote(filename)


def carica_url_da_file(links_file):
    """Legge una lista di URL da file di testo, ignorando righe vuote."""
    with open(links_file, "r", encoding="utf-8") as file_links:
        return [line.strip() for line in file_links if line.strip()]


def scarica_file_imerg(links_file, output_dir, wget_path, username, password):
    """
    Scarica file HDF5 IMERG da URL Earthdata con autenticazione.

    Legge lista di URL da file links_file, estrae il nome file da ciascun URL,
    controlla se già presenti localmente (skip se esiste), quindi scarica
    tramite wget con autenticazione HTTP basic (username/password).

    Retry automatico su errori di connessione (10 tentativi, timeout 30s).

    Args:
        links_file (str): percorso al file contenente uno URL per riga
        output_dir (str): directory di destinazione per i file HDF5 scaricati
        wget_path (str): percorso all'eseguibile wget
        username (str): username Earthdata
        password (str): password Earthdata

    Returns:
        None

    Output:
        File HDF5 scaricati in output_dir, uno per ciascun URL valido
    """
    if not os.path.exists(links_file):
        print(f"File link non trovato: {links_file}")
        return

    # Consente sia path assoluto sia comando disponibile nel PATH (es. "wget").
    if not os.path.exists(wget_path) and shutil.which(str(wget_path)) is None:
        print(f"Eseguibile wget non trovato: {wget_path}")
        return

    if not username or not password:
        print("Credenziali mancanti: impostare EARTHDATA_USERNAME e EARTHDATA_PASSWORD")
        return

    os.makedirs(output_dir, exist_ok=True)
    urls = carica_url_da_file(links_file)
    print(f"Trovati {len(urls)} URL da elaborare")

    for indice, url in enumerate(urls, start=1):
        nome_file = estrai_nome_file_da_url(url)
        percorso_output = os.path.join(output_dir, nome_file)

        if os.path.exists(percorso_output):
            print(f"[{indice}/{len(urls)}] Già presente: {nome_file}")
            continue

        print(f"[{indice}/{len(urls)}] Download: {nome_file}")
        comando = [
            wget_path,
            "--user", username,
            "--password", password,
            "-O", percorso_output,
            "--tries=10",
            "--timeout=30",
            "--retry-connrefused",
            url,
        ]

        try:
            subprocess.run(comando, check=True)
        except subprocess.CalledProcessError as errore:
            print(f"Errore nel download di {nome_file}: {errore}")


def ispeziona_hdf5(file_path):
    """Stampa la struttura interna di un file HDF5 per ispezione rapida."""
    if not os.path.exists(file_path):
        print(f"File HDF5 non trovato: {file_path}")
        return

    with h5py.File(file_path, "r") as file_hdf5:
        def print_h5(name, obj):
            print(name, obj)

        file_hdf5.visititems(print_h5)


def classifica_precipitazione(precip):
    """Classi: 0 secco, 1 debole, 2 moderata, 3 intensa, 4 molto intensa."""
    intervalli = [(0, 0.1), (0.1, 1), (1, 5), (5, 15), (15, np.inf)]
    classificata = np.full_like(precip, fill_value=-1, dtype=int)

    for classe, (basso, alto) in enumerate(intervalli):
        mask = (precip >= basso) & (precip < alto)
        classificata[mask] = classe

    return classificata


def estrai_datetime_da_nome_file(filename):
    """Estrae la data/ora di inizio acquisizione dal nome file IMERG."""
    date_str = filename.split("3IMERG.")[1][:8]
    start_part = [parte for parte in filename.split("-") if parte.startswith("S")][0]
    start_time = start_part[1:7]
    return datetime.strptime(date_str + start_time, "%Y%m%d%H%M%S")


def genera_mappe_precipitazione_grezza(input_dir, output_dir):
    """
    Genera mappe geospaziali (PNG) di precipitazione IMERG per ciascuna time slice di ogni file HDF5.

    Legge file HDF5 in input_dir, estrae lat, lon e array 3D precipitation (time x lat x lon),
    genera per ciascun time step una mappa cartografica (Cartopy PlateCarree).
    Include costa, confini, barra colorata in scala Blues (0-max mm/h).

    Args:
        input_dir (str): directory con file HDF5 scaricati (formato *.HDF5)
        output_dir (str): directory per salvataggio PNG (creata se non esiste)

    Returns:
        None

    Output:
        PNG per ciascun file e time slice: {filename_without_ext}_img{N}.png
        Esempio: 3B-MO.GPM.IMERG.v06B.20200101-S000000-E235959_img1.png
    """
    os.makedirs(output_dir, exist_ok=True)
    file_list = sorted(glob.glob(os.path.join(input_dir, "*.HDF5")))

    if not file_list:
        print("Nessun file HDF5 trovato per le mappe grezze")
        return

    for file_path in file_list:
        filename = os.path.basename(file_path)
        print(f"Elaborazione mappe grezze: {filename}")

        try:
            with h5py.File(file_path, "r") as file_hdf5:
                lat = file_hdf5["lat"][:]
                lon = file_hdf5["lon"][:]
                precip_all = file_hdf5["precipitation"][:]
        except Exception as errore:
            print(f"Errore nel file {filename}: {errore}")
            continue

        lon2d, lat2d = np.meshgrid(lon, lat, indexing="ij")
        lon_min, lon_max = lon.min(), lon.max()
        lat_min, lat_max = lat.min(), lat.max()

        date_str = filename.split("3IMERG.")[1][:8]
        date_fmt = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"

        for idx in range(precip_all.shape[0]):
            precip = precip_all[idx, :, :]

            parts = filename.split("-")
            start_time = parts[2][1:7]
            end_time = parts[3][1:7]
            start_fmt = f"{start_time[:2]}:{start_time[2:4]}:{start_time[4:]}"
            end_fmt = f"{end_time[:2]}:{end_time[2:4]}:{end_time[4:]}"

            fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": ccrs.PlateCarree()})
            vmin = 0
            vmax = max(precip.max(), 0.1)
            pcm = ax.pcolormesh(lon2d, lat2d, precip, shading="auto", cmap="Blues", vmin=vmin, vmax=vmax)

            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=":")
            ax.set_extent([lon_min, lon_max, lat_min, lat_max])
            ax.set_title(f"Precipitazione IMERG - {date_fmt} {start_fmt}-{end_fmt} UTC")

            cbar = fig.colorbar(pcm, ax=ax, orientation="vertical", label="Precipitazione (mm/h)", shrink=0.4)
            cbar.ax.tick_params(labelsize=10)
            cbar.ax.yaxis.label.set_size(12)

            out_name = os.path.splitext(filename)[0] + f"_img{idx + 1}.png"
            out_path = os.path.join(output_dir, out_name)
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()

    print(f"Mappe grezze completate: {output_dir}")


def genera_mappe_classi_orarie(input_dir, output_dir):
    """
    Aggrega per ora i dati IMERG e genera mappe per classe di precipitazione.

    Raggruppa file HDF5 per ora UTC, accumula precipitazione (somma pesata 0.5 per time step),
    classifica in 5 categorie (no pioggia, leggera, moderata, intensa, molto intensa),
    genera mappa cartografica con colori discreti per ciascun'ora.

    Args:
        input_dir (str): directory con file HDF5 IMERG (formato *.HDF5)
        output_dir (str): directory per salvataggio PNG (creata se non esiste)

    Returns:
        None

    Output:
        PNG per ciascun'ora UTC: {YYYY-MM-DD_HH00_classi.png}
        Mappa con colori: bianco=0, azzurro=1 (leggera), blu-scuro=2 (moderata), 
        arancio=3 (intensa), rosso=4 (molto intensa)
    """
    os.makedirs(output_dir, exist_ok=True)
    file_list = sorted(glob.glob(os.path.join(input_dir, "*.HDF5")))

    if not file_list:
        print("Nessun file HDF5 trovato per le mappe orarie")
        return

    gruppi_orari = defaultdict(list)
    lat = None
    lon = None

    for file_path in file_list:
        filename = os.path.basename(file_path)

        try:
            with h5py.File(file_path, "r") as file_hdf5:
                lat = file_hdf5["lat"][:]
                lon = file_hdf5["lon"][:]
                precip_all = file_hdf5["precipitation"][:]
        except Exception as errore:
            print(f"Errore nel file {filename}: {errore}")
            continue

        try:
            dt = estrai_datetime_da_nome_file(filename)
        except Exception:
            print(f"Nome file non riconosciuto, salto: {filename}")
            continue

        hourly_key = dt.replace(minute=0, second=0, microsecond=0)
        gruppi_orari[hourly_key].append((dt, precip_all[0, :, :]))

    if not gruppi_orari:
        print("Nessun gruppo orario valido trovato")
        return

    lon2d, lat2d = np.meshgrid(lon, lat, indexing="ij")
    cmap = mcolors.ListedColormap(["white", "lightblue", "dodgerblue", "orange", "red"])
    bounds = [0, 1, 2, 3, 4, 5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    for hour_key in sorted(gruppi_orari.keys()):
        entries = sorted(gruppi_orari[hour_key], key=lambda elem: elem[0])
        accum_1h = np.zeros_like(entries[0][1], dtype=float)

        for _, precip in entries:
            accum_1h += precip * 0.5

        if len(entries) == 1:
            accum_1h *= 2

        classes = classifica_precipitazione(accum_1h)

        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": ccrs.PlateCarree()})
        pcm = ax.pcolormesh(
            lon2d,
            lat2d,
            classes,
            cmap=cmap,
            norm=norm,
            shading="nearest",
            transform=ccrs.PlateCarree(),
        )

        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()])
        ax.set_title(f"Classi precipitazione IMERG - {hour_key:%Y-%m-%d %H:00} UTC")

        cbar = fig.colorbar(pcm, ax=ax, orientation="vertical", shrink=0.7, ticks=[0.5, 1.5, 2.5, 3.5, 4.5])
        cbar.ax.set_yticklabels([
            "Secco (0-0.1)",
            "Debole (0.1-1)",
            "Moderata (1-5)",
            "Intensa (5-15)",
            "Molto intensa (>15)",
        ])

        out_path = os.path.join(output_dir, f"imerg_classi_{hour_key:%Y%m%d_%H}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

        print(f"Mappa oraria salvata: {out_path}")

    print(f"Mappe orarie completate: {output_dir}")


if __name__ == "__main__":
    if RUN_DOWNLOAD:
        scarica_file_imerg(
            LINKS_FILE,
            DOWNLOAD_OUTPUT_DIR,
            WGET_PATH,
            EARTHDATA_USERNAME,
            EARTHDATA_PASSWORD,
        )

    if RUN_HDF5_INSPECTION:
        ispeziona_hdf5(SAMPLE_HDF5_FILE)

    if RUN_RAW_MAPS:
        genera_mappe_precipitazione_grezza(IMERG_INPUT_DIR, RAW_MAPS_OUTPUT_DIR)

    if RUN_HOURLY_CLASS_MAPS:
        genera_mappe_classi_orarie(IMERG_INPUT_DIR, HOURLY_CLASS_MAPS_OUTPUT_DIR)
