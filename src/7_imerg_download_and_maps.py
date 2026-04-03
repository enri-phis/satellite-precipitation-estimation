"""
Download and processing of IMERG data for external precipitation classification validation.

This module supports:
1. Downloading HDF5 files from Earthdata URLs (with authentication).
2. Inspecting the HDF5 structure of a sample file.
3. Generating geospatial raw precipitation maps for each file.
4. Hourly aggregation of precipitation class counts (no-rain/rain, intensity).

IMERG provides external precipitation data (ground truth) to compare
SEVIRI network predictions with independent satellite measurements.

Note: Earthdata credentials can be loaded from environment variables (EARTHDATA_USERNAME, EARTHDATA_PASSWORD).
All paths are placeholders and must be set before execution.
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

# Earthdata credentials: use environment variables, do not hardcode in the file.
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

# File to inspect (only if RUN_HDF5_INSPECTION=True)
SAMPLE_HDF5_FILE = IMERG_INPUT_DIR / "example.HDF5"
# =========================


def estrai_nome_file_da_url(url):
    """Extracts the final filename from an IMERG URL, even when passed as a query string."""
    parsed_url = urlparse(url)
    query = parse_qs(parsed_url.query)
    if "FILENAME" in query:
        filename = os.path.basename(query["FILENAME"][0])
    else:
        filename = os.path.basename(parsed_url.path)
    return unquote(filename)


def carica_url_da_file(links_file):
    """Reads a list of URLs from a text file, ignoring empty lines."""
    with open(links_file, "r", encoding="utf-8") as file_links:
        return [line.strip() for line in file_links if line.strip()]


def scarica_file_imerg(links_file, output_dir, wget_path, username, password):
    """
    Downloads IMERG HDF5 files from Earthdata URLs with authentication.

    Reads a list of URLs from links_file, extracts the filename from each URL,
    checks if already present locally (skip if exists), then downloads
    via wget with HTTP basic authentication (username/password).

    Automatic retry on connection errors (10 attempts, 30s timeout).

    Args:
        links_file (str): path to the file containing one URL per line
        output_dir (str): destination directory for downloaded HDF5 files
        wget_path (str): path to the wget executable
        username (str): Earthdata username
        password (str): Earthdata password

    Returns:
        None

    Output:
        HDF5 files downloaded into output_dir, one for each valid URL
    """
    if not os.path.exists(links_file):
        print(f"Link file not found: {links_file}")
        return

    # Allows both absolute path and command available in PATH (e.g., "wget").
    if not os.path.exists(wget_path) and shutil.which(str(wget_path)) is None:
        print(f"wget executable not found: {wget_path}")
        return

    if not username or not password:
        print("Missing credentials: set EARTHDATA_USERNAME and EARTHDATA_PASSWORD")
        return

    os.makedirs(output_dir, exist_ok=True)
    urls = carica_url_da_file(links_file)
    print(f"Found {len(urls)} URLs to process")

    for indice, url in enumerate(urls, start=1):
        nome_file = estrai_nome_file_da_url(url)
        percorso_output = os.path.join(output_dir, nome_file)

        if os.path.exists(percorso_output):
            print(f"[{indice}/{len(urls)}] Already present: {nome_file}")
            continue

        print(f"[{indice}/{len(urls)}] Downloading: {nome_file}")
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
            print(f"Error downloading {nome_file}: {errore}")


def ispeziona_hdf5(file_path):
    """Prints the internal structure of an HDF5 file for quick inspection."""
    if not os.path.exists(file_path):
        print(f"HDF5 file not found: {file_path}")
        return

    with h5py.File(file_path, "r") as file_hdf5:
        def print_h5(name, obj):
            print(name, obj)

        file_hdf5.visititems(print_h5)


def classifica_precipitazione(precip):
    """Classes: 0 dry, 1 light, 2 moderate, 3 intense, 4 very intense."""
    intervalli = [(0, 0.1), (0.1, 1), (1, 5), (5, 15), (15, np.inf)]
    classificata = np.full_like(precip, fill_value=-1, dtype=int)

    for classe, (basso, alto) in enumerate(intervalli):
        mask = (precip >= basso) & (precip < alto)
        classificata[mask] = classe

    return classificata


def estrai_datetime_da_nome_file(filename):
    """Extracts acquisition start date/time from an IMERG filename."""
    date_str = filename.split("3IMERG.")[1][:8]
    start_part = [parte for parte in filename.split("-") if parte.startswith("S")][0]
    start_time = start_part[1:7]
    return datetime.strptime(date_str + start_time, "%Y%m%d%H%M%S")


def genera_mappe_precipitazione_grezza(input_dir, output_dir):
    """
    Generates geospatial IMERG precipitation maps (PNG) for each time slice of each HDF5 file.

    Reads HDF5 files in input_dir, extracts lat, lon, and 3D precipitation array (time x lat x lon),
    then generates one cartographic map per time step (Cartopy PlateCarree).
    Includes coastline, borders, and colorbar in Blues scale (0-max mm/h).

    Args:
        input_dir (str): directory with downloaded HDF5 files (*.HDF5 format)
        output_dir (str): directory to save PNG files (created if missing)

    Returns:
        None

    Output:
        PNG for each file and time slice: {filename_without_ext}_img{N}.png
        Example: 3B-MO.GPM.IMERG.v06B.20200101-S000000-E235959_img1.png
    """
    os.makedirs(output_dir, exist_ok=True)
    file_list = sorted(glob.glob(os.path.join(input_dir, "*.HDF5")))

    if not file_list:
        print("No HDF5 files found for raw maps")
        return

    for file_path in file_list:
        filename = os.path.basename(file_path)
        print(f"Processing raw maps: {filename}")

        try:
            with h5py.File(file_path, "r") as file_hdf5:
                lat = file_hdf5["lat"][:]
                lon = file_hdf5["lon"][:]
                precip_all = file_hdf5["precipitation"][:]
        except Exception as errore:
            print(f"Error in file {filename}: {errore}")
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
            ax.set_title(f"IMERG Precipitation - {date_fmt} {start_fmt}-{end_fmt} UTC")

            cbar = fig.colorbar(pcm, ax=ax, orientation="vertical", label="Precipitation (mm/h)", shrink=0.4)
            cbar.ax.tick_params(labelsize=10)
            cbar.ax.yaxis.label.set_size(12)

            out_name = os.path.splitext(filename)[0] + f"_img{idx + 1}.png"
            out_path = os.path.join(output_dir, out_name)
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()

    print(f"Raw maps completed: {output_dir}")


def genera_mappe_classi_orarie(input_dir, output_dir):
    """
    Aggregates IMERG data hourly and generates precipitation class maps.

    Groups HDF5 files by UTC hour, accumulates precipitation (0.5 weighted sum per time step),
    classifies into 5 categories (no rain, light, moderate, intense, very intense),
    generates a cartographic map with discrete colors for each hour.

    Args:
        input_dir (str): directory with IMERG HDF5 files (*.HDF5 format)
        output_dir (str): directory to save PNG files (created if missing)

    Returns:
        None

    Output:
        PNG for each UTC hour: {YYYY-MM-DD_HH00_classi.png}
        Map with colors: white=0, light blue=1 (light), dark blue=2 (moderate),
        orange=3 (intense), red=4 (very intense)
    """
    os.makedirs(output_dir, exist_ok=True)
    file_list = sorted(glob.glob(os.path.join(input_dir, "*.HDF5")))

    if not file_list:
        print("No HDF5 files found for hourly maps")
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
            print(f"Error in file {filename}: {errore}")
            continue

        try:
            dt = estrai_datetime_da_nome_file(filename)
        except Exception:
            print(f"Unrecognized filename, skipping: {filename}")
            continue

        hourly_key = dt.replace(minute=0, second=0, microsecond=0)
        gruppi_orari[hourly_key].append((dt, precip_all[0, :, :]))

    if not gruppi_orari:
        print("No valid hourly group found")
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
        ax.set_title(f"IMERG Precipitation Classes - {hour_key:%Y-%m-%d %H:00} UTC")

        cbar = fig.colorbar(pcm, ax=ax, orientation="vertical", shrink=0.7, ticks=[0.5, 1.5, 2.5, 3.5, 4.5])
        cbar.ax.set_yticklabels([
            "Dry (0-0.1)",
            "Light (0.1-1)",
            "Moderate (1-5)",
            "Intense (5-15)",
            "Very intense (>15)",
        ])

        out_path = os.path.join(output_dir, f"imerg_classi_{hour_key:%Y%m%d_%H}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

        print(f"Hourly map saved: {out_path}")

    print(f"Hourly maps completed: {output_dir}")


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