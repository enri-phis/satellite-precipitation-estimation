"""
Script for loading and geographic filtering of satellite data.

This module handles loading raw satellite files in HDF5/MATLAB v7.3 format
and applies geographic filtering to select only images within the area of interest
defined by latitude and longitude intervals.

Workflow:
1. Reads raw .mat files (HDF5 v7.3) from input folder.
2. For each file, extracts 11 MSG spectral bands, DPR precipitation,
   geographic grids (lat/lon), and temporal metadata.
3. Applies geographic filtering to discard images outside the area of interest.
4. Saves filtered images as individual .mat files in a normalized output folder.

This step is the first in the preprocessing pipeline: it prepares raw data
before mask calculation and feature extraction.

Configuration:
- RAW_DATA_DIR: folder containing raw .mat files (HDF5 v7.3)
- FILTERED_OUTPUT_DIR: folder where to save filtered files
- LAT_RANGE, LON_RANGE: geographic intervals of interest
- GROUP_SIZE: number of images loaded at once (to limit RAM)
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
# Project paths: modify only this section to adapt the script to your environment.
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
    Load a block of images from an HDF5 v7.3 file (.mat).

    Each image consists of:
    - 11 MSG spectral bands (infrared and visible channels)
    - 1 DPR precipitation field
    - 1 latitude grid, 1 longitude grid
    - Temporal metadata (day of year, UTC hour)
    
    Args:
        directory (str): input folder containing the HDF5/MATLAB file.
        nome_file (str): name of the `.mat` file to read.
        start_idx (int): initial index of the image block.
        end_idx (int): final index of the image block.

    Returns:
        Dict[str, Dict[Any, np.ndarray]]: dictionary containing loaded data
        for each image in the block.
    """
    dati = {}
    percorso_file = os.path.join(directory, nome_file)
    print(f"Loading {nome_file}: images {start_idx + 1}-{end_idx}")

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

    print(f"Loaded {len(dati)} images")
    return dati


def filtro_geografico(
    dati: Dict[str, Dict[Any, np.ndarray]],
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
) -> Dict[str, Dict[Any, np.ndarray]]:
    """
    Filter images based on geographic area of interest.

    Keeps only images whose geographic bounding box intersects the area
    defined by lat_range and lon_range. Images outside the area are discarded.

    Args:
        dati (Dict[str, Dict[Any, np.ndarray]]): images loaded from source file.
        lat_range (Tuple[float, float]): allowed latitude interval.
        lon_range (Tuple[float, float]): allowed longitude interval.

    Returns:
        Dict[str, Dict[Any, np.ndarray]]: only images intersecting
        the geographic area of interest.
    """
    print(f"Geographic check on {len(dati)} images")
    dati_filtrati = {}

    for nome_immagine, dati_immagine in dati.items():
        # Check availability of geographic grids
        if LATITUDE_KEY not in dati_immagine or LONGITUDE_KEY not in dati_immagine:
            print(f"Skipped {nome_immagine}: latitude or longitude not available")
            continue

        latitudine = np.array(dati_immagine[LATITUDE_KEY])
        longitudine = np.array(dati_immagine[LONGITUDE_KEY])

        # Check dimension consistency
        if latitudine.shape != longitudine.shape:
            print(f"Skipped {nome_immagine}: dimension mismatch between latitude and longitude")
            continue

        # Create boolean masks for geographic filtering
        mask_lat = (lat_range[0] <= latitudine) & (latitudine <= lat_range[1])
        mask_lon = (lon_range[0] <= longitudine) & (longitudine <= lon_range[1])
        mask     = mask_lat & mask_lon

        # Keep only images with at least one pixel in the area
        if not mask.any():
            continue

        dati_filtrati[nome_immagine] = dati_immagine

    print(f"Accepted {len(dati_filtrati)} images after geographic filtering")
    return dati_filtrati


def salva_dati_filtrati(
    dati: Dict[Any, np.ndarray],
    output_dir: str,
    nome_file: str,
    gia_visti: Set[Tuple[int, int]],
) -> None:
    """
    Save a filtered image in `.mat` format, avoiding temporal duplicates.

    The uniqueness key is defined by the tuple `(dayOfYear, iTmOfDay)`.
    If an image with the same tuple has already been saved, it is ignored.

    Args:
        dati (Dict[Any, np.ndarray]): image content to save.
        output_dir (str): destination folder.
        nome_file (str): name of source file, used to construct output name.
        gia_visti (Set[Tuple[int, int]]): set of temporal tuples already exported.
    """
    os.makedirs(output_dir, exist_ok=True)

    giorno = int(np.squeeze(dati["dayOfYear"]))
    ora = int(np.squeeze(dati["iTmOfDay"]))
    chiave = (giorno, ora)

    if chiave in gia_visti:
        print(f"Duplicate image ignored: day {giorno}, hour {ora}")
        return

    gia_visti.add(chiave)
    nome_base       = f"{os.path.splitext(nome_file)[0]}_DOY{giorno}_TIME{ora}"
    output_file_mat = os.path.join(output_dir, f"{nome_base}.mat")
    dati_mat        = {str(key): array for key, array in dati.items()}

    savemat(output_file_mat, dati_mat)
    print(f"Saved {output_file_mat}")


def processa_gruppi(
    directory: str,
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
    output_dir: str,
    group_size: int = 100,
) -> Dict[str, Dict[Any, np.ndarray]]:
    """
    Process all files in directory in blocks to manage memory.

    For each HDF5 file:
    1. Load blocks of images (to limit RAM usage)
    2. Apply geographic filtering
    3. Save filtered images as separate .mat files

    Args:
        directory (str): input folder with raw `.mat` files.
        lat_range (Tuple[float, float]): latitude interval.
        lon_range (Tuple[float, float]): longitude interval.
        output_dir (str): output folder.
        group_size (int): number of images to process per block.

    Returns:
        Dict[str, Dict[Any, np.ndarray]]: filtered images collected during
        processing, useful for quick checks or final statistics.
    """
    gia_visti = set()

    print(f"Preparing output folder: {output_dir}")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    if not os.path.exists(directory):
        print(f"Folder not found: {directory}")
        return {}

    # List input files
    file_list = sorted(f for f in os.listdir(directory) if f.endswith(".mat"))
    filtered_data = {}
    print(f"Found {len(file_list)} input files")

    # Process each file in blocks
    for file_name in file_list:
        percorso_file = os.path.join(directory, file_name)
        with h5py.File(percorso_file, "r") as file_hdf5:
            numero_immagini = file_hdf5["msgData"].shape[0]

        print(f"Processing {file_name} ({numero_immagini} images)")

        # Load and process successive blocks
        for start_idx in range(0, numero_immagini, group_size):
            end_idx = start_idx + group_size
            dati_blocco = carica_blocchi_dati_v73(directory, file_name, start_idx, end_idx)
            dati_filtrati = filtro_geografico(dati_blocco, lat_range, lon_range)

            # Save each filtered image as separate file
            for _, dati in dati_filtrati.items():
                salva_dati_filtrati(dati, output_dir, file_name, gia_visti)

            filtered_data.update(dati_filtrati)

    print(f"Total filtered images: {len(filtered_data)}")
    return filtered_data
