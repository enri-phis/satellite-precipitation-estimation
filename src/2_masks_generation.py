"""
Script for generating geographic and temporal masks.

This module generates essential masks for satellite data filtering:

1. **Day/Night Mask**: physical criteria based on sunrise/sunset
   calculated with precise geographic coordinates (ephem). Includes temporal hysteresis
   (±2 hours from sunrise/sunset) to avoid instability at light/shadow transitions.

2. **Seasonal Mask**: divides data between summer season
   (spring/summer, DOY 152-243) and winter (fall/winter, DOY 335-59).
   Enables seasonal comparison of precipitation.

3. **Land/Sea Mask**: regrids a global land/sea map
   (from NetCDF file) to input data coordinates via nearest-neighbor interpolation.

They are used for:
- Filtering nighttime or unreliable data (night removal)
- Stratifying analysis by season
- Differentiated analysis on ocean vs land

Output: MATLAB .mat files containing masks (0/1 arrays) with dimensions
matching input data.

Configuration:
- DATA_DIRECTORY: folder with filtered .mat files (output of step 1)
- MASK_DIRECTORY: output folder for masks
- SEA_LAND_NETCDF_FILE: global land/sea map (NetCDF)
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
    """Extract day of year and time from filename for chronological ordering."""
    match = re.search(r"DOY(\d+)_TIME(\d+)", nome_file)
    if match:
        doy = int(match.group(1))
        time = int(match.group(2))
        return doy, time
    return float("inf"), float("inf")


def lista_file_mat_ordinati(data_directory: str):
    """Return `.mat` files sorted by day of year and acquisition time."""
    return sorted([f for f in os.listdir(data_directory) if f.endswith(".mat")], key=estrai_doy_time)


# --- Day/Night Mask ---

def calcola_fuso_orario(longitudine: float) -> int:
    """Approximate local time zone from longitude."""
    return round(longitudine / 15)


def get_sunrise_sunset(dt_utc: datetime, latitude: float, longitude: float):
    """Calculate local sunrise and sunset for a date and geographic coordinate."""
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
    """Classify a local instant as daytime using a conservative window between sunrise and sunset."""
    daytime_start = sunrise + timedelta(hours=2)
    daytime_end = sunset - timedelta(hours=2)
    return daytime_start <= local_datetime <= daytime_end


def genera_maschera_giorno_notte(data_directory: str, mask_directory: str) -> None:
    """
    Generate day/night mask based on precise astronomical calculations.
    
    For each pixel in each image:
    - Extract temporal metadata (day of year, UTC hour)
    - Convert hour to local time zone using longitude
    - Calculate sunrise/sunset with ephem library (astronomical ephemerides)
    - Assign value 1 (day) if in interval [sunrise+2h, sunset-2h]
    - Otherwise assign 0 (night)
    
    The ±2 hour hysteresis reduces diffuse illumination effects at twilight.
    """
    giorno_maschera = []
    notte_maschera = []

    mat_files = lista_file_mat_ordinati(data_directory)
    if not mat_files:
        print("No .mat files found for day/night mask.")
        return

    print(f"Day/night mask: found {len(mat_files)} files")

    for idx, file_name in enumerate(mat_files, start=1):
        file_path = os.path.join(data_directory, file_name)
        print(f"[{idx}/{len(mat_files)}] Processing {file_name}")

        try:
            dati = loadmat(file_path)

            if "dayOfYear" not in dati or "iTmOfDay" not in dati:
                print(f"Skipped {file_name}: temporal metadata missing")
                continue

            day_of_year = int(dati["dayOfYear"].flatten()[0])
            minute_of_day = int(dati["iTmOfDay"].flatten()[0]) * 15
            dt_utc = datetime(2020, 1, 1) + timedelta(days=day_of_year - 1, minutes=minute_of_day)

            if LATITUDE_KEY not in dati or LONGITUDE_KEY not in dati:
                print(f"Skipped {file_name}: latitude/longitude missing")
                continue

            latitudes = dati[LATITUDE_KEY].flatten()
            longitudes = dati[LONGITUDE_KEY].flatten()

            for lat, lon in zip(latitudes, longitudes):
                # Calculate time zone from longitude and convert UTC hour to local
                fuso = calcola_fuso_orario(lon)
                dt_local = dt_utc + timedelta(hours=fuso)
                # Get sunrise and sunset for specified position and date
                alba, tramonto = get_sunrise_sunset(dt_utc, lat, lon)

                # Classify pixel as day or night
                if is_daytime(dt_local, alba, tramonto):
                    giorno_maschera.append(1)
                    notte_maschera.append(0)
                else:
                    giorno_maschera.append(0)
                    notte_maschera.append(1)

        except Exception as errore:
            print(f"Error in file {file_name}: {errore}")

    savemat(os.path.join(mask_directory, DAY_MASK_FILENAME), {"giorno_maschera": np.array(giorno_maschera)})
    savemat(os.path.join(mask_directory, NIGHT_MASK_FILENAME), {"notte_maschera": np.array(notte_maschera)})
    print(f"Day/night mask saved. Total points: {len(giorno_maschera)}")


# --- Seasonal Mask ---

def get_season_by_day_of_year(day_of_year: int) -> int:
    """Assign climatological season used in project: summer, winter, or excluded period."""
    if 335 <= day_of_year <= 365 or 1 <= day_of_year <= 59:
        return 1  # winter
    if 152 <= day_of_year <= 243:
        return 0  # summer
    return 2  # spring/autumn (excluded)


def genera_maschera_stagionale(data_directory: str, mask_directory: str) -> None:
    """
    Generate seasonal mask based on day-of-year.
    
    - Summer: DOY 152-243 (June-August)
    - Winter: DOY 1-59 and 335-365 (January + December)
    - Excluded: spring and autumn (insufficient 2020 data)
    
    Season is calculated once per file (uses DOY of first pixel).
    """
    stagionalita_maschera = []

    mat_files = lista_file_mat_ordinati(data_directory)
    if not mat_files:
        print("No .mat files found for seasonal mask.")
        return

    print(f"Seasonal mask: found {len(mat_files)} files")

    for file_name in mat_files:
        file_path = os.path.join(data_directory, file_name)

        try:
            dati = loadmat(file_path)

            # Check availability of geographic data
            if LATITUDE_KEY in dati and LONGITUDE_KEY in dati and dati[LATITUDE_KEY].size > 0 and dati[LONGITUDE_KEY].size > 0:
                latitudine = dati[LATITUDE_KEY].flatten()
                day_of_year = int(dati["dayOfYear"][0])

                # Determine season from file date
                for _ in latitudine:
                    stagione = get_season_by_day_of_year(day_of_year)
                    stagionalita_maschera.append(stagione)
            else:
                print(f"File {file_name} without valid lat/lon")

        except Exception as errore:
            print(f"Error loading {file_name}: {errore}")

    output_path = os.path.join(mask_directory, SEASON_MASK_FILENAME)
    savemat(output_path, {"stagionalita_maschera": np.array(stagionalita_maschera)})
    print(f"Seasonal mask saved to {output_path}")


# --- Land/Sea Mask ---

def carica_sealand_mask(netcdf_file: str):
    """Load latitude, longitude, and land/sea mask from NetCDF/HDF5 file."""
    with h5py.File(netcdf_file, "r") as f:
        latitudes = np.array(f["latitude"][:])
        longitudes = np.array(f["longitude"][:])
        sealand_mask = np.array(f["lsm"][0, :, :])
    return latitudes, longitudes, sealand_mask


def regridding(latitudes_dati, longitudes_dati, mask, latitudes_mask, longitudes_mask):
    """Interpolate global mask onto data coordinates using nearest-neighbor."""
    grid_x, grid_y = np.meshgrid(longitudes_mask, latitudes_mask)
    points = np.array([grid_x.flatten(), grid_y.flatten()]).T
    mask_values = mask.flatten()
    return griddata(points, mask_values, (longitudes_dati, latitudes_dati), method="nearest")


def salva_mascherati(latitudes, longitudes, mare_mask, terra_mask, output_dir):
    """Save sea and land masks with associated coordinates."""
    sio.savemat(os.path.join(output_dir, SEA_MASK_FILENAME), {"mask": mare_mask, "latitudes": latitudes, "longitudes": longitudes})
    sio.savemat(os.path.join(output_dir, LAND_MASK_FILENAME), {"mask": terra_mask, "latitudes": latitudes, "longitudes": longitudes})


def genera_maschera_terra_mare(data_directory: str, mask_directory: str, netcdf_file: str) -> None:
    """
    Generate land/sea masks via regridding a global map.
    
    Procedure:
    1. Load global land/sea map from NetCDF file (lsm)
    2. Extract all lat/lon coordinates from input data
    3. Perform nearest-neighbor interpolation of map onto data
    4. Create two masks: one for sea (0→1) and one for land (1→1)
    """
    file_ordinati = lista_file_mat_ordinati(data_directory)
    if not file_ordinati:
        print("No .mat files found for land/sea mask.")
        return

    # Collect all geographic coordinates from input files
    latitudes_list = []
    longitudes_list = []

    for filename in file_ordinati:
        filepath = os.path.join(data_directory, filename)
        dati_mat = sio.loadmat(filepath)
        latitudes_list.append(np.array(dati_mat[LATITUDE_KEY]).flatten())
        longitudes_list.append(np.array(dati_mat[LONGITUDE_KEY]).flatten())

    latitudes_dati = np.concatenate(latitudes_list)
    longitudes_dati = np.concatenate(longitudes_list)

    # Load global land/sea map
    latitudes_mask, longitudes_mask, sealand_mask = carica_sealand_mask(netcdf_file)
    # Interpolate global map to data coordinates
    regridded_mask = regridding(latitudes_dati, longitudes_dati, sealand_mask, latitudes_mask, longitudes_mask)

    # Create two binary masks
    mare_mask = (regridded_mask == 0).astype(int)
    terra_mask = (regridded_mask == 1).astype(int)

    salva_mascherati(latitudes_dati, longitudes_dati, mare_mask, terra_mask, mask_directory)
    print("Land/sea masks saved.")


# --- Main Execution ---
