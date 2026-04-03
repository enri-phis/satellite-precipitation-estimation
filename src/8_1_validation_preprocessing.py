"""
Final SEVIRI validation preprocessing.

This module collects the preliminary steps required for validation:
- conversion of native SEVIRI files into NetCDF cropped to the study area;
- generation of channel 9 diagnostic maps;
- creation of seasonal, land/sea, and day/night masks;
- export of channels and masks in pickle format.

The operational behavior remains that of the original validation file, but the
preprocessing parts are isolated to improve readability and maintainability.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
import re
import warnings
from datetime import datetime, timedelta

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import xarray as xr
from pyproj import Proj
from satpy import Scene
from scipy.interpolate import griddata
from scipy.io import savemat

warnings.filterwarnings("ignore")

# =========================
# Configuration
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "validation"

NAT_ZIP_INPUT_DIR = DATA_ROOT / "nat_zip_input"
NETCDF_OUTPUT_DIR = DATA_ROOT / "netcdf_output"
CH9_MAP_OUTPUT_DIR = OUTPUT_ROOT / "ch9_maps"
MASK_OUTPUT_DIR = DATA_ROOT / "masks"
PROCESSED_PICKLE_DIR = DATA_ROOT / "processed_pickle_output"
SEALAND_NETCDF_FILE = DATA_ROOT / "support" / "sea_land_mask.nc"

# Run switches
RUN_NAT_TO_NETCDF = False
RUN_CH9_MAPS = False
RUN_MASKS = False
RUN_EXPORT_PICKLES = False

SEVIRI_CHANNELS = [
    "VIS006", "VIS008", "IR_016", "IR_039", "WV_062", "WV_073",
    "IR_087", "IR_097", "IR_108", "IR_120", "IR_134",
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def compute_latlon_actual(scn: Scene):
    """Compute the real geographic coordinates of the SEVIRI scene from the geostationary projection."""
    vis006 = scn["VIS006"]
    params = vis006.attrs["orbital_parameters"]

    proj_geo = Proj(
        proj="geos",
        h=params["satellite_actual_altitude"],
        lon_0=params["satellite_actual_longitude"],
        lat_0=params["satellite_actual_latitude"],
        sweep="y",
        a=6378169.0,
        b=6356583.8,
    )

    x = vis006.coords["x"].values
    y = vis006.coords["y"].values
    x_mesh, y_mesh = np.meshgrid(x, y)
    lon, lat = proj_geo(x_mesh, y_mesh, inverse=True)
    lon = np.where(np.isinf(lon), np.nan, lon).astype("float32")
    lat = np.where(np.isinf(lat), np.nan, lat).astype("float32")
    return lon, lat


def nat2net_crop_actual(source_dir, zip_name, out_dir, lat_min, lat_max, lon_min, lon_max, bbox=None):
    """Convert native SEVIRI ZIP files to NetCDF and apply geographic cropping."""
    import zipfile

    zip_path = os.path.join(source_dir, zip_name)
    with zipfile.ZipFile(zip_path) as archive:
        nat_name = next((name for name in archive.namelist() if name.endswith(".nat")), None)
        if nat_name is None:
            print(f"SKIP: no .nat in {zip_name}")
            return None, bbox
        archive.extract(nat_name, source_dir)

    nat_path = os.path.join(source_dir, nat_name)
    print(f"Processing {nat_path}")

    scene = Scene([nat_path], reader="seviri_l1b_native", reader_kwargs={"fill_disk": False})
    scene.load(SEVIRI_CHANNELS)

    lon, lat = compute_latlon_actual(scene)
    data_vars = {f"channel_{idx + 1}": (("y", "x"), scene[channel].values) for idx, channel in enumerate(SEVIRI_CHANNELS)}
    dataset = xr.Dataset(data_vars, coords={"lat": (("y", "x"), lat), "lon": (("y", "x"), lon)})

    if bbox is None:
        mask = (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            print(f"No pixels in bounding box for {zip_name}")
            os.remove(nat_path)
            return None, None
        bbox = (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)

    y0, y1, x0, x1 = bbox
    cropped = dataset.isel(y=slice(y0, y1), x=slice(x0, x1))

    match = re.search(r"(\d{14})", zip_name)
    if match:
        timestamp = datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
        cropped = cropped.assign(time=xr.DataArray(timestamp))

    ensure_dir(out_dir)
    out_file = os.path.join(out_dir, zip_name.replace(".zip", "_cropped_actual.nc"))
    cropped.to_netcdf(out_file)

    dataset.close()
    cropped.close()
    os.remove(nat_path)
    return out_file, bbox


def batch_convert_crop_actual(source_dir, out_dir, lat_min=43.4, lat_max=45.8, lon_min=9.15, lon_max=13.2):
    """Sequentially convert all native ZIP files in the source folder."""
    ensure_dir(out_dir)
    bbox = None
    for file_name in sorted(os.listdir(source_dir)):
        if file_name.endswith(".zip"):
            _, bbox = nat2net_crop_actual(source_dir, file_name, out_dir, lat_min, lat_max, lon_min, lon_max, bbox)


def generate_ch9_maps(input_dir, output_dir):
    """Generate geospatial channel 9 maps from preprocessed NetCDF files."""
    ensure_dir(output_dir)
    file_list = sorted(f for f in os.listdir(input_dir) if f.endswith(".nc"))

    for file_name in file_list:
        file_path = os.path.join(input_dir, file_name)
        try:
            dataset = xr.open_dataset(file_path)
            if "channel_9" not in dataset.variables:
                dataset.close()
                continue

            channel_9 = np.flip(dataset["channel_9"].values, axis=1)
            lat = dataset["lat"].values
            lon = np.fliplr(dataset["lon"].values)
            dataset.close()

            fig = plt.figure(figsize=(8, 6))
            ax = plt.axes(projection=ccrs.PlateCarree())
            image = ax.imshow(
                channel_9,
                origin="lower",
                cmap="Greys",
                extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                transform=ccrs.PlateCarree(),
            )
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=":")
            ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)
            plt.colorbar(image, ax=ax, orientation="vertical", pad=0.04, shrink=0.5)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

            out_path = os.path.join(output_dir, os.path.splitext(file_name)[0] + "_CH9.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
            plt.close()
        except Exception as error:
            print(f"Error generating CH9 map for {file_name}: {error}")


def get_season_by_day_of_year(day_of_year):
    if 335 <= day_of_year <= 365 or 1 <= day_of_year <= 59:
        return 1
    if 152 <= day_of_year <= 243:
        return 0
    return 2


def generate_season_mask(nc_dir, mask_dir):
    """Generate the seasonal mask for all NetCDF files."""
    ensure_dir(mask_dir)
    season_mask = []

    for file_name in sorted(f for f in os.listdir(nc_dir) if f.endswith(".nc")):
        dataset = xr.open_dataset(os.path.join(nc_dir, file_name))
        lat = dataset["lat"].values.flatten()
        timestamp = pd.to_datetime(str(dataset["time"].values))
        season = get_season_by_day_of_year(timestamp.dayofyear)
        season_mask.extend([season] * len(lat))
        dataset.close()

    savemat(os.path.join(mask_dir, "stagionalita_maschera_estate_inverno.mat"), {"stagionalita_maschera": np.array(season_mask)})


def load_sealand_mask(netcdf_file):
    """Load the global land/sea mask from a NetCDF file."""
    dataset = xr.open_dataset(netcdf_file)
    latitudes = dataset["latitude"].values
    longitudes = dataset["longitude"].values
    sealand_mask = dataset["lsm"].values[0, :, :]
    dataset.close()
    return latitudes, longitudes, sealand_mask


def generate_sealand_masks(nc_dir, mask_dir, sealand_file):
    """Regrid land/sea masks onto the coordinates of preprocessed SEVIRI data."""
    ensure_dir(mask_dir)
    lat_list = []
    lon_list = []

    for file_name in sorted(f for f in os.listdir(nc_dir) if f.endswith(".nc")):
        dataset = xr.open_dataset(os.path.join(nc_dir, file_name))
        lat_list.append(dataset["lat"].values.flatten())
        lon_list.append(dataset["lon"].values.flatten())
        dataset.close()

    lat_data = np.concatenate(lat_list)
    lon_data = np.concatenate(lon_list)

    lat_mask, lon_mask, lsm = load_sealand_mask(sealand_file)
    grid_x, grid_y = np.meshgrid(lon_mask, lat_mask)
    points = np.array([grid_x.flatten(), grid_y.flatten()]).T
    values = lsm.flatten()
    regridded = griddata(points, values, (lon_data, lat_data), method="nearest")

    sio.savemat(os.path.join(mask_dir, "Mare_mask_regridded.mat"), {"mask": (regridded == 0).astype(int), "latitudes": lat_data, "longitudes": lon_data})
    sio.savemat(os.path.join(mask_dir, "Terra_mask_regridded.mat"), {"mask": (regridded == 1).astype(int), "latitudes": lat_data, "longitudes": lon_data})


def _timezone_from_lon(lon):
    return round(lon / 15)


def _sunrise_sunset(dt_utc, lat, lon):
    import ephem

    observer = ephem.Observer()
    observer.lat = str(lat)
    observer.lon = str(lon)
    observer.date = dt_utc.date()
    observer.pressure = 0
    try:
        sunrise = observer.next_rising(ephem.Sun(), use_center=True)
        sunset = observer.next_setting(ephem.Sun(), use_center=True)
        return ephem.localtime(sunrise), ephem.localtime(sunset)
    except Exception:
        return None, None


def _is_day(dt_local, alba, tramonto):
    if alba is None or tramonto is None:
        return False
    return (alba + timedelta(hours=1.5)) <= dt_local <= (tramonto - timedelta(hours=1.5))


def generate_day_night_masks(nc_dir, mask_dir, day_threshold=70.0):
    """Classify each scene as day or night based on the percentage of illuminated pixels."""
    ensure_dir(mask_dir)
    day_mask = []
    night_mask = []

    for file_name in sorted(f for f in os.listdir(nc_dir) if f.endswith(".nc")):
        dataset = xr.open_dataset(os.path.join(nc_dir, file_name))
        latitudes = dataset["lat"].values.flatten()
        longitudes = dataset["lon"].values.flatten()
        dt_utc = pd.to_datetime(str(dataset["time"].values)).to_pydatetime()
        dataset.close()

        count_day = 0
        for lat, lon in zip(latitudes, longitudes):
            dt_local = dt_utc + timedelta(hours=_timezone_from_lon(lon))
            alba, tramonto = _sunrise_sunset(dt_utc, lat, lon)
            if _is_day(dt_local, alba, tramonto):
                count_day += 1

        is_day_scene = (100.0 * count_day / max(len(latitudes), 1)) >= day_threshold
        day_mask.extend([1 if is_day_scene else 0] * len(latitudes))
        night_mask.extend([0 if is_day_scene else 1] * len(latitudes))

    savemat(os.path.join(mask_dir, "giorno_maschera_70.mat"), {"giorno_maschera": np.array(day_mask)})
    savemat(os.path.join(mask_dir, "notte_maschera_70.mat"), {"notte_maschera": np.array(night_mask)})


def export_pickles_from_nc_and_masks(nc_dir, mask_dir, out_dir):
    """Export NetCDF channels, coordinates, and masks to pickle files aligned with the ML pipeline."""
    ensure_dir(out_dir)

    data_concat = {}
    time_concat = []
    lat_concat = []
    lon_concat = []

    for file_name in sorted(f for f in os.listdir(nc_dir) if f.endswith(".nc")):
        dataset = xr.open_dataset(os.path.join(nc_dir, file_name))

        for key in dataset.data_vars:
            data_concat.setdefault(key, []).append(dataset[key].values.flatten())

        n_pix = dataset["lat"].values.size
        time_concat.append(np.full(n_pix, dataset["time"].values.item()))
        lat_concat.append(dataset["lat"].values.T.flatten())
        lon_concat.append(dataset["lon"].values.T.flatten())
        dataset.close()

    for key, blocks in data_concat.items():
        arr = np.concatenate(blocks).ravel()
        match = re.match(r"(?:CH|channel)[ _]?(\d+)", key, re.IGNORECASE)
        out_key = f"CH_{match.group(1)}" if match else key
        with open(os.path.join(out_dir, f"{out_key}.pickle"), "wb") as file_handle:
            pickle.dump(arr, file_handle)

    with open(os.path.join(out_dir, "TIME.pickle"), "wb") as file_handle:
        pickle.dump(np.concatenate(time_concat).ravel().tolist(), file_handle)
    with open(os.path.join(out_dir, "lat.pickle"), "wb") as file_handle:
        pickle.dump(np.concatenate(lat_concat).ravel(), file_handle)
    with open(os.path.join(out_dir, "lon.pickle"), "wb") as file_handle:
        pickle.dump(np.concatenate(lon_concat).ravel(), file_handle)

    for mask_file in sorted(f for f in os.listdir(mask_dir) if f.endswith(".mat")):
        mat_data = sio.loadmat(os.path.join(mask_dir, mask_file))
        keys = [key for key in mat_data if not key.startswith("__")]
        if keys:
            with open(os.path.join(out_dir, f"{os.path.splitext(mask_file)[0]}.pickle"), "wb") as file_handle:
                pickle.dump(mat_data[keys[0]].ravel(), file_handle)


def main() -> None:
    if RUN_NAT_TO_NETCDF:
        batch_convert_crop_actual(NAT_ZIP_INPUT_DIR, NETCDF_OUTPUT_DIR)

    if RUN_CH9_MAPS:
        generate_ch9_maps(NETCDF_OUTPUT_DIR, CH9_MAP_OUTPUT_DIR)

    if RUN_MASKS:
        generate_season_mask(NETCDF_OUTPUT_DIR, MASK_OUTPUT_DIR)
        generate_sealand_masks(NETCDF_OUTPUT_DIR, MASK_OUTPUT_DIR, SEALAND_NETCDF_FILE)
        generate_day_night_masks(NETCDF_OUTPUT_DIR, MASK_OUTPUT_DIR)

    if RUN_EXPORT_PICKLES:
        export_pickles_from_nc_and_masks(NETCDF_OUTPUT_DIR, MASK_OUTPUT_DIR, PROCESSED_PICKLE_DIR)


if __name__ == "__main__":
    main()