"""
Feature engineering for precipitation classification.

This module extracts and processes advanced features from balanced datasets for ML model training.
It supports 4 class configurations (4/5/7/15) with multiple feature types:
- Base: raw SEVIRI channels (11 channels)
- Climate: day/night, season, land/sea masks (3 boolean features)
- Derived: channel differences, moving mean/std (window 5), spatial gradient
- Analytical: skewness, local entropy (5x5 kernel), spatial Laplacian

For the 5-class case, additional advanced features are available (skewness, entropy, Laplacian).
The module exports different configurations (feature block combinations) for comparative studies.

Data structure:
- Input: .pkl files (balanced) with precipitation DataFrame + raw DPR arrays
- Output: .pkl files (completed features) and multi-configuration dataset (5 classes)

Workflow: loading → feature calculation → saving by class/configuration
"""

from __future__ import annotations

import os
import pickle
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import generic_filter, uniform_filter, laplace
from scipy.stats import skew
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte


# =========================
# Configuration
# =========================
# Single configuration section: paths and run switches.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"

# Run switches
RUN_PREPARE_FEATURES_4 = False
RUN_PREPARE_FEATURES_5 = False
RUN_PREPARE_FEATURES_7 = False
RUN_PREPARE_FEATURES_15 = False
RUN_PREPARE_FEATURES_5_AVANZATE = False
RUN_EXPORT_FEATURE_CONFIGS_5 = False

PATH_DATI_PROCESSATI = DATA_ROOT / "processed"
PATH_BIL_4 = DATA_ROOT / "balanced" / "bilanciamento_4classi"
PATH_BIL_5 = DATA_ROOT / "balanced" / "bilanciamento_5classi"
PATH_BIL_7 = DATA_ROOT / "balanced" / "bilanciamento_7classi"
PATH_BIL_15 = DATA_ROOT / "balanced" / "bilanciamento_15classi"

PATH_ML_4 = DATA_ROOT / "ml" / "Dati_ML_plus"
PATH_ML_5 = DATA_ROOT / "ml" / "Dati_ML_plus_5classi"
PATH_ML_7 = DATA_ROOT / "ml" / "Dati_ML_plus_7classi"
PATH_ML_15 = DATA_ROOT / "ml" / "Dati_ML_plus_15classi"
PATH_ML_5_ADVANCED = DATA_ROOT / "features" / "features_5classi_avanzate"
PATH_ML_5_CONFIGS = DATA_ROOT / "features" / "configurazioni_features_5classi"

CHANNEL_PICKLES = [f"CH_{i}.pickle" for i in range(1, 12)]
BASE_PICKLES = ["DPR.pickle", "maschera_giorno.pickle", "maschera_notte.pickle"]
OPTIONAL_PICKLES = ["TERRA_maschera.pickle", "Mare_maschera.pickle", "STAGIONI_maschera.pickle"]


def load_pickle(path: str):
    """Load an object serialized in pickle format."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(path: str, data) -> None:
    """Save an object in pickle format, creating the destination folder if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def replace_nans_with_nearest(data: np.ndarray) -> np.ndarray:
    """Replace NaNs with the nearest valid value along the 1D vector."""
    data = data.astype(np.float32, copy=True)
    valid_idx = np.flatnonzero(~np.isnan(data))
    if len(valid_idx) == 0:
        return data
    invalid_idx = np.flatnonzero(np.isnan(data))
    for idx in invalid_idx:
        nearest = valid_idx[np.abs(valid_idx - idx).argmin()]
        data[idx] = data[nearest]
    return data


def calcola_media_std_mobile(data: np.ndarray, finestra: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and standard deviation over moving 1D windows.

    Args:
        data (np.ndarray): 1D data array
        finestra (int): window size (default 5)

    Returns:
        Tuple[np.ndarray, np.ndarray]: (mean, std); contain NaN when the window is incomplete
    """
    media_mobile = np.empty(len(data), dtype=np.float32)
    std_mobile = np.empty(len(data), dtype=np.float32)

    for indice in range(len(data)):
        inizio = max(0, indice - finestra // 2)
        fine = min(len(data), indice + finestra // 2 + 1)
        finestra_corrente = data[inizio:fine]

        if len(finestra_corrente) < finestra:
            media_mobile[indice] = np.nan
            std_mobile[indice] = np.nan
        else:
            media_mobile[indice] = np.mean(finestra_corrente)
            std_mobile[indice] = np.std(finestra_corrente)

    return media_mobile, std_mobile


def calcola_media_std_mobile_immagini(vettore_1d: np.ndarray, dim: int = 64, kernel_size: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and standard deviation over 2D windows (rectangular kernel) for image stacks.

    Args:
        vettore_1d (np.ndarray): 1D array, concatenation of N 64x64 images
        dim (int): spatial size (default 64)
        kernel_size (int): kernel size (default 5)

    Returns:
        Tuple[np.ndarray, np.ndarray]: (mean, std) for the whole stack, same shape as input
    """
    num_pixel = dim * dim
    num_immagini = vettore_1d.shape[0] // num_pixel
    vettore_media: List[np.ndarray] = []
    vettore_std: List[np.ndarray] = []

    for indice in range(num_immagini):
        start = indice * num_pixel
        end = (indice + 1) * num_pixel
        immagine = vettore_1d[start:end].reshape(dim, dim).astype(np.float32)
        media = uniform_filter(immagine, size=kernel_size, mode="nearest")
        mean_sq = uniform_filter(immagine ** 2, size=kernel_size, mode="nearest")
        std = np.sqrt(np.maximum(mean_sq - media ** 2, 0))
        vettore_media.append(media.flatten())
        vettore_std.append(std.flatten())

    return np.concatenate(vettore_media), np.concatenate(vettore_std)


def calcola_gradiente_spaziale(canale: np.ndarray, dim: int = 64) -> np.ndarray:
    """
    Compute spatial gradient magnitude for each image (x, y derivatives).

    Args:
        canale (np.ndarray): 1D array, concatenation of N 64x64 images (4096*N)
        dim (int): spatial image size (default 64)

    Returns:
        np.ndarray: 1D array of gradient magnitudes, same length as the original channel
    """
    n_pixel = dim * dim
    n_immagini = canale.shape[0] // n_pixel
    gradienti: List[np.ndarray] = []

    for indice in range(n_immagini):
        immagine = canale[indice * n_pixel:(indice + 1) * n_pixel].reshape(dim, dim)
        grad_x, grad_y = np.gradient(immagine)
        modulo_gradiente = np.sqrt(grad_x ** 2 + grad_y ** 2)
        gradienti.append(modulo_gradiente.flatten())

    return np.concatenate(gradienti)


def calcola_skewness_mobile(canale: np.ndarray, dim: int = 64, kernel_size: int = 5) -> np.ndarray:
    """
    Compute skewness (Fisher 3rd moment) over local windows for each image.

    Args:
        canale (np.ndarray): 1D array, concatenation of N 64x64 images (4096*N)
        dim (int): spatial image size (default 64)
        kernel_size (int): window size (default 5)

    Returns:
        np.ndarray: 1D array of skewness values, same length as the original channel
    """
    n_pixel = dim * dim
    n_immagini = canale.shape[0] // n_pixel
    risultati: List[np.ndarray] = []

    def skewness_funzione(finestra: np.ndarray) -> float:
        return float(skew(finestra))

    for indice in range(n_immagini):
        img = canale[indice * n_pixel:(indice + 1) * n_pixel].reshape(dim, dim)
        skew_img = generic_filter(img, skewness_funzione, size=kernel_size, mode="reflect")
        risultati.append(skew_img.flatten())

    return np.concatenate(risultati)


def calcola_entropia_mobile(canale: np.ndarray, dim: int = 64, kernel_size: int = 5) -> np.ndarray:
    """
    Compute moving entropy over 2D windows for each image of a channel.

    Args:
        canale (np.ndarray): 1D array, concatenation of N 64x64 images (4096*N)
        dim (int): spatial image size (default 64)
        kernel_size (int): disk kernel size (default 5)

    Returns:
        np.ndarray: 1D array of entropy values, same length as the original channel
    """
    n_pixel = dim * dim
    n_immagini = canale.shape[0] // n_pixel
    risultati: List[np.ndarray] = []

    for i in range(n_immagini):
        img_flat = canale[i * n_pixel:(i + 1) * n_pixel]
        img_2d = img_flat.reshape(dim, dim)
        
        # Normalize and convert to uint8 for entropy
        img_min = np.min(img_2d)
        img_max = np.max(img_2d)
        if img_max > img_min:
            img_norm = (img_2d - img_min) / (img_max - img_min)
        else:
            img_norm = img_2d - img_min
        img_uint8 = img_as_ubyte(img_norm)
        
        # Compute entropy with disk kernel
        entropia = entropy(img_uint8, disk(kernel_size // 2))
        risultati.append(entropia.flatten())

    return np.concatenate(risultati)


def calcola_laplaciano_spaziale(canale: np.ndarray, dim: int = 64) -> np.ndarray:
    """
    Compute the spatial Laplacian magnitude for each image.

    Args:
        canale (np.ndarray): 1D array, concatenation of N 64x64 images (4096*N)
        dim (int): spatial image size (default 64)

    Returns:
        np.ndarray: 1D array of Laplacian magnitudes, same length as the original channel
    """
    n_pixel = dim * dim
    n_immagini = canale.shape[0] // n_pixel
    risultati: List[np.ndarray] = []

    for i in range(n_immagini):
        img_flat = canale[i * n_pixel:(i + 1) * n_pixel]
        img_2d = img_flat.astype(np.float32).reshape(dim, dim)
        lap = laplace(img_2d, mode="reflect")
        risultati.append(np.abs(lap).flatten())

    return np.concatenate(risultati)


def prepara_features(cartella_input: str, cartella_output: str) -> None:
    """Prepare the base feature dataset for standard classification configurations."""
    nomi = CHANNEL_PICKLES + BASE_PICKLES + OPTIONAL_PICKLES
    data = {n.split(".")[0]: load_pickle(os.path.join(cartella_input, n)) for n in nomi if os.path.exists(os.path.join(cartella_input, n))}

    dpr = np.asarray(data["DPR"], dtype=np.float32).flatten()
    ch = {f"CH_{i}": np.asarray(data[f"CH_{i}"], dtype=np.float32).flatten() for i in range(1, 12)}
    gmask = np.asarray(data["maschera_giorno"], dtype=np.float32).flatten()
    nmask = np.asarray(data["maschera_notte"], dtype=np.float32).flatten()
    terra = np.asarray(data.get("TERRA_maschera", np.zeros_like(dpr)), dtype=np.float32).flatten()
    mare = np.asarray(data.get("Mare_maschera", np.zeros_like(dpr)), dtype=np.float32).flatten()
    stag = np.asarray(data.get("STAGIONI_maschera", np.zeros_like(dpr)), dtype=np.float32).flatten()

    non_valid = np.isnan(dpr) | (dpr < 0) | (ch["CH_4"] < 0)
    valid_idx = ~non_valid

    dpr = dpr[valid_idx]
    gmask = gmask[valid_idx]
    nmask = nmask[valid_idx]
    terra = terra[valid_idx]
    mare = mare[valid_idx]
    stag = stag[valid_idx]
    for key in ch:
        ch[key] = ch[key][valid_idx]

    features: Dict[str, np.ndarray] = {}
    for i in range(1, 12):
        c = ch[f"CH_{i}"]
        features[f"CH_{i}"] = c
        features[f"CH_{i}_terra"] = np.where(terra == 1, c, 0)
        features[f"CH_{i}_mare"] = np.where(mare == 1, c, 0)
        features[f"CH_{i}_estate"] = np.where(stag == 0, c, 0)
        features[f"CH_{i}_inverno"] = np.where(stag == 1, c, 0)
        m, s = calcola_media_std_mobile(c, finestra=5)
        features[f"mean_CH_{i}"] = replace_nans_with_nearest(m)
        features[f"std_CH_{i}"] = replace_nans_with_nearest(s)

    diff_pairs = [
        (4, 5), (4, 6), (4, 7), (4, 8), (4, 10), (4, 11),
        (5, 6), (5, 7), (5, 8), (5, 10), (5, 11),
        (6, 7), (6, 8), (6, 10), (6, 11),
        (7, 8), (7, 10), (7, 11),
        (8, 11), (9, 10), (10, 11),
    ]
    for a, b in diff_pairs:
        features[f"diff_CH_{a}_{b}"] = ch[f"CH_{a}"] - ch[f"CH_{b}"]

    df_features = pd.DataFrame(features)
    save_pickle(os.path.join(cartella_output, "df_features.pickle"), df_features)
    save_pickle(os.path.join(cartella_output, "DPR_ml.pickle"), dpr)
    save_pickle(os.path.join(cartella_output, "maschera_giorno_ml.pickle"), gmask)
    save_pickle(os.path.join(cartella_output, "maschera_notte_ml.pickle"), nmask)

    print(f"Features saved to: {cartella_output}")


def prepara_features_5_avanzate(
    cartella_input: str,
    cartella_output: str,
    includi_clima: bool = True,
    includi_statistiche_spaziali: bool = True,
    includi_gradienti: bool = True,
    includi_skewness: bool = True,
    includi_entropia: bool = True,
    includi_laplaciano: bool = True,
    kernel_sizes: Sequence[int] = (5, 3),
) -> None:
    """
    Prepare advanced features for 5-class classification with supplemental features.

    Loads 11 SEVIRI channels, masks (day/night, land/sea, seasons), and raw DPR.
    Builds features in selectable blocks:
    - climate: sparse multiplication by masks (e.g., CH_1_mare = CH_1 * sea_mask)
    - diff: differences between channel pairs (20 differences)
    - mean/std: moving statistics for kernel_sizes (default 5x5, 3x3)
    - gradient: spatial gradient computed via approximated Sobel
    - skewness: local value skewness (Fisher 3rd moment)
    - entropy: local entropy (Shannon, disk kernel 5)
    - laplacian: spatial Laplacian (discrete second derivative)

    Saves a single DataFrame with all selected features and masks in pickle format.

    Args:
        cartella_input (str): directory with raw channel and mask .pickle files
        cartella_output (str): output destination directory (created if it does not exist)
        includi_clima (bool): if True, add mask features (default True)
        includi_statistiche_spaziali (bool): if True, add mean/std (default True)
        includi_gradienti (bool): if True, add gradient (default True)
        includi_skewness (bool): if True, add skewness (default True)
        includi_entropia (bool): if True, add entropy (default True)
        includi_laplaciano (bool): if True, add laplacian (default True)
        kernel_sizes (Sequence[int]): kernel sizes for spatial statistics (default (5, 3))

    Returns:
        None

    Output files:
        - df_features.pickle: DataFrame with all features (columns)
        - DPR_ml.pickle, maschera_giorno_ml.pickle, maschera_notte_ml.pickle: raw arrays
    """
    nomi = CHANNEL_PICKLES + BASE_PICKLES + OPTIONAL_PICKLES
    loaded = {n.split(".")[0]: load_pickle(os.path.join(cartella_input, n)) for n in nomi if os.path.exists(os.path.join(cartella_input, n))}

    dpr = np.asarray(loaded["DPR"], dtype=np.float32).flatten()
    gmask = np.asarray(loaded["maschera_giorno"], dtype=np.float32).flatten()
    nmask = np.asarray(loaded["maschera_notte"], dtype=np.float32).flatten()
    terra = np.asarray(loaded.get("TERRA_maschera", np.zeros_like(dpr)), dtype=np.float32).flatten()
    mare = np.asarray(loaded.get("Mare_maschera", np.zeros_like(dpr)), dtype=np.float32).flatten()
    stag = np.asarray(loaded.get("STAGIONI_maschera", np.zeros_like(dpr)), dtype=np.float32).flatten()
    canali = {f"CH_{i}": np.asarray(loaded[f"CH_{i}"], dtype=np.float32).flatten() for i in range(1, 12)}

    data: Dict[str, np.ndarray] = {f"CH_{i}": canali[f"CH_{i}"] for i in range(1, 12)}

    if includi_clima:
        for i in range(1, 12):
            c = canali[f"CH_{i}"]
            data[f"CH_{i}_mare"] = np.where(mare == 1, c, 0)
            data[f"CH_{i}_terra"] = np.where(terra == 1, c, 0)
            data[f"CH_{i}_estate"] = np.where(stag == 0, c, 0)
            data[f"CH_{i}_inverno"] = np.where(stag == 1, c, 0)

    diff_pairs = [
        (4, 5), (4, 6), (4, 7), (4, 8), (4, 10), (4, 11),
        (5, 6), (5, 7), (5, 8), (5, 10), (5, 11),
        (6, 7), (6, 8), (6, 10), (6, 11),
        (7, 8), (7, 10), (7, 11),
        (8, 11), (9, 10), (10, 11),
    ]
    for a, b in diff_pairs:
        data[f"diff_CH_{a}_{b}"] = canali[f"CH_{a}"] - canali[f"CH_{b}"]

    if includi_statistiche_spaziali:
        for kernel_size in kernel_sizes:
            for i in range(1, 12):
                media, std = calcola_media_std_mobile_immagini(canali[f"CH_{i}"], dim=64, kernel_size=kernel_size)
                data[f"mean_{kernel_size}x{kernel_size}_CH_{i}"] = replace_nans_with_nearest(media)
                data[f"std_{kernel_size}x{kernel_size}_CH_{i}"] = replace_nans_with_nearest(std)

    if includi_gradienti:
        for i in range(1, 12):
            grad = calcola_gradiente_spaziale(canali[f"CH_{i}"], dim=64)
            data[f"gradient_CH_{i}"] = replace_nans_with_nearest(grad)

    if includi_skewness:
        for i in range(1, 12):
            skewness_arr = calcola_skewness_mobile(canali[f"CH_{i}"], dim=64, kernel_size=5)
            data[f"skewness_CH_{i}"] = replace_nans_with_nearest(skewness_arr)

    if includi_entropia:
        for i in range(1, 12):
            entropia_arr = calcola_entropia_mobile(canali[f"CH_{i}"], dim=64, kernel_size=5)
            data[f"entropy_CH_{i}"] = replace_nans_with_nearest(entropia_arr)

    if includi_laplaciano:
        for i in range(1, 12):
            laplaciano_arr = calcola_laplaciano_spaziale(canali[f"CH_{i}"], dim=64)
            data[f"laplacian_CH_{i}"] = replace_nans_with_nearest(laplaciano_arr)

    df_features = pd.DataFrame(data)
    save_pickle(os.path.join(cartella_output, "df_features.pickle"), df_features)
    save_pickle(os.path.join(cartella_output, "DPR_ml.pickle"), dpr)
    save_pickle(os.path.join(cartella_output, "maschera_giorno_ml.pickle"), gmask)
    save_pickle(os.path.join(cartella_output, "maschera_notte_ml.pickle"), nmask)

    print(f"Advanced 5-class features saved to: {cartella_output}")


def esporta_configurazioni_feature_5(cartella_feature: str, cartella_output: str) -> None:
    """
    Export all possible feature-block combinations from an advanced 5-class dataset.

    Reads the advanced DataFrame and groups columns into 9 semantic blocks:
    channels, climate, diff, mean (5x5), mean_k3 (3x3), gradient, skewness, entropy, laplacian.

    Generates all ${2^8} = 256$ combinations of blocks (excluding empty combinations),
    where channels are always included as the base. Each combination is saved as a separate .pkl.

    Useful for comparative studies: evaluate which block adds predictive value.

    Args:
        cartella_feature (str): directory containing df_features.pickle and raw masks (from prepara_features_5_avanzate)
        cartella_output (str): output directory for configuration .pkl files

    Returns:
        None

    Output:
        For each block combination: .pkl file with format "canali_[block1_block2_...].pkl"
        Example: "canali_clima_diff.pkl", "canali.pkl", "canali_entropy_laplacian.pkl", ...
        Also saves copies of raw masks (DPR_ml, maschera_giorno_ml, maschera_notte_ml)
    """
    df_features = load_pickle(os.path.join(cartella_feature, "df_features.pickle"))
    dpr = load_pickle(os.path.join(cartella_feature, "DPR_ml.pickle"))
    gmask = load_pickle(os.path.join(cartella_feature, "maschera_giorno_ml.pickle"))
    nmask = load_pickle(os.path.join(cartella_feature, "maschera_notte_ml.pickle"))

    blocchi = {
        "canali": [f"CH_{i}" for i in range(1, 12)],
        "clima": [c for c in df_features.columns if c.endswith(("_mare", "_terra", "_estate", "_inverno"))],
        "diff": [c for c in df_features.columns if c.startswith("diff_CH_")],
        "media": [c for c in df_features.columns if c.startswith(("mean_5x5_", "std_5x5_"))],
        "media_k3": [c for c in df_features.columns if c.startswith(("mean_3x3_", "std_3x3_"))],
        "gradient": [c for c in df_features.columns if c.startswith("gradient_")],
        "skewness": [c for c in df_features.columns if c.startswith("skewness_")],
        "entropy": [c for c in df_features.columns if c.startswith("entropy_")],
        "laplacian": [c for c in df_features.columns if c.startswith("laplacian_")],
    }

    # Generate all possible combinations of extra blocks (channels are always the base)
    extra_blocks = ["clima", "diff", "media", "media_k3", "gradient", "skewness", "entropy", "laplacian"]
    # Filter blocks that actually have columns present in the DataFrame
    extra_blocks = [b for b in extra_blocks if blocchi[b]]

    os.makedirs(cartella_output, exist_ok=True)
    for r in range(len(extra_blocks) + 1):
        for combo in combinations(extra_blocks, r):
            keys = ["canali"] + list(combo)
            nome_cfg = "_".join(keys)
            colonne: List[str] = []
            for key in keys:
                colonne.extend(blocchi[key])
            colonne = [c for c in colonne if c in df_features.columns]
            save_pickle(os.path.join(cartella_output, f"{nome_cfg}.pkl"), df_features[colonne].copy())
    print(f"Saved {sum(1 for r in range(len(extra_blocks)+1) for _ in combinations(extra_blocks, r))} configurations.")

    save_pickle(os.path.join(cartella_output, "DPR_ml.pickle"), dpr)
    save_pickle(os.path.join(cartella_output, "maschera_giorno_ml.pickle"), gmask)
    save_pickle(os.path.join(cartella_output, "maschera_notte_ml.pickle"), nmask)
    print(f"Feature configurations saved to: {cartella_output}")


def main() -> None:
    if RUN_PREPARE_FEATURES_4:
        prepara_features(PATH_BIL_4, PATH_ML_4)
    if RUN_PREPARE_FEATURES_5:
        prepara_features(PATH_BIL_5, PATH_ML_5)
    if RUN_PREPARE_FEATURES_7:
        prepara_features(PATH_BIL_7, PATH_ML_7)
    if RUN_PREPARE_FEATURES_15:
        prepara_features(PATH_BIL_15, PATH_ML_15)
    if RUN_PREPARE_FEATURES_5_AVANZATE:
        prepara_features_5_avanzate(PATH_DATI_PROCESSATI, PATH_ML_5_ADVANCED)
    if RUN_EXPORT_FEATURE_CONFIGS_5:
        esporta_configurazioni_feature_5(PATH_ML_5_ADVANCED, PATH_ML_5_CONFIGS)


if __name__ == "__main__":
    main()