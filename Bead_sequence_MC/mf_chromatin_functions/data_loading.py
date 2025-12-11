from pathlib import Path
from typing import Optional, Literal, Tuple, Union, Dict
import numpy as np

def get_region_and_mcool_from_setInvDict(
    coolDict: dict,
    sample: str,
    region_key: str,
    resolution: int,
    data_dir: Union[str, Path] = "data",
) -> Tuple[str, Path]:
    """
    Look up (a) the genomic region string (e.g. 'chr12:57,660,000-58,330,000')
    and (b) the .mcool filename for a given sample/region/resolution from setInvDict.coolDict.
    """
    data_dir = Path(data_dir)
    if sample not in coolDict:
        raise KeyError(f"Sample '{sample}' not found in coolDict.")
    mcool_name, region_dict = coolDict[sample]
    if region_key not in region_dict:
        raise KeyError(f"Region key '{region_key}' not found for sample '{sample}'.")
    res_key = str(resolution)
    if res_key not in region_dict[region_key]:
        raise KeyError(f"Resolution {resolution} not defined for '{sample}:{region_key}'.")
    region_entry = region_dict[region_key][res_key]

    # In this repo, region_entry[0] is a string 'chrX:start-end' (with thousands separators)
    region_string = region_entry[0]
    mcool_path = data_dir / mcool_name
    return region_string, mcool_path

def _read_csv_matrix(
    csv_path: Union[str, Path],
    dtype=np.float32,
    comment: str | None = None,
    fix_square: str = "auto",  # {'auto','crop','error'}
):
    """
    Read a contact matrix from CSV and return a square N x N array.

    Strategy:
    1) Try pandas with header inference (common in this repo's data) → often square.
    2) If not square, try header=None (raw numeric block).
       - If rows == cols: done.
       - If rows == cols+1: drop first row.
       - If cols == rows+1: drop first col.
    3) If still not square:
       - fix_square='crop' or 'auto' → crop to min(rows, cols) with a warning.
       - fix_square='error' → raise ValueError.
    """
    import pandas as pd

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    # Pass 1: header inference (matches original code path)
    df = pd.read_csv(csv_path, header=0, index_col=None, comment=comment)
    arr = df.values
    r, c = arr.shape

    if r == c:
        arr = np.asarray(arr, dtype=dtype)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Pass 2: raw read with no header
    df2 = pd.read_csv(csv_path, header=None, index_col=None, comment=comment)
    arr2 = df2.values
    r2, c2 = arr2.shape

    if r2 == c2:
        arr = arr2
    elif r2 == c2 + 1:
        # extra row (likely header previously treated as data)
        arr = arr2[1:, :]
    elif c2 == r2 + 1:
        # extra column (label/index column)
        arr = arr2[:, 1:]
    else:
        if fix_square in ("auto", "crop"):
            import warnings
            n = min(r2, c2)
            warnings.warn(
                f"CSV not square ({r2}x{c2}). Cropping to {n}x{n}. "
                "Set fix_square='error' to fail instead."
            )
            arr = arr2[:n, :n]
        else:
            raise ValueError(f"CSV at {csv_path} is not square (got {r2}x{c2}).")

    arr = np.asarray(arr, dtype=dtype)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)



def _read_cooler_matrix(
    mcool_path: Union[str, Path],
    region: str,
    resolution: int,
    balance: bool = True,
    dtype=np.float32,
) -> np.ndarray:
    """
    Read a cis contact matrix for `region` from an .mcool/.cool file.
    Requires `cooler` to be installed.
    """
    mcool_path = Path(mcool_path)
    if not mcool_path.exists():
        raise FileNotFoundError(mcool_path)

    try:
        import cooler  # lazy import
    except Exception as e:
        raise ImportError(
            "The 'cooler' package is required for Cooler/.mcool inputs. "
            "Install with: pip install cooler"
        ) from e

    # Cooler URI to the specific resolution group
    uri = f"{mcool_path}::resolutions/{resolution}"
    clr = cooler.Cooler(uri)

    # Fetch a dense square matrix for the region (cis)
    # Note: cooler will accept region strings like 'chr12:57,660,000-58,330,000'
    mat = clr.matrix(balance=balance, sparse=False).fetch(region)

    # Ensure numeric+finite
    mat = np.asarray(mat, dtype=dtype)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)

    # Sanity: square
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Cooler fetch returned a non-square block (shape {mat.shape}).")
    return mat


def read_contact_matrix(
    source: str,
    *,
    sample: str | None = None,
    region_key: str | None = None,
    resolution: int | None = None,
    coolDict: dict | None = None,
    data_dir: Union[str, Path] = "data",
    csv_path: Union[str, Path] | None = None,
    balance: bool = True,
    mcool_path: Union[str, Path] | None = None,
    region_string: str | None = None,
    dtype=np.float32,
):
    source = source.lower()
    data_dir = Path(data_dir)

    if source == "csv":
        if csv_path is None:
            if not (sample and region_key and resolution):
                raise ValueError("CSV mode needs either csv_path or (sample, region_key, resolution).")
            csv_path = data_dir / f"{sample}-Arima-allReps-filtered-{sample}+{region_key}+{resolution}.csv"
        return _read_csv_matrix(csv_path, dtype=dtype)

    elif source in {"cooler", "mcool", "cool"}:
        if (mcool_path is None or region_string is None):
            if coolDict is None or not (sample and region_key and resolution):
                raise ValueError(
                    "Cooler mode needs (mcool_path and region_string), or (coolDict+sample+region_key+resolution)."
                )
            region_string, mcool_path_auto = get_region_and_mcool_from_setInvDict(
                coolDict, sample, region_key, resolution, data_dir=data_dir
            )
            if mcool_path is None:
                mcool_path = mcool_path_auto
        return _read_cooler_matrix(mcool_path, region_string, resolution, balance=balance, dtype=dtype)

    else:
        raise ValueError("source must be one of {'csv','cooler'}.")

