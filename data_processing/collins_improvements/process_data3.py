import xarray as xr
import geopandas as gpd
import rioxarray
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from shapely.geometry import box
from pyproj import CRS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
import plotly.io as pio

# ------------------------ CONFIG ------------------------

LDAS_FOLDER = Path("C:/Users/colli/Downloads/ldasin")
DIVIDE_GPKG = r"C:\Users\colli\Downloads\hi_nextgen.gpkg"
OUTPUT_CSV = "multivar_test_nxgn_nonclip.csv"
VAR_LIST = ["T2D", "SWDOWN"]

# ------------------------ UTILS ------------------------

def load_sample_dataset(ldas_files):
    ds = xr.open_dataset(ldas_files[0], decode_times=True)
    cf_attrs = ds['crs'].attrs  # 'crs' for LDASOUT files. 'lambert_conformal_conic' for LDASIN files
    crs = CRS.from_cf(cf_attrs)

    ds = rename_dimensions(ds)
    ds = ds.rio.write_crs(crs, inplace=False)

    if ds['lambert_conformal_conic']:
        ny, nx = ds.dims['y'], ds.dims['x']
        x = np.linspace(-nx / 2 * 1000, nx / 2 * 1000, nx)
        y = np.linspace(-ny / 2 * 1000, ny / 2 * 1000, ny)
        ds = ds.assign_coords(x=("x", x), y=("y", y))
    
    return ds, crs, ds.rio.bounds()

def rename_dimensions(ds):
    if 'south_north' in ds.dims:
        ds = ds.rename({'south_north': 'y'})
    if 'west_east' in ds.dims:
        ds = ds.rename({'west_east': 'x'})
    return ds

def load_divides(filepath, crs, bounds):
    divides = gpd.read_file(filepath, layer="divides").to_crs(crs)
    intersects = divides[divides.intersects(box(*bounds))]
    return intersects

def prepare_variable(ds, var_name, crs):
    ds = rename_dimensions(ds)
    ds = ds.rio.write_crs(crs, inplace=False)
    if 'x' not in ds.coords or 'y' not in ds.coords:
        ny, nx = ds.dims['y'], ds.dims['x']
        x = np.linspace(-nx / 2 * 1000, nx / 2 * 1000, nx)
        y = np.linspace(-ny / 2 * 1000, ny / 2 * 1000, ny)
        ds = ds.assign_coords(x=("x", x), y=("y", y))
    var = ds[var_name]
    var.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    var.rio.write_crs(crs, inplace=True)
    return var

def extract_time(ds, fallback_ds):
    if 'Times' in ds.data_vars:
        val = ds['Times'].values.item()
        return pd.to_datetime(val.decode('utf-8') if isinstance(val, bytes) else val, format="%Y-%m-%d_%H:%M:%S")
    return pd.to_datetime(fallback_ds.time.values[0])


def load_pixel_indices(index_dir):
    index_dict = {}
    path = Path(index_dir)
    files = list(path.glob("divide_*.txt"))
    print(f"[DEBUG] Found {len(files)} index files in '{index_dir}'")

    for p in files:
        divide_id = p.stem.replace("divide_", "")
        with open(p, "r") as f:
            indices = [tuple(map(int, line.strip().split())) for line in f]
            index_dict[divide_id] = indices
    print(f"[INFO] Loaded pixel indices for {len(index_dict)} divides.")
    return index_dict

def process_ldas_files(ldas_files, index_map, var_names):
    """
    Extract mean values of multiple variables using pixel indices.
    Returns a DataFrame with time, divide_id, and one column per variable.
    """
    records = []
    for f in tqdm(ldas_files, desc="Processing files"):
        raw_ds = xr.open_dataset(f)
        timestamp = extract_time(raw_ds, raw_ds)

        # Prepare variables
        var_slices = {}
        for var_name in var_names:
            if var_name not in raw_ds:
                print(f"[WARN] Variable '{var_name}' not found in file {f.name}. Skipping.")
                continue
            var_data = raw_ds[var_name]
            if var_data.ndim == 3:
                if 'Time' in var_data.dims:
                    var_data = var_data.isel(Time=0)
                elif 'time' in var_data.dims:
                    var_data = var_data.isel(time=0)
            var_data = rename_dimensions(var_data)
            var_slices[var_name] = var_data

        for divide_id, pixel_list in index_map.items():
            row = {
                "time": timestamp,
                "divide_id": divide_id,
            }
            for var_name, var_data in var_slices.items():
                try:
                    vals = [var_data.values[i, j] for i, j in pixel_list]
                    mean_val = np.nanmean(vals)
                except Exception:
                    mean_val = np.nan
                row[var_name] = mean_val
            records.append(row)
    return pd.DataFrame(records)

def save_or_merge_csv(new_df, output_csv):  # Helper function to store 
    if Path(output_csv).exists():
        old_df = pd.read_csv(output_csv, parse_dates=["time"])
        merged_df = pd.merge(old_df, new_df, on=["time", "divide_id"], how="outer")
    else:
        merged_df = new_df
    merged_df = merged_df.sort_values(["divide_id", "time"])
    merged_df.to_csv(output_csv, index=False)
    print(f"[INFO] Saved merged output to: {output_csv}")

# ------------------------ MAIN ------------------------
def main():
    ldas_files = sorted(LDAS_FOLDER.glob("*.LDASIN_DOMAIN1"))

    print("[INFO] Loading precomputed pixel indices...")
    index_map = load_pixel_indices("pixel_indices_LDASIN_nxgen")  # Specify location of pixel index files that matches LDAS file type

    print(f"[INFO] Processing LDAS files using index-based extraction...{len(ldas_files)} files...{len(ldas_files)} LDAS files for {len(VAR_LIST)} variables")
    df = process_ldas_files(ldas_files, index_map, VAR_LIST)

    print("[INFO] Saving to CSV...")
    df = df.sort_values(["divide_id", "time"])
    df.to_csv(OUTPUT_CSV, index=False)
    print("[INFO] Done.")

if __name__ == "__main__":
    main()
