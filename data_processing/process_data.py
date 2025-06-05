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

LDAS_FOLDER = Path("./netcdf3")
DIVIDE_GPKG = "./hi_nextgen.gpkg"
OUTPUT_CSV = "multidate_RAINRATE_nextgen_timeseries.csv"
VAR_NAME = "T2D"

# ------------------------ UTILS ------------------------

def load_sample_dataset(ldas_files):
    ds = xr.open_dataset(ldas_files[0], decode_times=True)
    cf_attrs = ds['lambert_conformal_conic'].attrs
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
    """
    Load pixel indices from .txt files into a dictionary: {divide_id: [(i, j), ...]}
    """
    index_dict = {}
    for path in Path(index_dir).glob("divide_*.txt"):
        divide_id = int(path.stem.replace("divide_", ""))
        with open(path, "r") as f:
            indices = [tuple(map(int, line.strip().split())) for line in f]
            index_dict[divide_id] = indices
    print(f"[INFO] Loaded pixel indices for {len(index_dict)} divides.")
    return index_dict

def process_ldas_files(ldas_files, index_map, var_name):
    """
    Extract mean variable value using pixel indices instead of clipping polygons.
    """
    records = []
    for f in tqdm(ldas_files, desc="Processing files"):
        raw_ds = xr.open_dataset(f)
        var_data = raw_ds[var_name]

        # Handle dim renaming + coordinate assignment
        if 'south_north' in var_data.dims:
            var_data = var_data.rename({'south_north': 'y'})
        if 'west_east' in var_data.dims:
            var_data = var_data.rename({'west_east': 'x'})

        timestamp = extract_time(raw_ds, raw_ds)

        for divide_id, pixel_list in index_map.items():
            try:
                vals = [var_data.values[i, j] for i, j in pixel_list]
                mean_val = np.nanmean(vals)
            except Exception:
                mean_val = np.nan
            records.append({
                "time": timestamp,
                "divide_id": divide_id,
                var_name: mean_val
            })
    return pd.DataFrame(records)

# ------------------------ MAIN ------------------------
def main():
    ldas_files = sorted(LDAS_FOLDER.glob("*.LDASIN_DOMAIN1"))
    print("[INFO] Loading sample LDAS file...")
    sample_ds, real_crs, bounds = load_sample_dataset(ldas_files)

    print("[INFO] Loading precomputed pixel indices...")
    index_map = load_pixel_indices("pixel_indices")

    print("[INFO] Processing LDAS files using index-based extraction...")
    df = process_ldas_files(ldas_files, index_map, VAR_NAME)

    print("[INFO] Saving to CSV...")
    df = df.sort_values(["divide_id", "time"])
    df.to_csv(OUTPUT_CSV, index=False)
    print("[INFO] Done.")

if __name__ == "__main__":
    main()