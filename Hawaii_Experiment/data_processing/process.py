import xarray as xr
import geopandas as gpd
import rioxarray
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from shapely.geometry import box
from pyproj import CRS
import numpy as np
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import pyarrow.parquet as pq

# ------------------------ CONFIG (Now largely determined by arguments) ------------------------
# Default values - these can be overridden by command-line arguments
DEFAULT_LDAS_FOLDER = "data/LDASIN" # Common HPC structure might put data in a 'data' subfolder
DEFAULT_DIVIDE_GPKG = "data/hi_reference.gpkg"
DEFAULT_PIXEL_INDICES_DIR = "pixel_indices_LDASIN_ref"
DEFAULT_OUTPUT_DIR = "output_parquet_files" # Changed default to reflect parquet
DEFAULT_BATCH_SIZE = 32 # A common number for HPC nodes, adjust based on actual cores

# List of all variables to process
ALL_VAR_NAMES = [
    "LWDOWN", "PSFC", "Q2D", "RAINRATE", "SWDOWN", "T2D", "U2D", "V2D"
]

# ------------------------ UTILS ------------------------
def load_sample_dataset(ldas_files):
    """
    Loads a sample LDASIN dataset to infer CRS and bounds.
    Adjusted to handle original dimension names for rioxarray.
    """
    ds = xr.open_dataset(ldas_files[0], decode_times=True)
    
    if 'crs' in ds.data_vars:
        cf_attrs = ds['crs'].attrs
    elif 'lambert_conformal_conic' in ds.data_vars:
        cf_attrs = ds['lambert_conformal_conic'].attrs
    else:
        raise ValueError("Neither 'crs' nor 'lambert_conformal_conic' found for CRS definition.")
    
    crs = CRS.from_cf(cf_attrs)
    
    # Explicitly set spatial dims for rioxarray with original names
    ds = ds.rio.set_spatial_dims(x_dim="west_east", y_dim="south_north", inplace=False)
    ds = ds.rio.write_crs(crs, inplace=False)

    # Ensure x, y coordinates are present if not originally, using original dim names
    # This block now relies on rio.bounds() and existing dimensions.
    if 'west_east' not in ds.coords or 'south_north' not in ds.coords:
        ny, nx = ds.dims['south_north'], ds.dims['west_east']
        x = np.linspace(ds.rio.bounds()[0], ds.rio.bounds()[2], nx)
        y = np.linspace(ds.rio.bounds()[1], ds.rio.bounds()[3], ny)
        ds = ds.assign_coords(west_east=("west_east", x), south_north=("south_north", y))
        
    return ds, crs, ds.rio.bounds()

def load_divides(filepath, crs, bounds):
    """
    Loads watershed divides from a GeoPackage and filters them by dataset bounds.
    """
    divides = gpd.read_file(filepath, layer="divides").to_crs(crs)
    intersects = divides[divides.intersects(box(*bounds))]
    return intersects

def prepare_variable(ds, var_name, crs):
    """
    Prepares a specific variable from an xarray dataset for spatial operations.
    Adjusted to use original dimension names.
    (Note: This function might not be strictly necessary with the current pixel-based approach
    but is kept for completeness if future spatial operations are considered).
    """
    ds = ds.rio.set_spatial_dims(x_dim="west_east", y_dim="south_north", inplace=False)
    ds = ds.rio.write_crs(crs, inplace=False)
    if 'west_east' not in ds.coords or 'south_north' not in ds.coords:
        ny, nx = ds.dims['south_north'], ds.dims['west_east']
        x = np.linspace(ds.rio.bounds()[0], ds.rio.bounds()[2], nx)
        y = np.linspace(ds.rio.bounds()[1], ds.rio.bounds()[3], ny)
        ds = ds.assign_coords(west_east=("west_east", x), south_north=("south_north", y))
    var = ds[var_name]
    var.rio.set_spatial_dims(x_dim="west_east", y_dim="south_north", inplace=True)
    var.rio.write_crs(crs, inplace=True)
    return var

def extract_time(ds, fallback_ds):
    """
    Extracts the timestamp from an xarray Dataset, handling different time dimension names.
    """
    if 'time' in ds.coords:
        return pd.to_datetime(ds.time.values[0])
    if 'Times' in ds.data_vars:
        val = ds['Times'].values.item()
        return pd.to_datetime(val.decode('utf-8') if isinstance(val, bytes) else val, format="%Y-%m-%d_%H:%M:%S")
    return pd.to_datetime(fallback_ds.time.values[0]) 

def load_pixel_indices(index_dir):
    """
    Loads precomputed pixel indices for each watershed divide.
    """
    index_dict = {}
    path = Path(index_dir)
    if not path.exists():
        print(f"[ERROR] Pixel indices directory '{index_dir}' not found. Please ensure it exists and contains index files for your LDASIN grid.")
        return {}
    files = list(path.glob("divide_*.txt"))
    print(f"[DEBUG] Found {len(files)} index files in '{index_dir}'")
    if not files:
        print(f"[WARNING] No pixel index files found in '{index_dir}'. This might lead to empty results if indices aren't precomputed.")

    for p in files:
        divide_id = p.stem.replace("divide_", "")
        with open(p, "r") as f:
            indices = [tuple(map(int, line.strip().split())) for line in f]
        index_dict[divide_id] = indices
    print(f"[INFO] Loaded pixel indices for {len(index_dict)} divides.")
    return index_dict

def process_single_file(file_path, index_map, var_name):
    """
    Processes a single LDASIN file to extract mean variable values for each divide.
    Adjusted for original dimension names 'south_north' and 'west_east'.
    """
    try:
        raw_ds = xr.open_dataset(file_path)
        
        if var_name not in raw_ds.data_vars:
            print(f"[WARNING] Variable '{var_name}' not found in {file_path.name}. Skipping this variable for this file.")
            return []

        var_data = raw_ds[var_name]

        if var_data.ndim == 3:
            if 'time' in var_data.dims:
                var_data = var_data.isel(time=0)
            elif 'Times' in var_data.dims:
                var_data = var_data.isel(Time=0)
            else:
                print(f"[WARNING] 3D variable '{var_name}' in {file_path.name} does not have 'time' or 'Time' dimension. Attempting to select first slice, but this might be incorrect.")
                var_data = var_data.isel(0)

        timestamp = extract_time(raw_ds, raw_ds) 

        records = []
        for divide_id, pixel_list in index_map.items():
            try:
                if not pixel_list:
                    mean_val = np.nan
                else:
                    vals = [var_data.values[i, j] for i, j in pixel_list]
                    mean_val = np.nanmean(vals)
            except IndexError:
                print(f"[ERROR] Pixel indices out of bounds for {file_path.name}, variable {var_name}, divide {divide_id}. Check if pixel indices match the data grid with 'south_north' and 'west_east' dimensions.")
                mean_val = np.nan
            except Exception as e:
                print(f"  Error processing pixels for divide {divide_id} in {file_path.name} for {var_name}: {e}")
                mean_val = np.nan
            records.append({
                "time": timestamp, # Column name remains 'time'
                "divide_id": divide_id,
                var_name: mean_val
            })
        return records

    except Exception as e:
        print(f"Error processing {file_path.name} for variable {var_name}: {e}")
        return []

def process_ldas_files_for_variable(ldas_files, index_map, var_name, batch_size):
    """
    Orchestrates parallel processing of LDASIN files for a single variable.
    """
    all_records = []
    with ProcessPoolExecutor(max_workers=batch_size) as executor:
        futures = {executor.submit(process_single_file, f, index_map, var_name): f for f in ldas_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {var_name} files"):
            records = future.result()
            all_records.extend(records)
    return pd.DataFrame(all_records)

def pivot_variable(df, var_name):
    """
    Pivots the DataFrame to have 'time' as index, divide_ids as columns, and variable values.
    The index will retain the name 'time'.
    """
    df = df.drop_duplicates(subset=["time", "divide_id"])
    df_pivot = df.pivot(index="time", columns="divide_id", values=var_name)
    df_pivot.index = pd.to_datetime(df_pivot.index)
    # df_pivot.index.name will remain 'time' by default from the pivot operation
    return df_pivot

def save_or_merge_pivoted_parquet(new_pivoted_df, output_parquet_path):
    """
    If output_parquet exists, merge new pivoted data into it.
    If not, save new pivoted data as a new Parquet file.
    Ensures data is sorted by time.
    """
    if output_parquet_path.exists():
        try:
            existing_df = pd.read_parquet(output_parquet_path)
            existing_df.index = pd.to_datetime(existing_df.index)
            # No special renaming needed for existing_df.index.name if it was already 'time'
            
            new_pivoted_df.index = pd.to_datetime(new_pivoted_df.index)
            # No special renaming needed for new_pivoted_df.index.name as it's 'time' from pivot
            
            merged_df = existing_df.combine_first(new_pivoted_df)
            merged_df.update(new_pivoted_df) 
            merged_df.sort_index(inplace=True)

            merged_df.to_parquet(output_parquet_path, index=True) 
            print(f"[INFO] Merged and updated Parquet file: {output_parquet_path}")

        except Exception as e:
            print(f"[ERROR] Could not merge with existing Parquet file {output_parquet_path}: {e}. Overwriting file.")
            new_pivoted_df.to_parquet(output_parquet_path, index=True)
            print(f"[INFO] Overwrote Parquet file: {output_parquet_path}")
    else:
        new_pivoted_df.to_parquet(output_parquet_path, index=True)
        print(f"[INFO] New Parquet file created: {output_parquet_path}")


# ------------------------ MAIN ------------------------
def main():
    parser = argparse.ArgumentParser(description="Process LDASIN files for specified variables and save to Parquet.")
    parser.add_argument("--ldas_folder", type=str, default=DEFAULT_LDAS_FOLDER,
                        help=f"Path to the directory containing LDASIN files (default: {DEFAULT_LDAS_FOLDER})")
    parser.add_argument("--divide_gpkg", type=str, default=DEFAULT_DIVIDE_GPKG,
                        help=f"Path to the GeoPackage file containing watershed divides (default: {DEFAULT_DIVIDE_GPKG})")
    parser.add_argument("--pixel_indices_dir", type=str, default=DEFAULT_PIXEL_INDICES_DIR,
                        help=f"Path to the directory containing precomputed pixel indices (default: {DEFAULT_PIXEL_INDICES_DIR})")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to save output Parquet files (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Number of files to process in parallel (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--file_pattern", type=str, default="*.LDASIN_DOMAIN1",
                        help="Glob pattern for LDASIN files (e.g., '*.LDASIN_DOMAIN1', '*.LDASIN')")

    args = parser.parse_args()

    ldas_folder = Path(args.ldas_folder)
    divide_gpkg = Path(args.divide_gpkg)
    pixel_indices_dir = Path(args.pixel_indices_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    ldas_files = sorted(ldas_folder.glob(args.file_pattern))
    if not ldas_files:
        print(f"[ERROR] No LDASIN files found in {ldas_folder} with pattern '{args.file_pattern}'. Please check the path and file naming.")
        return

    print("[INFO] Loading precomputed pixel indices...")
    index_map = load_pixel_indices(pixel_indices_dir)
    if not index_map:
        print("[ERROR] No pixel indices loaded. Cannot proceed with processing.")
        return

    for var_name in ALL_VAR_NAMES:
        print(f"\n[INFO] Starting processing for variable: {var_name}")
        output_parquet_filename = f"{var_name}.parquet"
        output_parquet_path = output_dir / output_parquet_filename

        df = process_ldas_files_for_variable(ldas_files, index_map, var_name, args.batch_size)

        if not df.empty:
            print(f"[INFO] Pivoting data for {var_name}...")
            df_pivoted = pivot_variable(df, var_name)
            print(f"[INFO] Saving {var_name} to Parquet...")
            save_or_merge_pivoted_parquet(df_pivoted, output_parquet_path)
        else:
            print(f"[WARNING] No data extracted for variable {var_name}. Skipping Parquet creation.")

    print("\n[INFO] All variables processed. Done.")

if __name__ == "__main__":
    main()
