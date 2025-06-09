import argparse
import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from pyproj import CRS
import sys
from multiprocessing import Pool, cpu_count, freeze_support
import numpy as np
import json
import xarray as xr # Moved xarray import here for consistency

# --- Settings ---
# Fixed CRS for NWM Hawaii files. This will be used as a fallback if
# CRS inference from file attributes fails. These parameters are common
# for the NWM Hawaii domain's Lambert Conformal Conic projection.
HARDCODED_NWM_HAWAII_CRS_DICT = {
    "proj": "lcc",
    "lat_1": 25.0,            # Standard Parallel 1
    "lat_2": 25.0,            # Standard Parallel 2 (often same as lat_1 for single parallel LCC)
    "lon_0": -155.5,          # Longitude of Central Meridian
    "lat_0": 19.5,            # Latitude of Projection Origin
    "x_0": 0.0,               # False Easting
    "y_0": 0.0,               # False Northing
    "a": 6370000.0,           # Radius of Sphere (for spherical earth model, common in WRF/NWM)
    "b": 6370000.0,           # Also radius for spherical earth
    "units": "m",             # Units of coordinates
    "no_defs": True           # No default definitions
}

# Fixed Grid Parameters for NWM Hawaii files. These will be used as a fallback if
# DX, DY, XORIG, YORIG cannot be inferred from file attributes.
# Assuming 1km resolution for now. Adjust if your data is 250m or has other specifics.
HARDCODED_NWM_HAWAII_GRID_PARAMS = {
    "dx": 1000.0,             # X-direction grid spacing in meters (e.g., 1000 for 1km)
    "dy": 1000.0,             # Y-direction grid spacing in meters
    "x_orig": -895000.0,      # X-coordinate of the origin (e.g., bottom-left corner in projection units)
    "y_orig": -460000.0       # Y-coordinate of the origin
}


# --- Utility Functions ---

def get_global_nwm_spatial_params(sample_dataset: xr.Dataset) -> tuple[CRS, float, float, float, float]:
    """
    Infers the CRS and grid parameters (DX, DY, XORIG, YORIG) from a sample NetCDF dataset.
    This function is intended to be called once to get global parameters.

    Args:
        sample_dataset (xr.Dataset): A sample xarray Dataset to infer spatial info from.

    Returns:
        tuple: (real_crs, dx, dy, x_orig, y_orig) - the inferred CRS and grid parameters.
    Raises:
        ValueError: If CRS or critical grid parameters cannot be found from the sample dataset.
    """
    sample_dataset = sample_dataset.rename({'south_north': 'y', 'west_east': 'x'})

    real_crs = None
    
    # --- DIAGNOSTIC PRINTS (for initial setup) ---
    sys.stderr.write(f"\n[DEBUG] Sample Dataset Global Attributes: {sample_dataset.attrs}\n")
    if 'lambert_conformal_conic' in sample_dataset.variables:
        sys.stderr.write(f"[DEBUG] Sample 'lambert_conformal_conic' Variable Attributes: {sample_dataset['lambert_conformal_conic'].attrs}\n")
    else:
        sys.stderr.write("[DEBUG] 'lambert_conformal_conic' variable not found in sample dataset.\n")
    
    for var_name in sample_dataset.data_vars:
        if 'grid_mapping' in sample_dataset[var_name].attrs:
            sys.stderr.write(f"[DEBUG] Sample Data variable '{var_name}' has 'grid_mapping' attribute: {sample_dataset[var_name].attrs['grid_mapping']}\n")
        break # Only check the first data variable for grid_mapping
    # --- END DIAGNOSTIC PRINTS ---

    # --- CRS Extraction Attempts (Prioritized) ---
    # Attempt 1: From 'lambert_conformal_conic' variable attributes (CF-compliant)
    if 'lambert_conformal_conic' in sample_dataset.variables and sample_dataset['lambert_conformal_conic'].attrs:
        try:
            real_crs = CRS.from_cf(sample_dataset['lambert_conformal_conic'].attrs)
            if real_crs:
                print(f"[INFO] CRS inferred from 'lambert_conformal_conic' attributes: {real_crs}")
        except Exception as e:
            sys.stderr.write(f"[WARNING] Could not parse CRS from 'lambert_conformal_conic' attributes in sample file: {e}\n")
    
    # Attempt 2: Infer directly by rioxarray (general raster CRS inference)
    if not real_crs:
        try:
            real_crs = sample_dataset.rio.crs
            if real_crs:
                print(f"[INFO] CRS inferred directly by rioxarray from sample file: {real_crs}")
            else:
                sys.stderr.write(f"[WARNING] rioxarray couldn't guess projection directly from sample file. Trying other methods.\n")
        except Exception as e:
            sys.stderr.write(f"[WARNING] Direct rioxarray CRS inference failed for sample file ({e}). Trying other methods.\n")

    # Attempt 3: Find grid_mapping from a data variable (if a specific grid_mapping variable exists)
    if not real_crs:
        for var_name in sample_dataset.data_vars:
            if 'grid_mapping' in sample_dataset[var_name].attrs:
                grid_mapping_var_name = sample_dataset[var_name].attrs['grid_mapping']
                if grid_mapping_var_name in sample_dataset.variables:
                    try:
                        real_crs = CRS.from_cf(sample_dataset[grid_mapping_var_name].attrs)
                        if real_crs:
                            print(f"[INFO] CRS inferred from '{grid_mapping_var_name}' variable attributes in sample file: {real_crs}")
                            break
                    except Exception as e:
                        sys.stderr.write(f"[WARNING] Failed to infer CRS from grid_mapping var '{grid_mapping_var_name}' in sample file: {e}\n")
            break # Only need to check one data variable for grid_mapping

    # Attempt 4: Try to form CRS from global attributes (e.g., LCC parameters directly in global attrs)
    if not real_crs:
        global_attrs = sample_dataset.attrs
        try:
            if ('grid_mapping_name' in global_attrs and global_attrs['grid_mapping_name'] == 'lambert_conformal_conic') or \
               ('standard_parallel' in global_attrs and 'longitude_of_central_meridian' in global_attrs):
                
                temp_crs = CRS.from_cf(global_attrs)
                if temp_crs:
                    real_crs = temp_crs
                    print(f"[INFO] CRS inferred from global CF attributes in sample file: {real_crs}")
        except Exception as e:
            sys.stderr.write(f"[WARNING] Failed to infer CRS from global CF attributes in sample file: {e}\n")

    # Attempt 5: Fallback to hardcoded NWM Hawaii CRS
    if not real_crs:
        try:
            real_crs = CRS.from_dict(HARDCODED_NWM_HAWAII_CRS_DICT)
            print(f"[INFO] CRS inferred using hardcoded NWM Hawaii projection (fallback): {real_crs}")
        except Exception as e:
            sys.stderr.write(f"[WARNING] Failed to use hardcoded CRS even as fallback: {e}\n")

    if not real_crs:
        raise ValueError("Could not determine map projection from sample NetCDF file. CRS is essential for processing.")

    # --- Extract Grid Parameters (DX, DY, XORIG, YORIG) ---
    dx, dy, x_orig, y_orig = None, None, None, None

    # Source for grid parameters, prioritized
    lcc_attrs = sample_dataset['lambert_conformal_conic'].attrs if 'lambert_conformal_conic' in sample_dataset.variables else {}
    global_attrs = sample_dataset.attrs

    # Helper to find an attribute from a list of possible names
    def _find_attr(attrs, names): return next((attrs.get(n) for n in names), None)

    # 1. Try common WRF-Hydro attribute names (DX, DY, XORIG, YORIG) from lcc_attrs
    dx = _find_attr(lcc_attrs, ['DX', 'grid_spacing_x', 'geospatial_x_resolution'])
    dy = _find_attr(lcc_attrs, ['DY', 'grid_spacing_y', 'geospatial_y_resolution'])
    x_orig = _find_attr(lcc_attrs, ['XORIG', 'x_grid_origin', 'geospatial_x_min'])
    y_orig = _find_attr(lcc_attrs, ['YORIG', 'y_grid_origin', 'geospatial_y_min'])

    # 2. Try common WRF-Hydro attribute names from global_attrs
    if dx is None: dx = _find_attr(global_attrs, ['DX', 'grid_spacing_x', 'geospatial_x_resolution'])
    if dy is None: dy = _find_attr(global_attrs, ['DY', 'grid_spacing_y', 'geospatial_y_resolution'])
    if x_orig is None: x_orig = _find_attr(global_attrs, ['XORIG', 'x_grid_origin', 'geospatial_x_min'])
    if y_orig is None: y_orig = _find_attr(global_attrs, ['YORIG', 'y_grid_origin', 'geospatial_y_min'])

    # 3. Parse from 'GeoTransform' if available and other methods failed
    if (dx is None or dy is None or x_orig is None or y_orig is None) and 'GeoTransform' in lcc_attrs:
        try:
            gt = [float(x) for x in lcc_attrs['GeoTransform'].split()]
            x_orig_gt, dx_gt, y_max_gt, dy_gt = gt[0], gt[1], gt[3], gt[5]

            ny = sample_dataset.sizes.get('y', sample_dataset.sizes.get('south_north', 0))
            if ny == 0: raise ValueError("No 'y' dim size for GeoTransform calculation.")
            y_orig_gt = y_max_gt + ny * dy_gt # dy_gt is negative for north-up rasters

            if dx is None: dx = dx_gt
            if dy is None: dy = abs(dy_gt) # Take absolute value for grid spacing
            if x_orig is None: x_orig = x_orig_gt
            if y_orig is None: y_orig = y_orig_gt
            sys.stderr.write(f"[INFO] Successfully parsed GeoTransform from sample file for grid parameters.\n")
        except Exception as e:
            sys.stderr.write(f"[WARNING] Failed to parse 'GeoTransform' attribute from sample file: {e}\n")

    # 4. Fallback to hardcoded NWM Hawaii Grid Parameters
    if any(p is None for p in [dx, dy, x_orig, y_orig]):
        print(f"[INFO] Grid parameters not found in file. Using hardcoded NWM Hawaii grid parameters as fallback: DX={HARDCODED_NWM_HAWAII_GRID_PARAMS['dx']}, DY={HARDCODED_NWM_HAWAII_GRID_PARAMS['dy']}, XORIG={HARDCODED_NWM_HAWAII_GRID_PARAMS['x_orig']}, YORIG={HARDCODED_NWM_HAWAII_GRID_PARAMS['y_orig']}")
        dx = HARDCODED_NWM_HAWAII_GRID_PARAMS['dx']
        dy = HARDCODED_NWM_HAWAII_GRID_PARAMS['dy']
        x_orig = HARDCODED_NWM_HAWAII_GRID_PARAMS['x_orig']
        y_orig = HARDCODED_NWM_HAWAII_GRID_PARAMS['y_orig']


    # Ensure DX/DY are positive (resolutions should be)
    dx = abs(float(dx))
    dy = abs(float(dy))
    x_orig = float(x_orig)
    y_orig = float(y_orig)

    if any(p is None for p in [dx, dy, x_orig, y_orig]): # Final check if for some reason still None
        raise ValueError(f"Could not determine DX/DY or XORIG/YORIG from sample NetCDF file or hardcoded fallback. Found: DX:{dx}, DY:{dy}, XORIG:{x_orig}, YORIG:{y_orig}")
    
    return real_crs, dx, dy, x_orig, y_orig

def apply_spatial_info(dataset: xr.Dataset, real_crs: CRS, dx: float, dy: float, x_orig: float, y_orig: float) -> xr.Dataset:
    """
    Applies the given CRS and grid parameters to an xarray Dataset, assigning real-world
    coordinates to 'x' and 'y' dimensions.

    Args:
        dataset (xr.Dataset): The xarray Dataset to modify.
        real_crs (pyproj.CRS): The CRS to apply.
        dx (float): X-direction grid spacing.
        dy (float): Y-direction grid spacing.
        x_orig (float): X-coordinate of the origin (bottom-left corner).
        y_orig (float): Y-coordinate of the origin (bottom-left corner).

    Returns:
        xr.Dataset: The updated dataset with coordinates and CRS.
    """
    # Rename dimensions if necessary
    dataset = dataset.rename({'south_north': 'y', 'west_east': 'x'}) # Standardize dims

    nx = dataset.sizes['x'] if 'x' in dataset.sizes else dataset.sizes['west_east']
    ny = dataset.sizes['y'] if 'y' in dataset.sizes else dataset.sizes['south_north']

    # Create 1D coordinate arrays (cell centers)
    x_coords = x_orig + (np.arange(nx) + 0.5) * dx
    y_coords = y_orig + (np.arange(ny) + 0.5) * dy

    # Assign coordinates and set CRS
    dataset = dataset.assign_coords(x=x_coords, y=y_coords).rio.write_crs(real_crs, inplace=False)
    return dataset

def load_divide_pixels_from_json(file_path: Path) -> pd.DataFrame:
    """Loads divide pixel coordinates from a JSON file into a DataFrame."""
    if not file_path.is_file(): raise FileNotFoundError(f"Pixel data JSON not found: {file_path}")

    print(f"Loading divide pixel data from {file_path}...")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            if not isinstance(data, dict): raise TypeError("JSON top level must be a dictionary.")

            all_pixels_flat = []
            for divide_id, pixel_list in data.items():
                if not isinstance(pixel_list, list):
                    sys.stderr.write(f"[WARNING] Skipping divide '{divide_id}': pixel list is not a list.\n")
                    continue
                for i, pc in enumerate(pixel_list):
                    if isinstance(pc, list) and len(pc) == 2:
                        try: all_pixels_flat.append({'divide_id': divide_id, 'x': float(pc[0]), 'y': float(pc[1])})
                        except ValueError: sys.stderr.write(f"[WARNING] Skipping malformed pixel in '{divide_id}'@{i}: '{pc}'.\n")
                    else: sys.stderr.write(f"[WARNING] Skipping malformed pixel in '{divide_id}'@{i}: '{pc}'.\n")
            
            pixel_df = pd.DataFrame(all_pixels_flat) if all_pixels_flat else pd.DataFrame(columns=['divide_id', 'x', 'y'])
            print(f"Loaded {len(pixel_df)} pixels for {len(pixel_df['divide_id'].unique())} divides.")
            return pixel_df
    except json.JSONDecodeError as e: raise json.JSONDecodeError(f"Error parsing JSON {file_path}: {e}", e.doc, e.pos)
    except Exception as e: raise Exception(f"Unexpected error loading pixel data from {file_path}: {e}")

# --- Main Processing Logic ---

def process_one_netcdf_file(
    file_path: Path,
    all_divide_pixels_df: pd.DataFrame,
    global_crs: CRS,
    global_dx: float,
    global_dy: float,
    global_x_orig: float,
    global_y_orig: float,
    data_variables_to_extract: list[str]
) -> dict[str, list[dict]]:
    """Processes a single NetCDF file: extracts data for specific pixels."""
    data_records_by_variable = {var: [] for var in data_variables_to_extract}

    try:
        with xr.open_dataset(file_path, decode_coords=False, decode_times=True) as dataset:
            # Apply known global spatial info to this dataset
            try:
                dataset = apply_spatial_info(dataset, global_crs, global_dx, global_dy, global_x_orig, global_y_orig)
            except ValueError as e:
                sys.stderr.write(f"[ERROR] Failed to apply spatial info to {file_path.name}: {e}. Skipping.\n")
                return data_records_by_variable

            if 'x' not in dataset.coords or 'y' not in dataset.coords:
                sys.stderr.write(f"[ERROR] X or Y coords missing in {file_path.name} after spatial info application. Skipping.\n")
                return data_records_by_variable
            
            file_time_stamp = None
            time_dims = ['Time', 'time', 'valid_time', 'Times']
            for td in time_dims:
                if td in dataset.coords and pd.api.types.is_datetime64_any_dtype(dataset[td]):
                    file_time_stamp = pd.to_datetime(dataset[td].values[0])
                    break
                elif td in dataset.variables and dataset[td].dims == ('Time',) and td == 'valid_time': # Specific check for valid_time
                    file_time_stamp = pd.to_datetime(dataset[td].values[0])
                    break
                elif td == 'Times' and 'Times' in dataset.variables:
                    try:
                        times_decoded = [t.tobytes().decode('utf-8') for t in dataset['Times'].values]
                        if times_decoded: file_time_stamp = pd.to_datetime(times_decoded[0], format='%Y-%m-%d_%H:%M:%S', errors='coerce')
                        if file_time_stamp is not None and pd.isna(file_time_stamp): file_time_stamp = pd.to_datetime(times_decoded[0], errors='coerce')
                        if file_time_stamp is not None and pd.isna(file_time_stamp): sys.stderr.write(f"[WARNING] Unrecognized 'Times' format for {file_path.name}: {times_decoded[0]}\n")
                        else: break
                    except Exception: sys.stderr.write(f"[WARNING] Error decoding 'Times' for {file_path.name}.\n")
            
            if file_time_stamp is None or pd.isna(file_time_stamp):
                sys.stderr.write(f"[ERROR] No valid timestamp in {file_path.name}. Skipping.\n")
                return data_records_by_variable

            # Prepare data variables for pixel extraction
            vars_for_extraction = {}
            for var_name in data_variables_to_extract:
                if var_name in dataset.data_vars:
                    da = dataset[var_name]
                    # Ensure CRS is set on the DataArray too, for .sel method with coordinates
                    if not da.rio.crs: da = da.rio.write_crs(global_crs, inplace=False)
                    
                    # Ensure 2D (y, x) data for selection. If 3D (Time, y, x), take the first time slice.
                    if ('Time' in da.dims or 'time' in da.dims) and da.ndim == 3:
                        vars_for_extraction[var_name] = da.isel({d: 0 for d in da.dims if d in ['Time', 'time']}).squeeze()
                    elif da.ndim == 2 and 'y' in da.dims and 'x' in da.dims:
                        vars_for_extraction[var_name] = da
                    else:
                        sys.stderr.write(f"[WARNING] Skipping '{var_name}' in {file_path.name}: Unexpected dims {da.dims}.\n")
                else: sys.stderr.write(f"[WARNING] '{var_name}' not in {file_path.name}.\n")

            if all_divide_pixels_df.empty:
                sys.stderr.write(f"[WARNING] No pixels for extraction in {file_path.name}. Skipping.\n")
                return data_records_by_variable

            pixel_x_coords = xr.DataArray(all_divide_pixels_df['x'].values, dims='point')
            pixel_y_coords = xr.DataArray(all_divide_pixels_df['y'].values, dims='point')

            for var_name, var_data_array in vars_for_extraction.items():
                try:
                    selected_values_series = pd.Series(
                        var_data_array.sel(x=pixel_x_coords, y=pixel_y_coords, method='nearest').values,
                        index=all_divide_pixels_df.index
                    )
                    extracted_df = pd.DataFrame({
                        'divide_id': all_divide_pixels_df['divide_id'],
                        'value': selected_values_series
                    }).dropna(subset=['value'])

                    if not extracted_df.empty:
                        avg_values = extracted_df.groupby('divide_id')['value'].mean()
                        for div_id, avg_val in avg_values.items():
                            data_records_by_variable[var_name].append({"time": file_time_stamp, "divide_id": div_id, var_name: float(avg_val)})
                    else: sys.stderr.write(f"[INFO] No valid pixel values for '{var_name}' in {file_path.name}.\n")
                except Exception as e: sys.stderr.write(f"[ERROR] Pixel extraction for '{var_name}' in {file_path.name} failed: {e}\n")

    except Exception as e: sys.stderr.write(f"[ERROR] Failed to process {file_path.name}: {e}\n")
    return data_records_by_variable

# --- Main Execution ---

if __name__ == '__main__':
    freeze_support()

    parser = argparse.ArgumentParser(
        description="Process NWM NetCDF files for divide pixels, save as CSVs (HPC optimized).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # MODIFIED: Use nargs='+' to accept multiple directories
    parser.add_argument(
        "--input_netcdf_dirs", 
        nargs='+', # Accepts one or more directory paths
        default=["/scratch/dlfernando/netcdf_hawii"], # Default is now a list
        help="Space-separated list of directories containing NetCDF files (e.g., /path/to/1994 /path/to/1995)."
    )
    parser.add_argument("--pixel_data_file", type=str, default="/home/dlfernando/LDAS/all_divide_pixels.json", help="Path to JSON pixel data.")
    parser.add_argument("--output_csv_dir", type=str, default="/scratch/dlfernando/processed_nwm_data", help="Directory for CSV output.")
    parser.add_argument("--variables", nargs='+', default=["LWDOWN", "PSFC", "Q2D", "SWDOWN", "T2D", "U2D", "V2D", "RAINRATE"], help="Variables to extract.")
    parser.add_argument("--max_files", type=int, default=0, help="Max files to process (0 for all).")
    parser.add_argument("--num_processes", type=int, default=None, help="Number of cores (None for cpu_count - 1).")

    args = parser.parse_args()

    # Convert paths. input_netcdf_dirs is now a list of strings, convert each to Path.
    input_netcdf_folders = [Path(d) for d in args.input_netcdf_dirs] 
    pixel_data_file = Path(args.pixel_data_file)
    output_csv_folder = Path(args.output_csv_dir)
    variables_to_process = args.variables
    num_processes = args.num_processes
    max_files_to_process = args.max_files

    output_csv_folder.mkdir(parents=True, exist_ok=True)
    print(f"Output CSVs will be saved in: {output_csv_folder}")

    # MODIFIED: Collect files from ALL specified input directories
    all_netcdf_files = []
    for folder in input_netcdf_folders:
        if not folder.is_dir():
            sys.stderr.write(f"[WARNING] Input directory not found: {folder}. Skipping.\n")
            continue
        all_netcdf_files.extend(sorted(list(folder.glob("**/*.LDASIN_DOMAIN1*"))))

    if not all_netcdf_files:
        sys.exit(f"Error: No '.LDASIN_DOMAIN1' files found in any specified input directories: {args.input_netcdf_dirs}.")

    files_for_processing = all_netcdf_files[:max_files_to_process] if max_files_to_process > 0 and max_files_to_process < len(all_netcdf_files) else all_netcdf_files
    print(f"Processing {len(files_for_processing)} of {len(all_netcdf_files)} available files across all input directories.")
    if not files_for_processing: sys.exit("No NetCDF files selected for processing. Exiting.")

    print("\n--- Initial Setup: Understanding Data and Pixels ---")
    global_crs, global_dx, global_dy, global_x_orig, global_y_orig = None, None, None, None, None
    try:
        with xr.open_dataset(files_for_processing[0], decode_coords=False, decode_times=True) as sample_ds:
            global_crs, global_dx, global_dy, global_x_orig, global_y_orig = get_global_nwm_spatial_params(sample_ds)
        print(f"✅ Global NetCDF spatial parameters identified using sample file.")
    except Exception as e: sys.exit(f"Error: Couldn't read first NetCDF file to get global spatial parameters: {e}")

    try:
        all_divide_pixels_df = load_divide_pixels_from_json(pixel_data_file)
        if all_divide_pixels_df.empty: sys.exit("[WARNING] No pixel data loaded. Exiting.")
    except Exception as e: sys.exit(f"Error loading pixel data: {e}")

    print("\n--- Processing NetCDF Files (Parallel) ---")
    num_processes_to_use = num_processes if num_processes is not None else max(1, cpu_count() - 1)
    print(f"Using {num_processes_to_use} cores for faster processing.")

    all_combined_data_by_variable = {var: [] for var in variables_to_process}
    # Pass all necessary global spatial parameters to each worker
    task_args = [(f, all_divide_pixels_df, global_crs, global_dx, global_dy, global_x_orig, global_y_orig, variables_to_process) for f in files_for_processing]

    with Pool(processes=num_processes_to_use) as pool:
        results = list(tqdm(pool.starmap(process_one_netcdf_file, task_args), total=len(files_for_processing), desc="Processing NetCDF files"))
        for file_var_records_dict in results:
            for var_name, records in file_var_records_dict.items():
                if records: all_combined_data_by_variable[var_name].extend(records)

    print("\n--- Organizing and Saving Data to CSV Files ---")
    for var_name, records_list in all_combined_data_by_variable.items():
        if not records_list:
            print(f"[INFO] No valid data found for '{var_name}'. Skipping CSV.")
            continue

        df_for_current_run_variable = pd.DataFrame(records_list)
        df_for_current_run_variable['time'] = pd.to_datetime(df_for_current_run_variable['time'])

        print(f"Aggregating '{var_name}' data...")
        # Pivot current run's data
        current_run_time_series_df = df_for_current_run_variable.pivot_table(
            index='time', columns='divide_id', values=var_name, aggfunc='mean'
        ).reset_index()
        current_run_time_series_df.columns.name = None # Clean up the column name index

        csv_filename = output_csv_folder / f'{var_name}_TimeSeries.csv'
        
        # --- MODIFIED: Append/Merge logic for existing CSV ---
        if csv_filename.exists():
            print(f"[INFO] Merging data for '{var_name}' with existing file: {csv_filename}")
            existing_time_series_df = pd.read_csv(csv_filename, parse_dates=['time'])
            
            # Combine current and existing data
            # Use pd.concat and then drop duplicates based on 'time' to handle overwrites correctly
            combined_df = pd.concat([existing_time_series_df, current_run_time_series_df], ignore_index=True)
            
            # Drop duplicates based on the 'time' column, keeping the last occurrence (from the current run)
            # This handles cases where a timestamp might appear in both the existing and new data.
            # If you expect unique timestamps per file, `drop_duplicates` keeping `last` is safe.
            # If multiple runs could genuinely produce different values for the same timestamp,
            # you might need a more complex merge/aggregation strategy here.
            final_time_series_df = combined_df.drop_duplicates(subset='time', keep='last')
            
            # Ensure the columns are sorted (optional, but good for consistency)
            # Get all unique divide_ids from both old and new data to ensure all columns exist
            all_divide_ids = sorted(list(set(existing_time_series_df.columns.drop('time')).union(
                                        set(current_run_time_series_df.columns.drop('time')))))
            
            # Reindex to ensure all divide_id columns are present and in order
            final_time_series_df = final_time_series_df.set_index('time')[all_divide_ids].reset_index()
            final_time_series_df = final_time_series_df.sort_values(by='time').reset_index(drop=True)

            # Save the combined, deduplicated, and sorted data back to the CSV (overwriting the old one)
            final_time_series_df.to_csv(csv_filename, index=False)
            print(f"✅ Data for '{var_name}' (merged) saved to: {csv_filename}")
        else:
            # If the file doesn't exist, just save the current run's data
            current_run_time_series_df.to_csv(csv_filename, index=False)
            print(f"✅ Data for '{var_name}' saved to: {csv_filename}")

    print(f"\n🎉 All done! Processed data in: {output_csv_folder}")

    # Display sample data (modified to use the actual saved file if it exists)
    if output_csv_folder.exists() and any(output_csv_folder.glob("*_TimeSeries.csv")):
        # Try to read one of the generated CSVs to display a sample
        first_csv = next(output_csv_folder.glob("*_TimeSeries.csv"), None)
        if first_csv:
            print(f"\nHere's a peek at the combined time series data from {first_csv.name}:")
            try:
                sample_df = pd.read_csv(first_csv, parse_dates=['time'])
                print(sample_df.head().to_string())
            except Exception as e:
                print(f"Could not read sample CSV {first_csv.name} for display: {e}")
        else:
            print("No time series CSVs were generated. Double-check settings and pixel data.")
    else: 
        print("No time series data was generated for any variable. Double-check settings and pixel data.")

