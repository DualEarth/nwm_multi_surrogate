import xarray as xr
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from pathlib import Path
from pyproj import CRS
import rioxarray
import concurrent.futures

# ------------------------ CONFIG ------------------------
ldas_file = "./netcdf3/2006010100.LDASIN_DOMAIN1"
divide_gpkg = "./hi_reference.gpkg"
output_dir = Path("pixel_indices")
output_dir.mkdir(exist_ok=True)

# ------------------------ LOAD NETCDF ------------------------
print("[INFO] Loading NetCDF grid...")
ds = xr.open_dataset(ldas_file, engine="netcdf4")
cf_attrs = ds['lambert_conformal_conic'].attrs
crs = CRS.from_cf(cf_attrs)

if 'south_north' in ds.dims:
    ds = ds.rename({'south_north': 'y'})
if 'west_east' in ds.dims:
    ds = ds.rename({'west_east': 'x'})

ny, nx = ds.dims['y'], ds.dims['x']
x = np.linspace(-nx / 2 * 1000, nx / 2 * 1000, nx)
y = np.linspace(-ny / 2 * 1000, ny / 2 * 1000, ny)
ds = ds.assign_coords(x=("x", x), y=("y", y))
ds = ds.rio.write_crs(crs, inplace=False)

# ------------------------ COMPUTE CENTROIDS ------------------------
print("[INFO] Computing pixel centroids...")
xv, yv = np.meshgrid(ds.x.values, ds.y.values)
points = np.column_stack((xv.flatten(), yv.flatten()))
i_indices = np.repeat(np.arange(ny), nx)
j_indices = np.tile(np.arange(nx), ny)

# ------------------------ LOAD DIVIDES ------------------------
print("[INFO] Loading divides...")
divides = gpd.read_file(divide_gpkg, layer="divides").to_crs(crs)

# ------------------------ WORKER FUNCTION ------------------------
def process_divide(row_data):
    idx, divide = row_data
    divide_id = divide["ID"]
    geom = divide.geometry

    inside_mask = np.array([geom.contains(Point(x, y)) for x, y in points])
    selected_i = i_indices[inside_mask]
    selected_j = j_indices[inside_mask]

    output_file = output_dir / f"divide_{divide_id}.txt"
    with open(output_file, "w") as f:
        for i, j in zip(selected_i, selected_j):
            f.write(f"{i} {j}\n")

    return f"[✓] Divide {divide_id}: {len(selected_i)} pixels"

# ------------------------ MULTIPROCESSING EXECUTION ------------------------
if __name__ == "__main__":
    print("[INFO] Starting multiprocessing...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_divide, divides.iterrows()))

    print("\n".join(results))
    print("[✅ DONE] All divide pixel indices saved.")