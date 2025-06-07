import xarray as xr
import geopandas as gpd
import rioxarray
import pandas as pd
import matplotlib.pyplot as plt
import fiona

def inspect_gpkg(gpkg_path, display=False):
    """
    Reads and plots all layers in a GeoPackage file.

    Parameters:
    - gpkg_path (str): Path to the GeoPackage (.gpkg) file.
    """
    print(f"\nReading GeoPackage: {gpkg_path}")
    layer_names = fiona.listlayers(gpkg_path)
    all_layers = {}

    for layer in layer_names:
        print(f"\nReading layer: {layer}")
        gdf = gpd.read_file(gpkg_path, layer=layer)
        print(gdf)
        all_layers[layer] = gdf
    if display:
        for name, gdf in all_layers.items():
            if gdf.empty or not gdf.geometry.is_valid.all():
                print(f"Skipping layer '{name}': empty or invalid geometry.")
                continue
            try:
                print(f"Plotting {name}...")
                gdf.plot()
                plt.title(f"{name} Layer")
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Could not plot layer '{name}': {e}")


def inspect_netcdf(netcdf_file, display=False):
    """
    Loads and plots variables from a NetCDF file.

    Parameters:
    - netcdf_file (str): Path to the NetCDF file.
    """
    print(f"\nLoading NetCDF file: {netcdf_file}")
    ds = xr.open_dataset(netcdf_file)
    print("NetCDF file loaded successfully.")

    print("\nVariables in dataset:")
    print(list(ds.data_vars))

    def plot_variable(var_name, data_array):
        dims = data_array.dims
        print(f"\nProcessing variable: {var_name} with dimensions {dims}")

        try:
            if data_array.ndim == 0:
                print(f"Skipping '{var_name}': scalar (no plot needed).")

            elif data_array.ndim == 1:
                x = data_array[dims[0]]
                y = data_array.values
                plt.plot(x, y)
                plt.xlabel(dims[0])
                plt.ylabel(var_name)
                plt.title(f"{var_name} over {dims[0]}")
                plt.grid(True)
                plt.tight_layout()
                plt.show()

            elif data_array.ndim == 2:
                plt.imshow(data_array.values, aspect='auto', cmap='viridis')
                plt.colorbar(label=var_name)
                plt.title(f"{var_name} ({dims[0]} x {dims[1]})")
                plt.xlabel(dims[1])
                plt.ylabel(dims[0])
                plt.tight_layout()
                plt.show()

            elif data_array.ndim == 3:
                print(f"'{var_name}' is 3D — plotting first slice along 0-axis")
                plt.imshow(data_array.isel({dims[0]: 0}).values, aspect='auto', cmap='viridis')
                plt.colorbar(label=var_name)
                plt.title(f"{var_name} [{dims[0]}=0] ({dims[1]} x {dims[2]})")
                plt.xlabel(dims[2])
                plt.ylabel(dims[1])
                plt.tight_layout()
                plt.show()

            else:
                print(f"Skipping '{var_name}': {data_array.ndim}D is unsupported.")
        except Exception as e:
            print(f"Error plotting '{var_name}': {e}")

    for var_name in ds.data_vars:
        var_data = ds[var_name]
        try:
            df_var = var_data.to_dataframe().reset_index()
            print("\nSample of " + var_name + " timeseries data:")
            print(df_var)
            if display:
                plot_variable(var_name, var_data)
        except Exception as e:
            print(f"\nCould not convert '{var_name}' to DataFrame. Attempting to print scalar or summary:")
            try:
                print(f"{var_name}: {var_data.values.item()}")
            except:
                print(f"{var_name}: {var_data.values}")

gpkg_path = r"C:\Users\colli\Downloads\hi_reference.gpkg"
inspect_gpkg(gpkg_path, display=False)
netcdf_file = r"C:\Users\colli\Downloads\ldasin\2004010100.LDASIN_DOMAIN1"
inspect_netcdf(netcdf_file, display=False)
ds = xr.open_dataset(netcdf_file)
#print(ds.coords)
