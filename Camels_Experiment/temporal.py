import pickle
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import re
from multiprocessing import Pool, cpu_count
from functools import partial
import matplotlib.lines as mlines

# --- Utility Functions ---

def calculate_nse(obs: pd.Series, sim: pd.Series) -> float:
    df = pd.DataFrame({'obs': obs, 'sim': sim}).dropna()
    denominator = ((df['obs'] - df['obs'].mean())**2).sum()
    if denominator == 0:
        return np.nan
    return 1 - ((df['obs'] - df['sim'])**2).sum() / denominator

def rolling_nse_series(obs: pd.Series, sim: pd.Series, window: int) -> pd.Series:
    if len(obs) < window or len(sim) < window:
        return pd.Series([], dtype='float64')
    return pd.Series([calculate_nse(obs.iloc[i:i+window], sim.iloc[i:i+window])
                      if len(obs.iloc[i:i+window].dropna()) == window and
                         len(sim.iloc[i:i+window].dropna()) == window
                      else np.nan
                      for i in range(len(obs) - window + 1)],
                     index=obs.index[window - 1:])

def process_single_basin(basin_tuple, window: int):
    basin_id, entry = basin_tuple
    result = []
    try:
        ds = entry['1D']['xr']
        if isinstance(ds, xr.DataArray):
            ds = ds.to_dataset(name='data_var')
        df = ds.to_dataframe().reset_index()
        df['basin_id'] = str(basin_id)
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
        else:
            df['time'] = pd.date_range(start='1980-01-01', periods=len(df), freq='D')
        var_bases = set()
        for col in df.columns:
            m_obs = re.match(r"(.+)_obs$", col)
            m_sim = re.match(r"(.+)_sim$", col)
            if m_obs:
                var_bases.add(m_obs.group(1))
            elif m_sim:
                var_bases.add(m_sim.group(1))
        for var in var_bases:
            obs_col = f"{var}_obs"
            sim_col = f"{var}_sim"
            if obs_col in df.columns and sim_col in df.columns:
                obs = df[obs_col]
                sim = df[sim_col]
                r_nse = rolling_nse_series(obs, sim, window)
                if not r_nse.empty:
                    result.append(pd.DataFrame({
                        'time': df['time'].iloc[window - 1: window - 1 + len(r_nse)],
                        'rolling_nse': r_nse.values,
                        'basin_id': str(basin_id),
                        'variable': var
                    }))
    except Exception as e:
        print(f"Failed basin {basin_id}: {e}")
    return pd.concat(result) if result else None

def normalize_to_0_1(series: pd.Series) -> pd.Series:
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(0, index=series.index)
    return (series - min_val) / (max_val - min_val)

# --- Plotting Functions (titles removed, all fonts 12pt) ---

def plot_combined_nnse_extrapolation(df_all, extrap_csv_path, output_dir, selected_basins):
    if df_all.empty:
        print("Warning: No NSE data found for the specified basin list.")
        return

    df_extrap = pd.read_csv(extrap_csv_path)
    extrap_dist_col = 'distance_mahalanobis'
    if extrap_dist_col not in df_extrap.columns:
        raise KeyError(f"The required column '{extrap_dist_col}' was not found in the extrapolation data.")

    df_extrap['basin_id'] = df_extrap['basin_id'].astype(str).str.zfill(8)
    selected_basins_extrap = df_all['basin_id'].unique()
    df_extrap_filtered = df_extrap[df_extrap['basin_id'].isin(selected_basins_extrap)].copy()

    if df_extrap_filtered.empty:
        print("Warning: No extrapolation data found for the specified basin list.")
        return

    os.makedirs(output_dir, exist_ok=True)
    variables = df_all['variable'].unique()
    basin_colors = {'12013500': '#E41A1C', '06409000': '#377EB8', '06452000': '#4DAF4A'}
    basin_labels = {'12013500': "Best Basin (12013500)", '06409000': "Worst Basin (06409000)", '06452000': "Unique Basin (06452000)"}

    df_all['NNSE'] = 1 / (2 - df_all['rolling_nse'])
    df_extrap_filtered['extrap_dist_norm'] = normalize_to_0_1(df_extrap_filtered[extrap_dist_col])

    for var in variables:
        plt.figure(figsize=(16, 8))
        nse_handles, mahal_handles = [], []

        df_all['timestep_idx'] = df_all.groupby('basin_id').cumcount()
        df_extrap_filtered['timestep_idx'] = df_extrap_filtered.groupby('basin_id').cumcount()
        df_merged_full = pd.merge(
            df_all[['basin_id', 'variable', 'time', 'NNSE', 'timestep_idx']],
            df_extrap_filtered[['basin_id', 'timestep_idx', 'extrap_dist_norm']],
            on=['basin_id', 'timestep_idx'], how='inner'
        )

        for basin_id in sorted(selected_basins_extrap):
            df_merged = df_merged_full[(df_merged_full['basin_id'] == basin_id) &
                                       (df_merged_full['variable'] == var)].copy()
            if df_merged.empty:
                continue
            color = basin_colors.get(basin_id, 'black')
            current_label = basin_labels.get(basin_id, basin_id)

            plt.plot(df_merged['time'], df_merged['NNSE'], linewidth=1.5, color=color, alpha=0.9, linestyle='-')
            plt.plot(df_merged['time'], df_merged['extrap_dist_norm'], linewidth=1.5, color=color, alpha=0.9, linestyle='--')

            nse_handles.append(mlines.Line2D([], [], color=color, linestyle='-', label=f'{current_label} (NNSE)'))
            mahal_handles.append(mlines.Line2D([], [], color=color, linestyle='--', label=f'{current_label} (Mahalanobis)'))

        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.grid(True, linestyle=':')
        plt.ylim(0, 1)
        plt.tick_params(axis='both', which='major', labelsize=13)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        plt.legend(handles=nse_handles + mahal_handles, title="Trends by Basin", loc='upper right', fontsize=13, title_fontsize=12)

        safe_var_name = re.sub(r'[^A-Za-z0-9_.]+', '_', var)
        plot_path = os.path.join(output_dir, f'combined_trends_all_basins_by_date_{safe_var_name}.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved combined plot for all basins to: {plot_path}")


def plot_individual_nnse_extrapolation(df_all, extrap_csv_path, output_dir):
    if df_all.empty:
        print("Warning: No NSE data found for the specified basins.")
        return

    df_extrap = pd.read_csv(extrap_csv_path)
    extrap_dist_col = 'distance_mahalanobis'
    if extrap_dist_col not in df_extrap.columns:
        raise KeyError(f"The required column '{extrap_dist_col}' was not found in the extrapolation data.")

    df_extrap['basin_id'] = df_extrap['basin_id'].astype(str).str.zfill(8)
    selected_basins_extrap = df_all['basin_id'].unique()
    df_extrap_filtered = df_extrap[df_extrap['basin_id'].isin(selected_basins_extrap)].copy()

    if df_extrap_filtered.empty:
        print("Warning: No extrapolation data found for the specified basins.")
        return

    os.makedirs(output_dir, exist_ok=True)
    variables = df_all['variable'].unique()
    colors_vibrant = ['#E41A1C', '#377EB8']  # Red, Blue

    df_all['NNSE'] = 1 / (2 - df_all['rolling_nse'])
    df_extrap_filtered['extrap_dist_norm'] = normalize_to_0_1(df_extrap_filtered[extrap_dist_col])

    for var in variables:
        for basin_id in sorted(selected_basins_extrap):
            fig, ax = plt.subplots(figsize=(16, 8))
            df_basin = df_all[(df_all['basin_id'] == basin_id) & (df_all['variable'] == var)].copy()
            df_extrap_basin = df_extrap_filtered[df_extrap_filtered['basin_id'] == basin_id].copy()
            if df_basin.empty or df_extrap_basin.empty:
                plt.close(fig)
                continue

            df_basin['timestep_idx'] = range(len(df_basin))
            df_extrap_basin['timestep_idx'] = range(len(df_extrap_basin))
            df_merged = pd.merge(
                df_basin[['time', 'NNSE', 'timestep_idx']],
                df_extrap_basin[['extrap_dist_norm', 'timestep_idx']],
                on='timestep_idx', how='inner'
            )

            ax.plot(df_merged['time'], df_merged['NNSE'], linewidth=2.5, color=colors_vibrant[0], linestyle='-', label='NNSE')
            ax.plot(df_merged['time'], df_merged['extrap_dist_norm'], linewidth=2.5, color=colors_vibrant[1], linestyle='--', label='Normalized Mahalanobis Distance')

            ax.set_xlabel('Date', fontsize=13)
            ax.set_ylabel('Value', fontsize=13)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.set_ylim(0, 1)
            ax.tick_params(axis='both', which='major', labelsize=13)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            fig.autofmt_xdate()
            ax.legend(fontsize=13, loc='upper right')

            plt.tight_layout()
            safe_var_name = re.sub(r'[^A-Za-z0-9_.]+', '_', var)
            plot_path = os.path.join(output_dir, f'basin_{basin_id}_trends_{safe_var_name}.png')
            plt.savefig(plot_path, dpi=300)
            plt.close(fig)
            print(f"Saved plot for basin {basin_id} to: {plot_path}")


# --- Main Execution Block ---
if __name__ == "__main__":
    PICKLE_PATH = r"C:\Users\deonf\Model_work\runs\my_camels_run\lstm_camels_custom_0408_130226\test\model_epoch003\test_results.p"
    EXTRAP_CSV_PATH = r"C:\Users\deonf\Model_work\runs\my_camels_run\lstm_camels_custom_0408_130226\internal_states\test\all_basins_colored_detailed.csv"

    INDIVIDUAL_OUTPUT_DIR = os.path.join(os.path.dirname(PICKLE_PATH), "individual_nnse_plots_updated")
    COMBINED_OUTPUT_DIR = os.path.join(os.path.dirname(PICKLE_PATH), "combined_nnse_plots_updated")

    WINDOW = 30
    N_CORES = 12
    selected_basins = ['12013500', '06409000', '06452000']

    print("Loading data and computing rolling NNSE for specified basins...")
    with open(PICKLE_PATH, "rb") as f:
        data = pickle.load(f)

    basin_items = [(k, v) for k, v in data.items() if str(k) in selected_basins]

    if not basin_items:
        print("Error: None of the specified basins were found in the test results.")
    else:
        with Pool(processes=N_CORES) as pool:
            func = partial(process_single_basin, window=WINDOW)
            results = pool.map(func, basin_items)
        df_all = pd.concat([r for r in results if r is not None], ignore_index=True)

        os.makedirs(INDIVIDUAL_OUTPUT_DIR, exist_ok=True)
        df_all.to_csv(os.path.join(INDIVIDUAL_OUTPUT_DIR, "rolling_nse_selected_basins.csv"), index=False)
        print(f"Rolling NSE results saved to: {os.path.join(INDIVIDUAL_OUTPUT_DIR, 'rolling_nse_selected_basins.csv')}")

        print("\nPlotting individual NNSE and Mahalanobis distance for each basin...")
        plot_individual_nnse_extrapolation(df_all, EXTRAP_CSV_PATH, INDIVIDUAL_OUTPUT_DIR)

        print("\nPlotting combined NNSE and Mahalanobis distance for all basins...")
        plot_combined_nnse_extrapolation(df_all, EXTRAP_CSV_PATH, COMBINED_OUTPUT_DIR, selected_basins)

    print("\nScript execution finished. Check the output directories for plots.")
