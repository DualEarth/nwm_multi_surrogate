import pickle
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader
import argparse
import pandas as pd

from neuralhydrology.evaluation import metrics
from neuralhydrology.nh_run import start_run, eval_run
from neuralhydrology.utils.config import Config
from neuralhydrology.evaluation.evaluate import start_evaluation 
from neuralhydrology.evaluation.metrics import calculate_all_metrics


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a NeuralHydrology run and plot/metric obs vs. sim"
    )
    p.add_argument("run_dir", type=Path,
                   help="Directory of the trained run (e.g. runs/.../your_run_1706_145615)")
    p.add_argument("--period", choices=["train", "validation", "test"],
                   default="test", help="Which period to evaluate")
    p.add_argument("--epoch", type=int, default=None,
                   help="Epoch number to evaluate (e.g. 15). Uses last if omitted.")
    return p.parse_args()


def find_epoch_folder(run_dir: Path, period: str, epoch: int = None) -> Path:
    """Return the Path to the model_epochXXX folder to load."""
    period_folder = run_dir / period
    epoch_folders = sorted(d for d in period_folder.iterdir()
                           if d.is_dir() and d.name.startswith("model_epoch"))
    if not epoch_folders:
        raise RuntimeError(f"No epoch folders found under {period_folder}")
    if epoch is None:
        return epoch_folders[-1]
    target = period_folder / f"model_epoch{epoch:03d}"
    if not target.exists():
        raise RuntimeError(f"Epoch folder {target.name} not found")
    return target


def find_results_file(epoch_folder: Path) -> Path:
    """Locate the single .p results file in the epoch folder."""
    p_files = list(epoch_folder.glob("*.p"))
    if not p_files:
        raise RuntimeError(f"No .p results file in {epoch_folder}")
    return p_files[0]


def run_and_load_results(run_dir: Path, period: str, epoch: int = None) -> dict:
    """Run `eval_run`, then load and return the results dict."""
    # this writes out the .p file(s)
    eval_run(run_dir=run_dir, period=period)
    # find the folder and file we just produced
    epoch_folder = find_epoch_folder(run_dir, period, epoch)
    results_file = find_results_file(epoch_folder)
    # load pickle
    with open(results_file, "rb") as fp:
        return pickle.load(fp)


def plot_obs_sim(dates, obs, sim, title: str, ylabel: str, save_fig: bool = False, out_dir: Path = None):
    """Plot observed vs. simulated time series."""
    plt.figure(figsize=(10, 4))
    plt.plot(dates, obs, label="Observed")
    plt.plot(dates, sim, label="Simulated")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    if save_fig and out_dir is not None:
        fig_path = out_dir / f"{title.replace(' ', '_')}.png"
        plt.savefig(fig_path)
        print(f"Saved figure to {fig_path}")
    else:
        plt.show()

def compute_and_print_metrics(obs, sim, dates=None):
    """
    Wrap obs/sim into xr.DataArray with a 'date' dimension so that
    calculate_all_metrics finds a 'date' coord and can compute the full suite.
    """
    def to_dataarray(arr, dates):
        # 1) Extract raw data and coordinate values
        if isinstance(arr, xr.DataArray):
            return arr.rename({'time': 'date'})  # if it already exists, just rename
        elif isinstance(arr, pd.Series):
            data = arr.values
            coord_vals = arr.index.values
        elif isinstance(arr, np.ndarray):
            data = arr
            if dates is not None:
                coord_vals = (dates.values
                              if isinstance(dates, pd.DatetimeIndex)
                              else np.array(dates))
            else:
                coord_vals = np.arange(data.shape[0])
        else:
            raise ValueError(f"Cannot convert type {type(arr)} to DataArray")

        # 2) Build DataArray with a 'date' dimension
        return xr.DataArray(data, coords=[coord_vals], dims=["date"])

    # Convert both obs and sim
    da_obs = to_dataarray(obs, dates)
    da_sim = to_dataarray(sim, dates)

    # Drop NaNs so metrics only sees valid pairs
    mask = (~np.isnan(da_obs)) & (~np.isnan(da_sim))
    da_obs = da_obs.where(mask, drop=True)
    da_sim = da_sim.where(mask, drop=True)

    # Need at least two points
    if da_obs.size < 2:
        print("    Not enough data points to compute metrics (need ≥2).")
        return

    # Compute full suite, fallback to NSE on any error
    try:
        values = metrics.calculate_all_metrics(da_obs, da_sim)
    except Exception as e:
        print(f"    Warning: full‐suite metrics failed ({e}). Computing NSE only.")
        try:
            nse_val = metrics.nse(da_obs, da_sim)
            print(f"    NSE: {nse_val:.3f}")
        except Exception as e2:
            print(f"    Even NSE failed: {e2}")
        return

    # Print results
    for key, val in values.items():
        print(f"    {key}: {val:.3f}")


def process_and_plot(results: dict, save_figures: bool = False, fig_dir: Path = None):
    for basin_id, basin_res in results.items():
        print(f"\n=== Basin {basin_id} ===")
        for freq, res in basin_res.items():
            print(f"\n-- Frequency: {freq} --")
            xr_ds = res["xr"]
            dates = xr_ds.indexes["date"]  # pandas.DatetimeIndex

            obs_vars = [v for v in xr_ds.data_vars if v.endswith("_obs")]
            for obs_var in obs_vars:
                var_name = obs_var[:-4]
                sim_var = f"{var_name}_sim"
                if sim_var not in xr_ds.data_vars:
                    continue

                # Plot at horizon 0
                obs0 = xr_ds[obs_var].sel(time_step=0).values
                sim0 = xr_ds[sim_var].sel(time_step=0).values
                title = f"{basin_id} [{freq}] {var_name} (h=0)"
                plot_obs_sim(dates, obs0, sim0, title, var_name,
                             save_fig=save_figures, out_dir=fig_dir)

                # Compute metrics at final horizon (-1)
                print(f"\nMetrics for {var_name} at final horizon:")
                obs_last = xr_ds[obs_var].sel(time_step=-1).values
                sim_last = xr_ds[sim_var].sel(time_step=-1).values
                compute_and_print_metrics(obs_last, sim_last, dates=dates)

def main():
    args = parse_args()
    # Optional: create a directory to save figures
    # fig_dir = args.run_dir / "evaluation_figures"
    # fig_dir.mkdir(exist_ok=True)

    # 1) run evaluation & load results
    results = run_and_load_results(args.run_dir, args.period, args.epoch)

    # 2) process, plot, and print metrics
    process_and_plot(results, save_figures=False)


if __name__ == "__main__":
    main()

