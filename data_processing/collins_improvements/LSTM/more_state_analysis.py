import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.covariance import EmpiricalCovariance
from umap import UMAP
import statsmodels.api as sm
from torch.utils.data import DataLoader
from scipy.spatial import ConvexHull
from matplotlib.path import Path as MplPath
import pyvista as pv
import logging
# from tensorly.decomposition import tucker
# import tensorly as tl
import torch
import torch.nn as nn


from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.datautils.utils import load_scaler
from neuralhydrology.modelzoo.customlstm import CustomLSTM
from neuralhydrology.utils.config import Config
from neuralhydrology.modelzoo.customlstm import CudaLSTM

LOGGER = logging.getLogger(__name__)

class StateReducer(nn.Module):
    """
    A simple neural network module to reduce the dimensionality of LSTM hidden states.
    Uses a stack of linear layers with ReLU activations for non-linear reduction.
    """
    def __init__(self, original_dim: int = 32, reduced_dim: int = 3):
        super().__init__()
        if reduced_dim >= original_dim:
            raise ValueError(f"Reduced dimension ({reduced_dim}) must be less than original dimension ({original_dim}).")
        # Example of a non-linear reduction block
        # You can customize these layers based on your needs
        self.reducer = nn.Sequential(
            nn.Linear(original_dim, 128),  # First intermediate layer
            nn.ReLU(),
            nn.Linear(128, 64),           # Second intermediate layer
            nn.ReLU(),
            nn.Linear(64, reduced_dim)    # Output layer to the reduced dimension
            # No activation on the final layer if you want raw values for convex hull
        )
        LOGGER.info(f"StateReducer initialized: {original_dim}D -> {reduced_dim}D")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the reducer.
        Args:
            x (torch.Tensor): Input tensor, expected to be an LSTM state (batch_size, original_dim).
                              Note: neuralhydrology's h_n/c_n is (num_layers, batch_size, hidden_size).
                              We'll assume we take the last layer's state.
        Returns:
            torch.Tensor: The reduced-dimensional representation (batch_size, reduced_dim).
        """
        # If x comes from h_n/c_n, its shape is (num_layers, batch_size, hidden_size).
        # We need to select the last layer's state for reduction.
        # Assuming last layer is the one we want to reduce:
        if x.dim() == 3: # If shape is (num_layers, batch_size, hidden_size)
            x = x[-1, :, :] # Take the last layer's state: (batch_size, hidden_size)
        elif x.dim() != 2: # Should be (batch_size, hidden_size) or (num_layers, batch_size, hidden_size)
            raise ValueError(f"Unexpected input shape for StateReducer: {x.shape}")
        return self.reducer(x)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze LSTM cell states and diagnostics from a trained CustomLSTM model"
    )
    parser.add_argument(
        "--run_dir", type=Path, required=True,
        help="Directory containing training artifacts (model, scaler, etc.)"
    )
    parser.add_argument(
        "--epoch", type=int, required=True,
        help="Epoch number of the model checkpoint to load (e.g., 3 for model_epoch003.pt)"
    )
    parser.add_argument(
        "--config", type=Path, default="quick_config.yml",
        help="YAML config file for CustomLSTM"
    )
    parser.add_argument(
        "--start_date", type=str, default="2005-12-01",
        help="Start date for the time series (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1000,
        help="Batch size for the test DataLoader"
    )
    return parser.parse_args()


def load_data_and_model(run_dir: Path, epoch: int, config_file: Path, batch_size: int):
    # Load configuration and model
    cfg = Config(config_file)
    model_path = run_dir / f"model_epoch{epoch:03d}.pt"
    weights = torch.load(str(model_path), map_location="cuda:0")

    cuda_model = CudaLSTM(cfg=cfg)
    cuda_model.load_state_dict(weights)
    model = CustomLSTM(cfg=cfg)
    model.copy_weights(cuda_model)
    model.eval()

    # Load scaler and dataset
    scaler = load_scaler(run_dir)
    dataset = get_dataset(cfg, is_train=False, period="test", scaler=scaler)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn  
    )

    # Run inference
    outputs = []
    with torch.no_grad():
        for batch in loader:
            outputs.append(model(batch))
    print('CustomLSTM output:', list(outputs[0].keys())) 
    print('Cell state shape:', outputs[0]['c_n'].shape, "  #[batch size, sequence length, hidden size]") 
    #print('Prediction shape:', outputs[0]['y_hat'].shape) 

    return cfg, dataset, outputs, loader, model

def compute_cell_states_and_dates(loader, model):
    """
    Runs the model over loader, returning:
      - cell_states: np.ndarray, shape (N_windows, hidden_size)
      - dates:       np.ndarray of np.datetime64, shape (N_windows,)
    """
    all_states = []
    all_dates  = []

    with torch.no_grad():
        for batch in loader:
            # forward
            out = model(batch)

            #    extract the timestamp at the *end* of each output window
            #    batch['date'] has shape (batch_size, seq_length_out)
            #    torch.Tensor of dtype datetime64 or int64 (ns since epoch)
            dt = batch['date'][:, -1]
            all_dates.append(dt)

    dates = np.concatenate(all_dates) # (N_windows,)
    return dates

def compute_cell_states(outputs):
    # Concatenate final cell states across batches
    cells = [out['c_n'][:, -1, :] for out in outputs]
    return torch.cat(cells, dim=0)

def compute_full_cell_states(outputs):
    """
    From model outputs list, each out['c_n'] is (batch, seq_len, hidden_size).
    Returns a single array of shape (N_windows, seq_len, hidden_size).
    """
    # Stack along the window axis:
    arr = np.concatenate([out['c_n'].cpu().numpy() for out in outputs], axis=0)
    return arr

def plot_cell_state_dynamics(cell_states, start_date: str):
    dates = pd.date_range(start=start_date, periods=cell_states.shape[0], freq="3h")
    fig, ax = plt.subplots(figsize=(15, 4))

    # Plot all units faintly and the 8th unit clearly, with legend
    ax.plot(dates, cell_states.numpy(), alpha=0.1, label=None)
    ax.plot(dates, cell_states[:, 10].numpy(), label='Unit 8')

    ax.set_ylabel("cell state")
    ax.set_title("Cell State Dynamics over Time")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_diagnostic_panels(dataset, outputs, cfg):
    # Grab first sample for diagnostics
    sample = dataset[0]
    x_d = sample['x_d']
    to_plot = []
    labels = []
    for feat, tensor in x_d.items():
        arr = tensor.squeeze().cpu().numpy()
        to_plot.append(arr)
        labels.append(feat)

    plot_array = np.stack(to_plot, axis=1)

    first_out = outputs[0]

    fig, axes = plt.subplots(4, 2, figsize=(20, 14), sharex=True)
    axes[0, 0].set_title('Input values')
    lines = axes[0, 0].plot(plot_array)
    axes[0, 0].legend(lines, labels, frameon=False)

    length, num_lines  = first_out['c_n'][0].shape
    axes[1, 0].set_title(f'Cell state: {num_lines} lines of {length} length')
    axes[1, 0].plot(first_out['c_n'][0])

    axes[0, 1].set_title('Hidden state')
    axes[0, 1].plot(first_out['h_n'][0])

    axes[1, 1].set_title('Output gate')
    axes[1, 1].plot(first_out['o'][0])

    axes[2, 0].set_title('Forget gate')
    axes[2, 0].plot(first_out['f'][0])

    axes[2, 1].set_title('Input gate')
    axes[2, 1].plot(first_out['i'][0])

    axes[3, 0].set_title('Cell input activation')
    axes[3, 0].plot(first_out['g'][0])

    axes[3, 1].set_title('Prediction')
    axes[3, 1].plot(first_out['y_hat'][0])   

    fig.tight_layout()
    return fig, axes


# --- Additional Analyses --- 
def compute_autocorrelation(cell_states: np.ndarray, unit: int, nlags: int=500):
    """
    Computes autocorrelation for a specific cell unit.
    Returns acf values of length nlags+1.
    """
    series = cell_states[:, unit]
    return sm.tsa.acf(series, nlags=nlags)


def plot_autocorrelation(acf_vals: np.ndarray, unit: int):
    """
    Quickly plots the ACF values up to a specified lag.
    """
    fig, ax = plt.subplots(figsize=(6,4))
    ax.stem(range(len(acf_vals)), acf_vals)#, use_line_collection=True)
    ax.set_xlabel('Lag'); ax.set_ylabel('ACF'); ax.set_title(f'Autocorrelation: Unit {unit}')
    return fig, ax


def compute_mahalanobis(train_states: np.ndarray, test_states: np.ndarray):
    """
    Mahalanobis distance of each test state to the training distribution.
    - Mahalanobis distance measures how many standard deviations a point is away from the mean of a distribution, considering the shape and orientation of the data distribution.
    """
    cov = EmpiricalCovariance().fit(train_states)
    return cov.mahalanobis(test_states)


def plot_novelty(dists: np.ndarray, dates: pd.DatetimeIndex):
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(dates, dists, '-k')
    thresh = np.percentile(dists, 95)
    ax.axhline(thresh, color='r', linestyle='--', label='95th pct')
    ax.set_title('Cell-State Novelty (Mahalanobis)'); ax.legend()
    return fig, ax


def reduce_and_plot_states(cell_states: np.ndarray, train_idx: int):
    """
    UMAP embedding for train vs test cell states.
    train_idx: number of training timesteps
    """
    emb = UMAP(n_components=2).fit_transform(cell_states)
    fig, ax2 = plt.subplots(figsize=(6,6))
    ax2.scatter(emb[:train_idx,0], emb[:train_idx,1], c='gray', alpha=0.3, label='train')
    ax2.scatter(emb[train_idx:,0], emb[train_idx:,1], c='C1', label='test')
    ax2.legend(); ax2.set_title('UMAP of Cell States')
    #return fig, ax2
    fig, ax = plt.subplots(1, 2, figsize=(12,6), sharex=True, sharey=True)
    ax[0].scatter(emb[:train_idx,0], emb[:train_idx,1], c='gray', alpha=0.3, label='train')
    ax[0].legend(); ax[0].set_title('UMAP of Cell States Separated')
    ax[1].scatter(emb[train_idx:,0], emb[train_idx:,1], c='C1', label='test')
    ax[1].legend()
    plt.tight_layout()
    return fig, ax, ax2

def umap_embed(train_cs: np.ndarray,
               test_cs: np.ndarray,
               n_components: int = 2, init='pca'): # init='random'
    """
    Fit UMAP on train_cs and transform both train_cs and test_cs.
    Returns: train_emb, test_emb  (each shape (N, n_components))
    """
    umap_model = UMAP(
        n_components=n_components,
        init=init,            # 'random' or 'pca' instead of 'spectral'
        n_neighbors=15,       # you can also tune this lower if needed
        min_dist=0.1,
        metric='euclidean'
    )
    train_emb = umap_model.fit_transform(train_cs)
    test_emb  = umap_model.transform(test_cs)
    return train_emb, test_emb

def plot_umap(train_emb: np.ndarray,
              test_emb: np.ndarray):
    """
    Plot side-by-side UMAP of train vs. test embeddings.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    ax1.scatter(train_emb[:,0], train_emb[:,1], 
                color='gray', alpha=0.4, s=10)
    ax1.set_title("Train UMAP")
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")
    ax1.grid(True)

    ax2.scatter(test_emb[:,0], test_emb[:,1], 
                color='C1', alpha=0.6, s=10)
    ax2.set_title("Test UMAP")
    ax2.set_xlabel("UMAP 1")
    ax2.grid(True)

    plt.tight_layout()
    return fig, (ax1, ax2)


def event_triggered_average(cell_states: np.ndarray, driver: np.ndarray, threshold: float, pre: int, post: int, unit: int=7):
    """
    Event-triggered average of a cell unit around driver threshold crossings.
    """
    idx = np.where(driver > threshold)[0]
    windows = []
    for i in idx:
        if i-pre>=0 and i+post<len(cell_states):
            windows.append(cell_states[i-pre:i+post+1, unit])
    return np.mean(windows, axis=0)


def plot_event_response(avg_response: np.ndarray, pre: int, post: int, unit: int=7):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(np.arange(-pre, post+1), avg_response)
    ax.axvline(0, color='r', linestyle='--')
    ax.set_xlabel('Lag steps'); ax.set_ylabel('Cell state')
    ax.set_title(f'Event-Triggered Response: Unit {unit}')
    return fig, ax

def convex_hull_membership(train_emb: np.ndarray,
                           test_emb: np.ndarray):
    """
    Computes convex hull on train_emb and returns a mask
    telling which test_emb points fall inside.
    """
    hull      = ConvexHull(train_emb)
    polygon   = MplPath(train_emb[hull.vertices])
    inside    = polygon.contains_points(test_emb)  # bool array (len test_emb)
    return hull, inside

def plot_convex_hull(train_emb: np.ndarray,
                     test_emb: np.ndarray,
                     hull,
                     inside_mask: np.ndarray):
    """
    Plot the hull edges on train_emb and scatter
    train vs. test inside/outside, with counts in the legend.
    """
    fig, ax = plt.subplots(figsize=(6,6))

    # Hull edges
    for simplex in hull.simplices:
        ax.plot(train_emb[simplex,0], train_emb[simplex,1], 'k-')

    # Counts
    n_train        = train_emb.shape[0]
    n_inside       = inside_mask.sum()
    n_outside      = (~inside_mask).sum()

    # Scatter
    ax.scatter(train_emb[:,0], train_emb[:,1],
               c='gray', alpha=0.3, s=10,
               label=f"train ({n_train})")
    ax.scatter(test_emb[inside_mask,0], test_emb[inside_mask,1],
               c='C1', alpha=0.6, s=10,
               label=f"test inside ({n_inside})")
    ax.scatter(test_emb[~inside_mask,0], test_emb[~inside_mask,1],
               c='C3', alpha=0.6, s=10,
               label=f"test outside ({n_outside})")

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Convex Hull Membership")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig, ax

def plot_convex_hull_pyvista(train_data, test_data, reducer):
    train_mesh = np.vstack(train_data)
    test_mesh = np.vstack(test_data)
    reducer.eval()
    with torch.no_grad():
        t_train = torch.from_numpy(train_mesh).float()
        t_test  = torch.from_numpy(test_mesh).float()
        train_reduced = reducer(t_train).cpu().numpy()
        test_reduced  = reducer(t_test).cpu().numpy()

    hull = ConvexHull(train_reduced)
    mesh = pv.PolyData(train_reduced)
    faces = []
    for simplex in hull.simplices:
        faces.extend([3, *simplex])
    mesh.faces = np.array(faces)

    n_train = train_reduced.shape[0]
    n_test  = test_reduced.shape[0]
    print("output test shape:", test_reduced.shape)
    print("test shape before reduction:", test_mesh.shape)

    plotter = pv.Plotter()
    # Add convex hull mesh
    actor_hull  = plotter.add_mesh(mesh, color="red", opacity=0.1, show_edges=True, label="Convex Hull")
    # Add train points (blue)
    actor_train = plotter.add_points(train_reduced, color="blue", point_size=5, render_points_as_spheres=True)#, label=f"Train ({n_train} pts)")
    # Add validation points (cyan)
    actor_valid = plotter.add_points(validate_data, color="dark_blue", point_size=5, render_points_as_spheres=True, label="Validation")
    # Add test points (green)
    actor_test  = plotter.add_points(test_reduced, color="green", point_size=5, render_points_as_spheres=True)#, label=f"Test ({n_test} pts)")
    # # === Add legend ===
    # plotter.add_legend()

    # --- define toggle callbacks ---
    def toggle_hull(checked):
        actor_hull.SetVisibility(checked)
    def toggle_train(checked):
        actor_train.SetVisibility(checked)
    def toggle_test(checked):
        actor_test.SetVisibility(checked)

    # --- add checkbox widgets to control each layer ---
    # position is in normalized viewport coords (0–1)
    plotter.add_checkbox_button_widget(toggle_hull,
                                       value=True,
                                       position=(10,  10),
                                       size=25)
    plotter.add_checkbox_button_widget(toggle_train,
                                       value=True,
                                       position=(10,  45),
                                       size=25)
    plotter.add_checkbox_button_widget(toggle_test,
                                       value=True,
                                       position=(10,  80),
                                       size=25)

    # label them so the user knows which is which
    plotter.add_text("Convex Hull",  position=(40,  10), color="yellow")
    plotter.add_text(f"Train ({n_train} pts)", position=(40,  45), color="blue")
    plotter.add_text(f"Test ({n_test} pts)",  position=(40,  80), color="green")

    plotter.show()

def tucker_decomposition_3d(cell_states: np.ndarray,
                            ranks: tuple):
    """
    Performs Tucker decomposition on a 3-D tensor:
      cell_states: (N_windows, seq_len, hidden_size)
      ranks:       (R_time, R_seq, R_hidden)
    Returns:
      core (R_time, R_seq, R_hidden),
      factors [U_time, U_seq, U_hidden]
    """
    # Ensure using NumPy backend
    tl.set_backend('numpy')
    core, factors = tucker(cell_states, ranks=ranks)
    return core, factors


def compute_rsm_3d(cell_states: np.ndarray) -> np.ndarray:
    """
    Flattens each window into a row vector, then computes
    the (N_windows x N_windows) correlation matrix.
    Returns RSM.
    """
    N, T, H = cell_states.shape
    flat = cell_states.reshape(N, T * H)
    return np.corrcoef(flat)

# 1) Extract per‐window driver series from your DataLoader
def extract_driver_series(loader, feature_names, device):
    """
    From a DataLoader over the TEST set, pulls out for each window
    the last time‐step value of each feature in feature_names.
    
    Returns:
      drivers: dict var -> np.ndarray of shape (N_windows,)
    """
    all_vals = {feat: [] for feat in feature_names}
    for batch in loader:
        # x_d: dict feat -> tensor of shape (batch_size, seq_len)
        for feat in feature_names:
            arr = batch['x_d'][feat][:, -1]  # last time step
            all_vals[feat].append(arr.cpu().numpy())
    # concatenate batches
    drivers = {feat: np.concatenate(all_vals[feat], axis=0)
               for feat in feature_names}
    return drivers


# 2) Correlation matrix
def compute_correlation_matrix(cell_states: np.ndarray,
                               drivers: np.ndarray) -> np.ndarray:
    """
    Compute Pearson correlations between each cell dimension and each driver.

    Parameters
    ----------
    cell_states : ndarray, shape (N, H)
        N windows, H hidden units.
    drivers : ndarray, shape (N, D)
        N windows, D driver series.

    Returns
    -------
    corr : ndarray, shape (H, D)
        corr[i,j] = corrcoef(cell_states[:,i], drivers[:,j])[0,1]
    """
    # Make sure both are 2-D
    X = np.asarray(cell_states)
    Y = np.asarray(drivers)
    Y = Y[:, :, 0] 
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got X.ndim={X.ndim}, Y.ndim={Y.ndim}. Y shape ={Y.shape}")

    N = min(X.shape[0], Y.shape[0])
    X = X[:N]
    Y = Y[:N]

    H = X.shape[1]
    D = Y.shape[1]
    corr = np.zeros((H, D), dtype=float)

    # Compute per-pair Pearson
    for i in range(H):
        xi = X[:, i]
        # subtract mean once
        xi_m = xi - xi.mean()
        si   = xi_m.std()
        for j in range(D):
            yj = Y[:, j]
            yj_m = yj - yj.mean()
            sj   = yj_m.std()
            if si == 0 or sj == 0:
                corr[i, j] = 0.0
            else:
                corr[i, j] = np.dot(xi_m, yj_m) / (si * sj * (N - 1))
    return corr

def plot_correlation_matrix(corr: np.ndarray,
                            driver_names: list,
                            unit_indices: list = None):
    """
    Plot a heatmap of correlations between LSTM cell states and driver variables.

    Parameters
    ----------
    corr : np.ndarray
        Array of shape (H, D) where H is hidden_size and D is number of drivers.
    driver_names : list of str
        Names of the driver columns (length D).
    unit_indices : list of int, optional
        Which cell‐state indices to label on the y‐axis. Defaults to all.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    H, D = corr.shape
    fig, ax = plt.subplots(figsize=(D * 1.2 + 2, H * 0.2 + 2))

    im = ax.imshow(corr, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
    cbar = fig.colorbar(im, ax=ax, label='Pearson Correlation')

    # X-axis: driver names
    ax.set_xticks(np.arange(D))
    ax.set_xticklabels(driver_names, rotation=45, ha='right')

    # Y-axis: cell-state indices
    if unit_indices is None:
        unit_indices = list(range(H))
    ax.set_yticks(unit_indices)
    ax.set_yticklabels(unit_indices)

    ax.set_xlabel("Driver variable")
    ax.set_ylabel("Cell state index")
    ax.set_title("Correlation: Cell States vs. Drivers")
    ax.grid(False)

    plt.tight_layout()
    return fig, ax

# 3) Event‐triggered average for one cell unit
def event_triggered_average(cell_series: np.ndarray,
                            event_indices: np.ndarray,
                            pre: int,
                            post: int) -> np.ndarray:
    """
    cell_series:    (N_windows,) one cell-state time series
    event_indices:  array of ints where events occur
    pre, post:      number of windows before/after to include
    returns avg:    (pre+post+1,) the mean trajectory
    """
    windows = []
    N = len(cell_series)
    for i in event_indices:
        if i >= pre and i + post < N:
            windows.append(cell_series[i-pre : i+post+1])
    if not windows:
        return np.array([])
    return np.mean(windows, axis=0)

def main():
    args = parse_args()
    cfg, dataset, outputs, loader, model = load_data_and_model(
        args.run_dir, args.epoch, args.config, args.batch_size
    )

    test_cs = compute_cell_states(outputs).numpy()
    train_cs = np.load(f"{args.run_dir}/cell_states/train_c_n.npy")
    print("Train states shape: ", train_cs.shape)
    train_flat = train_cs[:, -1, :]  
    full_test_cs = compute_full_cell_states(outputs)
    # print("full test_data  shape:", full_test_cs.shape)
    # flattened_test = np.vstack(full_test_cs)
    # print("combined test shape:", flattened_test.shape)

    # Basic interanl state plots
    plot_cell_state_dynamics(torch.from_numpy(test_cs), args.start_date)
    plot_diagnostic_panels(dataset, outputs, cfg)

    # Autocorrelation 
    acf_vals = compute_autocorrelation(test_cs, unit=9)
    plot_autocorrelation(acf_vals, unit=9)
    plt.show()

    # Novelty mahalanobis
    dists = compute_mahalanobis(train_flat, test_cs)
    dates = compute_cell_states_and_dates(loader, model)
    plot_novelty(dists, dates)

    # Embed and UMAP
    train_emb, test_emb = umap_embed(train_flat, test_cs)
    plot_umap(train_emb, test_emb)
    
    # Hull membership
    hull, inside_mask = convex_hull_membership(train_emb, test_emb)
    plot_convex_hull(train_emb, test_emb, hull, inside_mask)
    plt.show
    # 3D interactive Convex hull
    reducer = StateReducer(original_dim=train_cs.shape[2], reduced_dim=3)
    plot_convex_hull_pyvista(train_cs, test_cs, reducer=reducer)

    feature_names = cfg.dynamic_inputs  # e.g. ['prcp','tmin','tmax']
    drivers_dict  = extract_driver_series(loader, feature_names, cfg.device)
    # make a matrix drivers_mat of shape (N_windows, D)
    drivers_mat = np.stack([drivers_dict[f] for f in feature_names], axis=1)

    # correlation matrix
    corr = compute_correlation_matrix(test_cs, drivers_mat)
    plot_correlation_matrix(corr, feature_names)

    # event triggered response
    # variable = drivers_dict['Q2D']
    # thresh = np.percentile(variable, 95)
    # events = np.where(variable > thresh)[0]

    # # choose a cell unit to inspect, e.g. unit 7
    # unit = 100
    # avg_traj = event_triggered_average(
    #     test_cs[:, unit], events, pre=5, post=10
    # )
    # if avg_traj.size:
    #     fig2, ax2 = plot_event_response(avg_traj, pre=5, post=10, unit=unit)
    #     plt.show()
    # else:
    #     print("No events found in test period.") 

    # core, factors = tucker_decomposition_3d(train_cs, ranks=(5, 3, 10))
    # # Optional: plot the first few modes
    # U_time, U_seq, U_hidden = factors
    # # e.g.:
    # fig, ax = plt.subplots()
    # for i in range(min(3, U_time.shape[1])):
    #     ax.plot(U_time[:, i], label=f"Mode {i}")
    # ax.set_title("Tucker: Temporal Factors")
    # ax.legend()
    # plt.show()

    # rsm = compute_rsm_3d(train_cs)
    # fig, ax = plt.subplots(figsize=(6,6))
    # im = ax.imshow(rsm, cmap="viridis", vmin=-1, vmax=1)
    # plt.colorbar(im, ax=ax)
    # ax.set_title("Representational Similarity Matrix")
    # plt.show()
    plt.show()

if __name__ == '__main__':
    main()
    
