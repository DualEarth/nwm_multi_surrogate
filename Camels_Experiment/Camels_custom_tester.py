import sys
import logging
import re
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from neuralhydrology.utils.config import Config
from neuralhydrology.modelzoo import get_model
from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.datautils.utils import load_scaler
from neuralhydrology.evaluation import start_evaluation  # <- added

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


class StateReducer(nn.Module):
    def __init__(self, original_dim: int = 256, reduced_dim: int = 3):
        super().__init__()
        if reduced_dim >= original_dim:
            raise ValueError(f"Reduced dimension ({reduced_dim}) must be less than original dimension ({original_dim}).")
        self.reducer = nn.Sequential(
            nn.Linear(original_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, reduced_dim)
        )
        LOGGER.info(f"StateReducer initialized: {original_dim}D -> {reduced_dim}D")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            # Case 1: shape (num_layers, batch_size, hidden_size)
            if x.shape[0] <= 4 and x.shape[1] > 1:  # likely (layers, batch, hidden)
                x = x[-1, :, :]  # get last layer only
            else:
                # Case 2: (N, 1, hidden) from flattened sequence outputs
                x = x.squeeze(1)  # shape becomes (N, hidden)
        if x.dim() != 2:
            raise ValueError(f"Unexpected input shape for StateReducer: {x.shape}")
        return self.reducer(x)
    

def find_latest_epoch(run_dir: Path) -> Optional[int]:
    epoch_files = list(run_dir.glob("model_epoch*.pt"))
    if not epoch_files:
        return None
    epochs = [
        int(m.group(1))
        for f in epoch_files
        if (m := re.search(r"model_epoch(\d+)\.pt", f.name))
    ]
    return max(epochs) if epochs else None


def load_config(run_dir: Path) -> Config:
    cfg_path = run_dir / "config.yml"
    with open(cfg_path, "r") as f:
        return Config(yaml.safe_load(f))


def prepare_model_and_reducer(cfg: Config, run_dir: Path, epoch: int, device: torch.device):
    model = get_model(cfg).to(device)
    model_path = run_dir / f"model_epoch{epoch:03d}.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    reducer_dim = cfg.as_dict().get("trainer", {}).get("reduced_state_dimension", 3)
    reducer = StateReducer(cfg.hidden_size, reducer_dim).to(device)
    reducer.eval()

    reducer_path = run_dir / f"state_reducer_epoch{epoch:03d}.pth"
    try:
        reducer.load_state_dict(torch.load(reducer_path, map_location=device))
        LOGGER.info(f"Loaded reducer weights from {reducer_path}")
    except FileNotFoundError:
        LOGGER.warning(f"Reducer weights not found at {reducer_path}. Using untrained reducer.")

    return model, reducer


def prepare_test_loader_for_basin(cfg: Config, run_dir: Path, basin_id: str, scaler):
    dataset = get_dataset(cfg=cfg, is_train=False, period="test", basin=basin_id, scaler=scaler)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=dataset.collate_fn,
    )


def collect_reduced_states_per_basin(
    model: nn.Module,
    state_reducer: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    cfg: Config,
    epoch: int,
    basin_id: str,
) -> None:
    model.eval()
    state_reducer.eval()
    all_reduced_states = []

    LOGGER.info(f"Collecting reduced states for basin {basin_id} (epoch {epoch})...")

    with torch.no_grad():
        pbar = tqdm(test_loader, file=sys.stdout, desc=f"Basin {basin_id} Epoch {epoch}")
        for batch in pbar:
            for key, val in batch.items():
                if key.startswith("x_d") and isinstance(val, dict):
                    batch[key] = {k: v.to(device) for k, v in val.items()}
                elif key != "date":
                    batch[key] = val.to(device)

            batch = model.pre_model_hook(batch, is_train=False)
            preds = model(batch)

            c_n = preds.get("c_n")
            if c_n is None:
                LOGGER.warning("No 'c_n' in predictions; skipping batch.")
                continue

            c_n = c_n.detach().to(device)
            reduced = state_reducer(c_n).detach().cpu()
            all_reduced_states.append(reduced)

    save_dir = Path(cfg.run_dir) / "internal_states" / "test"
    save_dir.mkdir(parents=True, exist_ok=True)

    if all_reduced_states:
        tensor_all = torch.cat(all_reduced_states, dim=0)
        torch.save(
            tensor_all,
            save_dir / f"c_n_reduced_epoch{epoch:03d}_basin_{basin_id.replace('.', '_')}.pt",
        )
        LOGGER.info(f"Saved reduced states tensor shape {tensor_all.shape} for basin {basin_id}")


def main():
    if len(sys.argv) != 2:
        LOGGER.error("Usage: python eval_plot.py <run_dir>")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    if not run_dir.exists():
        LOGGER.error(f"Run directory {run_dir} does not exist.")
        sys.exit(1)

    cfg = load_config(run_dir)
    cfg.run_dir = str(run_dir)

    # --- ADDED: run NeuralHydrology evaluation to generate metrics CSV ---
    start_evaluation(cfg)

    basin_list_file = Path("C:/Users/deonf/Model_work/data/CAMELS_US/camel_test_basins.txt")
    if not basin_list_file.exists():
        LOGGER.error(f"Basin list file {basin_list_file} does not exist.")
        sys.exit(1)

    scaler = load_scaler(run_dir)
    with open(basin_list_file, "r") as f:
        basin_ids = [line.strip() for line in f if line.strip()]

    if not basin_ids:
        LOGGER.error(f"No basin IDs found in the basin list file {basin_list_file}")
        sys.exit(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    epoch = find_latest_epoch(run_dir)
    if epoch is None:
        LOGGER.error("No model epoch files found.")
        sys.exit(1)

    model, reducer = prepare_model_and_reducer(cfg, run_dir, epoch, device)

    for basin_id in basin_ids:
        try:
            test_loader = prepare_test_loader_for_basin(cfg, run_dir, basin_id, scaler)
            collect_reduced_states_per_basin(model, reducer, test_loader, device, cfg, epoch, basin_id)
        except Exception as e:
            LOGGER.error(f"Error processing basin {basin_id}: {e}")

    LOGGER.info("🎉 Finished collecting reduced states for all basins.")


if __name__ == "__main__":
    main()
