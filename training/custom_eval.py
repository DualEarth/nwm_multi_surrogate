import sys
import logging
from pathlib import Path
import yaml
import torch
from typing import List, Dict
import pandas as pd

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from neuralhydrology.utils.config import Config
from neuralhydrology.evaluation.evaluate import start_evaluation
from neuralhydrology.modelzoo import get_model
from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.datautils.utils import load_scaler

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


class StateReducer(nn.Module):
    def __init__(self, original_dim: int = 256, reduced_dim: int = 3):
        super().__init__()
        if reduced_dim >= original_dim:
            raise ValueError("Reduced dimension must be less than original.")
        self.reducer = nn.Sequential(
            nn.Linear(original_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, reduced_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x[-1, :, :]  # take last layer
        elif x.dim() != 2:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        return self.reducer(x)


def load_config(run_dir: Path) -> Config:
    cfg_path = run_dir / "config.yml"
    with open(cfg_path, 'r') as f:
        cfg = Config(yaml.safe_load(f))
    cfg.run_dir = run_dir
    cfg.device = str(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    return cfg


def load_model(cfg: Config, run_dir: Path, epoch: int) -> nn.Module:
    model = get_model(cfg).to(cfg.device)
    model_path = run_dir / f"model_epoch{epoch:03d}.pt"
    model.load_state_dict(torch.load(model_path, map_location=cfg.device))
    model.eval()
    LOGGER.info(f"Loaded model from: {model_path}")
    return model


def load_reducer(cfg: Config, run_dir: Path, epoch: int) -> StateReducer:
    reducer_dim = cfg.as_dict().get('trainer', {}).get('reduced_state_dimension', 3)
    reducer = StateReducer(cfg.hidden_size, reducer_dim).to(cfg.device)
    reducer_path = run_dir / f"state_reducer_epoch{epoch:03d}.pth"
    try:
        reducer.load_state_dict(torch.load(reducer_path, map_location=cfg.device))
        LOGGER.info(f"Loaded reducer weights from: {reducer_path}")
    except FileNotFoundError:
        LOGGER.warning("Reducer weights not found. Using untrained reducer.")
    reducer.eval()
    return reducer


def get_test_basins(cfg: Config) -> List[str]:
    test_basin_file = Path(cfg.test_basin_file)
    with open(test_basin_file, 'r') as f:
        return [line.strip() for line in f.readlines()]


def collect_reduced_states(
    model: nn.Module,
    reducer: StateReducer,
    cfg: Config,
    basins: List[str],
    run_dir: Path,
    epoch: int
):
    scaler = load_scaler(run_dir)
    out_dir = run_dir / "reduced_internal_states" / "test"
    out_dir.mkdir(parents=True, exist_ok=True)

    for basin in tqdm(basins, desc="Saving reduced states per basin"):
        dataset = get_dataset(cfg=cfg, is_train=False, period="test", basin=basin, scaler=scaler)
        test_loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=dataset.collate_fn
        )

        all_states = []
        with torch.no_grad():
            for data in test_loader:
                move_data_to_device(data, cfg.device)
                data = model.pre_model_hook(data, is_train=False)
                predictions = model(data)

                if "c_n" not in predictions:
                    LOGGER.warning(f"No c_n found for basin {basin}.")
                    continue

                c_n = predictions["c_n"].detach()
                reduced = reducer(c_n.to(cfg.device)).detach().cpu()
                all_states.append(reduced)

        if all_states:
            basin_tensor = torch.cat(all_states, dim=0)
            torch.save(basin_tensor, out_dir / f"{basin}_c_n_reduced_epoch{epoch:03d}.pt")


def move_data_to_device(data: Dict, device: torch.device):
    for key in data:
        if key.startswith('x_d'):
            data[key] = {k: v.to(device) for k, v in data[key].items()}
        elif key != 'date':
            data[key] = data[key].to(device)


def collect_reduced_states_for_test_basins(run_dir: Path, epoch: int):
    cfg = load_config(run_dir)
    model = load_model(cfg, run_dir, epoch)
    reducer = load_reducer(cfg, run_dir, epoch)
    test_basins = get_test_basins(cfg)

    # Trigger evaluation to generate metrics
    metrics_csv_path = run_dir / f"metrics_epoch{epoch:03d}_test.csv"
    start_evaluation(cfg=cfg, run_dir=run_dir, epoch=epoch, period="test")
    LOGGER.info(f"Evaluation complete. Metrics saved to: {metrics_csv_path}")

    # Collect and save reduced internal states
    collect_reduced_states(model, reducer, cfg, test_basins, run_dir, epoch)
    LOGGER.info("✅ Finished collecting reduced internal states for all test basins.")


if __name__ == "__main__":
    run_dir = Path(sys.argv[1])
    epoch = (
        int(sys.argv[2])
        if len(sys.argv) > 2 else
        max(int(f.name.split("epoch")[-1].split(".pt")[0]) for f in run_dir.glob("model_epoch*.pt"))
    )

    collect_reduced_states_for_test_basins(run_dir, epoch)
