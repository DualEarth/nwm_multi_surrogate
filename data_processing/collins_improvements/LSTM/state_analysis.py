import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from neuralhydrology.datasetzoo import get_dataset, camelsus
from neuralhydrology.datautils.utils import load_scaler
from neuralhydrology.modelzoo.customlstm import CustomLSTM
from neuralhydrology.utils.config import Config
from neuralhydrology.modelzoo.customlstm import CudaLSTM



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
#    model.load_state_dict(weights)
    model
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
    print('Cell state shape:', outputs[0]['c_n'].shape) # [batch size, sequence length, hidden size]

    return cfg, dataset, outputs


def compute_cell_states(outputs):
    # Concatenate final cell states across batches
    cells = [out['c_n'][:, -1, :] for out in outputs]
    return torch.cat(cells, dim=0)


def plot_cell_state_dynamics(cell_states, start_date: str):
    dates = pd.date_range(start=start_date, periods=cell_states.shape[0], freq="3h")
    fig, ax = plt.subplots(figsize=(15, 4))

    # Plot all units faintly and the 8th unit clearly, with legend
    ax.plot(dates, cell_states.numpy(), alpha=0.1, label=None)
    ax.plot(dates, cell_states[:, 7].numpy(), label='Unit 8')

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

    axes[1, 0].set_title('Cell state')
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

    fig.delaxes(axes[3, 1])
    # Label axes for all panels
    # for row in axes:
    #     for ax in row:
    #         ax.set_xlabel('Time step')
    #         ax.set_ylabel('Value')
    fig.tight_layout()
    return fig, axes


def main():
    args = parse_args()
    cfg, dataset, outputs = load_data_and_model(
        args.run_dir, args.epoch, args.config, args.batch_size
    )

    cell_states = compute_cell_states(outputs)

    # Plot and show figures
    plot_cell_state_dynamics(cell_states, args.start_date)
    plot_diagnostic_panels(dataset, outputs, cfg)

    #plt.show()


if __name__ == '__main__':
    main()
