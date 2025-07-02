import logging
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from typing import Optional, List

from neuralhydrology.training.basetrainer import BaseTrainer
from neuralhydrology.utils.config import Config
from neuralhydrology.datasetzoo import get_dataset
from torch.utils.data import DataLoader

LOGGER = logging.getLogger(__name__)


class StateReducer(nn.Module):
    """Module to reduce LSTM cell state dimensionality."""
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
        LOGGER.info(f"StateReducer initialized: {original_dim}D -> {reduced_dim}D")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x[-1, :, :] if x.shape[0] <= 4 else x.squeeze(1)
        if x.dim() != 2:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        return self.reducer(x)


class CustomTrainer(BaseTrainer):
    """Trainer that logs and reduces LSTM cell states."""
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        LOGGER.info("CustomTrainer initialized.")

        trainer_cfg = cfg.as_dict().get('trainer', {})
        self._save_internal_states_flag = trainer_cfg.get('save_internal_states', False)
        self._reduced_state_dimension = trainer_cfg.get('reduced_state_dimension', 3)

        self.state_reducer = StateReducer(
            original_dim=self.cfg.hidden_size,
            reduced_dim=self._reduced_state_dimension
        ).to(self.device)

    def initialize_training(self):
        """Initialize training loaders and attach reducer parameters to optimizer."""
        super().initialize_training()

        if self.optimizer is not None:
            reducer_params = list(self.state_reducer.parameters())
            if not set(map(id, reducer_params)).issubset(
                set(map(id, [p for g in self.optimizer.param_groups for p in g['params']]))
            ):
                self.optimizer.add_param_group({'params': reducer_params,
                                                'lr': self.optimizer.param_groups[0]['lr']})
                LOGGER.info("StateReducer parameters added to optimizer.")

        self._init_loader("validation")
        self._init_loader("test")

    def _init_loader(self, period: str):
        """Helper to initialize validation/test DataLoader."""
        dataset = None
        try:
            scaler = getattr(self.loader.dataset, 'scaler', None) if hasattr(self.loader, 'dataset') else None
            id_map = getattr(self.loader.dataset, 'id_to_int', None)
            dataset = get_dataset(self.cfg, is_train=False, period=period, scaler=scaler, id_to_int=id_map)
        except Exception as e:
            LOGGER.error(f"{period.capitalize()} DataLoader init error: {e}")

        loader_attr = f"{period}_loader"
        if dataset is not None:
            setattr(self, loader_attr, DataLoader(
                dataset,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers,
                pin_memory=self.device.type == "cuda",
                collate_fn=dataset.collate_fn
            ))
            LOGGER.info(f"{period.capitalize()} DataLoader initialized.")
        else:
            setattr(self, loader_attr, None)
            LOGGER.info(f"No {period} dataset found; skipping loader.")

    def train_and_validate(self):
        for epoch in range(self._epoch + 1, self._epoch + self.cfg.epochs + 1):
            self._update_lr(epoch)
            self._train_epoch(epoch)
            self._log_epoch(epoch)

            if epoch % self.cfg.save_weights_every == 0:
                self._save_model(epoch)

            if self.validator and epoch % self.cfg.validate_every == 0:
                self._validate_epoch(epoch)

        if self.cfg.log_tensorboard:
            self.experiment_logger.stop_tb()

    def _update_lr(self, epoch: int):
        if epoch in self.cfg.learning_rate:
            lr = self.cfg.learning_rate[epoch]
            LOGGER.info(f"Setting LR to {lr}")
            for group in self.optimizer.param_groups:
                group["lr"] = lr

    def _save_model(self, epoch: int):
        self._save_weights_and_optimizer(epoch)
        reducer_path = self.cfg.run_dir / f"state_reducer_epoch{epoch:03d}.pth"
        torch.save(self.state_reducer.state_dict(), reducer_path)
        LOGGER.info(f"Saved StateReducer weights to {reducer_path}")

    def _log_epoch(self, epoch: int):
        avg_losses = self.experiment_logger.summarise()
        loss_str = ", ".join(f"{k}: {v:.5f}" for k, v in avg_losses.items())
        LOGGER.info(f"Epoch {epoch} avg loss: {loss_str}")

    def _validate_epoch(self, epoch: int):
        LOGGER.info(f"--- Running Validation for Epoch {epoch} ---")
        self.validator.evaluate(
            epoch=epoch,
            save_results=self.cfg.save_validation_results,
            save_all_output=self.cfg.save_all_output,
            metrics=self.cfg.metrics,
            model=self.model,
            experiment_logger=self.experiment_logger.valid()
        )

        if self._save_internal_states_flag:
            self._collect_states(epoch, split="validation")

        valid_metrics = self.experiment_logger.summarise()
        msg = f"Epoch {epoch} validation loss: {valid_metrics.get('avg_total_loss', float('nan')):.5f}"
        metric_msg = ", ".join(f"{k}: {v:.5f}" for k, v in valid_metrics.items() if k != "avg_total_loss")
        if metric_msg:
            msg += " -- Metrics: " + metric_msg
        LOGGER.info(msg)

    def _train_epoch(self, epoch: int):
        self.model.train()
        self.state_reducer.train()

        collect_states = self._save_internal_states_flag and (epoch == self.cfg.epochs)
        cell_states, reduced_states = [], []

        pbar = tqdm(self.loader, file=sys.stdout, disable=self._disable_pbar)
        pbar.set_description(f'# Epoch {epoch} (Train)')

        nan_count = 0
        for i, data in enumerate(pbar):
            if self._max_updates_per_epoch and i >= self._max_updates_per_epoch:
                break
            self._move_to_device(data)
            data = self.model.pre_model_hook(data, is_train=True)
            predictions = self.model(data)

            if collect_states and "c_n" in predictions:
                c_n = predictions["c_n"].detach()
                cell_states.append(c_n.cpu())
                reduced_states.append(self.state_reducer(c_n).detach().cpu())

            loss_val, losses = self.loss_obj(predictions, data)
            if torch.isnan(loss_val):
                nan_count += 1
                if nan_count > self._allow_subsequent_nan_losses:
                    raise RuntimeError("Loss NaN repeatedly. Stopping training.")
                continue

            nan_count = 0
            self.optimizer.zero_grad()
            loss_val.backward()
            if self.cfg.clip_gradient_norm:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_gradient_norm)
            self.optimizer.step()
            pbar.set_postfix_str(f"Loss: {loss_val.item():.4f}")
            self.experiment_logger.log_step(**{k: v.item() for k, v in losses.items()})

        if collect_states:
            self._save_states(epoch, cell_states, reduced_states, split="train")

    def _move_to_device(self, data: dict):
        for key in data:
            if key.startswith("x_d"):
                data[key] = {k: v.to(self.device) for k, v in data[key].items()}
            elif not key.startswith("date"):
                data[key] = data[key].to(self.device)

    def _collect_states(self, epoch: int, split: str = "test"):
        loader = getattr(self, f"{split}_loader")
        if loader is None:
            LOGGER.warning(f"{split.capitalize()} loader is None; skipping state collection.")
            return

        self.model.eval()
        self.state_reducer.eval()
        cell_states, reduced_states = [], []

        with torch.no_grad():
            pbar = tqdm(loader, file=sys.stdout, disable=self._disable_pbar)
            pbar.set_description(f'# Epoch {epoch} ({split.capitalize()} States)')
            for data in pbar:
                self._move_to_device(data)
                data = self.model.pre_model_hook(data, is_train=False)
                predictions = self.model(data)
                if "c_n" in predictions:
                    c_n = predictions["c_n"].detach()
                    cell_states.append(c_n.cpu())
                    reduced_states.append(self.state_reducer(c_n).detach().cpu())

        self._save_states(epoch, cell_states, reduced_states, split=split)

    def _save_states(self, epoch: int, cell_states: List[torch.Tensor], reduced_states: List[torch.Tensor], split: str):
        save_dir = self.cfg.run_dir / "internal_states" / split
        save_dir.mkdir(parents=True, exist_ok=True)

        if cell_states:
            all_c_n = torch.cat(cell_states, dim=0)
            torch.save(all_c_n, save_dir / f"c_n_epoch{epoch:03d}_{split}.pt")
            LOGGER.info(f"Saved {split} cell states for epoch {epoch}")

        if reduced_states:
            all_reduced = torch.cat(reduced_states, dim=0)
            torch.save(all_reduced, save_dir / f"c_n_reduced_epoch{epoch:03d}_{split}.pt")
            LOGGER.info(f"Saved {split} reduced cell states for epoch {epoch}")
