import logging
import torch
from pathlib import Path
from tqdm import tqdm
import sys
import traceback

from typing import Optional, List

from neuralhydrology.training.basetrainer import BaseTrainer
from neuralhydrology.utils.config import Config
from neuralhydrology.datasetzoo import get_dataset
from torch.utils.data import DataLoader
import torch.nn as nn

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
    
class CustomTrainer(BaseTrainer):
    def __init__(self, cfg: Config):
        super(CustomTrainer, self).__init__(cfg)
        LOGGER.info("!!! CustomTrainer __init__ called !!!")

        trainer_config_dict = cfg.as_dict().get('trainer', {})
        self._save_internal_states_flag = trainer_config_dict.get('save_internal_states', False)
        self._reduced_state_dimension = trainer_config_dict.get('reduced_state_dimension', 3)  # Default 3 dims

        LOGGER.info(f"CustomTrainer will save internal states: {self._save_internal_states_flag}")
        LOGGER.info(f"Reduced state dimension: {self._reduced_state_dimension}")

        self.state_reducer = StateReducer(
            original_dim=self.cfg.hidden_size,
            reduced_dim=self._reduced_state_dimension
        ).to(self.device)

    def initialize_training(self):
        super().initialize_training()

        if self.optimizer is not None:
            optimizer_param_ids = {id(p) for group in self.optimizer.param_groups for p in group['params']}
            reducer_param_ids = {id(p) for p in self.state_reducer.parameters()}
            if not reducer_param_ids.issubset(optimizer_param_ids):
                self.optimizer.add_param_group({'params': list(self.state_reducer.parameters()),
                                                'lr': self.optimizer.param_groups[0]['lr']})
                LOGGER.info("StateReducer parameters added to optimizer.")
            else:
                LOGGER.debug("StateReducer parameters already in optimizer.")

        # Initialize validation and test loaders as usual
        data_scaler = None
        data_id_to_int = None
        if hasattr(self, 'loader') and self.loader is not None and hasattr(self.loader, 'dataset'):
            data_scaler = getattr(self.loader.dataset, 'scaler', None)
            data_id_to_int = getattr(self.loader.dataset, 'id_to_int', None)

        try:
            self.validation_dataset = get_dataset(self.cfg, is_train=False, period="validation", basin=None,
                                                  scaler=data_scaler, id_to_int=data_id_to_int)
            if self.validation_dataset is not None:
                self.validation_loader = DataLoader(
                    self.validation_dataset,
                    batch_size=self.cfg.batch_size,
                    shuffle=False,
                    num_workers=self.cfg.num_workers,
                    pin_memory=self.device.type == "cuda",
                    collate_fn=self.validation_dataset.collate_fn
                )
                LOGGER.info("Validation DataLoader initialized.")
            else:
                self.validation_loader = None
                LOGGER.info("No validation dataset found; skipping validation loader.")
        except Exception as e:
            LOGGER.error(f"Validation DataLoader init error: {e}")
            self.validation_loader = None

        try:
            self.test_dataset = get_dataset(self.cfg, is_train=False, period="test", basin=None,
                                            scaler=data_scaler, id_to_int=data_id_to_int)
            if self.test_dataset is not None:
                self.test_loader = DataLoader(
                    self.test_dataset,
                    batch_size=self.cfg.batch_size,
                    shuffle=False,
                    num_workers=self.cfg.num_workers,
                    pin_memory=self.device.type == "cuda",
                    collate_fn=self.test_dataset.collate_fn
                )
                LOGGER.info("Test DataLoader initialized.")
            else:
                self.test_loader = None
                LOGGER.info("No test dataset found; skipping test loader.")
        except Exception as e:
            LOGGER.error(f"Test DataLoader init error: {e}")
            self.test_loader = None

    def _train_epoch(self, epoch: int):
        self.model.train()
        self.state_reducer.train()
        n_iter = min(self._max_updates_per_epoch, len(self.loader)) if self._max_updates_per_epoch else None
        pbar = tqdm(self.loader, file=sys.stdout, disable=self._disable_pbar, total=n_iter)
        pbar.set_description(f'# Epoch {epoch} (Train)')

        collect_states = self._save_internal_states_flag and (epoch == self.cfg.epochs)
        if collect_states:
            epoch_cell_states = []
            epoch_reduced_states = []
        else:
            epoch_cell_states = None
            epoch_reduced_states = None

        nan_count = 0
        for i, data in enumerate(pbar):
            if self._max_updates_per_epoch and i >= self._max_updates_per_epoch:
                break
            for key in data.keys():
                if key.startswith('x_d'):
                    data[key] = {k: v.to(self.device) for k, v in data[key].items()}
                elif not key.startswith('date'):
                    data[key] = data[key].to(self.device)

            data = self.model.pre_model_hook(data, is_train=True)
            predictions = self.model(data)

            if collect_states and "c_n" in predictions:
                c_n_state = predictions["c_n"].detach()
                LOGGER.debug(f"Train c_n shape before reduction: {c_n_state.shape}")
                epoch_cell_states.append(c_n_state.cpu())
                reduced_state = self.state_reducer(c_n_state).detach().cpu()
                epoch_reduced_states.append(reduced_state)

            loss_val, all_losses = self.loss_obj(predictions, data)
            if torch.isnan(loss_val):
                nan_count += 1
                if nan_count > self._allow_subsequent_nan_losses:
                    raise RuntimeError("Loss was NaN multiple times; stopping training.")
                LOGGER.warning(f"Loss NaN count {nan_count}; skipping step.")
            else:
                nan_count = 0
                self.optimizer.zero_grad()
                loss_val.backward()
                if self.cfg.clip_gradient_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_gradient_norm)
                self.optimizer.step()

            pbar.set_postfix_str(f"Loss: {loss_val.item():.4f}")
            self.experiment_logger.log_step(**{k: v.item() for k, v in all_losses.items()})

        if collect_states and (epoch_cell_states or epoch_reduced_states):
            save_dir = self.cfg.run_dir / "internal_states" / "train"
            save_dir.mkdir(parents=True, exist_ok=True)

            if epoch_cell_states:
                all_cell_states = torch.cat(epoch_cell_states, dim=0)
                LOGGER.info(f"DEBUG - Train original states shape epoch {epoch}: {all_cell_states.shape}")
                torch.save(all_cell_states, save_dir / f"c_n_epoch{epoch:03d}.pt")
                LOGGER.info(f"Saved training cell states epoch {epoch}")

            if epoch_reduced_states:
                all_reduced_states = torch.cat(epoch_reduced_states, dim=0)
                LOGGER.info(f"DEBUG - Train reduced states shape epoch {epoch}: {all_reduced_states.shape}")
                torch.save(all_reduced_states, save_dir / f"c_n_reduced_epoch{epoch:03d}.pt")
                LOGGER.info(f"Saved training reduced cell states epoch {epoch}")

    def train_and_validate(self):
        for epoch in range(self._epoch + 1, self._epoch + self.cfg.epochs + 1):
            if epoch in self.cfg.learning_rate:
                lr = self.cfg.learning_rate[epoch]
                LOGGER.info(f"Setting LR to {lr}")
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

            self._train_epoch(epoch)

            avg_losses = self.experiment_logger.summarise()
            loss_str = ", ".join(f"{k}: {v:.5f}" for k, v in avg_losses.items())
            LOGGER.info(f"Epoch {epoch} avg loss: {loss_str}")

            if epoch % self.cfg.save_weights_every == 0:
                self._save_weights_and_optimizer(epoch)
                # Save StateReducer weights separately
                reducer_path = self.cfg.run_dir / f"state_reducer_epoch{epoch:03d}.pth"
                torch.save(self.state_reducer.state_dict(), reducer_path)
                LOGGER.info(f"Saved StateReducer weights to {reducer_path}")

            if self.validator is not None and epoch % self.cfg.validate_every == 0:
                LOGGER.info(f"--- Running Validation for Epoch {epoch} ---")
                self.validator.evaluate(epoch=epoch,
                                        save_results=self.cfg.save_validation_results,
                                        save_all_output=self.cfg.save_all_output,
                                        metrics=self.cfg.metrics,
                                        model=self.model,
                                        experiment_logger=self.experiment_logger.valid())

                if self._save_internal_states_flag:
                    self._collect_validation_states(epoch=epoch)

                valid_metrics = self.experiment_logger.summarise()
                val_msg = f"Epoch {epoch} average validation loss: "
                val_msg += f"{valid_metrics.get('avg_total_loss', float('nan')):.5f}"
                if self.cfg.metrics:
                    other_metrics = {k: v for k, v in valid_metrics.items() if k != 'avg_total_loss'}
                    if other_metrics:
                        val_msg += " -- Validation metrics: "
                        val_msg += ", ".join(f"{k}: {v:.5f}" for k, v in other_metrics.items())
                LOGGER.info(val_msg)

        if self.cfg.log_tensorboard:
            self.experiment_logger.stop_tb()

    def _collect_validation_states(self, epoch: int):
        self.model.eval()
        self.state_reducer.eval()
        epoch_cell_states = []
        epoch_reduced_states = []

        with torch.no_grad():
            if self.validation_loader is None:
                LOGGER.warning("Validation loader is None; skipping validation states collection.")
                return

            pbar = tqdm(self.validation_loader, file=sys.stdout, disable=self._disable_pbar)
            pbar.set_description(f'# Epoch {epoch} (Validation States)')
            for data in pbar:
                for key in data.keys():
                    if key.startswith('x_d'):
                        data[key] = {k: v.to(self.device) for k, v in data[key].items()}
                    elif not key.startswith('date'):
                        data[key] = data[key].to(self.device)

                data = self.model.pre_model_hook(data, is_train=False)
                predictions = self.model(data)

                if "c_n" in predictions:
                    c_n_state = predictions["c_n"].detach()
                    LOGGER.debug(f"Validation c_n shape before reduction: {c_n_state.shape}")
                    epoch_cell_states.append(c_n_state.cpu())
                    reduced_state = self.state_reducer(c_n_state).detach().cpu()
                    epoch_reduced_states.append(reduced_state)

        if epoch_cell_states or epoch_reduced_states:
            save_dir = self.cfg.run_dir / "internal_states" / "validation"
            save_dir.mkdir(parents=True, exist_ok=True)

            if epoch_cell_states:
                all_cell_states = torch.cat(epoch_cell_states, dim=0)
                torch.save(all_cell_states, save_dir / f"c_n_epoch{epoch:03d}_validation.pt")
                LOGGER.info(f"Saved validation cell states for epoch {epoch}")

            if epoch_reduced_states:
                all_reduced_states = torch.cat(epoch_reduced_states, dim=0)
                torch.save(all_reduced_states, save_dir / f"c_n_reduced_epoch{epoch:03d}_validation.pt")
                LOGGER.info(f"Saved validation reduced cell states for epoch {epoch}")
