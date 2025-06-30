import logging
import pickle
import re
import sys
import warnings
from collections import defaultdict
from typing import List, Dict, Union
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
import torch
import xarray
from numba import NumbaPendingDeprecationWarning
from numba import njit, prange
from ruamel.yaml import YAML
from torch.utils.data import Dataset
from tqdm import tqdm


from neuralhydrology.datasetzoo.basedataset import BaseDataset 
from neuralhydrology.datautils import utils
from neuralhydrology.utils.config import Config
from neuralhydrology.utils.errors import NoTrainDataError, NoEvaluationDataError
from neuralhydrology.utils import samplingutils

LOGGER = logging.getLogger(__name__)

combine_strategy= 'auto'
input_dir= 'in'
target_dir= 'out'
datetime_column= 'date'

class CsvDataset(BaseDataset):
    """
    Custom dataset loading inputs and targets from separate CSVs per basin,
    then aligning and merging them before handing off to BaseDataset machinery.

    Config requirements:
      dataset: csvdataset
      data_dir: /path/to/root
      input_dir: subfolder for inputs (e.g. 'inputs')
      target_dir: subfolder for targets (e.g. 'targets')
      datetime_column: name of that column in all CSVs (e.g. 'date')
      dynamic_inputs: list of input column names
      target_variables: list of target column names
      combine_strategy: 'auto'|'align_on_input'|'align_on_common'

    Example config snippet:
    ```yaml
    dataset: csvdataset
    data_dir: /data/my_basins
    input_dir: inputs
    target_dir: targets
    datetime_column: date
    dynamic_inputs:
      - LWDOWN
      - PSFC
      - SWDOWN
    target_variables:
      - ALBEDO
      - FIRA
      - FSA
    combine_strategy: auto
    ```
    """

    def __init__(
        self,
        cfg,
        is_train: bool,
        period: str,
        basin: str = None,
        additional_features: list = None,
        id_to_int: dict = None,
        scaler: dict = None
    ):
        # store combine strategy
        self.strategy = getattr(cfg, 'combine_strategy', 'auto')
        super(CsvDataset, self).__init__(
            cfg=cfg,
            is_train=is_train,
            period=period,
            basin=basin,
            additional_features=additional_features or [],
            id_to_int=id_to_int or {},
            scaler=scaler or {}
        )

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        data_dir = Path(self.cfg.data_dir)
        # locate files
        in_file = find_csv_for_basin(data_dir, self.cfg.input_dir, basin)
        tar_file = find_csv_for_basin(data_dir, self.cfg.target_dir, basin)

        # load into DataFrames
        df_in = load_csv(in_file, self.cfg.datetime_column)
        df_tar = load_csv(tar_file, self.cfg.datetime_column)

        # select only requested columns
        df_in = df_in[self.cfg.dynamic_inputs]
        df_tar = df_tar[self.cfg.target_variables]

        # align
        df_all = align_dataframes(df_in, df_tar, strategy=self.strategy)
        df_all = df_all.dropna(axis=0, how='any')  # drop any rows still missing

        return df_all

    def _load_attributes(self) -> pd.DataFrame:
        # no static attributes by default
        return pd.DataFrame()


def find_csv_for_basin(data_dir: Path, subdir: str, basin: str) -> Path:
    """
    Search `data_dir/subdir` for a CSV file containing `basin` in its stem.
    Raises FileNotFoundError if none or multiple found.
    """
    folder = data_dir / subdir
    files = list(folder.glob('*.csv'))
    matches = [f for f in files if basin in f.stem]
    if not matches:
        raise FileNotFoundError(f"No CSV for basin {basin} in {folder}")
    if len(matches) > 1:
        LOGGER.warning(f"Multiple CSVs for basin {basin} in {folder}, using first: {matches}")
    return matches[0]


def load_csv(path: Path, datetime_col: str) -> pd.DataFrame:
    """
    Load CSV at `path`, parse dates in `datetime_col`, set as DatetimeIndex, sorted.
    """
    df = pd.read_csv(path, parse_dates=[datetime_col])
    df = df.set_index(datetime_col).sort_index()
    df.index = pd.to_datetime(df.index)
    return df


def align_dataframes(
    df_inputs: pd.DataFrame,
    df_targets: pd.DataFrame,
    strategy: str = 'auto'
) -> pd.DataFrame:
    """
    Align inputs & targets using one of:
    - 'auto': if freq matches, simple join; else reindex targets to inputs index and join.
    - 'align_on_input': reindex targets to inputs.index via forward-fill then join.
    - 'align_on_common': inner join on intersection of timestamps.
    """
    if strategy == 'align_on_common':
        return df_inputs.join(df_targets, how='inner')

    # detect freq equivalence
    inp_freq = pd.infer_freq(df_inputs.index)
    tar_freq = pd.infer_freq(df_targets.index)
    if strategy == 'auto' and inp_freq == tar_freq and inp_freq is not None:
        return df_inputs.join(df_targets, how='inner')

    # fallback: align on input timestamps
    df_t_reindexed = df_targets.reindex(df_inputs.index, method='ffill')
    return df_inputs.join(df_t_reindexed, how='inner')
