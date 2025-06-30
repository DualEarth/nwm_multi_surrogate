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

class ParquetDataset(BaseDataset):
    """
    New Configs
      - target_parquet_dir 
      - input_parquet_dir
      - datetime_column
      - combine_strategy

    Load per-basin data from two master Parquet files:
      - one containing all dynamic_inputs (with a 'divide_id' column)
      - one containing all target_variables (also with 'divide_id')
    Slices out each basin’s rows by divide_id, aligns inputs & targets, then
    defers to BaseDataset for the rest of the NH pipeline.
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
        # how to align timestamps
        self.strategy = getattr(cfg, 'combine_strategy', 'auto')
        # which column holds the datetime
        self.datetime_col = getattr(cfg, 'datetime_column', 'date')
        self.default_freq = "1h"

        inp_dir = Path(cfg.data_dir) / cfg.input_parquet_dir
        out_dir = Path(cfg.data_dir) / cfg.target_parquet_dir

        # Read each parquet file once and store its DataFrame
        self._in_dfs  = [pd.read_parquet(p) for p in sorted(inp_dir.glob("*.parquet"))]
        self._out_dfs = [pd.read_parquet(p) for p in sorted(out_dir.glob("*.parquet"))]

        # Convert datetime column in each to actual datetime dtype
        for df in self._in_dfs + self._out_dfs:
            df[self.datetime_col] = pd.to_datetime(df[self.datetime_col])

        super(ParquetDataset, self).__init__(
            cfg=cfg,
            is_train=is_train,
            period=period,
            basin=basin,
            additional_features=additional_features or [],
            id_to_int=id_to_int or {},
            scaler=scaler or {}
        )

    def _load_basin_data(self, basin: str) -> pd.DataFrame:

        # 1) For each input-DataFrame, take only rows matching this basin
        in_slices = []
        for df in self._in_dfs:
            sub = df[df["divide_id"] == basin]
            if not sub.empty:
                in_slices.append(sub)
        if not in_slices:
            raise ValueError(f"No input data found for basin {basin}")
        # then concat them in chronological order
        df_in = pd.concat(in_slices, axis=0).sort_values(self.datetime_col)

        out_slices = []
        for df in self._out_dfs:
            sub = df[df["divide_id"] == basin]
            if not sub.empty:
                out_slices.append(sub)
        if not out_slices:
            raise ValueError(f"No target data found for basin {basin}")
        df_out = pd.concat(out_slices, axis=0).sort_values(self.datetime_col)


        # 2) set time index
        df_in  = df_in .set_index(self.datetime_col).sort_index()
        df_out = df_out.set_index(self.datetime_col).sort_index()

        # 3) pick only the columns you declared in your YAML
        df_in  = df_in [self.cfg.dynamic_inputs]
        df_out = df_out[self.cfg.target_variables]

        # 4) align (inner‐join on common timestamps, or ffill if needed)
        df_all = align_dataframes(df_in, df_out, strategy=self.strategy)

        # Attempt to infer frequency
        freq = pd.infer_freq(df_all.index)
        if freq is None:
            # LOGGER.warning(
            #     f"No frequency inferred for basin {basin}; "
            #     f"defaulting to '{self.default_freq}' and forcing index."
            # )
            # Round any timestamps to the default and asfreq
            df_all = df_all[~df_all.index.duplicated(keep='first')]
            df_all.index = df_all.index.round(self.default_freq)
            df_all = df_all.asfreq(self.default_freq)


        # 5) drop any rows still missing values
        df_all = df_all.dropna(how='any')

        return df_all

    def _load_attributes(self) -> pd.DataFrame:
        # no static attributes in this simple loader
        return pd.DataFrame()


def align_dataframes(
    df_inputs: pd.DataFrame,
    df_targets: pd.DataFrame,
    strategy: str = 'auto'
) -> pd.DataFrame:
    """
    - 'auto': if both have the same pandas-inferred freq, do an inner join;
              else ffill targets to inputs.index then join.
    - 'align_on_input': always ffill targets to inputs.index then join.
    - 'align_on_common': inner join on the intersection of timestamps.
    """
    if strategy == 'align_on_common':
        return df_inputs.join(df_targets, how='inner')

    inp_freq = pd.infer_freq(df_inputs.index)
    tar_freq = pd.infer_freq(df_targets.index)
    if strategy == 'auto' and inp_freq == tar_freq and inp_freq is not None:
        return df_inputs.join(df_targets, how='inner')

    # fallback: forward‐fill targets to inputs' index
    df_t = df_targets.reindex(df_inputs.index, method='ffill')
    return df_inputs.join(df_t, how='inner')
