import os
from pathlib import Path
from typing import Optional, Any, List, Dict
import pandas as pd
from functools import reduce

from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.utils.config import Config

# Global cache for preloaded parquet data
_GLOBAL_PARQUET_DATA_CACHE: Dict[tuple, Dict[str, Dict[str, pd.DataFrame]]] = {}
_CACHE_INITIALIZED_FOR_DIRS: Dict[tuple, bool] = {}


class ParquetDataset(BaseDataset):
    def __init__(
        self,
        cfg: Config,
        is_train: bool,
        period: str,
        basin: Optional[str] = None,
        scaler: Optional[Any] = None,
        **kwargs,
    ):
        self._validate_config(cfg)
        self.parquet_ldasin_dir = Path(cfg.parquet_ldasin_dir)
        self.parquet_ldasout_dir = Path(cfg.parquet_ldasout_dir)
        self.basin = basin
        self._cache_key = (str(self.parquet_ldasin_dir), str(self.parquet_ldasout_dir))

        if self._cache_key not in _CACHE_INITIALIZED_FOR_DIRS:
            self._preload_all_parquet_data()
            _CACHE_INITIALIZED_FOR_DIRS[self._cache_key] = True

        super().__init__(
            cfg=cfg,
            is_train=is_train,
            period=period,
            basin=basin,
            scaler=scaler,
            **kwargs,
        )

    def _validate_config(self, cfg: Config):
        if not hasattr(cfg, "parquet_ldasin_dir") or not hasattr(cfg, "parquet_ldasout_dir"):
            raise ValueError("Config must include 'parquet_ldasin_dir' and 'parquet_ldasout_dir'")

    def _preload_all_parquet_data(self):
        if self._cache_key not in _GLOBAL_PARQUET_DATA_CACHE:
            _GLOBAL_PARQUET_DATA_CACHE[self._cache_key] = {"ldasin": {}, "ldasout": {}}

        current_cache = _GLOBAL_PARQUET_DATA_CACHE[self._cache_key]
        total_mem_mb = 0

        total_mem_mb += self._cache_parquet_files(self.parquet_ldasin_dir, current_cache["ldasin"])
        total_mem_mb += self._cache_parquet_files(self.parquet_ldasout_dir, current_cache["ldasout"])

    def _cache_parquet_files(self, directory: Path, cache_dict: Dict[str, pd.DataFrame]) -> float:
        total_mem_mb = 0
        for file in os.listdir(directory):
            if file.endswith(".parquet"):
                var_name = file[:-8]
                file_path = directory / file
                try:
                    df = pd.read_parquet(file_path)
                    df.rename(columns={"time": "date"}, inplace=True)
                    df["date"] = pd.to_datetime(df["date"]).dt.round("h")
                    df.set_index("date", inplace=True)
                    cache_dict[var_name] = df
                    total_mem_mb += df.memory_usage(deep=True).sum() / (1024 ** 2)
                except Exception as e:
                    print(f"Warning: Failed to load parquet file {file_path}: {e}")
        return total_mem_mb

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        cached_data = _GLOBAL_PARQUET_DATA_CACHE.get(self._cache_key)
        if cached_data is None:
            raise RuntimeError(f"Parquet cache not populated for directories: {self._cache_key}")

        df_in_list = self._extract_basin_dfs(cached_data["ldasin"], basin)
        df_out_list = self._extract_basin_dfs(cached_data["ldasout"], basin)

        if not df_in_list and not df_out_list:
            raise ValueError(f"No data found for basin '{basin}' in input or output datasets.")

        df_in = self._merge_dfs_on_index(df_in_list)
        df_out = self._merge_dfs_on_index(df_out_list)

        if not df_in.empty and not df_out.empty:
            df_merged = pd.merge(df_in, df_out, left_index=True, right_index=True, how="inner")
        elif not df_in.empty:
            df_merged = df_in
        elif not df_out.empty:
            df_merged = df_out
        else:
            raise ValueError(f"No combined data found for basin '{basin}' after merging.")

        return df_merged[~df_merged.index.duplicated(keep="first")]

    def _extract_basin_dfs(self, data_dict: Dict[str, pd.DataFrame], basin: str) -> List[pd.DataFrame]:
        dfs = []
        for var_name, df_full in data_dict.items():
            if basin in df_full.columns:
                dfs.append(df_full[[basin]].rename(columns={basin: var_name}))
        return dfs

    def _merge_dfs_on_index(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        if not dfs:
            return pd.DataFrame()
        return reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how="inner"), dfs)

    def _load_static_attributes(self) -> pd.DataFrame:
        return pd.DataFrame()

    def __getitem__(self, index):
        return super().__getitem__(index)
