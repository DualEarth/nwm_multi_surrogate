import pandas as pd
from datetime import datetime
from typing import List
import os
import numpy as np
from pathlib import Path

def reformat_timestamps(csv_path, desired_format="%m/%d/%Y"):
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d %H:%M:%S")
    df['date'] = df['date'].dt.strftime(desired_format)
    df.to_csv(csv_path, index=False)
    return df

# def average_by_date(df, value_col, date_col="time"):
#     """Average the values for each unique date."""
#     daily_avg = df.groupby(date_col)[value_col].mean().reset_index()
#     return daily_avg

def average_by_date(df, start_idx, end_idx, date_col="date"):
    """
    Average a range of columns (by index) grouped by date.

    Parameters:
    - df: pandas DataFrame
    - start_idx: index of the first column to average
    - end_idx: index of the last column to average (inclusive)
    - date_col: name of the date column to group by

    Returns:
    - A DataFrame with one row per date and averaged columns.
    """
    value_cols = df.columns[start_idx:end_idx + 1]
    daily_avg = df.groupby(date_col)[value_cols].mean().reset_index()
    return daily_avg

def select_columns_from_csv(input_path: str, output_path: str, columns: List[str]) -> None:
    """
    Selects specific columns from a CSV and saves them to a new CSV.

    :param input_path: Path to the input CSV file.
    :param output_path: Path to save the new CSV file.
    :param columns: List of column names to keep.
    """
    df = pd.read_csv(input_path)
    df_selected = df[columns]
    df_selected.to_csv(output_path, index=False)
    print(f"Saved selected columns to: {output_path}")

def merge_csvs_on_time_column(file_paths: List[str], time_column: str, output_path: str) -> None:
    """
    Merges multiple CSV files on a common time column.

    :param file_paths: List of CSV file paths to merge.
    :param time_column: The column name representing timestamps in each file.
    :param output_path: Path to save the merged CSV.
    """
    merged_df = None

    for path in file_paths:
        df = pd.read_csv(path)
        df[time_column] = pd.to_datetime(df[time_column])
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on=time_column, how='outer')

    merged_df.sort_values(by=time_column, inplace=True)
    merged_df.to_csv(output_path, index=False)
    print(f"Merged CSV saved to: {output_path}")

def rename_column_in_csv(input_path: str, output_path: str, old_name: str, new_name: str) -> None:
    """
    Renames a column in a CSV file.

    :param input_path: Path to the input CSV.
    :param output_path: Path to save the modified CSV.
    :param old_name: The current column name.
    :param new_name: The new column name.
    """
    df = pd.read_csv(input_path)
    df.rename(columns={old_name: new_name}, inplace=True)
    df.to_csv(output_path, index=False)
    print(f"Renamed column '{old_name}' to '{new_name}' and saved to: {output_path}")

def rename_column_in_parqe(input_path: str, output_path: str, old_name: str, new_name: str) -> None:
    """
    Renames a column in a CSV file.

    :param input_path: Path to the input CSV.
    :param output_path: Path to save the modified CSV.
    :param old_name: The current column name.
    :param new_name: The new column name.
    """
    df = pd.read_parquet(input_path)
    df.rename(columns={old_name: new_name}, inplace=True)
    df.to_parquet(output_path, index=False)
    print(f"Renamed column '{old_name}' to '{new_name}' and saved to: {output_path}")

def add_empty_column_to_csv(csv_path: str, new_column_name: str) -> None:
    """
    Adds an empty column (with just the column name) to the end of an existing CSV file.

    :param csv_path: Path to the CSV file to modify.
    :param new_column_name: Name of the new column to add.
    """
    df = pd.read_csv(csv_path)
    df[new_column_name] = pd.NA  # Add empty column
    df.to_csv(csv_path, index=False)  # Overwrite file with updated content

def delete_column_from_csv(csv_path: str, column_name: str) -> None:
    """
    Deletes a column and all its entries from an existing CSV file.

    :param csv_path: Path to the CSV file to modify.
    :param column_name: Name of the column to delete.
    """
    df = pd.read_csv(csv_path)
    if column_name in df.columns:
        df.drop(columns=[column_name], inplace=True)
        df.to_csv(csv_path, index=False)
        print(f"Column '{column_name}' deleted from {csv_path}.")
    else:
        print(f"Column '{column_name}' not found in {csv_path}.")

def fill_columns_with_nan(input_path, columns_to_clear, nan_or_zero=True, output_path=None):
    """
    Open a CSV file and replace all values in the specified column(s) with NaN.

    Parameters:
        input_path (str): Path to the original CSV file.
        columns_to_clear (str or list of str): Column name(s) to be filled with NaN.
        output_path (str, optional): Path to save the updated CSV. If None, overwrite input_path.

    Returns:
        pd.DataFrame: The updated DataFrame with specified columns filled with NaN.
    """
    # Read the CSV
    df = pd.read_csv(input_path)

    # Convert single string to list
    if isinstance(columns_to_clear, str):
        columns_to_clear = [columns_to_clear]

    # Replace values in the specified columns with NaN
    for col in columns_to_clear:
        if nan_or_zero == True and col in df.columns:
            df[col] = np.nan
        elif nan_or_zero == False and col in df.columns:
            df[col] = 0
        else:
            print(f"Warning: Column '{col}' not found in {input_path}. Skipping.")

    # Save updated CSV
    if output_path is None:
        output_path = input_path  # Overwrite original
    df.to_csv(output_path, index=False)

    return df


def find_and_fix_irregular_timestamps(
    parquet_path: str,
    basin_id: str,
    datetime_column: str = "time",
    divide_id_column: str = "divide_id",
    default_freq: str = "1H",
    fix_timestamps: bool = False
):
    """
    Identify rows whose timestamp breaks the regular frequency pattern for a given basin,
    and optionally round them to the nearest default frequency.

    Args:
        parquet_path: Path to the Parquet file (inputs or targets).
        basin_id: The divide_id of the basin to inspect.
        datetime_column: Column name containing timestamps.
        divide_id_column: Column name for basin identifiers.
        default_freq: Frequency string to use when no freq is inferable or when fixing.
        fix_timestamps: If True, round anomalous timestamps to nearest default_freq.
    Returns:
        anomalies: DataFrame of rows with irregular timestamps and their time deltas.
        df_basin: The (optionally fixed) per-basin DataFrame.
    """
    # Load only relevant columns
    df = pd.read_parquet(parquet_path, columns=[divide_id_column, datetime_column])
    # Filter to this basin
    df_basin = df[df[divide_id_column] == basin_id].copy()
    if df_basin.empty:
        raise ValueError(f"No rows for basin {basin_id}")
    # Ensure datetime and sort
    df_basin[datetime_column] = pd.to_datetime(df_basin[datetime_column])
    df_basin = df_basin.sort_values(datetime_column)
    # Compute deltas
    deltas = df_basin[datetime_column].diff()
    # Find expected (mode) delta
    mode_delta = deltas.mode().iloc[0]
    print(f"Mode (most common) delta for basin {basin_id}: {mode_delta}")
    # Identify anomalies
    anomalies = df_basin[deltas != mode_delta][[datetime_column]].copy()
    anomalies["delta"] = deltas[deltas != mode_delta]
    print(f"Found {len(anomalies)} irregular timestamps:")
    print(anomalies.head(10))
    
    # Optionally fix the anomalies
    if fix_timestamps:
        print("\nRounding irregular timestamps to nearest default frequency:", default_freq)
        # Round timestamp column
        rounded = df_basin[datetime_column].dt.round(default_freq)
        # Assign rounded to anomalies only
        df_basin.loc[anomalies.index, datetime_column] = rounded.loc[anomalies.index]
        # Re-sort and reindex to default_freq
        df_basin = df_basin.set_index(datetime_column).asfreq(default_freq).reset_index()
        print("Timestamps rounded and DataFrame reindexed to", default_freq)
    
    return anomalies, df_basin


def get_valid_divide_ids(parquet_path):
    """
    Reads a parquet file and returns a DataFrame containing only the rows
    where all columns except 'time' and 'divide_id' are non-null.
    The returned DataFrame has 'time' and 'divide_id' (as strings).
    """
    df = pd.read_parquet(parquet_path)
    cols_to_check = [col for col in df.columns if col not in ['date', 'divide_id']]
    valid_df = df.dropna(subset=cols_to_check)
    # Ensure divide_id is string
    valid_df['divide_id'] = valid_df['divide_id'].astype(str)
    return valid_df[['date', 'divide_id']]

def get_common_divide_ids(df1, df2):
    """
    Takes two DataFrames with a 'divide_id' column and returns a new DataFrame
    listing the divide_ids present in both inputs.
    """
    set1 = set(df1['divide_id'])
    set2 = set(df2['divide_id'])
    common = sorted(set1.intersection(set2))
    return pd.DataFrame({'divide_id': common})

INPUT_CSV = r"C:\Users\colli\Downloads\neurohydro\pivoted\OUT\2005_ALBEDO.csv"
TEMP = r"C:\Users\colli\Downloads\neurohydro\data\temp.csv"
OUTPUT_CSV = r"C:\Users\colli\Downloads\neurohydro\data\out\2005_cat1019.csv"
INPUT_FOLDER = r"C:\Users\colli\Downloads\neurohydro\pivoted\OUT"
VAR_NAMES = ['ALBEDO', 'FIRA', 'FSA', 'TRAD'] #['LWDOWN','PSFC','Q2D','RAINRATE','SWDOWN','T2D','U2D','V2D']
PARQUET_FILE = "C:/Users/colli/OneDrive/Documents/PlatformIO/Projects/hydrology/in_parq/2007_LDASIN_nxgn.parquet"
OUTPUT_PARQUET= "./temp2.parquet"

"""Creating new files with variables from the pivoted dataset"""
input_files = sorted([
    os.path.join(INPUT_FOLDER, f)
    for f in os.listdir(INPUT_FOLDER)
    if f.endswith(".csv")
])

"""Reformat files, dates, and converting to daily averges"""
# df = pd.read_csv(r"C:\Users\colli\Downloads\neurohydro\data\in\2005_cat100_3h.csv")
# df = df.sort_values("date")
# df.to_csv(r"C:\Users\colli\Downloads\neurohydro\data\temp.csv", index=False)
# df = reformat_timestamps(r"C:\Users\colli\Downloads\neurohydro\data\in\2005_cat1_3h.csv", desired_format="%m/%d/%Y %H:%M")
# df = average_by_date(df, 1, 12) #3274
# df.to_csv(r"C:\Users\colli\Downloads\neurohydro\data\in\2005_cat1_3h.csv", index=False)


"""Create files with data from existing files"""
# select_columns_from_csv(INPUT_CSV, OUTPUT_CSV, ["time"])
# for input_csv, var_name in zip(input_files, VAR_NAMES):
#     select_columns_from_csv(input_csv, TEMP, ["time", "cat-1019"])
#     merge_csvs_on_time_column([OUTPUT_CSV, TEMP], "time", OUTPUT_CSV)
#     rename_column_in_csv(OUTPUT_CSV, OUTPUT_CSV, "cat-1019", var_name)

# rename_column_in_csv(OUTPUT_CSV, OUTPUT_CSV, "time", "date")
# reformat_timestamps(OUTPUT_CSV, desired_format="%m/%d/%Y %H:%M:%S")
#rename_column_in_parqe(PARQUET_FILE, PARQUET_FILE, "time", "date")

'''Clearing columns'''
# fill_columns_with_nan(
#     input_path="./data/temp.csv",
#     nan_or_zero=False,
#     columns_to_clear=['padding']
# )

# for input_csv in input_files:
#     for var_name in VAR_NAMES:
#         delete_column_from_csv(input_csv, var_name)

"""Sorting by date/time"""
# import pandas as pd
# df = pd.read_csv("./data/2005_cat1_in.csv", parse_dates=["date"])
# df = df.sort_values("date")  # Sort if not sorted

# print(df["date"].head())
# print(pd.infer_freq(df["date"])) 

'''find and fix timestep anomolies'''
#anomalies, fixed_df = find_and_fix_irregular_timestamps(
# anomalies = find_and_fix_irregular_timestamps(
#     parquet_path="C:/Users/colli/OneDrive/Documents/PlatformIO/Projects/hydrology/2006_LDASIN_nxgn.parquet",
#     basin_id="cat-1",
#     datetime_column="time",
#     fix_timestamps=False
# )

'''Get common divide_ids between files'''
# df1_valid = get_valid_divide_ids('C:/Users/colli/OneDrive/Documents/PlatformIO/Projects/hydrology/in_parq/2006_LDASIN_nxgn.parquet')
# df2_valid = get_valid_divide_ids('C:/Users/colli/OneDrive/Documents/PlatformIO/Projects/hydrology/out_parq/2006_LDASOUT_nxgn.parquet')
# common_df = get_common_divide_ids(df1_valid, df2_valid)
# print(common_df.size)
# for did in common_df['divide_id'].sample(n=60, random_state=42):
#     print(did)

