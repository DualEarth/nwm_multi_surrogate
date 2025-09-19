import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from scipy.stats import pearsonr


def _load_data(path, description):
    print(f"Loading {description} from: {path}")
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: {description} file not found at {path}")
        return None
    except Exception as e:
        print(f"Error loading {description} CSV: {e}")
        return None

def _preprocess_and_merge_basins(df_extrapolation, df_metrics, extrapolation_col):
    if df_extrapolation is None or df_metrics is None:
        return None
    df_extrapolation['basin'] = df_extrapolation['basin_id'].astype(str)
    df_metrics['basin'] = df_metrics['basin'].astype(str).str.replace(r'\.0$', '', regex=True)

    if 'cumulative extrapolation distance' in df_extrapolation.columns:
        df_extrapolation.rename(columns={'cumulative extrapolation distance': extrapolation_col}, inplace=True)

    df_merged = pd.merge(df_extrapolation, df_metrics, on='basin', how='inner')
    if df_merged.empty:
        print("Warning: No matching basin IDs found.")
        return None
    print(f"Successfully merged data for {len(df_merged)} basins.")
    return df_merged

def _filter_zero_extrapolation(df_merged, extrapolation_col):
    if df_merged is None:
        return None
    initial_count = len(df_merged)
    df_merged[extrapolation_col] = pd.to_numeric(df_merged[extrapolation_col], errors='coerce').fillna(0)
    df_filtered = df_merged[df_merged[extrapolation_col] > 0].copy()
    removed_count = initial_count - len(df_filtered)
    if removed_count > 0:
        print(f"Removed {removed_count} basin(s) with zero '{extrapolation_col}'.")
    if df_filtered.empty:
        print("Warning: All basins removed.")
        return None
    return df_filtered

# --- Helper to fix metric labels --- #
def _format_metric_label(metric_col):
    label = metric_col.replace('_', ' ')
    label = label.replace('Nse', 'NSE').replace('Rmse', 'RMSE')
    return label

# --- Corrected Helper to calculate and display correlation --- #
def _calculate_and_display_correlation(ax, plot_df, x_col, y_col):
    # Create a clean DataFrame for correlation calculation
    corr_df = plot_df[[x_col, y_col]].dropna()

    # Check for sufficient data points and variability
    if len(corr_df) >= 2 and corr_df[x_col].nunique() > 1 and corr_df[y_col].nunique() > 1:
        try:
            r, p = pearsonr(corr_df[x_col], corr_df[y_col])
            corr_text = f'Pearson r: {r:.2f}\nP-value: {p:.3e}'
        except Exception:
            corr_text = "Correlation: N/A"
    else:
        corr_text = "Correlation: N/A (Insufficient or constant data)"

    ax.text(0.05, 0.95, corr_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.8, alpha=0.8))

def _plot_single_metric(df_filtered, extrapolation_col, metric_col, output_plot_dir):
    if metric_col not in df_filtered.columns:
        print(f"Metric '{metric_col}' not found. Skipping.")
        return

    plot_df = df_filtered[[extrapolation_col, metric_col, 'basin']].copy()
    plot_df[extrapolation_col] = pd.to_numeric(plot_df[extrapolation_col], errors='coerce')
    plot_df[metric_col] = pd.to_numeric(plot_df[metric_col], errors='coerce')
    plot_df.dropna(subset=[extrapolation_col, metric_col], inplace=True)
    if plot_df.empty:
        print(f"No valid data for '{metric_col}'. Skipping.")
        return

    # Research paper styling
    sns.set_theme(style="whitegrid", font_scale=1.5)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    sns.regplot(
        data=plot_df,
        x=extrapolation_col,
        y=metric_col,
        lowess=True,
        scatter=True,
        color='skyblue',
        scatter_kws={'s': 60, 'alpha': 0.8, 'edgecolor': 'black', 'linewidths': 0.5},
        line_kws={'color': 'red', 'linestyle': '--', 'linewidth': 2}
    )

    _calculate_and_display_correlation(ax, plot_df, extrapolation_col, metric_col)
    
    label = _format_metric_label(metric_col)
    ax.set_xscale('log')
    ax.set_xlabel("Sum Extrapolation Distance", fontsize=14, fontweight='bold')
    ax.set_ylabel(label, fontsize=14, fontweight='bold')
    ax.set_title(f'{label} vs Sum Extrapolation Distance',
                 fontsize=16, fontweight='bold', pad=15)

    sns.despine()
    plt.tight_layout()
    plot_filename = os.path.join(output_plot_dir, f"{metric_col}_vs_{extrapolation_col.replace(' ', '_')}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved: {plot_filename}")

# --- Corrected Final NSE plot function --- #
def _plot_final_nse_vs_sum_extrapolation(df, extrapolation_col, output_dir):
    nse_cols = [col for col in df.columns if col.endswith("_NSE")]
    if not nse_cols:
        print("No NSE columns found.")
        return

    final_data = []
    for basin, row in df.groupby("basin").first().iterrows():
        sum_extrap = pd.to_numeric(row[extrapolation_col], errors="coerce")
        if sum_extrap <= 1e-2:
            continue

        nse_values = [
            pd.to_numeric(row[col], errors='coerce')
            for col in nse_cols
            if pd.notna(row[col]) and row[col] > 0
        ]
        if nse_values:
            final_data.append({
                "basin": basin,
                "sum_extrapolation": sum_extrap,
                "avg_nse": np.mean(nse_values)
            })

    if not final_data:
        print("No basins passed the filter for avg NSE plot. Skipping.")
        return

    plot_df = pd.DataFrame(final_data)

    # Check if there are enough data points before plotting and calculating correlation
    if len(plot_df) < 2:
        print("Insufficient data for final NSE plot. Skipping correlation calculation and plot.")
        return

    sns.set_theme(style="whitegrid", font_scale=1.5)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    sns.regplot(
        data=plot_df,
        x="sum_extrapolation",
        y="avg_nse",
        lowess=True,
        scatter=True,
        color='skyblue',
        scatter_kws={'s': 60, 'alpha': 0.8, 'edgecolor': 'black', 'linewidths': 0.5},
        line_kws={'color': 'red', 'linestyle': '--', 'linewidth': 2}
    )

    # Re-check for sufficient data for correlation after filtering
    if len(plot_df) >= 2 and plot_df["sum_extrapolation"].nunique() > 1 and plot_df["avg_nse"].nunique() > 1:
        r, p = pearsonr(plot_df["sum_extrapolation"], plot_df["avg_nse"])
        corr_text = f"Pearson r: {r:.2f}\nP-value: {p:.3e}"
    else:
        corr_text = "Correlation: N/A (Insufficient data)"
        
    ax.text(
        0.05, 0.95,
        corr_text,
        transform=ax.transAxes, fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.8, alpha=0.8)
    )

    ax.set_xscale('log')
    ax.set_xlabel("Sum Extrapolation Distance", fontsize=14, fontweight='bold')
    ax.set_ylabel("Average NSE (NSE>0 only)", fontsize=14, fontweight='bold')
    ax.set_title("Average NSE vs Sum Extrapolation Distance", fontsize=16, fontweight='bold', pad=15)

    sns.despine()
    plt.tight_layout()
    out_path = os.path.join(output_dir, "final_avgNSE_vs_sumExtrapolation.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Final NSE plot saved: {out_path}")

# ---------------- Main function ---------------- #
def plot_all_with_final(extrapolation_summary_path, test_metrics_path,
                        extrapolation_col='sum extrapolation distance',
                        metrics_cols=None, output_plot_dir="extrapolation_plots"):
    if metrics_cols is None:
        print("No metrics columns specified.")
        return
    os.makedirs(output_plot_dir, exist_ok=True)
    df_extrapolation = _load_data(extrapolation_summary_path, "extrapolation summary")
    df_metrics = _load_data(test_metrics_path, "test metrics")
    df_merged = _preprocess_and_merge_basins(df_extrapolation, df_metrics, extrapolation_col)
    if df_merged is None:
        return
    df_filtered = _filter_zero_extrapolation(df_merged, extrapolation_col)
    if df_filtered is None:
        return

    for metric_col in metrics_cols:
        _plot_single_metric(df_filtered, extrapolation_col, metric_col, output_plot_dir)

    _plot_final_nse_vs_sum_extrapolation(df_filtered, extrapolation_col, output_plot_dir)

# ---------------- Run script ---------------- #
if __name__ == "__main__":
    extrapolation_summary_csv = r"C:\Users\deonf\Model_work\runs\my_parquet_run\lstm_parquet_custom_0807_091323\internal_states\test\extrapolation_summary_per_basin.csv"
    test_metrics_csv = r"C:\Users\deonf\Model_work\runs\my_parquet_run\lstm_parquet_custom_0807_091323\test\model_epoch030\test_metrics.csv"
    output_plots_directory = r"C:\Users\deonf\Model_work\runs\my_parquet_run\lstm_parquet_custom_0807_091323\test\model_epoch030\plots"

    y_axis_metrics = [
        'ACCET_NSE', 'ACCET_RMSE', 'EDIR_NSE', 'EDIR_RMSE',
        'FIRA_NSE', 'FIRA_RMSE', 'FSA_NSE', 'FSA_RMSE',
        'HFX_NSE', 'HFX_RMSE', 'LH_NSE', 'LH_RMSE',
        'QRAIN_NSE', 'QRAIN_RMSE', 'COSZ_NSE', 'COSZ_RMSE',
        'TRAD_NSE', 'TRAD_RMSE'
    ]

    plot_all_with_final(
        extrapolation_summary_path=extrapolation_summary_csv,
        test_metrics_path=test_metrics_csv,
        extrapolation_col='sum extrapolation distance',
        metrics_cols=y_axis_metrics,
        output_plot_dir=output_plots_directory
    )
