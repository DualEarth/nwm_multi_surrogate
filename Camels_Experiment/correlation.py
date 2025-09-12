import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import seaborn as sns
import os
from scipy.stats import pearsonr, spearmanr
import numpy as np
import statsmodels.formula.api as smf

# -------------------- CSV Loading --------------------
def load_csv(path, label, parse_dates=None):
    try:
        df = pd.read_csv(path, parse_dates=parse_dates)
        print(f"Loaded {label}: {len(df)} rows")
        return df
    except Exception as e:
        print(f"Failed to load {label}: {e}")
        return None

# -------------------- Aggregation --------------------
def aggregate_extrapolated_mahal(df_all_states):
    df_extrap = df_all_states[df_all_states['is_extrapolated'] == True].copy()
    agg = df_extrap.groupby('basin_id')['mahalanobis_distance'].agg(
        sum_mahalanobis_distance_extrapolated='sum'
    ).reset_index()
    agg['basin_id'] = agg['basin_id'].astype(str)
    print(f"Aggregated extrapolated mahalanobis distances for {len(agg)} basins.")
    return agg

# -------------------- Metrics Merge --------------------
def merge_test_metrics(metrics1, metrics2):
    metrics1['basin'] = metrics1['basin'].astype(str)
    if metrics2 is not None:
        metrics2['basin'] = metrics2['basin'].astype(str)
        merged_metrics = pd.merge(metrics1, metrics2, on='basin', how='outer', suffixes=('_m1', '_m2'))
        merged_metrics['NSE'] = merged_metrics['NSE_m1'].combine_first(merged_metrics['NSE_m2'])
        merged_metrics.drop(columns=['NSE_m1', 'NSE_m2'], inplace=True)
        return merged_metrics
    else:
        return metrics1

def merge_all_metrics(summary_df, extrap_df, merged_metrics):
    summary_df['basin'] = summary_df['basin_id'].astype(str)
    extrap_df['basin'] = extrap_df['basin_id'].astype(str)
    merged_metrics['basin'] = merged_metrics['basin'].astype(str)
    merged = pd.merge(summary_df, extrap_df[['basin', 'sum_mahalanobis_distance_extrapolated']], on='basin', how='left')
    merged = pd.merge(merged, merged_metrics, on='basin', how='inner')
    print(f"Final merged data has {len(merged)} rows.")
    return merged

# -------------------- Clean Plotting Function --------------------
def plot_xy_clean(df, x_col, y_col, output_dir, xlabel=None, ylabel= "NSE"):
    df_plot = df[[x_col, y_col]].dropna()
    if df_plot.empty:
        print(f"No data to plot {y_col} vs {x_col}")
        return
    
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(figsize=(8, 5))

    # Scatter points colored by y-value
    sc = ax.scatter(df_plot[x_col], df_plot[y_col], c=df_plot[y_col], cmap='viridis', s=50, alpha=0.7)

    # LOESS regression (blue)
    sns.regplot(data=df_plot, x=x_col, y=y_col, lowess=True,
                scatter=False, ax=ax, line_kws={'color': 'blue'}, ci=None)

    # Polynomial regression degree 2 (red)
    sns.regplot(data=df_plot, x=x_col, y=y_col, order=2,
                scatter=False, ax=ax, line_kws={'color': 'red'}, ci=None)

    ax.set_xscale('log')

    # Colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(ylabel if ylabel else y_col, fontsize=12)

    # Stats
    try:
        r_pearson, _ = pearsonr(df_plot[x_col], df_plot[y_col])
        r_spearman, _ = spearmanr(df_plot[x_col], df_plot[y_col])
    except Exception:
        r_pearson = r_spearman = np.nan

    linear_model = smf.ols(f"{y_col} ~ {x_col}", data=df_plot).fit()
    linear_r2 = linear_model.rsquared

    stats_text = (f"Pearson r: {r_pearson:.2f}\n"
                  f"Spearman r: {r_spearman:.2f}\n"
                  f"Linear $R^2$: {linear_r2:.2f}")
    stats_patch = mpatches.Patch(color='none', label=stats_text)

    # Legend handles
    point_marker = mlines.Line2D([], [], color='green', marker='o', linestyle='None',
                                 markersize=6, label='test basin')
    loess_line_handle = mlines.Line2D([], [], color='blue', label='LOESS')
    poly_line_handle = mlines.Line2D([], [], color='red', label='Polynomial (deg=2)')
    handles = [point_marker, loess_line_handle, poly_line_handle, stats_patch]

    # Legend stays in upper right
    ax.legend(handles=handles, loc='upper right', fontsize=12, handlelength=1, handletextpad=0.5)

    ax.set_xlabel(xlabel if xlabel else x_col.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel(ylabel if ylabel else y_col.replace('_', ' ').title(), fontsize=12)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{y_col.replace(' ', '_')}_vs_{x_col.replace(' ', '_')}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Saved plot to {filepath}")

# -------------------- Main Workflow --------------------
def main():
    summary_csv = r"C:\Users\deonf\Model_work\runs\my_camels_run\collection\analysis\summary.csv"
    all_states_csv = r"C:\Users\deonf\Model_work\runs\my_camels_run\collection\analysis\all_states.csv"
    test_metrics_csv1 = r"C:\Users\deonf\Model_work\runs\my_camels_run\lstm_camels_custom_0708_141507\test\model_epoch030\test_metrics.csv"
    test_metrics_csv2 = r"C:\Users\deonf\Model_work\runs\my_camels_run\lstm_camels_custom_0808_164906\test\model_epoch030\test_metrics.csv"
    output_dir = r"C:\Users\deonf\Model_work\runs\my_camels_run\analysis_plots_final"

    df_summary = load_csv(summary_csv, "Summary")
    df_all_states = load_csv(all_states_csv, "All States", parse_dates=['date'])
    df_metrics1 = load_csv(test_metrics_csv1, "Test Metrics 1")
    df_metrics2 = load_csv(test_metrics_csv2, "Test Metrics 2")

    if df_summary is None or df_all_states is None or df_metrics1 is None:
        print("Missing required data, exiting.")
        return

    df_extrap = aggregate_extrapolated_mahal(df_all_states)
    merged_metrics = merge_test_metrics(df_metrics1, df_metrics2)
    df_merged = merge_all_metrics(df_summary, df_extrap, merged_metrics)

    df_filtered = df_merged[df_merged['NSE'] > 0].copy()
    print(f"Basins with NSE > 0: {df_filtered['basin'].nunique()}")

    plot_specs = [
        ('sum_distance_convex_hull', 'NSE', 
         'Sum of Extrapolation Distance from Convex Hull (log scale)'),

        ('sum_distance_hull_center', 'NSE', 
         'Sum Distance from Geometric Hull Center (log scale)'),

        ('sum_mahalanobis_distance', 'NSE', 
         'Sum of Mahalanobis Distance (log scale)'),

        ('sum_mahalanobis_distance_extrapolated', 'NSE', 
         'Sum of Mahalanobis Distance for Extrapolated States (log scale)'),
    ]

    sns.set_theme(style="whitegrid")

    for x_col, y_col, xlabel in plot_specs:
        if x_col not in df_filtered.columns or y_col not in df_filtered.columns:
            print(f"Skipping plot due to missing columns: {x_col}, {y_col}")
            continue

        if x_col == 'sum_distance_convex_hull':
            df_plot = df_filtered[df_filtered[x_col] >= 1e-2]
        else:
            df_plot = df_filtered

        plot_xy_clean(df_plot, x_col, y_col, output_dir, xlabel=xlabel)

    print("✅ Done.")

if __name__ == "__main__":
    main()
