import torch
import numpy as np
import pandas as pd
import os
import re
import time
from sklearn.covariance import LedoitWolf
from scipy.spatial import ConvexHull
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- Helpers ---

def extract_basin_id_from_filename(filename, pattern):
    match = re.search(pattern, filename)
    return match.group(1) if match else None

def compute_mahalanobis_ledoitwolf_no_pca(train_data, test_data):
    cov_estimator = LedoitWolf().fit(train_data)
    VI = cov_estimator.get_precision()
    diffs = test_data - cov_estimator.location_
    sq_mahal = np.sum(diffs @ VI * diffs, axis=1)
    mahal_dist = np.sqrt(np.clip(sq_mahal, 0, None))
    return mahal_dist, cov_estimator

def calculate_basin_spread_volume(test_data, basin_ids):
    unique_basins = np.unique(basin_ids)
    spread_data = []

    min_points_for_hull = 4
    
    for basin in unique_basins:
        mask = basin_ids == basin
        basin_points = test_data[mask]
        
        volume = 0.0
        if basin_points.shape[0] >= min_points_for_hull:
            try:
                hull = ConvexHull(basin_points)
                volume = hull.volume
            except Exception as e:
                print(f"Warning: Could not compute convex hull volume for basin {basin}. Points may be degenerate. Error: {e}")
                volume = 0.0
        
        spread_data.append({"basin_id": basin, "basin_spread_volume": volume})
    
    return pd.DataFrame(spread_data)

def load_nse_scores(basin_ids, nse_file_path, nse_basin_column, nse_score_column):
    try:
        nse_df = pd.read_csv(nse_file_path)
        nse_df[nse_basin_column] = nse_df[nse_basin_column].astype(str)
        nse_data = nse_df.set_index(nse_basin_column)[nse_score_column].to_dict()
        print(f"Successfully loaded NSE data from '{nse_file_path}' using columns '{nse_basin_column}' and '{nse_score_column}'.")
    except KeyError:
        print(f"Error: Could not find columns '{nse_basin_column}' or '{nse_score_column}' in the NSE file. Please check the column names.")
        return None
    except Exception as e:
        print(f"Error loading NSE file: {e}")
        return None

    nse_values = np.array([nse_data.get(str(b), np.nan) for b in basin_ids])
    return nse_values

def generate_colors(N):
    colors = plt.get_cmap("tab20").colors
    num_colors = len(colors)
    full_colors = [colors[i % num_colors] for i in range(N)]
    return full_colors

# --- Main function ---

def analyze_and_visualize_no_pca(
    train_file, test_dir, nse_file_path,
    output_summary_csv, output_detailed_csv,
    output_anomalies_csv, mahal_threshold,
    nse_basin_column, nse_score_column,
    highlight_basins=None,
    test_file_pattern=r'_basin_(\d+\.?\d*)\.pt$'
):
    if highlight_basins is not None:
        print("[INFO] 'highlight_basins' parameter is ignored. All basins will be colored distinctly.")

    t0 = time.perf_counter()

    try:
        train_data = torch.load(train_file, map_location='cpu').numpy()
    except Exception as e:
        print(f"[ERROR] loading training data: {e}")
        return
    train_data = train_data[~np.isnan(train_data).any(axis=1)]
    if train_data.size == 0:
        print("[ERROR] training data empty after NaN removal.")
        return
    if train_data.shape[1] != 3:
        print(f"[WARNING] Expected 3D training data, got {train_data.shape[1]}. Visualization assumes 3D.")

    print(f"Training data: {train_data.shape[0]} points of dim {train_data.shape[1]}")

    test_points_with_meta = []
    found_files = 0
    for fname in os.listdir(test_dir):
        if re.search(test_file_pattern, fname):
            found_files += 1
            basin_id = extract_basin_id_from_filename(fname, test_file_pattern)
            if basin_id is None:
                print(f"Could not extract basin ID from {fname}, skipping.")
                continue
            fpath = os.path.join(test_dir, fname)
            try:
                data_raw = torch.load(fpath, map_location='cpu').numpy()
                data_clean = data_raw[~np.isnan(data_raw).any(axis=1)]
                if data_clean.shape[1] != train_data.shape[1]:
                    print(f"Skipping {fname}: dim mismatch (train {train_data.shape[1]} vs test {data_clean.shape[1]})")
                    continue
                for idx_pt, pt in enumerate(data_clean):
                    test_points_with_meta.append({
                        "basin_id": basin_id,
                        "index_in_basin_file": idx_pt,
                        "coordinates": pt
                    })
            except Exception as e:
                print(f"Skipping {fname} due to error: {e}")

    if not test_points_with_meta:
        if found_files == 0:
            print(f"No files matching pattern '{test_file_pattern}' found.")
        else:
            print("No valid test points loaded.")
        return

    test_data = np.array([x["coordinates"] for x in test_points_with_meta])
    basin_ids = np.array([x["basin_id"] for x in test_points_with_meta])
    indices_in_files = np.array([x["index_in_basin_file"] for x in test_points_with_meta])
    print(f"Loaded {test_data.shape[0]} test points from {found_files} files.")

    mahal_dist, _ = compute_mahalanobis_ledoitwolf_no_pca(train_data, test_data)
    print("Computed Mahalanobis distances (Ledoit-Wolf).")
    
    # Load NSE scores
    nse_values = load_nse_scores(basin_ids, nse_file_path, nse_basin_column, nse_score_column)

    df_detailed = pd.DataFrame({
        "basin_id": basin_ids,
        "point_index_in_basin_file": indices_in_files,
        "distance_mahalanobis": mahal_dist,
        "NSE_score": nse_values,
    })
    for d in range(test_data.shape[1]):
        df_detailed[f"coord_c{d+1}"] = test_data[:, d]
    df_detailed.to_csv(output_detailed_csv, index=False)
    print(f"Saved detailed CSV to: {output_detailed_csv}")

    anomalous_points = df_detailed[df_detailed["distance_mahalanobis"] > mahal_threshold]
    anomalies_to_save = anomalous_points[[
        'basin_id', 'point_index_in_basin_file',
        'coord_c1', 'coord_c2', 'coord_c3',
        'distance_mahalanobis'
    ]].copy()
    anomalies_to_save.rename(columns={
        'coord_c1': 'coordinate_1',
        'coord_c2': 'coordinate_2',
        'coord_c3': 'coordinate_3'
    }, inplace=True)
    anomalies_to_save.to_csv(output_anomalies_csv, index=False)
    print(f"Saved anomalies CSV with {len(anomalous_points)} points to: {output_anomalies_csv}")

    diffusion_df = df_detailed.groupby("basin_id").agg(
        total_points=("basin_id", "count"),
        mean_mahalanobis_distance=("distance_mahalanobis", "mean"),
        num_anomalies_above_threshold=("distance_mahalanobis", lambda x: int((x > mahal_threshold).sum())),
        mean_nse_score=("NSE_score", "mean")
    ).reset_index()
    
    spread_volume_df = calculate_basin_spread_volume(test_data, basin_ids)
    
    summary_df = pd.merge(diffusion_df, spread_volume_df, on="basin_id", how="left")
    
    try:
        nse_df = pd.read_csv(nse_file_path)
        nse_df[nse_basin_column] = nse_df[nse_basin_column].astype(str)
        summary_df = pd.merge(summary_df, nse_df, left_on="basin_id", right_on=nse_basin_column, how="left")
        summary_df.drop(columns=[nse_basin_column], inplace=True)
    except Exception as e:
        print(f"Could not load or merge NSE data from {nse_file_path}: {e}")
    
    summary_df.to_csv(output_summary_csv, index=False)
    print(f"Saved summary CSV to: {output_summary_csv}")

    # --- Convex hull and visualization ---
    
    pv.global_theme.font.size = 14
    pv.set_plot_theme("document")
    plotter = pv.Plotter(window_size=(1100, 800))

    # Manually create a list to hold all legend labels
    legend_labels = []

    hull_input = train_data
    if hull_input.shape[0] < hull_input.shape[1] + 1:
        print("Too few training points to build a convex hull for visualization.")
        hull = None
    else:
        try:
            hull = ConvexHull(hull_input)
        except Exception as e:
            print(f"Failed to build hull: {e}")
            hull = None

    if hull is not None:
        mesh = pv.PolyData(hull_input)
        faces_np = np.hstack([[3, *face] for face in hull.simplices])
        mesh.faces = faces_np
        plotter.add_mesh(mesh, color="brown", opacity=0.2, show_edges=True)
        legend_labels.append(["Convex Hull (Train)", "brown"])

    plotter.add_points(hull_input, color="blue", point_size=4, render_points_as_spheres=True)
    legend_labels.append(["Train Data", "blue"])

    if test_data.size > 0:
        unique_basins = np.unique(basin_ids)
        num_basins = len(unique_basins)
        colors = generate_colors(num_basins)
        
        for i, basin in enumerate(unique_basins):
            mask = basin_ids == basin
            basin_points = test_data[mask]
            
            color = colors[i]
            
            plotter.add_points(
                basin_points,
                color=color,
                point_size=7,
                render_points_as_spheres=True
            )
            # Add each basin's label to the list
            legend_labels.append([f"Basin {basin}", color])
    else:
        print("No test points to visualize.")
        
    # Final call to add the legend with the manually built list
    plotter.add_legend(labels=legend_labels, bcolor="white", border=True, size=(0.15, 0.1))
    
    plotter.show(title="States colored by Basin")

    t1 = time.perf_counter()
    print(f"Total runtime: {t1 - t0:.2f} seconds")


if __name__ == "__main__":
    train_file_path = r"C:\Users\deonf\Model_work\runs\my_camels_run\lstm_camels_custom_1208_140157\internal_states\train\c_n_reduced_epoch030.pt"
    test_data_directory = r"C:\Users\deonf\Model_work\runs\my_camels_run\lstm_camels_custom_1208_140157\internal_states\test"
    
    nse_file = r"C:\Users\deonf\Model_work\runs\my_camels_run\lstm_camels_custom_1208_140157\test\model_epoch030\test_metrics.csv"
    
    nse_basin_column_name = 'basin'
    nse_score_column_name = 'NSE'

    output_summary_csv_path = os.path.join(test_data_directory, "all_basins_colored_summary_with_nse.csv")
    output_detailed_csv_path = os.path.join(test_data_directory, "all_basins_colored_detailed_with_nse.csv")
    output_anomalies_csv_path = os.path.join(test_data_directory, "all_basins_colored_anomalies_with_nse.csv")
    
    mahal_distance_threshold = 11.0
    file_pattern = r'_basin_(\d+\.?\d*)\.pt$'

    analyze_and_visualize_no_pca(
        train_file_path,
        test_data_directory,
        nse_file,
        output_summary_csv_path,
        output_detailed_csv_path,
        output_anomalies_csv_path,
        mahal_distance_threshold,
        nse_basin_column=nse_basin_column_name,
        nse_score_column=nse_score_column_name,
        highlight_basins=[],
        test_file_pattern=file_pattern
    )
