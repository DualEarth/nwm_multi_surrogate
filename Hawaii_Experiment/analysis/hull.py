import torch
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import pyvista as pv
import pandas as pd
import os
import re
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import inv
import osqp
from scipy.sparse import csc_matrix
from multiprocessing import Pool, cpu_count

# --- Helper Functions (Defined FIRST) ---

# Helper function to extract basin ID, using the provided pattern
def extract_basin_id_from_filename(filename, pattern):
    match = re.search(pattern, filename)
    return match.group(1) if match else None

def mahalanobis_filter(points, percentile=99.0):
    """
    Filters points based on their Mahalanobis distance from the data's mean.
    Points with Mahalanobis distances above the specified percentile are removed.

    Args:
        points (np.ndarray): A NxD array of data points.
        percentile (float): The percentile cutoff for Mahalanobis distance (e.g., 99.6).

    Returns:
        np.ndarray: The filtered array of points.
    """
    if points.shape[0] < points.shape[1]:
        print("Warning: Not enough points for a stable covariance matrix. Skipping Mahalanobis filter.")
        return points
    if points.shape[0] < 2: # Need at least 2 points for covariance calculation
        return points

    mean = np.mean(points, axis=0)
    covariance = np.cov(points, rowvar=False) + np.eye(points.shape[1]) * 1e-6

    try:
        inv_covariance = inv(covariance)
    except np.linalg.LinAlgError:
        print("Warning: Covariance matrix is singular. Skipping Mahalanobis filter.")
        return points

    diff = points - mean
    mahalanobis_distances = np.sqrt(np.sum((diff @ inv_covariance) * diff, axis=1))

    cutoff = np.percentile(mahalanobis_distances, percentile)
    mask = mahalanobis_distances <= cutoff

    print(f"Mahalanobis Filter: Retained {np.sum(mask)}/{points.shape[0]} points (>{percentile} percentile cutoff: {cutoff:.4f})")
    return points[mask]

def diffusion_filter(points, k=20, density_percentile=99.9):
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    distances, _ = nbrs.kneighbors(points)
    avg_distances = np.mean(distances, axis=1)
    cutoff = np.percentile(avg_distances, density_percentile)
    mask = avg_distances <= cutoff
    print(f"Diffusion Filter: Retained {np.sum(mask)}/{points.shape[0]} points (>{density_percentile} percentile cutoff: {cutoff:.4f})")
    return points[mask]

def calculate_closest_point_on_hull_qp(point, hull_vertices):
    """
    Calculates the Euclidean distance from a point to its closest point on the convex hull
    defined by hull_vertices using Quadratic Programming.

    Args:
        point (np.ndarray): The 1xD query point.
        hull_vertices (np.ndarray): NxD array of points defining the convex hull.

    Returns:
        float: The Euclidean distance from the point to the closest point on the hull.
               Returns np.inf if the QP solver fails or if there are insufficient vertices.
    """
    num_vertices = hull_vertices.shape[0]
    num_dimensions = hull_vertices.shape[1]

    if num_vertices < num_dimensions + 1:
        if num_vertices == 1:
            return np.linalg.norm(point - hull_vertices[0])
        elif num_vertices == 0:
            return np.inf

    # Problem: minimize 0.5 * alpha.T @ P @ alpha + q.T @ alpha
    P_qp = 2 * (hull_vertices @ hull_vertices.T)

    # Adding a small identity matrix to the diagonal of P_qp for numerical stability
    regularization_term = np.eye(num_vertices) * 1e-9 # A small value like 1e-9 or 1e-6
    P_qp += regularization_term

    q_qp = -2 * (hull_vertices @ point)

    P_osqp = csc_matrix(P_qp)

    # Constraints: alpha_i >= 0 and sum(alpha_i) == 1
    A_constraints = np.vstack([
        np.eye(num_vertices),
        np.ones((1, num_vertices))
    ])
    A_osqp = csc_matrix(A_constraints)

    l = np.hstack([np.zeros(num_vertices), 1.0])
    u = np.hstack([np.full(num_vertices, np.inf), 1.0])

    prob = osqp.OSQP()
    try:
        prob.setup(P=P_osqp, q=q_qp, A=A_osqp, l=l, u=u, verbose=False)

        # Solve problem
        res = prob.solve()

        if res.info.status == 'solved':
            alpha_optimal = res.x
            closest_point_on_hull = np.sum(alpha_optimal[:, None] * hull_vertices, axis=0)
            distance = np.linalg.norm(point - closest_point_on_hull)
            return distance
        else:
            # print(f"OSQP solver failed to converge. Status: {res.info.status} (Code: {res.info.status_val})")
            return np.inf
    except Exception as e:
        # print(f"Error during QP calculation (setup or solve): {e}")
        return np.inf

# --- Main Plotting Function ---
def plot_convexhull_and_test_states(train_file, validate_file, test_dir,
                                     output_csv_path, detailed_csv_path=None,
                                     test_file_pattern=r'_basin_(\d+)_0\.pt$',
                                     filter_type='mahalanobis', filter_percentile=99.6,
                                     distance_calc_method='qp'):
    # Initialize train_data and validate_data to None
    train_data = None
    validate_data = None

    try:
        train_data = torch.load(train_file, map_location='cpu').numpy()
        validate_data = torch.load(validate_file, map_location='cpu').numpy()
    except FileNotFoundError as e:
        print(f"Error: One of the specified training/validation files was not found: {e}")
        return
    except Exception as e:
        print(f"Error loading training/validation data: {e}")
        return

    # --- Data validation and dimension definition (moved here) ---
    if train_data is None or validate_data is None:
        print("Error: Training or validation data not loaded. Exiting.")
        return

    num_dimensions = train_data.shape[1]

    if num_dimensions != 3:
        print(f"Error: Expected 3D data, but got {num_dimensions}D. Exiting.")
        return

    # Remove NaN values (if any)
    train_data = train_data[~np.isnan(train_data).any(axis=1)]
    validate_data = validate_data[~np.isnan(validate_data).any(axis=1)]

    combined_data = np.vstack([train_data, validate_data])
    print(f"Original combined training + validation points: {combined_data.shape[0]}")

    # Apply Filtering
    if filter_type == 'mahalanobis':
        combined_data_filtered = mahalanobis_filter(combined_data, percentile=filter_percentile)
    elif filter_type == 'diffusion':
        combined_data_filtered = diffusion_filter(combined_data, density_percentile=filter_percentile)
    elif filter_type == 'none':
        combined_data_filtered = combined_data
        print("No filtering applied to combined training + validation points.")
    else:
        print(f"Warning: Unknown filter type '{filter_type}'. No filtering applied.")
        combined_data_filtered = combined_data

    # Check for sufficient points for convex hull after filtering
    if combined_data_filtered.shape[0] < num_dimensions + 1:
        print(f"Warning: Too few valid points after filtering ({combined_data_filtered.shape[0]} points, min {num_dimensions + 1} required for {num_dimensions}D hull).")
        print("Skipping convex hull calculation and distance metrics. Only plotting points.")
        hull = None
        tri = None
    else:
        hull = ConvexHull(combined_data_filtered)
        tri = Delaunay(combined_data_filtered)
        print(f"Convex hull created with {len(hull.simplices)} faces from {combined_data_filtered.shape[0]} filtered points.")

    # --- Load all test data from the directory, filtered by pattern ---
    all_test_data_list = []
    # To store original global indices for test points (before filtering and per basin)
    # This will help us reconstruct the full test_data array along with its basin IDs
    original_test_data_with_meta = [] # Store (basin_id, original_index_in_file, data_point)

    found_test_files_count = 0

    for fname in os.listdir(test_dir):
        if re.search(test_file_pattern, fname):
            found_test_files_count += 1
            basin_id = extract_basin_id_from_filename(fname, test_file_pattern)
            if basin_id is None:
                print(f"Could not extract basin ID from {fname}, skipping.")
                continue

            fpath = os.path.join(test_dir, fname)
            try:
                basin_data_raw = torch.load(fpath, map_location='cpu').numpy()
                # Store dates if available for this basin, assuming sequential matching
                # For this function, we don't have direct date mapping per point,
                # but we'll record basin_id and point index.
                
                # Filter NaN values
                non_nan_mask = ~np.isnan(basin_data_raw).any(axis=1)
                basin_data = basin_data_raw[non_nan_mask]

                if basin_data.shape[1] == num_dimensions:
                    for idx_in_basin_file, point in enumerate(basin_data):
                        # The 'original_index_in_file' here corresponds to the index AFTER NaN removal
                        original_test_data_with_meta.append({
                            "basin_id": basin_id,
                            "index_in_basin_file": idx_in_basin_file, # Corresponds to the order of test states for that basin
                            "coordinates": point # The actual 3D coordinates
                        })
                else:
                    print(f"Skipping {fname}: Not {num_dimensions}D data (shape: {basin_data.shape}) after NaN removal.")
            except Exception as e:
                print(f"Skipping {fname} due to loading error: {e}")

    if not original_test_data_with_meta:
        if found_test_files_count == 0:
            print(f"No files matching pattern '{test_file_pattern}' found in '{test_dir}'. Exiting.")
        else:
            print("No valid test data (3D, non-NaN) found among matched files. Exiting.")
        return

    # Reconstruct test_data and basin_map_global_indices from the meta list
    test_data = np.array([item["coordinates"] for item in original_test_data_with_meta])
    basin_map_global_indices = np.array([item["basin_id"] for item in original_test_data_with_meta])
    original_indices_in_basin_files = np.array([item["index_in_basin_file"] for item in original_test_data_with_meta])


    print(f"Total test points loaded from directory (matching pattern): {test_data.shape[0]}")

    # Identify points outside the convex hull
    is_outside = np.full(test_data.shape[0], False)
    if tri is not None:
        is_outside = tri.find_simplex(test_data) == -1

    points_inside = test_data[~is_outside]
    points_outside = test_data[is_outside]
    outside_global_indices_array = np.where(is_outside)[0] # Global indices within the 'test_data' array

    print(f"Test points inside hull: {points_inside.shape[0]}")
    print(f"Test points outside hull: {points_outside.shape[0]}")

    # --- Distance Calculation for points outside the hull and CSV output ---
    cumulative_distances = {basin_id: 0.0 for basin_id in set(basin_map_global_indices)}
    extrapolated_counts = {basin_id: 0 for basin_id in set(basin_map_global_indices)}
    detailed_records = [] # This will store all the details for the detailed CSV

    if points_outside.shape[0] > 0 and hull is not None:
        if distance_calc_method == 'qp':
            print(f"Calculating QP distances for {points_outside.shape[0]} points using multiprocessing...")

            hull_vertices_for_qp = hull.points[hull.vertices]
            tasks = [(p_out, hull_vertices_for_qp) for p_out in points_outside]

            num_processes = cpu_count() - 1 if cpu_count() > 1 else 1
            print(f"Using {num_processes} processes for distance calculation.")

            with Pool(processes=num_processes) as pool:
                distances_results = pool.starmap(calculate_closest_point_on_hull_qp, tasks)

            for i, dist in enumerate(distances_results):
                p_out_global_idx = outside_global_indices_array[i] # Get the global index of this outside point
                
                basin_id = basin_map_global_indices[p_out_global_idx]
                original_idx = original_indices_in_basin_files[p_out_global_idx]
                coords = test_data[p_out_global_idx] # Get the actual coordinates of this point

                if dist != np.inf:
                    cumulative_distances[basin_id] += dist
                    extrapolated_counts[basin_id] += 1
                    
                    # Add detailed record including coordinates
                    detailed_records.append({
                        "basin_id": basin_id,
                        "point_index_in_basin_file": original_idx,
                        "distance_to_hull": dist,
                        "coord_c1": coords[0],
                        "coord_c2": coords[1],
                        "coord_c3": coords[2]
                    })
            print("QP distance calculation complete.")
        elif distance_calc_method == 'delaunay':
            print("Using Delaunay simplex check for outside points. No specific distance calculated for 'delaunay' method in this version.")
            # For Delaunay, we just count them as outside. If you need distances,
            # you'd need to re-integrate specific distance-to-face logic.
            for p_out_global_idx in outside_global_indices_array:
                basin_id = basin_map_global_indices[p_out_global_idx]
                original_idx = original_indices_in_basin_files[p_out_global_idx]
                coords = test_data[p_out_global_idx]

                extrapolated_counts[basin_id] += 1
                # For Delaunay, distance might not be calculated here, so set to NaN or a placeholder
                detailed_records.append({
                    "basin_id": basin_id,
                    "point_index_in_basin_file": original_idx,
                    "distance_to_hull": np.nan, # Or 0 if you consider it an inside point effectively for distance
                    "coord_c1": coords[0],
                    "coord_c2": coords[1],
                    "coord_c3": coords[2]
                })
        else:
            print(f"Warning: Unknown distance calculation method '{distance_calc_method}'. No distances calculated.")
    else:
        print("No points outside the hull or no hull formed; skipping distance calculation and detailed CSV creation.")


    # Prepare data for summary CSV
    summary_data = []
    all_known_basins = sorted(list(set(basin_map_global_indices))) # Use all basins found in test_data
    for basin_id in all_known_basins:
        num_extrapolated = extrapolated_counts.get(basin_id, 0)
        total_distance = cumulative_distances.get(basin_id, 0.0)
        mean_distance = total_distance / num_extrapolated if num_extrapolated > 0 else 0.0
        summary_data.append({
            "basin_id": basin_id,
            "cumulative extrapolation distance": total_distance,
            "number of extrapolation distance per basin": num_extrapolated,
            "mean extrapolation per basin": mean_distance
        })

    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(output_csv_path, index=False)
    print(f"Saved basin extrapolation summary to: {output_csv_path}")

    # Prepare and save detailed CSV if requested
    if detailed_csv_path and detailed_records:
        df_detailed = pd.DataFrame(detailed_records)
        df_detailed.to_csv(detailed_csv_path, index=False)
        print(f"Saved detailed per-point extrapolation distances and coordinates to: {detailed_csv_path}")
    elif detailed_csv_path and not detailed_records:
        print(f"No detailed extrapolation records generated to save for {detailed_csv_path}.")


    # Visualization
    pv.set_plot_theme("document")
    plotter = pv.Plotter(window_size=(1024, 768))

    if hull is not None:
        mesh = pv.PolyData(combined_data_filtered)
        faces_np = np.hstack([[3, *face] for face in hull.simplices])
        mesh.faces = faces_np
        plotter.add_mesh(mesh, color="yellow", opacity=0.15, show_edges=True, label="Convex Hull (Filtered Train+Validation)")

    # Ensure data is not empty before plotting
    if train_data.shape[0] > 0:
        plotter.add_points(train_data, color="blue", point_size=5, render_points_as_spheres=True, label="Original Train Data")
    if validate_data.shape[0] > 0:
        plotter.add_points(validate_data, color="darkblue", point_size=5, render_points_as_spheres=True, label="Original Validation Data")

    if points_inside.shape[0] > 0:
        plotter.add_points(points_inside, color="green", point_size=5, render_points_as_spheres=True, label="Test Data (Inside Hull)")
    if points_outside.shape[0] > 0:
        plotter.add_points(points_outside, color="red", point_size=8, render_points_as_spheres=True, label="Test Data (Outside Hull)")

    plotter.add_legend()
    plotter.show(title=f"Convex Hull with {filter_type.capitalize()} Filter ({filter_percentile}% Retained) and Test States")

if __name__ == "__main__":
    # Define your file paths and the pattern here
    train_file_path = r"C:\Users\deonf\Model_work\runs\my_parquet_run\lstm_parquet_custom_0807_091323\internal_states\train\c_n_reduced_epoch030.pt"
    validate_file_path = r"C:\Users\deonf\Model_work\runs\my_parquet_run\lstm_parquet_custom_0807_091323\internal_states\validation\c_n_reduced_epoch030_validation.pt"
    test_data_directory = r"C:\Users\deonf\Model_work\runs\my_parquet_run\lstm_parquet_custom_0807_091323\internal_states\test"

    # Define output CSV paths
    output_summary_csv = os.path.join(test_data_directory, "extrapolation_summary_per_basin.csv")
    output_detailed_csv = os.path.join(test_data_directory, "extrapolation_detailed_points.csv") # Optional: Set to None if not needed

    # This is the pattern that will be used to filter files in the test_data_directory
    # Ensure this pattern correctly extracts the basin ID.
    # From your provided path: c_n_reduced_epoch006_basin_80000100000015_0.pt
    # The pattern should capture '80000100000015'
    file_pattern_to_match = r'_basin_(\d+)_0\.pt$' # Updated to match "basin_ID.0.pt" or "basin_ID_0.pt"

    # --- Configuration for filtering ---
    chosen_filter_type = 'none' # Options: 'none', 'diffusion', 'mahalanobis'
    chosen_filter_percentile = 99.0     # Relevant for 'mahalanobis' and 'diffusion'

    # --- Configuration for distance calculation for out-of-hull points ---
    chosen_distance_calc_method = 'qp' # Ensure this is 'qp' for QP-based distances

    plot_convexhull_and_test_states(
        train_file_path,
        validate_file_path,
        test_data_directory,
        output_csv_path=output_summary_csv,
        detailed_csv_path=output_detailed_csv,
        test_file_pattern=file_pattern_to_match,
        filter_type=chosen_filter_type,
        filter_percentile=chosen_filter_percentile,
        distance_calc_method=chosen_distance_calc_method
    )
