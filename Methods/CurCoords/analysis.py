from matplotlib.path import Path
import numpy as np
from scipy.spatial import ConvexHull, KDTree


def find_nearest_points(ground_truth_points, cartesian_x, cartesian_y, grid_s, grid_t, method='inverse_distance', epsilon=1e-6):
    lut_x_flat = cartesian_x.flatten()
    lut_y_flat = cartesian_y.flatten()
    lut_points = np.column_stack([lut_x_flat, lut_y_flat])

    s_flat = grid_s.flatten()
    t_flat = grid_t.flatten()
        
    kdtree = KDTree(lut_points)
        
    distances, indices = kdtree.query(ground_truth_points, k=4)
    s_coords, t_coords = weighted_interpolation(
        distances, indices, s_flat, t_flat, method=method, epsilon=epsilon
    )
    
    return s_coords, t_coords

def weighted_interpolation(distances, indices, s_flat, t_flat, method='inverse_square', epsilon=1e-6):
    n_points = distances.shape[0]
        
    s_values = s_flat[indices]
    t_values = t_flat[indices]
    
    exact_matches = np.any(distances < epsilon, axis=1)
    exact_match_indices = np.argmin(distances, axis=1)
    
    if method == 'inverse_distance':
        weights = 1.0 / (distances + epsilon)
    
    elif method == 'inverse_square':
        weights = 1.0 / (distances**2 + epsilon)
    
    elif method == 'gaussian':
        sigma = np.mean(distances, axis=1, keepdims=True)
        weights = np.exp(-distances**2 / (2 * sigma**2 + epsilon))
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    total_weights = np.sum(weights, axis=1, keepdims=True)
    normalized_weights = weights / total_weights
    
    s_interp = np.sum(normalized_weights * s_values, axis=1)
    t_interp = np.sum(normalized_weights * t_values, axis=1)
    
    if np.any(exact_matches):
        s_exact = s_values[np.arange(n_points), exact_match_indices]
        t_exact = t_values[np.arange(n_points), exact_match_indices]
        
        s_interp[exact_matches] = s_exact[exact_matches]
        t_interp[exact_matches] = t_exact[exact_matches]
    
    return s_interp, t_interp

def filter_points_inside_hull(ground_truth_mask, cartesian_x, cartesian_y):
    y_indices, x_indices = np.where(ground_truth_mask > 0)
    ground_truth_points = np.column_stack([x_indices, y_indices])
    lut_x_flat = cartesian_x.flatten()
    lut_y_flat = cartesian_y.flatten()
    lut_points = np.column_stack([lut_x_flat, lut_y_flat])
    
    hull = ConvexHull(lut_points)
    
    hull_path = Path(lut_points[hull.vertices])
    
    inside_mask = hull_path.contains_points(ground_truth_points)
    
    points_inside = ground_truth_points[inside_mask]
    points_outside = ground_truth_points[~inside_mask]

    return points_inside, points_outside, hull, hull_path
