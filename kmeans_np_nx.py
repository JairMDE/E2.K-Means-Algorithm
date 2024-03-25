import numpy as np
import numexpr as ne

def calculate_distance_numpy(centroid, data):
    """Calculate the Euclidean distance using numpy for efficiency."""
    diff_squared = (data - centroid) ** 2
    # Using numexpr to sum the squared differences
    distances_squared = ne.evaluate("sum(diff_squared, axis=1)")
    return np.sqrt(distances_squared)

def assign_points_to_clusters_numpy(centroids, data):
    """Assign each point to the nearest centroid using numpy."""
    distances = np.stack([calculate_distance_numpy(centroid, data) for centroid in centroids], axis=1)
    return np.argmin(distances, axis=1)

def update_centroids_numpy(clusters, data, k):
    """Recalculate centroids using numpy."""
    new_centroids = np.array([data[clusters == i].mean(axis=0) for i in range(k)])
    return new_centroids

def k_means_numpy(data, k, max_iterations=100):
    """The K-Means clustering algorithm optimized with Numpy and Numexpr."""
    data = np.array(data)
    # Initialize centroids randomly from the dataset
    centroids = data[np.random.choice(data.shape[0], k, replace=False), :]
    for iteration in range(max_iterations):
        clusters = assign_points_to_clusters_numpy(centroids, data)
        new_centroids = update_centroids_numpy(clusters, data, k)
        if np.all(centroids == new_centroids):
            break  # Centroids have stabilized
        centroids = new_centroids
    return centroids, clusters
