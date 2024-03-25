import random

def calculate_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)) ** 0.5

def assign_points_to_clusters(centroids, data):
    """Assign each point to the nearest centroid."""
    clusters = [[] for _ in centroids]
    for point in data:
        distances = [calculate_distance(point, centroid) for centroid in centroids]
        nearest_centroid_index = distances.index(min(distances))
        clusters[nearest_centroid_index].append(point)
    return clusters

def update_centroids(clusters):
    """Recalculate centroids as the mean of all points in a cluster."""
    new_centroids = []
    for cluster in clusters:
        new_centroid = [sum(dimensions) / len(cluster) for dimensions in zip(*cluster)]
        new_centroids.append(new_centroid)
    return new_centroids

def k_means(data, k, max_iterations=100):
    """The K-Means clustering algorithm."""
    # Initialize centroids randomly from the dataset
    centroids = random.sample(data, k)
    for iteration in range(max_iterations):
        clusters = assign_points_to_clusters(centroids, data)
        new_centroids = update_centroids(clusters)
        if centroids == new_centroids:
            break  # Centroids have stabilized
        centroids = new_centroids
    return centroids, clusters
