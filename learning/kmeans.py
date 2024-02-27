import numpy as np
import matplotlib.pyplot as plt


# Calculate the distance between two points
def distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


# K-means algorithm
def kmeans(data, k, max_iters=100):
    # Randomly initialize k centroids
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    clusters = np.zeros(len(data))

    for _ in range(max_iters):
        # Assign each sample to the nearest centroid
        for i, sample in enumerate(data):
            distances = [distance(sample, centroid) for centroid in centroids]
            cluster = np.argmin(distances)
            clusters[i] = cluster

        # Update centroids
        for i in range(k):
            cluster_points = [data[j] for j in range(len(data)) if clusters[j] == i]
            centroids[i] = np.mean(cluster_points, axis=0)

    return clusters, centroids


# Generate test data
np.random.seed(42)
data = np.random.randn(100, 2)

# Perform K-means clustering
k = 3
clusters, centroids = kmeans(data, k)

# Visualize the results
for i in range(k):
    cluster_points = np.array([data[j] for j in range(len(data)) if clusters[j] == i])
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1])

plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='red', s=100)
plt.title('K-means Clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
