#4. Implement K means clustering algorithm.

import numpy as np
import matplotlib.pyplot as plt

def initialize_centroids(X, k):

    np.random.seed(42)
    random_indices = np.random.choice(X.shape[0], k, replace=False)
    centroids = X[random_indices]
    return centroids

def assign_clusters(X, centroids):
 
    num_samples = X.shape[0]
    num_clusters = centroids.shape[0]
    labels = np.zeros(num_samples, dtype=int)

    for i in range(num_samples):
        distances = np.linalg.norm(X[i] - centroids, axis=1)
        labels[i] = np.argmin(distances)

    return labels

def update_centroids(X, labels, k):
 
    centroids = np.zeros((k, X.shape[1]))

    for i in range(k):
        points = X[labels == i]
        if len(points) > 0:
            centroids[i] = np.mean(points, axis=0)

    return centroids

def k_means(X, k, max_iters=100, tol=1e-4):

    centroids = initialize_centroids(X, k)

    for _ in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)

        # Check for convergence
        if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < tol):
            break

        centroids = new_centroids

    return centroids, labels

def plot_clusters(X, centroids, labels, k):

    plt.figure(figsize=(8, 6))

   
    for i in range(k):
        plt.scatter(X[labels == i, 0], X[labels == i, 1], label=f'Cluster {i+1}')

    plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='black', marker='X', label='Centroids')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title('K-Means Clustering')
    plt.show()

np.random.seed(42)
X = np.random.rand(300, 2)

k = 3
centroids, labels = k_means(X, k)


plot_clusters(X, centroids, labels, k)
