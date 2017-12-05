import numpy as np
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from preprocess import load_data

if __name__ == "__main__":
    num_features = 16
    num_components = 3

    data, labels = load_data('data/data-splits/data.train', num_features)
    pca = PCA(n_components=num_components, whiten=True)

    X = pca.fit_transform(data)
    labels = np.array(['b' if label else 'r' for label in labels])

    if num_components == 2:
        plt.scatter(X[:, 0], X[:, 1], c=labels, alpha=0.5)
        plt.show()

        plt.scatter(X[labels == 'b', 0], X[labels == 'b', 1], c=labels[labels == 'b'])
        plt.show()

        plt.scatter(X[labels == 'r', 0], X[labels == 'r', 1], c=labels[labels == 'r'])
        plt.show()
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels)
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[labels == 'b', 0], X[labels == 'b', 1], X[labels == 'b', 2], c=labels[labels == 'b'])
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[labels == 'r', 0], X[labels == 'r', 1], X[labels == 'r', 2], c=labels[labels == 'r'])
        plt.show()
