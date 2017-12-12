import numpy as np
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from preprocess import load_data

if __name__ == "__main__":
    n_features = 16
    n_components = 3

    data, labels = load_data('data/data-splits/data.train', n_features=n_features)
    pca = PCA(n_components=n_components, whiten=True)

    x = pca.fit_transform(data)
    labels = np.array(['b' if label else 'r' for label in labels])

    if n_components == 2:
        plt.scatter(x[:, 0], x[:, 1], c=labels, alpha=0.5)
        plt.show()

        plt.scatter(x[labels == 'b', 0], x[labels == 'b', 1], c=labels[labels == 'b'])
        plt.show()

        plt.scatter(x[labels == 'r', 0], x[labels == 'r', 1], c=labels[labels == 'r'])
        plt.show()
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=labels)
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x[labels == 'b', 0], x[labels == 'b', 1], x[labels == 'b', 2], c=labels[labels == 'b'])
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x[labels == 'r', 0], x[labels == 'r', 1], x[labels == 'r', 2], c=labels[labels == 'r'])
        plt.show()
