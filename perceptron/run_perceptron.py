import numpy as np

from preprocess import load_data
from perceptron import train_dev_test

if __name__ == '__main__':
    # Parameters -----------------------------------------------------------------------------#
    params_r = [{'r': 0.01, 'mu': None}, {'r': 0.1, 'mu': None}, {'r': 1.0, 'mu': None}]
    params_mu = [{'r': None, 'mu': 0.01}, {'r': None, 'mu': 0.1}, {'r': None, 'mu': 1.0}]
    params_both = [{'r': 0.01, 'mu': 0.01}, {'r': 0.1, 'mu': 0.01}, {'r': 1.0, 'mu': 0.01},
                   {'r': 0.01, 'mu': 0.1}, {'r': 0.1, 'mu': 0.1}, {'r': 1.0, 'mu': 0.1},
                   {'r': 0.01, 'mu': 1.0}, {'r': 0.1, 'mu': 1.0}, {'r': 1.0, 'mu': 1.0}]

    # Other parameters
    n_features = 16
    w = np.random.uniform(-0.01, 0.01, n_features + 1)

    # Load the training and testing data
    train_data, train_labels = load_data('../data/data-splits/data.train',
                                         n_features=n_features, neg_labels=True, bias=True)
    test_data, test_labels = load_data('../data/data-splits/data.test',
                                       n_features=n_features, neg_labels=True, bias=True)

    # Create cross-validation sets
    cv_data = np.array_split(np.hstack((train_data, train_labels)), 6)
    dev_data = cv_data[-1]
    cv_data = cv_data[:-1]

    # ---------- Simple perceptron ---------- #
    train_dev_test(w, cv_data, train_data, train_labels, dev_data, test_data, test_labels,
                   n_features, 'simple perceptron', params_r)

    # ---------- Dynamic perceptron ---------- #
    train_dev_test(w, cv_data, train_data, train_labels, dev_data, test_data, test_labels,
                   n_features, 'dynamic perceptron', params_r, dynamic=True)

    # ---------- Margin perceptron ---------- #
    train_dev_test(w, cv_data, train_data, train_labels, dev_data, test_data, test_labels,
                   n_features, 'margin perceptron', params_both, dynamic=True)

    # ---------- Average perceptron ---------- #
    train_dev_test(w, cv_data, train_data, train_labels, dev_data, test_data, test_labels,
                   n_features, 'average perceptron', params_r, average=True)

    # ---------- Aggressive perceptron ---------- #
    train_dev_test(w, cv_data, train_data, train_labels, dev_data, test_data, test_labels,
                   n_features, 'aggressive perceptron', params_mu, aggressive=True)
