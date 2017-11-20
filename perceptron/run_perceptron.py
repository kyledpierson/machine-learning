import numpy as np
from perceptron import train_dev_test
from preprocess import discretize, binarize, bins, equal_bins, load_data

if __name__ == '__main__':
    # Parameters -----------------------------------------------------------------------------#
    preprocessor = lambda x: x
    params_r = [{'r': 0.01, 'mu': None}, {'r': 0.1, 'mu': None}, {'r': 1.0, 'mu': None}]
    params_mu = [{'r': None, 'mu': 0.01}, {'r': None, 'mu': 0.1}, {'r': None, 'mu': 1.0}]
    params_both = [{'r': 0.01, 'mu': 0.01}, {'r': 0.1, 'mu': 0.01}, {'r': 1.0, 'mu': 0.01},
                   {'r': 0.01, 'mu': 0.1}, {'r': 0.1, 'mu': 0.1}, {'r': 1.0, 'mu': 0.1},
                   {'r': 0.01, 'mu': 1.0}, {'r': 0.1, 'mu': 1.0}, {'r': 1.0, 'mu': 1.0}]

    # Other parameters
    np.random.seed(1)
    num_features = 16
    w = np.random.uniform(-0.01, 0.01, num_features + 1)

    # Load the training and testing data
    train_data = load_data('../data-splits/data.train', num_features, bias=True, preprocessor=preprocessor)
    test_data = load_data('../data-splits/data.test', num_features, bias=True, preprocessor=preprocessor)

    # Create cross-validation sets
    cv_sets = np.array_split(train_data, 6)
    dev_data = cv_sets[-1]
    cv_sets = cv_sets[:-1]

    # ---------- Simple perceptron ---------- #
    train_dev_test(w, cv_sets, train_data, dev_data, test_data, num_features,
                   'simple perceptron', preprocessor, params_r)

    # ---------- Dynamic perceptron ---------- #
    train_dev_test(w, cv_sets, train_data, dev_data, test_data, num_features,
                   'dynamic perceptron', preprocessor, params_r, dynamic=True)

    # ---------- Margin perceptron ---------- #
    train_dev_test(w, cv_sets, train_data, dev_data, test_data, num_features,
                   'margin perceptron', preprocessor, params_both, dynamic=True)

    # ---------- Average perceptron ---------- #
    train_dev_test(w, cv_sets, train_data, dev_data, test_data, num_features,
                   'average perceptron', preprocessor, params_r, average=True)

    # ---------- Aggressive perceptron ---------- #
    train_dev_test(w, cv_sets, train_data, dev_data, test_data, num_features,
                   'aggressive perceptron', preprocessor, params_mu, aggressive=True)
