import numpy as np
from hw5_code import train_dev_test
from preprocess import discretize, binarize, bins, equal_bins, load_data

if __name__ == '__main__':
    # Parameters -----------------------------------------------------------------------------#
    preprocessor = lambda x: equal_bins(x, 2)
    steps = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    params = [[{'gamma': a, 'C': b} for a in steps] for b in steps]
    params = np.array(params).flatten()

    # Other parameters
    num_features = 16
    w = np.random.uniform(-0.01, 0.01, num_features + 1)

    # Load the training and testing data
    train_data = load_data('../data/data-splits/data.train', num_features,
                           bias=True, preprocessor=preprocessor, neg_labels=True)
    test_data = load_data('../data/data-splits/data.test', num_features,
                          bias=True, preprocessor=preprocessor, neg_labels=True)

    # Create cross-validation sets
    cv_sets = np.array_split(train_data, 6)
    dev_data = cv_sets[-1]
    cv_sets = cv_sets[:-1]

    # ---------- Simple perceptron ---------- #
    train_dev_test(w, cv_sets, train_data, dev_data, test_data,
                   num_features, preprocessor, params)
