import numpy as np
from pandas import qcut
from scipy.stats import mode

from id3 import id3, evaluate_forest, classify
from preprocess import load_data, write_predictions

if __name__ == "__main__":
    # Parameters -----------------------------------------------------------------------------#
    preprocessor = lambda data: qcut(data, 2, labels=False)
    num_trees = 100
    data_size = 0.75
    feature_size = 0.75
    split_size = 0.5

    # Load the training and testing data
    n_features = 16
    train_data, train_labels = load_data('../data/data-splits/data.train',
                                         n_features=n_features, preprocessor=preprocessor)
    test_data, test_labels = load_data('../data/data-splits/data.test',
                                       n_features=n_features, preprocessor=preprocessor)

    # Other parameters
    data_indices = range(train_data.shape[0])
    feature_indices = range(n_features)
    data_size = int(data_size * train_data.shape[0])
    feature_size = int(feature_size * n_features)
    split_size = int(split_size * n_features)
    max_depth = n_features + 2  # No pruning

    # Train the trees ------------------------------------------------------------------------#
    trees = []
    for i in range(num_trees):
        ri = np.random.choice(data_indices, size=data_size, replace=True)
        rf = np.random.choice(feature_indices, size=n_features - feature_size, replace=False)

        tree, depth = id3(train_data[ri], train_labels[ri],
                          max_depth=max_depth, split_size=split_size)
        trees.append(tree)

        if (i + 1) % (num_trees / 10) == 0:
            print(str((i + 1) / num_trees * 100) + '%')

    # Classify the test set
    test_acc = evaluate_forest(test_data, test_labels, trees)
    print(test_acc)


    # Write predictions ----------------------------------------------------------------------#
    def predictor(row):
        return mode(list(map(lambda tree: classify(row, tree), trees)))[0][0]


    write_predictions('rf', predictor,
                      n_features=n_features, preprocessor=preprocessor)
