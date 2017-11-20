import numpy as np
from scipy.stats import mode

from decision_tree import id3, classify
from preprocess import discretize, binarize, bins, equal_bins
from preprocess import load_data, write_predictions

if __name__ == "__main__":
    # Parameters -----------------------------------------------------------------------------#
    preprocessor = lambda data: equal_bins(data, 2)
    num_trees = 100
    data_size = 0.75
    feature_size = 0.75
    split_size = 0.5

    # Load the training and testing data
    num_features = 16
    train = load_data('../data-splits/data.train', num_features, preprocessor=preprocessor)
    test = load_data('../data-splits/data.test', num_features, preprocessor=preprocessor)

    # Other parameters
    np.random.seed(1)
    data_indices = range(len(train))
    feature_indices = range(num_features)
    data_size = int(data_size * len(train))
    feature_size = int(feature_size * num_features)
    split_size = int(split_size * num_features)
    max_depth = num_features + 2  # No pruning

    # Train the trees ------------------------------------------------------------------------#
    trees = []
    for i in range(num_trees):
        ri = np.random.choice(data_indices, size=data_size, replace=True)
        rf = np.random.choice(feature_indices, size=num_features - feature_size, replace=False)

        data = train[ri]
        tree, depth = id3(data, rf, max_depth, 0, split_size)
        trees.append(tree)

        if (i + 1) % (num_trees / 10) == 0:
            print(str((i + 1) / num_trees * 100) + '%')

    # Classify the test set
    classified = 0
    for row in test:
        labels = list(map(lambda tree: classify(row, tree), trees))
        label = mode(labels)[0][0]

        if label == bool(row[-1]):
            classified = classified + 1

    accuracy = float(classified) / float(len(test))
    print(accuracy)

    # Write predictions ----------------------------------------------------------------------#
    write_predictions('rf', lambda row: mode(list(map(
        lambda tree: classify(row, tree), trees)))[0][0], num_features, preprocessor=preprocessor)
