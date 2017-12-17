import numpy as np
from pandas import qcut
from scipy.stats import mode

from id3 import id3, evaluate_forest, classify
from preprocess import load_data, write_output, write_predictions

if __name__ == "__main__":
    preprocessor = lambda data: qcut(data, 2, labels=False)
    num_trees = 1000
    data_size = 0.5
    feature_size = 0.75
    split_size = 0.5

    n_features = 16
    train_data, train_labels = load_data('../data/data-splits/data.train',
                                         n_features=n_features, preprocessor=preprocessor)
    test_data, test_labels = load_data('../data/data-splits/data.test',
                                       n_features=n_features, preprocessor=preprocessor)

    data_indices = range(train_data.shape[0])
    feature_indices = range(n_features)
    data_size = int(data_size * train_data.shape[0])
    feature_size = int(feature_size * n_features)
    split_size = int(split_size * n_features)
    max_depth = n_features + 2  # No pruning

    trees = []
    for i in range(num_trees):
        ri = np.random.choice(data_indices, size=data_size, replace=True)
        rf = np.random.choice(feature_indices, size=n_features - feature_size, replace=False)

        tree, depth = id3(train_data[ri], train_labels[ri],
                          used_features=rf, max_depth=max_depth, split_size=split_size)
        trees.append(tree)

        if (i + 1) % (num_trees / 10) == 0:
            print(str((i + 1) / num_trees * 100) + '%')

    np.save('../data/trees', trees)
    train_acc = evaluate_forest(train_data, train_labels, trees, '../data/new_train')
    test_acc = evaluate_forest(test_data, test_labels, trees, '../data/new_test')


    def predictor(row):
        return mode(list(map(lambda tree: classify(row, tree), trees)))[0][0]


    write_output('Random forest', None, None, train_acc, test_acc)
    write_predictions('rf', predictor,
                      n_features=n_features, preprocessor=preprocessor)
