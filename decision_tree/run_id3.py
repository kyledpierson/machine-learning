import numpy as np
from pandas import qcut

from id3 import id3, evaluate_tree, classify
from preprocess import load_data, write_output, write_predictions

if __name__ == "__main__":
    preprocessor = lambda data: qcut(data, 2, labels=False)

    n_features = 16
    train_data, train_labels = load_data(
        '../data/data-splits/data.train', n_features=n_features, preprocessor=preprocessor)
    test_data, test_labels = load_data(
        '../data/data-splits/data.test', n_features=n_features, preprocessor=preprocessor)

    cv_data = np.array_split(np.hstack((train_data, train_labels)), 5)

    max_acc = 0
    opt_depth = 0
    for i in range(2, n_features + 2):
        acc = []

        for j in range(len(cv_data)):
            cv_test = cv_data[j]
            cv_train = np.vstack(cv_data[:j] + cv_data[j + 1:])

            tree, depth = id3(cv_train[:, :-1], cv_train[:, -1], max_depth=i)

            cv_acc = evaluate_tree(cv_test[:, :-1], cv_test[:, -1], tree)
            acc.append(cv_acc)

        avg_acc = np.mean(acc)
        if avg_acc > max_acc:
            opt_depth = i
            max_acc = avg_acc

    tree, depth = id3(train_data, train_labels, max_depth=opt_depth)
    train_acc = evaluate_tree(train_data, train_labels, tree)
    test_acc = evaluate_tree(test_data, test_labels, tree)

    write_output('ID3', opt_depth, max_acc, train_acc, test_acc)
    write_predictions('id3', lambda row: classify(row, tree),
                      n_features=n_features, preprocessor=preprocessor)
