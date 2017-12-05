import numpy as np
from pandas import qcut

from id3 import id3, evaluate_tree, classify
from preprocess import load_data, write_predictions

if __name__ == "__main__":
    # Parameters -----------------------------------------------------------------------------#
    preprocessor = lambda data: qcut(data, 2, labels=False)

    # Load the training and testing data
    n_features = 16
    train_data, train_labels = load_data(
        '../data/data-splits/data.train', n_features=n_features, preprocessor=preprocessor)
    test_data, test_labels = load_data(
        '../data/data-splits/data.test', n_features=n_features, preprocessor=preprocessor)

    # Create cross-validation sets
    cv_data = np.array_split(np.hstack((train_data, train_labels)), 5)

    # Train the tree at different depths -----------------------------------------------------#
    max_acc = 0
    opt_depth = 0
    for i in range(2, n_features + 2):
        acc = []

        # 5-fold cross-validation
        for j in range(len(cv_data)):
            cv_test = cv_data[j]
            cv_train = np.vstack(cv_data[:j] + cv_data[j + 1:])

            # Train on the training set
            tree, depth = id3(cv_train[:, :-1], cv_train[:, -1], max_depth=i)

            # Test on the test set
            cv_acc = evaluate_tree(cv_test[:, :-1], cv_test[:, -1], tree)
            acc.append(cv_acc)

        avg_acc = np.mean(acc)
        if avg_acc > max_acc:
            opt_depth = i
            max_acc = avg_acc

        print('Depth limit:   ' + str(i))
        print('Actual depth:  ' + str(depth))
        print('Accuracy mean: ' + str(avg_acc))
        print('Accuracy std:  ' + str(np.std(acc)) + '\n')

    # Train with the optimum depth -----------------------------------------------------------#
    tree, depth = id3(train_data, train_labels, max_depth=opt_depth)
    test_acc = evaluate_tree(train_data, train_labels, tree)

    print('Optimal depth: ' + str(opt_depth))
    print('Test accuracy: ' + str(test_acc) + '%\n')

    # Write predictions ----------------------------------------------------------------------#
    write_predictions('id3', lambda row: classify(row, tree),
                      n_features=n_features, preprocessor=preprocessor)
