import numpy as np

from decision_tree import id3, classify
from preprocess import discretize, binarize, bins, equal_bins
from preprocess import load_data, write_predictions

if __name__ == "__main__":
    # Parameters -----------------------------------------------------------------------------#
    preprocessor = lambda data: equal_bins(data, 2)

    # Load the training and testing data
    num_features = 16
    train = load_data('../data-splits/data.train', num_features, preprocessor=preprocessor)
    test = load_data('../data-splits/data.test', num_features, preprocessor=preprocessor)

    # Create cross-validation sets
    cv_sets = np.array_split(train, 5)

    # Train the tree at different depths -----------------------------------------------------#
    max_accuracy = 0
    optimal_depth = 2
    for i in range(2, num_features + 2):
        accuracy = []

        # 5-fold cross-validation
        for j in range(len(cv_sets)):
            test_data = cv_sets[j]
            train_data = np.vstack(cv_sets[:j] + cv_sets[j + 1:])

            # Train on the training set
            tree, depth = id3(train_data, [], i, 0, num_features)

            # Classify the test set
            classified = 0
            for row in test_data:
                if bool(row[-1]) == classify(row, tree):
                    classified = classified + 1

            accuracy.append(float(classified) / float(len(test_data)))

        new_accuracy = np.mean(accuracy)
        if new_accuracy > max_accuracy:
            optimal_depth = i
            max_accuracy = new_accuracy

        print('Depth limit:   ' + str(i))
        print('Actual depth:  ' + str(depth))
        print('Accuracy mean: ' + str(new_accuracy))
        print('Accuracy std:  ' + str(np.std(accuracy)) + '\n')

    # Train with the optimum depth -----------------------------------------------------------#
    tree, depth = id3(train, [], optimal_depth, 0, num_features)

    classified = 0
    for row in test:
        if bool(row[-1]) == classify(row, tree):
            classified = classified + 1
    accuracy = float(classified) / float(len(test)) * 100

    print('Optimal depth: ' + str(optimal_depth))
    print('Test accuracy: ' + str(accuracy) + '%\n')

    # Write predictions ----------------------------------------------------------------------#
    write_predictions('id3', lambda row: classify(row, tree), num_features, preprocessor=preprocessor)
