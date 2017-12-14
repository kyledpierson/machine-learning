import numpy as np
from csv import writer

from model import cross_validate, train, classify
from decision_tree.id3 import classify as id3_classify
from preprocess import load_data, write_output, write_predictions


def update_weights(x, y, w, param):
    gamma = param['gamma']
    C = param['C']

    if y * np.dot(x, w) <= 1:
        w = (1 - gamma) * w + gamma * C * y * x.reshape((-1, 1))
    else:
        w = (1 - gamma) * w

    return w


def update_params(new_params, old_params, t):
    new_params['gamma'] = old_params['gamma'] / (1 + t)
    return new_params


if __name__ == '__main__':
    gammas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    costs = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    params = [[{'gamma': g, 'C': c} for g in gammas] for c in costs]
    params = np.array(params).flatten()
    # max_param = {'gamma': 1, 'C': 0.001}
    # max_param = {'gamma': 10, 'C': 10}
    max_param = None
    over_trees = True

    if over_trees:
        n_features = 1000
    else:
        n_features = 16
    weights = np.zeros((n_features + 1, 1))

    if over_trees:
        train_data = np.load('../data/new_train.npy')
        test_data = np.load('../data/new_test.npy')
        train_labels = train_data[:, -1].reshape((-1, 1))
        train_labels[train_labels == 0] = -1
        train_data = np.hstack((train_data[:, :-1], np.ones((train_data.shape[0], 1))))
        test_labels = test_data[:, -1].reshape((-1, 1))
        test_data = np.hstack((test_data[:, :-1], np.ones((test_data.shape[0], 1))))
        test_labels[test_labels == 0] = -1

    else:
        train_data, train_labels = load_data(
            '../data/data-splits/data.train', n_features=n_features, neg_labels=True, bias=True)
        test_data, test_labels = load_data(
            '../data/data-splits/data.test', n_features=n_features, neg_labels=True, bias=True)

    cv_data = np.array_split(np.hstack((train_data, train_labels)), 5)

    cv_acc = 0
    if max_param is None:
        cv_acc, max_param = cross_validate(cv_data, weights, update_weights, params, update_params)

    max_weights = train(train_data, train_labels, weights, update_weights, max_param, update_params)
    train_acc = classify(train_data, train_labels, max_weights)

    test_acc = classify(test_data, test_labels, max_weights)


    def predictor(row):
        label = np.sign(np.dot(row, max_weights))
        if label == -1:
            label = 0
        return label


    write_output('SVM', max_param, cv_acc, train_acc, test_acc)

    if over_trees:
        with open('../data/data-splits/data.eval.id') as file:
            eval_id = [int(line) for line in file]
        eval_data, _ = load_data('../data/data-splits/data.eval.anon',
                                 n_features=n_features, neg_labels=True, bias=True)

        trees = np.load('../data/trees.npy')
        new_data = []

        for i in range(eval_data.shape[0]):
            predictions = list(map(lambda tree: id3_classify(eval_data[i], tree), trees))
            predictions.append(1)
            new_data.append(predictions)

        eval_data = np.array(new_data).astype(int)

        with open('../data/sample-solutions/svm_trees_predictions.csv', 'w', newline='') as predictions:
            predictions_csv = writer(predictions)
            predictions_csv.writerow(['Id', 'Prediction'])

            for row_num in range(len(eval_id)):
                row_id = eval_id[row_num]
                row = eval_data[row_num]
                label = predictor(row)

                predictions_csv.writerow([row_id, int(label)])
    else:
        write_predictions('svm', predictor, n_features, neg_labels=True, bias=True)
