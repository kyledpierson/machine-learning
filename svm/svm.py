import numpy as np

from model import cross_validate, train, classify
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
    gammas = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    costs = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    params = [[{'gamma': g, 'C': c} for g in gammas] for c in costs]
    params = np.array(params).flatten()
    max_param = {'gamma': 1, 'C': 0.001}

    n_features = 16
    weights = np.zeros((n_features + 1, 1))

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
    write_predictions('svm', predictor, n_features, neg_labels=True, bias=True)
