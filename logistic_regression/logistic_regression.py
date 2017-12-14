import numpy as np

from model import cross_validate, train, classify
from preprocess import load_data, write_output, write_predictions


def sigmoid(e):
    if e > 100:
        return 0
    return 1 / (1 + np.exp(e))


def update_weights(x, y, w, params):
    gamma = params['gamma']
    sigma = params['sigma']

    update = sigmoid(y * np.dot(x, w)) * -y * x.reshape((-1, 1)) + 2 * w / sigma
    w = w - gamma * update

    return w


def update_params(new_params, old_params, t):
    new_params['gamma'] = old_params['gamma'] / (1 + t)
    return new_params


if __name__ == '__main__':
    gammas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    sigmas = [0.1, 1, 10, 100, 1000, 10000]
    params = [[{'gamma': g, 'sigma': s} for g in gammas] for s in sigmas]
    params = np.array(params).flatten()
    max_param = {'gamma': 1, 'sigma': 0.1}

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


    write_output('Logistic regression', max_param, cv_acc, train_acc, test_acc)
    write_predictions('logreg', predictor, n_features=n_features, neg_labels=True, bias=True)
