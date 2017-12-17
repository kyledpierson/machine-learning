import copy
import numpy as np


def cross_validate(cv_data, weights, update_weights, params, update_params):
    max_acc = 0
    max_param = None

    for param in params:
        print(param)
        acc = []

        for i in range(len(cv_data)):
            cv_test = cv_data[i]
            cv_train = np.vstack(cv_data[:i] + cv_data[i + 1:])

            new_weights = train(cv_train[:, :-1], cv_train[:, -1],
                                weights, update_weights, param, update_params)
            test_acc = classify(cv_test[:, :-1], cv_test[:, -1], new_weights)
            acc.append(test_acc)

        acc = np.mean(acc)
        if acc > max_acc:
            max_acc = acc
            max_param = param

    return max_acc, max_param


def train(data, labels, weights, update_weights, param, update_params, epochs=10):
    t = 0
    new_weights = weights
    new_param = copy.copy(param)

    combined = np.hstack((data, labels))
    for e in range(epochs):
        np.random.shuffle(combined)
        data = combined[:, :-1]
        labels = combined[:, -1]

        for i in range(data.shape[0]):
            new_weights = update_weights(data[i], labels[i], new_weights, new_param)
            t += 1
            new_param = update_params(new_param, param, t)

    return new_weights


def classify(data, labels, weights):
    predict = np.sign(np.dot(data, weights))
    diff = np.count_nonzero(predict - labels.reshape((-1, 1)))

    accuracy = 1 - diff / float(data.shape[0])
    return accuracy
