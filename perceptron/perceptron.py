import numpy as np
from preprocess import write_output, write_predictions


def update_weights(w, x, y, r, mu, aggressive):
    predict = np.dot(x, w)

    if (mu and y * predict < mu) or (not mu and np.sign(predict) != y):
        if aggressive:
            r = (mu - y * predict) / (np.dot(x, x) + 1)
        return w + r * y * x, 1

    return w, 0


def train(w, data, labels, r, dynamic, mu, aggressive):
    t = 0
    avg_w = w
    new_w = w
    new_r = r

    for e in range(10):
        # np.random.shuffle(data)

        for i in range(data.shape[0]):
            new_w, update = update_weights(new_w, data[i], labels[i],
                                           new_r, mu, aggressive)
            avg_w = avg_w + new_w
            if dynamic:
                t = t + 1
                new_r = r / (1 + t)

    return new_w, avg_w


def test(w, data, labels):
    correct = 0
    n_samples = data.shape[0]

    for i in range(n_samples):
        if np.sign(np.dot(data[i], w)) == labels[i]:
            correct = correct + 1

    accuracy = float(correct) / float(n_samples)
    return accuracy


def cross_validate(w, params, cv_data, dynamic, average, aggressive):
    max_acc = 0
    max_param = None

    for param in params:
        acc = []
        r = param['r']
        mu = param['mu']

        for i in range(len(cv_data)):
            test_data = cv_data[i]
            train_data = np.vstack(cv_data[:i] + cv_data[i + 1:])

            new_w, new_a = train(w, train_data[:, :-1], train_data[:, -1], r, dynamic, mu, aggressive)

            if average:
                new_w = new_a

            test_acc = test(new_w, test_data[:, :-1], test_data[:, -1])
            acc.append(test_acc)

        acc = np.mean(acc)
        if acc > max_acc:
            max_acc = acc
            max_param = param

    return max_acc, max_param


def train_dev(w, param, train_data, train_labels, test_data, test_labels,
              dynamic, average, aggressive):
    r = param['r']
    mu = param['mu']

    max_w = w
    max_acc = 0

    t = 0
    avg_w = w
    new_w = w
    new_r = r
    updates = 0
    for e in range(20):
        # np.random.shuffle(train_data)

        for i in range(train_data.shape[0]):
            new_w, update = update_weights(new_w, train_data[i], train_labels[i],
                                           new_r, mu, aggressive)
            avg_w = avg_w + new_w
            if dynamic:
                t = t + 1
                new_r = r / (1 + t)

            updates = updates + update

        if not average:
            avg_w = new_w
        acc = test(avg_w, test_data, test_labels)

        if acc > max_acc:
            max_w = avg_w
            max_acc = acc

    return max_w, max_acc


def train_dev_test(
        w, cv_data, train_data, train_labels, dev_data, test_data, test_labels,
        n_features, name, params, dynamic=False, average=False, aggressive=False):
    cv_acc, max_param = cross_validate(w, params, cv_data, dynamic, average, aggressive)

    max_w, train_acc = train_dev(
        w, max_param, train_data, train_labels, dev_data[:, :-1], dev_data[:, -1],
        dynamic, average, aggressive)

    test_acc = test(max_w, test_data, test_labels)

    def predictor(row):
        label = np.sign(np.dot(row, max_w))
        if label == -1:
            label = 0
        return label

    write_output(name, max_param, cv_acc, train_acc, test_acc)
    write_predictions(name[:3], predictor, n_features=n_features, neg_labels=True, bias=True)
