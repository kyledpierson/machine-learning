import numpy as np
from preprocess import write_predictions


# ---------- HELPER FUNCTIONS ---------- #
def update_weights(w, x, y, r, mu, aggressive):
    predict = np.dot(w, x)

    if (mu and y * predict < mu) or (not mu and np.sign(predict) != y):
        if aggressive:
            r = (mu - y * predict) / (np.dot(x, x) + 1)
        return w + r * y * x, 1

    return w, 0


# ---------- IMPLEMENTATIONS ---------- #
def train(w, data, r, dynamic, mu, aggressive):
    t = 0
    avg_w = w
    new_w = w
    new_r = r

    for e in range(10):
        np.random.shuffle(data)

        for row in data:
            new_w, update = update_weights(new_w, row[:-1], row[-1], new_r, mu, aggressive)
            avg_w = avg_w + new_w
            if dynamic:
                t = t + 1
                new_r = r / (1 + t)

    return new_w, avg_w


def test(w, data):
    correct = 0

    for row in data:
        if np.sign(np.dot(w, row[:-1])) == row[-1]:
            correct = correct + 1

    accuracy = float(correct) / float(len(data))
    return accuracy


def cross_validate(w, params, cv_sets, dynamic, average, aggressive):
    max_acc = 0
    max_param = None

    # Try all hyper-parameters
    for param in params:
        acc = []
        r = param['r']
        mu = param['mu']

        # 5-fold cross-validation
        for i in range(len(cv_sets)):
            # Prepare the data
            test_data = cv_sets[i]
            train_data = np.vstack(cv_sets[:i] + cv_sets[i + 1:])

            # Train and test to get accuracy
            new_w, new_a = train(w, train_data, r, dynamic, mu, aggressive)

            if average:
                new_w = new_a
            acc.append(test(new_w, test_data))

        # Update hyper-parameter if it is better
        acc = np.mean(acc)
        if acc > max_acc:
            max_acc = acc
            max_param = param

    return max_acc, max_param


def train_dev(w, param, train_data, test_data, dynamic, average, aggressive):
    r = param['r']
    mu = param['mu']

    max_w = w
    max_acc = 0
    max_updates = 0

    t = 0
    avg_w = w
    new_w = w
    new_r = r
    updates = 0
    for e in range(20):
        np.random.shuffle(train_data)

        # Update for every row in the training data
        for row in train_data:
            new_w, update = update_weights(new_w, row[:-1], row[-1], new_r, mu, aggressive)
            avg_w = avg_w + new_w
            if dynamic:
                t = t + 1
                new_r = r / (1 + t)

            updates = updates + update

        # Test on the development set
        if not average:
            avg_w = new_w
        acc = test(avg_w, test_data)

        # If this epoch is better, set the max weight vector and accuracy
        if acc > max_acc:
            max_w = avg_w
            max_acc = acc
            max_updates = updates

        print('  ' + str(e + 1) + '  ' + str(acc) + '  ' + str(updates))

    print('Updates to best epoch: ' + str(max_updates))
    return max_w, max_acc


def train_dev_test(w, cv_sets, train_data, dev_data, test_data, num_features, name,
                   preprocessor, params, dynamic=False, average=False, aggressive=False):
    print(name)
    max_acc, max_param = cross_validate(w, params, cv_sets, dynamic, average, aggressive)

    print('Best parameters:       ' + str(max_param))
    print('Mac CV accuracy:       ' + str(max_acc))

    new_data = train_data
    max_w, max_acc = train_dev(w, max_param, new_data, dev_data, dynamic, average, aggressive)
    print('Max dev accuracy:      ' + str(max_acc))

    max_acc = test(max_w, test_data)
    print('Test accuracy:         ' + str(max_acc) + '\n')

    # Write out the predictions
    def predictor(row):
        label = np.sign(np.dot(max_w, row[:-1]))
        if label == -1:
            label = 0
        return label

    write_predictions(name[:3], predictor, num_features, bias=True, preprocessor=preprocessor)
