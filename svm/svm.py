import numpy as np
from preprocess import write_predictions


# ---------- HELPER FUNCTIONS ---------- #
def update_weights(w, x, y, gamma, C):
    if y * np.dot(w, x) <= 1:
        w = (1 - gamma) * w + gamma * C * y * x
    else:
        w = (1 - gamma) * w

    return w


# ---------- IMPLEMENTATIONS ---------- #
def train(w, data, gamma, C):
    t = 0
    new_w = w
    new_gamma = gamma

    for e in range(10):
        # np.random.shuffle(data)

        for row in data:
            new_w = update_weights(new_w, row[:-1], row[-1], new_gamma, C)
            t += 1
            new_gamma = gamma / (1 + gamma * t / C)

    return new_w


def test(w, data):
    correct = 0

    for row in data:
        if np.sign(np.dot(w, row[:-1])) == row[-1]:
            correct = correct + 1

    accuracy = float(correct) / float(data.shape[0])
    return accuracy


def cross_validate(w, params, cv_sets):
    max_acc = 0
    max_param = None

    # Try all hyper-parameters
    for param in params:
        acc = []
        gamma = param['gamma']
        C = param['C']

        # 5-fold cross-validation
        for i in range(len(cv_sets)):
            # Prepare the data
            test_data = cv_sets[i]
            train_data = np.vstack(cv_sets[:i] + cv_sets[i + 1:])

            # Train and test to get accuracy
            new_w = train(w, train_data, gamma, C)
            test_acc = test(new_w, test_data)
            acc.append(test_acc)

        # Update hyper-parameter if it is better
        acc = np.mean(acc)
        if acc > max_acc:
            max_acc = acc
            max_param = param

    return max_acc, max_param


def train_dev(w, param, train_data, test_data):
    gamma = param['gamma']
    C = param['C']
    new_w = w
    new_gamma = gamma

    t = 0
    max_w = w
    max_acc = 0
    for e in range(20):
        # np.random.shuffle(train_data)

        # Update for every row in the training data
        for row in train_data:
            new_w = update_weights(new_w, row[:-1], row[-1], new_gamma, C)
            t += 1
            new_gamma = gamma / (1 + gamma * t / C)

        acc = test(new_w, test_data)

        # If this epoch is better, set the max weight vector and accuracy
        if acc > max_acc:
            max_w = new_w
            max_acc = acc

    return max_w, max_acc


def train_dev_test(w, cv_sets, train_data, dev_data, test_data,
                   num_features, preprocessor, params):
    max_acc, max_param = cross_validate(w, params, cv_sets)

    print('Best parameters:  ' + str(max_param))
    print('Mac CV accuracy:  ' + str(max_acc))

    new_data = train_data
    max_w, max_acc = train_dev(w, max_param, new_data, dev_data)
    print('Max dev accuracy: ' + str(max_acc))

    max_acc = test(max_w, test_data)
    print('Test accuracy:    ' + str(max_acc) + '\n')

    # Write out the predictions
    def predictor(row):
        label = np.sign(np.dot(max_w, row[:-1]))
        if label == -1:
            label = 0
        return label

    write_predictions('svm', predictor, num_features, bias=True, preprocessor=preprocessor, neg_labels=True)
