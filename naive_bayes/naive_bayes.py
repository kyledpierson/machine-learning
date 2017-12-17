import numpy as np
from csv import writer
from preprocess import load_data, write_output


def cross_validate(cv_data, lams):
    counts = []
    max_acc = 0
    max_lam = 1

    for i in range(len(cv_data)):
        counts.append(count_pos(cv_data[i][:, :-1], cv_data[i][:, -1]))
    counts = np.array(counts)

    for lam in lams:
        acc = []
        for i in range(len(cv_data)):
            cv_test = cv_data[i]
            cv_train = np.vstack(cv_data[:i] + cv_data[i + 1:])

            indices = [j for j in range(len(cv_data)) if j != i]
            cv_counts = np.sum(counts[indices], axis=0)

            like, like_inv = naive_bayes(cv_train.shape[0], cv_counts, lam)
            test_acc = classify(cv_test[:, :-1], cv_test[:, -1], like, like_inv)
            acc.append(test_acc)

        acc = np.mean(acc)
        if acc > max_acc:
            max_acc = acc
            max_lam = lam

    return max_acc, max_lam


def count_pos(data, labels):
    n_features = data.shape[1]
    pos = np.count_nonzero(labels)

    # For each feature f, store:
    # 0 - the number of positive labels
    # 1 - the number of samples with positive f
    # 2 - the number of positive labels for positive f
    counts = np.zeros((n_features, 3))
    counts[:, 0] = pos
    counts[:, 1] = np.count_nonzero(data, axis=0)

    for i in range(n_features):
        feature = data[:, i]
        pos_label = np.count_nonzero(labels[np.nonzero(feature)[0]])
        counts[i, 2] = pos_label

    return counts


def naive_bayes(n_samples, counts, lam):
    pos = counts[0, 0]

    pos_prior = pos / n_samples
    neg_prior = 1 - pos_prior

    pos_like = ((counts[:, 2] + lam) / (pos + 2 * lam)).reshape((-1, 1))
    neg_like = ((counts[:, 1] - counts[:, 2] + lam) / (n_samples - pos + 2 * lam)).reshape((-1, 1))

    like = np.vstack(([neg_prior, pos_prior], np.hstack((neg_like, pos_like))))
    like_inv = np.vstack(([neg_prior, pos_prior], np.hstack((1 - neg_like, 1 - pos_like))))

    return np.log(like), np.log(like_inv)


def classify(data, labels, like, like_inv):
    n_samples, n_features = data.shape

    samples_inv = np.ones((n_samples, n_features)) - data
    samples_inv = np.hstack((np.zeros((n_samples, 1)), samples_inv))
    samples = np.hstack((np.ones((n_samples, 1)), data))

    probs = np.dot(samples, like) + np.dot(samples_inv, like_inv)
    probs = probs.argmax(axis=1).reshape((-1, 1))

    error = np.count_nonzero(probs - labels.reshape((-1, 1)))
    accuracy = 1 - error / float(n_samples)
    return accuracy


if __name__ == '__main__':
    n_features = 16
    lams = [2, 1.5, 1, 0.5]

    train_data, train_labels = load_data(
        '../data/data-splits/data.train', n_features=n_features, pos_labels=True)
    test_data, test_labels = load_data(
        '../data/data-splits/data.train', n_features=n_features, pos_labels=True)
    cv_data = np.array_split(np.hstack((train_data, train_labels)), 5)

    cv_acc, lam = cross_validate(cv_data, lams)

    counts = count_pos(train_data, train_labels)
    like, like_inv = naive_bayes(train_data.shape[0], counts, lam)
    train_acc = classify(train_data, train_labels, like, like_inv)
    test_acc = classify(test_data, test_labels, like, like_inv)

    write_output('Naive Bayes', lam, cv_acc, train_acc, test_acc)

    # Write predictions
    with open('../data/data-splits/data.eval.id') as file:
        eval_id = [int(line) for line in file]
    eval_data, _ = load_data('../data/data-splits/data.eval.anon', n_features=n_features, pos_labels=True)

    n_samples, n_features = eval_data.shape

    samples_inv = np.ones((n_samples, n_features)) - eval_data
    samples_inv = np.hstack((np.zeros((n_samples, 1)), samples_inv))
    samples = np.hstack((np.ones((n_samples, 1)), eval_data))

    probs = np.dot(samples, like) + np.dot(samples_inv, like_inv)
    probs = probs.argmax(axis=1).reshape((-1, 1))

    with open('../data/sample-solutions/nbayes_predictions.csv', 'w', newline='') as predictions:
        predictions_csv = writer(predictions)
        predictions_csv.writerow(['Id', 'Prediction'])

        for row_num in range(len(eval_id)):
            row_id = eval_id[row_num]
            label = probs[row_num]

            predictions_csv.writerow([row_id, int(label)])
