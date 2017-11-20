import numpy as np
from csv import writer
from pandas import cut, qcut
from sklearn.datasets import load_svmlight_file


def discretize(data):
    return data.astype(int)


def binarize(data):
    for i in range(len(data[0])):
        data[:, i] = data[:, i] > np.mean(data[:, i])
    return data


def bins(data, nbins):
    for i in range(len(data[0])):
        data[:, i] = cut(data[:, i], nbins, labels=False)
    return data


def equal_bins(data, nbins):
    for i in range(len(data[0])):
        data[:, i] = qcut(data[:, i], nbins, labels=False)
    return data


def load_data(file, num_features, bias=False, preprocessor=lambda x: x):
    # Load data
    data = load_svmlight_file(file, n_features=num_features)
    labels = data[1]
    data = data[0].toarray()

    # Threshold or scale
    data = preprocessor(data)

    # Append bias and labels
    if bias:
        data = np.hstack((data, np.ones((len(labels), 1))))
    data = np.hstack((data, np.reshape(labels, (-1, 1))))  # .astype(int)

    return data


def write_predictions(name, predictor, num_features, bias=False, preprocessor=lambda x: x):
    with open('../data-splits/data.eval.id') as file:
        eval_id = [int(line) for line in file]
    eval_set = load_data('../data-splits/data.eval.anon', num_features, bias, preprocessor)

    with open('../sample-solutions/' + name + '_predictions.csv', 'w', newline='') as predictions:
        predictions_csv = writer(predictions)
        predictions_csv.writerow(['Id', 'Prediction'])

        for row_num in range(len(eval_id)):
            row_id = eval_id[row_num]
            row = eval_set[row_num]
            label = predictor(row)

            predictions_csv.writerow([row_id, int(label)])
