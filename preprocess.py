import numpy as np
from csv import writer
from sklearn.datasets import load_svmlight_file


def load_data(file, n_features=None, neg_labels=False, pos_labels=False, preprocessor=lambda x: x, bias=False):
    # Load data
    data = load_svmlight_file(file, n_features=n_features)

    # Use 0 or -1 for false labels
    labels = data[1]
    if neg_labels:
        labels[labels == 0] = -1
    if pos_labels:
        labels[labels == -1] = 0
    labels = np.reshape(labels, (-1, 1))

    # Threshold or scale by columns
    data = data[0].toarray()
    for i in range(data.shape[1]):
        data[:, i] = preprocessor(data[:, i])

    # Append bias and labels
    if bias:
        data = np.hstack((data, np.ones((len(labels), 1))))

    return data, labels


def write_predictions(name, predictor, n_features=None, neg_labels=False, pos_labels=False,
                      preprocessor=lambda x: x, bias=False):
    with open('../data/data-splits/data.eval.id') as file:
        eval_id = [int(line) for line in file]
    eval_data, _ = load_data('../data/data-splits/data.eval.anon',
                             n_features=n_features, neg_labels=neg_labels, pos_labels=pos_labels,
                             preprocessor=preprocessor, bias=bias)

    with open('../data/sample-solutions/' + name + '_predictions.csv', 'w', newline='') as predictions:
        predictions_csv = writer(predictions)
        predictions_csv.writerow(['Id', 'Prediction'])

        for row_num in range(len(eval_id)):
            row_id = eval_id[row_num]
            row = eval_data[row_num]
            label = predictor(row)

            predictions_csv.writerow([row_id, int(label)])
