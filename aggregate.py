import numpy as np
from csv import writer
from scipy.stats import mode

learners = [
    'agg',
    'ave',
    'dyn',
    'id3',
    'logreg',
    'mar',
    'nbayes',
    'rf',
    'sim',
    'svm'
]

all_labels = None
for learner in learners:
    labels = []
    with open('data/sample-solutions/' + learner + '_predictions.csv') as file:
        for line in file:
            id, label = line.split(',')
            if not label.startswith('Prediction'):
                labels.append(int(label))

    if all_labels is None:
        all_labels = labels
    else:
        all_labels = np.vstack((all_labels, labels))

all_labels = mode(all_labels)[0].flatten()

with open('data/data-splits/data.eval.id') as file:
    eval_id = [int(line) for line in file]

with open('data/sample-solutions/bagged_predictions.csv', 'w', newline='') as predictions:
    predictions_csv = writer(predictions)
    predictions_csv.writerow(['Id', 'Prediction'])

    for row_num in range(len(eval_id)):
        row_id = eval_id[row_num]
        label = all_labels[row_num]
        predictions_csv.writerow([row_id, label])
