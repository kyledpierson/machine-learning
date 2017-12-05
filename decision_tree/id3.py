import numpy as np
from scipy.stats import mode


def id3(data, labels, used_features=None, max_depth=np.inf, depth=0, split_size=np.inf):
    if used_features is None:
        used_features = []

    n_features = data.shape[1]
    h, pos, neg = entropy(labels)

    if depth == max_depth - 1 or len(used_features) == n_features:
        return pos > neg, 1
    elif pos is 0:
        return False, 1
    elif neg is 0:
        return True, 1
    else:
        max_i = None
        max_gain = -1
        max_values = None

        usable = [i for i in range(n_features) if i not in used_features]
        if split_size < len(usable):
            usable = np.random.choice(usable, size=split_size, replace=False)

        for i in usable:
            features = data[:, i]
            gain, values = info_gain(features, h, labels)
            if gain > max_gain:
                max_i = i
                max_gain = gain
                max_values = values

        new_depth = 1
        tree = {max_i: {}}
        for key in max_values:
            tree[max_i][str(key)], d = id3(
                data[max_values[key]], labels[max_values[key]],
                np.append(used_features, [max_i]), max_depth, depth + 1, split_size)
            if d > new_depth:
                new_depth = d

        return tree, new_depth + 1


def evaluate_tree(data, labels, tree):
    classified = 0
    n_samples = data.shape[0]

    for i in range(n_samples):
        if classify(data[i], tree) == bool(labels[i]):
            classified = classified + 1

    accuracy = float(classified) / float(n_samples)
    return accuracy


def evaluate_forest(data, labels, trees):
    classified = 0
    n_samples = data.shape[0]

    for i in range(n_samples):
        predictions = list(map(lambda tree: classify(data[i], tree), trees))
        prediction = mode(predictions)[0][0]

        if prediction == bool(labels[i]):
            classified = classified + 1

    accuracy = float(classified) / float(n_samples)
    return accuracy


# ==================== HELPER FUNCTIONS ==================== #
def entropy(labels):
    # Get the positive and negative labels
    pos = np.count_nonzero(labels)
    neg = len(labels) - pos

    # Compute entropy
    left = 0
    right = 0
    if pos > 0:
        left = (pos / len(labels)) * np.log2(pos / len(labels))
    if neg > 0:
        right = (neg / len(labels)) * np.log2(neg / len(labels))

    h = -left - right, pos, neg
    return h


def info_gain(features, h, labels):
    # Split into all possible values for the feature
    uniques, indices, counts = np.unique(features, return_inverse=True, return_counts=True)

    # Compute information gain
    summed_h = 0
    split_indices = {}
    for i in range(len(uniques)):
        weight = counts[i] / len(labels)
        new_labels = labels[indices == i]
        summed_h = summed_h + weight * entropy(new_labels)[0]

        split_indices[uniques[i]] = indices == i

    g = h - summed_h
    return g, split_indices


def classify(sample, tree):
    if type(tree) is bool:
        return tree

    feature = next(iter(tree))
    value = str(sample[feature])

    if value not in tree[feature]:
        return np.random.uniform() > 0.5

    node = tree[feature][value]
    result = node if type(node) is bool else classify(sample, tree[feature][value])
    return result
