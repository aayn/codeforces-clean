from collections import Counter
from functools import reduce


def sample_cleaner(filters, raw_data, labels):
    "Applies a list of filters to raw_data to remove undesirable samples."
    for filter in filters:
        raw_data, labels = filter(raw_data, labels)
    return raw_data, labels


def small_class(raw_data, labels, threshold=20):
    """Removes samples and classes for classes that have less than
    `threshold` number of samples."""
    counts = Counter(labels)
    data, n_labels = [], []
    for i, l in enumerate(labels):
        if counts[l] >= threshold:
            data.append(raw_data[i])
            n_labels.append(l)
    return data, n_labels


def small_class_multi(raw_data, labels, threshold=20):
    flat_labels = reduce(lambda acc, x: acc + list(x), labels, [])
    counts = Counter(flat_labels)
    labels = [[l for l in label if counts[l] >= 100] for label in labels]
    data, n_labels = [], []
    for d, l in zip(raw_data, labels):
        if l:
            data.append(d)
            n_labels.append(l)
    return data, n_labels    
    