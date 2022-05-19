import collections

import pandas as pd
from sklearn.utils import check_array


def probas_to_metric(metric, probas, classes, **labels):
    print(classes)
    for idx, class_ in enumerate(classes):
        for proba in probas:
            metric.labels(class_=class_, **labels).observe(proba[idx])


def flatten(nested_iterable):
    """
    Helper function, returns an iterator of flattened values from an arbitrarily nested iterable
    >>> list(flatten([['test1', 'test2'], ['a', 'b', ['c', 'd']]]))
    ['test1', 'test2', 'a', 'b', 'c', 'd']
    >>> list(flatten(['test1', ['test2']]))
    ['test1', 'test2']
    """
    for el in nested_iterable:
        if isinstance(el, collections.abc.Iterable) and not isinstance(
            el, (str, bytes)
        ):
            yield from flatten(el)
        else:
            yield el


def get_feature_names(X):
    if isinstance(X, pd.DataFrame):
        return X.columns
    else:
        X = check_array(X, force_all_finite=False)
        return list(range(X.shape[1]))
