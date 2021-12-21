from skprometheus.preprocessing import OneHotEncoder
import numpy as np
from prometheus_client import REGISTRY
import pandas as pd


def test_OneHotEncoder():
    one_hot = OneHotEncoder()
    X = np.array([
        [1, 3, 4, 6],
        [2, 3, 4, 5],
        [4, 5, 6, 6],
        [0, 0, 0, 0],
        [6, 7, 8, 9]
    ])

    assert 'model_categorical' in [m.name for m in REGISTRY.collect()]
    one_hot.fit(X)
    one_hot.transform(X)

    assert REGISTRY.get_sample_value('model_categorical_total', {'feature': '2', 'category': '4'}) == 2
    assert REGISTRY.get_sample_value('model_categorical_total', {'feature': '3', 'category': '9'}) == 1


def test_OneHotEncoder_pandas():
    one_hot_pd = OneHotEncoder()
    X = np.array([
        [1, 3, 4, 6],
        [2, 3, 4, 5],
        [4, 5, 6, 6],
        [0, 0, 0, 0],
        [6, 7, 8, 9]
    ])

    df = pd.DataFrame.from_records(X, columns=['A', 'B', 'C', 'D'])
    one_hot_pd.fit(df)
    one_hot_pd.transform(df)

    assert REGISTRY.get_sample_value('model_categorical_total', {'feature': 'C', 'category': '4'}) == 2


def test_OneHotEncoder_missing():
    one_hot = OneHotEncoder(handle_unknown='ignore')
    X = np.array([
        [1, 3, 4, 6],
        [2, 3, 4, 5],
        [4, 5, 6, 6],
        [0, 0, 0, 0],
        [6, 7, 8, 9]
    ])

    one_hot.fit(X)

    X_test = np.array([
        [1, 10, 4, 6],
        [2, 23, 4, 77],
        [4, 5, 100, 6],
    ])

    one_hot.transform(X_test)

    assert REGISTRY.get_sample_value('model_categorical_total', {'feature': '1', 'category': 'missing'}) == 2
