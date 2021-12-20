from skprometheus.preprocessing import OneHotEncoder
import numpy as np
from prometheus_client import REGISTRY
from tests.utils import unregister_collectors


def test_OneHotEncoder():
    one_hot = OneHotEncoder()
    X = np.array([
        [1, 3, 4, 6],
        [2, 3, 4, 5],
        [4, 5, 6, 6],
        [0, 0, 0, 0],
        [6, 7, 8, 9]
    ])

    assert 'model_categorical_count' in [m.name for m in REGISTRY.collect()]
    one_hot.fit(X)
    one_hot.transform(X)

    assert REGISTRY.get_sample_value('model_categorical_count_total', {'feature': '2', 'category': '4'}) == 2
    assert REGISTRY.get_sample_value('model_categorical_count_total', {'feature': '3', 'category': '9'}) == 1

