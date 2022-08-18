import pytest

from skprometheus.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
import numpy as np
from prometheus_client import REGISTRY
import pandas as pd

from skprometheus.utils import flatten
from tests.conftest import general_checks, select_tests, transformer_checks


@pytest.mark.parametrize(
    "test_fn",
    select_tests(
        flatten([general_checks, transformer_checks]),
        exclude=["check_fit2d_predict1d"],
    )
)
def test_standard_checks(test_fn):
    trf = OneHotEncoder()
    test_fn(OneHotEncoder.__name__, trf)


def test_OneHotEncoder():
    one_hot = OneHotEncoder()
    X = np.array([
        [1, 3, 4, 6],
        [2, 3, 4, 5],
        [4, 5, 6, 6],
        [0, 0, 0, 0],
        [6, 7, 8, 9]
    ])

    one_hot.fit(X)
    one_hot.transform(X)
    assert 'skprom_model_categorical' in [m.name for m in REGISTRY.collect()]

    assert REGISTRY.get_sample_value('skprom_model_categorical_total', {'feature': '2', 'category': '4'}) == 2
    assert REGISTRY.get_sample_value('skprom_model_categorical_total', {'feature': '3', 'category': '9'}) == 1


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

    assert REGISTRY.get_sample_value('skprom_model_categorical_total', {'feature': 'C', 'category': '4'}) == 2


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

    assert REGISTRY.get_sample_value('skprom_model_categorical_total', {'feature': '1', 'category': 'missing'}) == 2


def test_OrdinalEncoder():
    ordinal = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
    x = np.array([['ndhbfg', 'akshf'],
                ['abhvg', 'likrghfb'],
                ['ndhbfg', 'lsbvjl']], dtype=np.str_)

    ordinal.fit(x)
    ordinal.transform(x)

    assert 'skprom_model_categorical' in [m.name for m in REGISTRY.collect()]

    assert REGISTRY.get_sample_value('skprom_model_categorical_total', {'feature': '0', 'category': 'ndhbfg'}) == 2
    assert REGISTRY.get_sample_value('skprom_model_categorical_total', {'feature': '1', 'category': 'likrghfb'}) == 1


def test_OrdinalEncoder_pandas():
    ordinal_pd = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
    x = np.array([['ndhbfg', 'akshf'],
                ['abhvg', 'likrghfb'],
                ['ndhbfg', 'lsbvjl']], dtype=np.str_)

    df = pd.DataFrame.from_records(x, columns=['X', 'Y'])
    ordinal_pd.fit(df)
    ordinal_pd.transform(df)

    assert REGISTRY.get_sample_value('skprom_model_categorical_total', {'feature': 'X', 'category': 'ndhbfg'}) == 2


def test_OrdinalEncoder_missing():
    ordinal = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
    x = np.array([['ndhbfg', 'akshf'],
                ['abhvg', 'likrghfb'],
                ['ndhbfg', 'lsbvjl']], dtype=np.str_)

    ordinal.fit(x)

    x_test = np.array([['aaaa', 'bbbvbg'],
                    ['abhvg', 'likrghfb'],
                    ['ndhbfg', 'lsbvjl']], dtype=np.str_)

    ordinal.transform(x_test)

    assert REGISTRY.get_sample_value('skprom_model_categorical_total', {'feature': '0', 'category': 'missing'}) == 1
    assert REGISTRY.get_sample_value('skprom_model_categorical_total', {'feature': '1', 'category': 'missing'}) == 1

def test_LabelEncoder():
    label_enc = LabelEncoder()
    Y = np.array(['A', 'B', 'C', 'B', 'E', 'D', 'E', 'E'], dtype = np.str_). reshape((-1, 1))

    label_enc.fit(Y)
    label_enc.transform(Y)

    assert 'skprom_label_categorical' in [m.name for m in REGISTRY.collect()]

    assert REGISTRY.get_sample_value('skprom_label_categorical_total', {'Y': 'A'}) == 1
    assert REGISTRY.get_sample_value('skprom_label_categorical_total', {'Y': 'E'}) == 3