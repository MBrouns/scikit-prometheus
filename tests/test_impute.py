import pytest

from skprometheus.impute import SimpleImputer, MissingIndicator, KNNImputer
from skprometheus.utils import flatten
import numpy as np
from prometheus_client import REGISTRY
import pandas as pd
from tests.conftest import general_checks, transformer_checks, select_tests


IMPUTERS = [SimpleImputer, MissingIndicator, KNNImputer]


@pytest.fixture()
def missing_values():
    return np.array([
        [np.nan, 3, 4, 6],
        [np.nan, np.nan, 4, 5],
        [np.nan, 5, 6, np.nan],
        [np.nan, 0, 0, np.nan],
        [np.nan, 7, 8, 9]
    ])


@pytest.mark.parametrize(
    "test_fn",
    select_tests(
        flatten([general_checks, transformer_checks]),
    ),
)
@pytest.mark.parametrize(
    "trf",
    IMPUTERS
)
def test_standard_checks_imputer(test_fn, trf):
    test_fn(trf.__class__.__name__, trf())


@pytest.mark.parametrize(
    "method",
    IMPUTERS
)
def test_imputer(missing_values, method):
    imputer = method()
    X = missing_values

    imputer.fit(X)
    imputer.transform(X)
    assert 'skprom_imputed' in [m.name for m in REGISTRY.collect()]

    assert REGISTRY.get_sample_value(
        'skprom_imputed_total', {'feature': '0', 'method': imputer.__class__.__name__}
        ) == 5
    assert REGISTRY.get_sample_value(
        'skprom_imputed_total', {'feature': '1', 'method': imputer.__class__.__name__}
        ) == 1
    assert REGISTRY.get_sample_value(
        'skprom_imputed_total', {'feature': '3', 'method': imputer.__class__.__name__}
        ) == 2


@pytest.mark.parametrize(
    "method",
    IMPUTERS
)
def test_simple_imputer_pandas(missing_values, method):
    imputer = method()
    X = missing_values

    df = pd.DataFrame.from_records(X, columns=['A', 'B', 'C', 'D'])

    imputer.fit(df)
    imputer.transform(df)
    assert 'skprom_imputed' in [m.name for m in REGISTRY.collect()]

    assert REGISTRY.get_sample_value(
        'skprom_imputed_total', {'feature': 'A', 'method': imputer.__class__.__name__}
        ) == 5
    assert REGISTRY.get_sample_value(
        'skprom_imputed_total', {'feature': 'B', 'method': imputer.__class__.__name__}
        ) == 1
    assert REGISTRY.get_sample_value(
        'skprom_imputed_total', {'feature': 'D', 'method': imputer.__class__.__name__}
        ) == 2
