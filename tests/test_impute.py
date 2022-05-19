from skprometheus.impute import SimpleImputer
import numpy as np
from prometheus_client import REGISTRY
import pandas as pd


def test_simple_imputer():
    imputer = SimpleImputer()
    X = np.array([
        [np.nan, 3, 4, 6],
        [np.nan, np.nan, 4, 5],
        [np.nan, 5, 6, np.nan],
        [np.nan, 0, 0, np.nan],
        [np.nan, 7, 8, 9]
    ])

    imputer.fit(X)
    imputer.transform(X)
    assert 'skprom_imputed' in [m.name for m in REGISTRY.collect()]

    assert REGISTRY.get_sample_value('skprom_imputed_total', {'feature': '0', 'method': 'SimpleImputer'}) == 5
    assert REGISTRY.get_sample_value('skprom_imputed_total', {'feature': '1', 'method': 'SimpleImputer'}) == 1
    assert REGISTRY.get_sample_value('skprom_imputed_total', {'feature': '3', 'method': 'SimpleImputer'}) == 2


def test_simple_imputer_pandas():
    imputer = SimpleImputer()
    X = np.array([
        [np.nan, 3, 4, 6],
        [np.nan, np.nan, 4, 5],
        [np.nan, 5, 6, np.nan],
        [np.nan, 0, 0, np.nan],
        [np.nan, 7, 8, 9]
    ])

    df = pd.DataFrame.from_records(X, columns=['A', 'B', 'C', 'D'])

    imputer.fit(df)
    imputer.transform(df)
    assert 'skprom_imputed' in [m.name for m in REGISTRY.collect()]

    assert REGISTRY.get_sample_value('skprom_imputed_total', {'feature': 'A', 'method': 'SimpleImputer'}) == 5
    assert REGISTRY.get_sample_value('skprom_imputed_total', {'feature': 'B', 'method': 'SimpleImputer'}) == 1
    assert REGISTRY.get_sample_value('skprom_imputed_total', {'feature': 'D', 'method': 'SimpleImputer'}) == 2
