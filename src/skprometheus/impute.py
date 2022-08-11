from functools import wraps

import numpy as np
from sklearn import impute
from sklearn.impute._base import _BaseImputer

from skprometheus.metrics import MetricRegistry
from skprometheus.utils import get_feature_names


class _BaseSkPrometheusImputer(_BaseImputer):
    def __init__(self, transformer):
        self.transformer = transformer
        super().__init__(missing_values=self.transformer.missing_values, add_indicator=self.transformer.add_indicator)

    def transform(self):
        pass


class SimpleImputer(_BaseSkPrometheusImputer):
    def __init__(self, *args, **kwargs):
        super().__init__(transformer=impute.SimpleImputer(*args, **kwargs))


class SimpleImputer(impute.SimpleImputer):
    @wraps(impute.SimpleImputer.__init__, assigned=["__signature__"])
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        MetricRegistry.add_counter("imputed", "the number of values imputed", additional_labels=("method", "feature"))

    def transform(self, X):
        features = get_feature_names(X)

        missing = np.isnan(X).sum(axis=0)
        for idx, feature in enumerate(features):
            MetricRegistry.imputed(feature=feature, method="SimpleImputer").inc(missing[idx])

        return super().transform(X)
