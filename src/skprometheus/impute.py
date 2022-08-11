from functools import wraps

import numpy as np
from sklearn import impute

from skprometheus.metrics import MetricRegistry
from skprometheus.utils import get_feature_names


class SimpleImputer(impute.SimpleImputer):
    def __new__(cls, *args, **kwargs):
        MetricRegistry.add_counter("imputed", "the number of values imputed", additional_labels=("method", "feature"))
        return super(SimpleImputer, cls).__new__(cls)

    @wraps(impute.SimpleImputer.__init__, assigned=["__signature__"])
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def transform(self, X):
        features = get_feature_names(X)

        missing = np.isnan(X).sum(axis=0)
        for idx, feature in enumerate(features):
            MetricRegistry.imputed(feature=feature, method="SimpleImputer").inc(missing[idx])

        return super().transform(X)
