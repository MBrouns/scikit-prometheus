from functools import wraps

import numpy as np
from sklearn import impute

from skprometheus.metrics import MetricRegistry
from skprometheus.utils import get_feature_names


def register_imputer_metrics(X, method):
    """Register the number of missing values per feature.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data to complete.
    method: {string}, the imputer method that is being registered
    """
    features = get_feature_names(X)

    missing = np.isnan(X).sum(axis=0)
    for idx, feature in enumerate(features):
        MetricRegistry.imputed(feature=feature, method=method).inc(missing[idx])


class SimpleImputer(impute.SimpleImputer):
    @wraps(impute.SimpleImputer.__init__, assigned=["__signature__"])
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        MetricRegistry.add_counter("imputed", "the number of values imputed", additional_labels=("method", "feature"))

    def transform(self, X):
        register_imputer_metrics(X, method="SimpleImputer")
        return super().transform(X)


class IterativeImputer(impute.IterativeImputer):
    @wraps(impute.IterativeImputer.__init__, assigned=["__signature__"])
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        MetricRegistry.add_counter("imputed", "the number of values imputed", additional_labels=("method", "feature"))

    def transform(self, X):
        register_imputer_metrics(X, method="IterativeImputer")
        return super().transform(X)


class MissingIndicator(impute.MissingIndicator):
    @wraps(impute.MissingIndicator.__init__, assigned=["__signature__"])
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        MetricRegistry.add_counter("imputed", "the number of values imputed", additional_labels=("method", "feature"))

    def transform(self, X):
        register_imputer_metrics(X, method="MissingIndicator")
        return super().transform(X)


class KNNImputer(impute.KNNImputer):
    @wraps(impute.KNNImputer.__init__, assigned=["__signature__"])
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        MetricRegistry.add_counter("imputed", "the number of values imputed", additional_labels=("method", "feature"))

    def transform(self, X):
        register_imputer_metrics(X, method="KNNImputer")
        return super().transform(X)

