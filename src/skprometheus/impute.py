from functools import wraps

import numpy as np
from sklearn import impute
from sklearn.base import TransformerMixin

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


# class SimpleImputer(impute.SimpleImputer):
#     @wraps(impute.SimpleImputer.__init__, assigned=["__signature__"])
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         MetricRegistry.add_counter("imputed", "the number of values imputed", additional_labels=("method", "feature"))
#
#     def transform(self, X):
#         register_imputer_metrics(X, method="SimpleImputer")
#         return super().transform(X)


# class IterativeImputer(impute.IterativeImputer):
#     @wraps(impute.IterativeImputer.__init__, assigned=["__signature__"])
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         MetricRegistry.add_counter("imputed", "the number of values imputed", additional_labels=("method", "feature"))
#
#     def transform(self, X):
#         register_imputer_metrics(X, method="IterativeImputer")
#         return super().transform(X)


# class MissingIndicator(impute.MissingIndicator):
#     @wraps(impute.MissingIndicator.__init__, assigned=["__signature__"])
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         MetricRegistry.add_counter("imputed", "the number of values imputed", additional_labels=("method", "feature"))
#
#     def transform(self, X):
#         register_imputer_metrics(X, method="MissingIndicator")
#         return super().transform(X)


# class KNNImputer(impute.KNNImputer):
#     @wraps(impute.KNNImputer.__init__, assigned=["__signature__"])
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         MetricRegistry.add_counter("imputed", "the number of values imputed", additional_labels=("method", "feature"))
#
#     def transform(self, X):
#         register_imputer_metrics(X, method="KNNImputer")
#         return super().transform(X)


class ImputerCreated(type):
    def __new__(mcs, name, bases, d):
        class_obj = super().__new__(mcs, name, bases, d)

        # define __init__
        setattr(class_obj, '__init__', ImputerCreated.init(class_obj, bases[0]))
        setattr(class_obj, 'transform', ImputerCreated.transform(class_obj, bases[0]))

        return class_obj

    @staticmethod
    def init(class_obj, base):
        @wraps(base.__init__, assigned=["__signature__"])
        def new_init(self, *args, **kwargs):
            super(class_obj, self).__init__(*args, **kwargs)
            MetricRegistry.add_counter("imputed", "the number of values imputed",
                                       additional_labels=("method", "feature"))

        return new_init

    @staticmethod
    def transform(class_obj, base):
        def new_transform(self, X):
            register_imputer_metrics(X, method=base.__name__)
            return super(class_obj, self).transform(X)

        return new_transform


KNNImputer = ImputerCreated("KNNImputer", (impute.KNNImputer,), {})
MissingIndicator = ImputerCreated("MissingIndicator", (impute.MissingIndicator,), {})
SimpleImputer = ImputerCreated("SimpleImputer", (impute.SimpleImputer,), {})

if __name__ == "__main__":
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