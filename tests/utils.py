import pickle
import time
from copy import copy
from types import SimpleNamespace

import numpy as np
from prometheus_client import REGISTRY
from sklearn.base import BaseEstimator, ClassifierMixin

from skprometheus.metrics import MetricRegistry


class FixedLatencyClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, latency):
        self.latency = latency

    def fit(self, X, y):
        return self

    def predict(self, X):
        time.sleep(self.latency)
        return np.full(len(X), 0.3)


class FixedProbasClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, probas, classes_):
        self.probas = probas
        self.classes_ = classes_

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.full(len(X), 0.3)

    def predict_proba(self, X):
        return self.probas


class ErrorClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self):
        ...

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        raise ValueError()


def metric_exists(metric_name, registry=REGISTRY):
    ...


def pickle_load_populates_metric_registry(name, estimator):
    metrics = {m._name for m in REGISTRY._collector_to_names.keys()}
    pkl = pickle.dumps(estimator)
    unregister_collectors()
    pickle.loads(pkl)
    assert {m._name for m in REGISTRY._collector_to_names.keys()} == metrics


def unregister_collectors():
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)

    # Resetting attributes of MetricRegistry to avoid state transfer between tests
    # TODO: Maybe find less ugly solution in future?
    MetricRegistry.metrics_initialized = False
    MetricRegistry.current_labels = {}
    MetricRegistry.labels = set()
    MetricRegistry.metrics = SimpleNamespace()
