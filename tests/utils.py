import time

import numpy as np
from prometheus_client import REGISTRY
from sklearn.base import BaseEstimator, ClassifierMixin


class FixedLatencyClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, latency):
        self.latency = latency

    def fit(self, X, y):
        return self

    def predict(self, X):
        time.sleep(self.latency)
        return np.full(len(X), 0.3)


def metric_exists(metric_name, registry=REGISTRY):
    ...
