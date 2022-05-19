from prometheus_client import Counter, Histogram
from skprometheus.prom_client_utils import add_labels


class _MetricRegistry:
    """Object for initiation and upkeep of metrics. Necessary to avoid assignment and labeling conflicts."""
    def __init__(self):
        self.labels = set()
        self.metrics_initialized = False
        self.categorical_metrics_initialized = False
        self.current_labels = {}
        self.DEFAULT_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25,
                                        0.5, 0.75, 1., 2.5, 5., 7.5, 10., float('inf'))
        self.DEFAULT_PROBA_BUCKETS = (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)

    def set_labels(self, labels):
        if self.metrics_initialized:
            raise ValueError('can only add labels before metrics are initialized')

        self.labels = set.union(self.labels, labels)

    def _init_metrics(self):
        if self.metrics_initialized:
            return
        self.metrics_initialized = True
        self._model_predict = Counter(
            "model_predict",
            "Amount of instances that the model made predictions for.",
            labelnames=tuple(self.labels)
        )
        self._model_predict_latency = Histogram(
            "model_predict_latency_seconds",
            "Time in seconds it takes to call `predict` on the model",
            labelnames=tuple(self.labels),
            buckets=self.DEFAULT_LATENCY_BUCKETS
        )
        self._model_predict_proba = Histogram(
            "model_predict_probas",
            "Prediction probability for each class of the model",
            labelnames=tuple(self.labels) + ("class_",),
            buckets=self.DEFAULT_PROBA_BUCKETS
        )
        self._model_exception = Counter(
            "model_exception",
            "Amount of exceptions during predict step.",
            labelnames=tuple(self.labels)
        )

    def _init_categorical_metrics(self):
        if self.categorical_metrics_initialized:
            return
        self.categorical_metrics_initialized = True
        self._model_categorical = Counter(
            "model_categorical",
            "Counts category occurrence for each categorical feature.",
            labelnames=tuple(self.labels) + ("feature", "category")
        )

    def label(self, **labels):
        self.current_labels = labels
        return self

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.current_labels = {}

    def model_predict_total(self):
        self._init_metrics()
        return add_labels(self._model_predict, self.current_labels)

    def model_predict_latency(self):
        self._init_metrics()
        return add_labels(self._model_predict_latency, self.current_labels)

    def model_predict_proba(self, **labels):
        self._init_metrics()
        labels = dict(labels, **self.current_labels)
        return add_labels(self._model_predict_proba, labels)

    def model_exception(self):
        self._init_metrics()
        return add_labels(self._model_exception, self.current_labels)

    def model_categorical(self, **labels):
        self._init_categorical_metrics()
        labels = dict(labels, **self.current_labels)
        return add_labels(self._model_categorical, labels)


MetricRegistry = _MetricRegistry()
