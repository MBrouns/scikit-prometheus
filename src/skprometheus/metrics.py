from prometheus_client import Counter, Histogram
from skprometheus.prom_client_utils import add_labels


# Do we want the metricRegistry to be automatically loaded unless specified otherwise?  --> Yes


class _MetricRegistry:
    def __init__(self):
        self.labels = set()
        self.metrics_initialized = False
        self.current_labels = {}
        self.DEFAULT_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1., 2.5, 5., 7.5, 10., float('inf'))
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
            labelnames=tuple(self.labels) + ("class",),
            buckets=self.DEFAULT_PROBA_BUCKETS
        )
        self._model_exception = Counter(
            "model_exception",
            "Amount of exceptions during predict step.",
            labelnames=tuple(self.labels)
        )

    def label(self, **labels):
        self.current_labels = labels

    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        self.current_labels = {}

    @property
    def model_predict_total(self):
        self._init_metrics() #_init method for each metric?s
        return add_labels(self._model_predict, self.current_labels)

    @property
    def model_predict_latency(self):
        self._init_metrics()
        return add_labels(self._model_predict_latency, self.current_labels)

    @property
    def model_predict_proba(self):
        self._init_metrics()
        return add_labels(self._model_predict_proba, self.current_labels)

    @property
    def model_exception(self):
        self._init_metrics()
        return add_labels(self._model_exception, self.current_labels)


MetricRegistry = _MetricRegistry()


def reset_metric_registry():
    global MetricRegistry
    MetricRegistry = _MetricRegistry()
