from collections import Counter



# This should be a singleton class (https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python)
class _MetricRegistry:
    def __init__(self):
        self.labels = set()
        self.metrics_initialized = False
        self.current_labels = {}

    def add_labels(self, **labels):
        if self.metrics_initialized:
            raise ValueError('can only add labels before metrics are initialized')

        self.labels += set(labels)

    def _init_metrics(self):
        if self.metrics_initialized:
            return
        self.metrics_initialized = True
        self._model_predict_total = Counter(
            "model_predict_total",
            "Amount of instances that the model made predictions for.",
            labelnames=list(self.labels)
        )

    def label(self, **labels):
        self.current_labels = labels

    def __enter__(self, ...):
        ...

    def __exit__(self, ...):
        self.current_labels = {}


    @property
    def model_predict_total(self):
        self._init_metrics()
        return self._model_predict_total.labels(**self.current_labels)



MetricRegistry = _MetricRegistry()
