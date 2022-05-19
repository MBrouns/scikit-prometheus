from types import SimpleNamespace

from prometheus_client import Counter, Histogram


class _MetricRegistry:
    """Object for initiation and upkeep of metrics. Necessary to avoid assignment and labeling conflicts."""

    def __init__(self):
        self.prefix = 'skprom_'
        self.labels = set()
        self.current_labels = {}
        self.metrics_initialized = False
        self.categorical_metrics_initialized = False
        self.metrics = SimpleNamespace()

    def _add_metric(self, metric_type, name, description, additional_labels, **metric_kwargs):
        if hasattr(self.metrics, name):
            return
        additional_labels = additional_labels or tuple()
        setattr(self.metrics, name, metric_type(
            self.prefix + name,
            description,
            tuple(self.labels) + additional_labels,
            **metric_kwargs
            )
        )

    def add_histogram(self, name, description, *, buckets, additional_labels=None):
        self._add_metric(Histogram, name, description, additional_labels=additional_labels, buckets=buckets)

    def add_counter(self, name, description, *, additional_labels=None):
        self._add_metric(Counter, name, description, additional_labels=additional_labels)

    def _init_metrics(self):
        if self.metrics_initialized:
            return
        self.metrics_initialized = True

    def set_labels(self, labels):
        if self.metrics_initialized:
            raise ValueError('can only add labels before metrics are initialized')

        self.labels = set.union(self.labels, labels)

    def label(self, **labels):
        self.current_labels = labels
        return self

    def __getattr__(self, item):
        metric = getattr(self.metrics, item)
        # note the order, only init metrics if we can actually find a metric with that name
        self._init_metrics()

        def with_labels(**additional_labels):
            labels = dict(additional_labels, **self.current_labels)
            if not labels:
                return metric
            return metric.labels(**labels)

        return with_labels

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.labels = {}


MetricRegistry = _MetricRegistry()
