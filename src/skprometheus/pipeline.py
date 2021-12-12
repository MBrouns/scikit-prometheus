from sklearn import pipeline
from sklearn.utils.metaestimators import available_if
from prometheus_client import Histogram


def _final_estimator_has(attr):
    """Check that final_estimator has `attr`.

    Used together with `avaliable_if` in `Pipeline`."""

    def check(self):
        # raise original `AttributeError` if `attr` does not exist
        getattr(self._final_estimator, attr)
        return True

    return check


DEFAULT_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1., 2.5, 5., 7.5, 10., float('inf'))


class Pipeline(pipeline.Pipeline):
    """
    A pipeline that adds metrics to the prometheus metric registry.
    """

    def __init__(
        self,
        steps,
        memory=None,
        verbose=False,
        *,
        prom_labels=None,
        latency_buckets=DEFAULT_LATENCY_BUCKETS
    ):
        self.prom_labels = prom_labels
        self.latency_buckets = latency_buckets
        self._model_predict_latency = Histogram(
            "model_predict_latency_seconds",
            "Time in seconds it takes to call `predict` on the model",
            labelnames=tuple(self.prom_labels.keys()) if self.prom_labels is not None else (),
            buckets=latency_buckets
        )
        super().__init__(steps=steps, memory=memory, verbose=verbose)

    @available_if(_final_estimator_has("predict"))
    def predict(self, X, **predict_params):
        with self._model_predict_latency.time():
            return super().predict(X, **predict_params)
