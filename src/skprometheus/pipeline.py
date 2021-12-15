from sklearn import pipeline
from sklearn.utils.metaestimators import available_if
from prometheus_client import Histogram, Counter
from skprometheus.utils import probas_to_metric


def _final_estimator_has(attr):
    """Check that final_estimator has `attr`.

    Used together with `avaliable_if` in `Pipeline`."""

    def check(self):
        # raise original `AttributeError` if `attr` does not exist
        getattr(self._final_estimator, attr)
        return True

    return check


def make_pipeline(*steps, memory=None, verbose=False):
    """
    Construct a :class:`Pipeline` from the given estimators.

    This is a shorthand for the :class:`Pipeline` constructor; it does not
    require, and does not permit, naming the estimators. Instead, their names
    will be set to the lowercase of their types automatically.
    """
    return Pipeline(pipeline._name_estimators(steps), memory=memory, verbose=verbose)


DEFAULT_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1., 2.5, 5., 7.5, 10., float('inf'))
DEFAULT_PROBA_BUCKETS = (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)


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
        prom_labels={},
        latency_buckets=DEFAULT_LATENCY_BUCKETS,
        proba_buckets=DEFAULT_PROBA_BUCKETS,
    ):
        self.prom_labels = prom_labels
        self.latency_buckets = latency_buckets
        self.proba_buckets = proba_buckets
        self._model_predict_total = Counter(
            "model_predict_total",
            "Amount of instances that the model made predictions for.",
            labelnames=tuple(self.prom_labels.keys()) if self.prom_labels is not None else ()
        )
        self._model_predict_latency = Histogram(
            "model_predict_latency_seconds",
            "Time in seconds it takes to call `predict` on the model",
            labelnames=tuple(self.prom_labels.keys()) if self.prom_labels is not None else (),
            buckets=latency_buckets
        )
        self._model_predict_proba = Histogram(
            "model_predict_probas",
            "Prediction probability for each class of the model",
            labelnames=(tuple(self.prom_labels.keys()) if self.prom_labels is not None else ()) + ("class_",),
            buckets=proba_buckets
        )
        super().__init__(steps=steps, memory=memory, verbose=verbose)

    @available_if(_final_estimator_has( "predict"))
    def predict(self, X, **predict_params):
        self._model_predict_total.inc()
        last_step = self.steps[-1][1]

        if hasattr(last_step, "predict_proba"):
            predict_probas = super().predict_proba(X, **predict_params) #**predict_parms not applicable here?
            #probas_to_metric(self._model_predict_proba, predict_probas, last_step.classes_, **self.prom_labels)
            for idx, class_ in enumerate(last_step.classes_):
                for proba in predict_probas: #TODO extract this to utils func
                    self._model_predict_proba.labels(class_=class_, **self.prom_labels).observe(proba[idx])

        with self._model_predict_latency.time():
            return super().predict(X, **predict_params)



# Try, except for model_exception_total?