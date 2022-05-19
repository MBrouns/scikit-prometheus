from sklearn import pipeline
from sklearn.utils.metaestimators import available_if
from skprometheus.prom_client_utils import observe_many
from skprometheus.metrics import MetricRegistry


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

    This method also does not allow the setting of additional labels to your prometheus metrics.
    """
    return Pipeline(pipeline._name_estimators(steps), memory=memory, verbose=verbose)


class Pipeline(pipeline.Pipeline):
    """
    A pipeline that adds metrics to the prometheus metric registry.
    """

    @available_if(_final_estimator_has("predict"))
    def predict(self, X, **predict_params):
        """
        Predict method that adds the model latency and model probabilities to
        prometheus metric registry.
        """
        try:
            with MetricRegistry.model_predict_latency().time():
                X_transformed = X
                for _, _, transformer in self._iter(with_final=False):
                    X_transformed = transformer.transform(X_transformed)

                final_step = self.steps[-1][1]

                if hasattr(final_step, "predict_proba"):
                    predict_probas = final_step.predict_proba(X_transformed, **predict_params)
                    for idx, class_ in enumerate(final_step.classes_):
                        observe_many(
                            MetricRegistry.model_predict_proba(class_=class_),
                            predict_probas[:, idx]
                        )
                predictions = final_step.predict(X_transformed, **predict_params)
                MetricRegistry.model_predict_total().inc(len(predictions))
                return predictions
        except Exception as err:
            MetricRegistry.model_exception().inc()
            raise err
