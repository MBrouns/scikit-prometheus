import numpy as np
from prometheus_client import REGISTRY
from skprometheus.pipeline import Pipeline
from tests.utils import FixedLatencyClassifier, FixedProbasClassifier
import pytest


@pytest.fixture(autouse=True)
def unregister_collectors():
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)


def test_pipeline_latency():
    pipeline = Pipeline([
        ('clf', FixedLatencyClassifier(0.095))
    ])

    assert 'model_predict_latency_seconds' in [m.name for m in REGISTRY.collect()]
    pipeline.predict(np.ones((15, 3)))

    assert REGISTRY.get_sample_value('model_predict_latency_seconds_bucket', {'le': '0.1'}) == 1


def test_pipeline_probas():
    probas = np.array([
            [0.2, 0.2, 0.5, 0.1],
            [0.4, 0.6, 0., 0.]
        ])
    classes = [0, 1, 2, 3]

    pipeline = Pipeline([
        ('clf', FixedProbasClassifier(probas, classes))
    ])

    assert 'model_predict_probas' in [m.name for m in REGISTRY.collect()]
    pipeline.predict(np.ones((15, 3)))

    assert REGISTRY.get_sample_value('model_predict_probas_bucket', {'le': '0.6', 'class': '0'}) == 2
    assert REGISTRY.get_sample_value('model_predict_probas_bucket', {'le': '0.3', 'class': '2'}) == 1
