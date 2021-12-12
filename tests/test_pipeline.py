import numpy as np
from prometheus_client import REGISTRY

from skprometheus.pipeline import Pipeline
from tests.utils import FixedLatencyClassifier


def test_pipeline_latency():
    pipeline = Pipeline([
        ('clf', FixedLatencyClassifier(0.095))
    ])

    assert 'model_predict_latency_seconds' in [m.name for m in REGISTRY.collect()]
    pipeline.predict(np.ones((15, 3)))

    assert REGISTRY.get_sample_value('model_predict_latency_seconds_bucket', {'le': '0.1'}) == 1
