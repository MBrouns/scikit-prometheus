import numpy as np
from prometheus_client import REGISTRY
from skprometheus.pipeline import Pipeline
from tests.utils import FixedLatencyClassifier, FixedProbasClassifier, ErrorClassifier
from skprometheus.metrics import MetricRegistry

MetricRegistry.set_labels(['Test'])

def test_pipeline_latency():
    with MetricRegistry:
        MetricRegistry.label(Test='latency')
        pipeline = Pipeline([
            ('clf', FixedLatencyClassifier(0.095))
        ])
        pipeline.predict(np.ones((15, 3)))

        assert 'model_predict_latency_seconds' in [m.name for m in REGISTRY.collect()]

        assert REGISTRY.get_sample_value('model_predict_latency_seconds_bucket', {'le': '0.1', 'Test': 'latency'}) == 1


def test_pipeline_probas():
    probas = np.array([
            [0.2, 0.2, 0.5, 0.1],
            [0.4, 0.6, 0., 0.]
        ])
    classes = [0, 1, 2, 3]
    with MetricRegistry:
        MetricRegistry.label(Test='probas')
        pipeline = Pipeline([
            ('clf', FixedProbasClassifier(probas, classes))
        ])
        pipeline.predict(np.ones((15, 3)))
        print([m for m in REGISTRY.collect()])
        assert False

    assert 'model_predict_probas' in [m.name for m in REGISTRY.collect()]


    assert REGISTRY.get_sample_value('model_predict_probas_bucket', {'le': '0.6', 'class': '0', 'Test': 'probas'}) == 2
    assert REGISTRY.get_sample_value('model_predict_probas_bucket', {'le': '0.3', 'class': '2', 'Test': 'probas'}) == 1


def test_pipeline_exceptions():
    pipeline = Pipeline([
        ('clf', ErrorClassifier())
    ])
    for i in range(4):
        try:
            pipeline.predict(np.ones((15, 3)))
        except ValueError:
            continue

    assert REGISTRY.get_sample_value('model_exception_total') == 4
