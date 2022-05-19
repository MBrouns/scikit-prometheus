import numpy as np
from prometheus_client import REGISTRY
from skprometheus.pipeline import Pipeline
from tests.utils import FixedLatencyClassifier, FixedProbasClassifier, ErrorClassifier
from skprometheus.metrics import MetricRegistry


def test_pipeline_latency():
    MetricRegistry.set_labels(['Test'])
    with MetricRegistry.label(Test='latency'):
        pipeline = Pipeline([
            ('clf', FixedLatencyClassifier(0.095))
        ])
        pipeline.predict(np.ones((15, 3)))

        assert 'skprom_model_predict_latency_seconds' in [m.name for m in REGISTRY.collect()]
        assert REGISTRY.get_sample_value(
            'skprom_model_predict_latency_seconds_bucket',
            {'le': '0.1', 'Test': 'latency'}
        ) == 1


def test_pipeline_probas():
    probas = np.array([
            [0.2, 0.2, 0.5, 0.1],
            [0.4, 0.6, 0., 0.]
        ])
    classes = [0, 1, 2, 3]

    pipeline = Pipeline([
        ('clf', FixedProbasClassifier(probas, classes))
    ])
    pipeline.predict(np.ones((15, 3)))

    assert 'skprom_model_predict_probas' in [m.name for m in REGISTRY.collect()]
    assert REGISTRY.get_sample_value('skprom_model_predict_probas_bucket', {'le': '0.6', 'class_': '0'}) == 2
    assert REGISTRY.get_sample_value('skprom_model_predict_probas_bucket', {'le': '0.3', 'class_': '2'}) == 1


def test_pipeline_exceptions():
    MetricRegistry.set_labels(['Test'])
    with MetricRegistry.label(Test='exceptions'):
        pipeline = Pipeline([
            ('clf', ErrorClassifier())
        ])
        for i in range(4):
            try:
                pipeline.predict(np.ones((15, 3)))
            except ValueError:
                continue

        assert REGISTRY.get_sample_value('skprom_model_exception_total', {'Test': 'exceptions'}) == 4


def test_pipeline_count():
    pipeline = Pipeline([
        ('clf', FixedLatencyClassifier(0.095))
    ])
    pipeline.predict(np.ones((15, 3)))
    pipeline.predict(np.ones((22, 3)))

    assert 'skprom_model_predict' in [m.name for m in REGISTRY.collect()]
    assert REGISTRY.get_sample_value('skprom_model_predict_total') == 37
