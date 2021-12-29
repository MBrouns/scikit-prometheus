import pytest
from prometheus_client import REGISTRY
from skprometheus.metrics import MetricRegistry


@pytest.fixture(autouse=True)
def unregister_collectors():
    """
    Fixture for cleaning registers before each test. Both prometheus_client.REGISTRY and
    skprometheus.metrics.MetricRegistry are cleaned.
    """
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)

    # Resetting attributes of MetricRegistry to avoid state transfer between tests
    # TODO: Maybe find less ugly solution in future?
    MetricRegistry.metrics_initialized = False
    MetricRegistry.current_labels = {}
    MetricRegistry.labels = set()
