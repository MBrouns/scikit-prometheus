import pytest
from prometheus_client import REGISTRY


@pytest.fixture(autouse=True)
def unregister_collectors():
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)