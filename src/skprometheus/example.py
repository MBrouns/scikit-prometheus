from skprometheus.pipeline import make_pipeline
from skprometheus.metrics import MetricRegistry


mr = MetricRegistry.setLabels(['model'])
mr2 = MetricRegistry.setLabels(['foo'])

mr is mr2



with mr.label(model='example-model-1'):
    model = make_pipeline(
        ...
    )



with mr.label(model='example-model-2'):
    model2 = make_pipeline(
        ...
    )
