import numpy as np
from sklearn.impute import SimpleImputer

from skprometheus.metrics import MetricRegistry
from skprometheus.utils import get_feature_names


class SimpleImputer(SimpleImputer):
    def __init__(
        self,
        *,
        missing_values=np.nan,
        strategy="mean",
        fill_value=None,
        verbose=0,
        copy=True,
        add_indicator=False,
    ):
        super().__init__(
            missing_values=missing_values,
            strategy=strategy,
            fill_value=fill_value,
            verbose=verbose,
            copy=copy,
            add_indicator=add_indicator,
        )
        MetricRegistry.add_counter('imputed', "the number of values imputed", additional_labels=('method', 'feature'))

    def transform(self, X):
        features = get_feature_names(X)

        missing = np.isnan(X).sum(axis=0)
        for idx, feature in enumerate(features):
            MetricRegistry.imputed(feature=feature, method='SimpleImputer').inc(missing[idx])

        return super().transform(X)
