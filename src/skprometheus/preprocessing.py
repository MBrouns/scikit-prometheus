from functools import wraps

from sklearn import preprocessing
from skprometheus.metrics import MetricRegistry
from skprometheus.utils import get_feature_names


class OneHotEncoder(preprocessing.OneHotEncoder):
    """
    OneHotEncoder that adds metrics to the prometheus metric registry.
    """
    @wraps(preprocessing.OneHotEncoder.__init__, assigned=["__signature__"])
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        MetricRegistry.add_counter(
            "model_categorical",
            "Counts category occurrence for each categorical feature.",
            additional_labels=("feature", "category"),
        )

    def transform(self, X):
        """
        Transform method that adds the count for each category in each feature to the prometheus
        metric registry.
        """
        transformed_X = super().transform(X)
        features = get_feature_names(X)

        # Use inverse method on transformed_X to get all missing values back as 'None'
        categories = self.inverse_transform(transformed_X)

        for idx, row in enumerate(categories.T):
            for category in row:
                if not category:
                    category = "missing"
                MetricRegistry.model_categorical(feature=str(features[idx]), category=str(category)).inc()

        return transformed_X
