from sklearn import preprocessing
from prometheus_client import Counter
from sklearn.utils.validation import _get_feature_names


class OneHotEncoder(preprocessing.OneHotEncoder):
    """
    OneHotEncoder that adds metrics to the prometheus metric registry.
    """
    def __init__(self, *args, prom_labels=None, **kwargs):
        self.prom_labels = prom_labels or {}
        self.model_categorical_count = Counter(
            "model_categorical_count",
            "Counts category occurrence for each categorical feature.",
            labelnames=tuple(self.prom_labels.keys()) + ("feature", "category")
        )
        super().__init__(*args, **kwargs)

    def transform(self, X):
        """
        Transform method that adds the count for each category in each feature to the prometheus
        metric registry.
        """
        transformed_X = super().transform(X)
        features = _get_feature_names(X)
        if not features:
            features = [col_nr for col_nr in range(X.shape[1])]

        # Use inverse method on transformed_X to get all missing values back as 'None'
        categories = self.inverse_transform(transformed_X)

        for idx, row in enumerate(categories.T):
            for category in row:
                if not category:
                    category = "missing"
                self.model_categorical_count.labels(feature=str(features[idx]), category=str(category)).inc()

        return transformed_X
