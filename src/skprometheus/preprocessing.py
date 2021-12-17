from sklearn import preprocessing
from prometheus_client import Counter


class OneHotEncoder(preprocessing.OneHotEncoder):
    def __init__(self, *args, prom_labels=None, **kwargs):
        self.prom_labels = prom_labels or {}
        self.model_categorical_total = Counter(
            "model_categorical_total",
            "Counts category occurrence for each categorical feature.",
            labelnames=tuple(self.prom_labels.keys()) + ("feature", "category")
        )
        super().__init__(*args, **kwargs)

    def transform(self, X):
        transformed_X = super().transform(X)

        for row in self.inverse_transform(transformed_X): # inverse_transform to get missing back as none --> handig?
            for item in row:
                self.model_categorical_total.inc() #with labels

        return transformed_X
