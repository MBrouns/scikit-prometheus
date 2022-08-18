from tkinter import Y
from functools import wraps

from sklearn import preprocessing
from skprometheus.metrics import MetricRegistry
from skprometheus.utils import get_feature_names


def feature_category_count(X, categories):

    features = get_feature_names(X)

    for idx, row in enumerate(categories.T):
        for category in row:
            if category is None:
                category = "missing"
                
            MetricRegistry.model_categorical(feature=str(features[idx]), category=str(category)).inc()


def label_count(labels):

    for label in labels:
        if label[0] is None:
            label[0] = "missing"
        MetricRegistry.label_categorical(Y=str(label[0])).inc()
           

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

        # Use inverse method on transformed_X to get all missing values back as 'None'
        categories = self.inverse_transform(transformed_X)

        feature_category_count(X, categories)

        return transformed_X


class OrdinalEncoder(preprocessing.OrdinalEncoder):
    """
    OrdinalEncoder that adds metrics to the prometheus metric registry.
    """
    @wraps(preprocessing.OrdinalEncoder.__init__, assigned=["__signature__"])
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

        # Use inverse method on transformed_X to get all missing values back as 'None'
        categories = self.inverse_transform(transformed_X)

        feature_category_count(X, categories)

        return transformed_X


class LabelEncoder(preprocessing.OrdinalEncoder):
    """
    LabelEncoder that adds metrics to the prometheus metric registry.
    """
    @wraps(preprocessing.LabelEncoder.__init__, assigned=["__signature__"])
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        MetricRegistry.add_counter(
            "label_categorical",
            "Counts category occurrence for each target label.",
            additional_labels=tuple("Y"),
        )

    def transform(self, Y):
        """
        Transform method that adds the count for each label to the prometheus
        metric registry.
        """
        transformed_Y = super().transform(Y)

        # Use inverse method on transformed_X to get all missing values back as 'None'
        labels = self.inverse_transform(transformed_Y)

        label_count(labels)

        return transformed_Y