# scikit-prometheus

The main goal for this project is to make it easy to add basic Prometheus instrumentation to your `sklearn` models. This way we can lower the entry barrier for instrumenting ML systems and thereby raise the quality of ML in production.



## Examples


### Overall pipeline monitoring
```python
from skprometheus.pipeline import make_pipeline

clf_pipeline = make_pipeline(
  StandardScaler(),
  LogisticRegression(),
)
```

This will expose the following prometheus metrics

#### `model_predict_probas` (Histogram) 
For every instance that passed through `predict`, the predicted probabilities are logged in a histogram. This histogram has 10 bins, evenly spaced between 0.0 and 1.0 

#### `model_predict_latency_ms` (Histogram)
For every instance that passed through `predict`, the time it took to make a prediction is logged in a histogram. This histogram has 10 bins, evenly spaced between 0 and 250 ms. 

We don't automatically infer the buckets of the histogram. The disadvantage there is that a new model fit will therefore have new hist buckets which make tracking different models over time difficult

#### `model_predict_total` (Counter)
The amount of instances that the model made predictions for.

#### `model_exception_total` (Counter)
The amount of times the model threw an exception while predicting for new instances

### Categorical encoder monitoring
In the example below we've used the `OneHotEncoder`, but this also works for the `OrdinalEncoder`

```python
from skprometheus.pipeline import make_pipeline
from skprometheus.preprocessing import OneHotEncoder

clf_pipeline = make_pipeline(
  OneHotEncoder(),
  LogisticRegression(),
)
```

#### `model_categorical_total{feature='...', category='...', action='transform'}` (Counter) 
Counts the number of times each `category` is observed for each `feature` when `transform` is called on the encoder
Note that apart from the categories observed in the training data, an additional category called `missing` is added as a label. This counter gets increased
whenever transform is called on data that contains a category that wasn't observed during training time.

#### `model_categorical_total{feature='...', category='...', action='fit'}` (Gauge) 
Counts the number of times each `category` is observed for each `feature` when the encoder is fitted. TODO: fit time metrics need to be worked out further!







