import pytest
from sklearn.utils import estimator_checks

from tests.utils import pickle_load_populates_metric_registry, unregister_collectors


@pytest.fixture(autouse=True)
def _unregister_collectors():
    """
    Fixture for cleaning registers before each test. Both prometheus_client.REGISTRY and
    skprometheus.metrics.MetricRegistry are cleaned.
    """
    unregister_collectors()


transformer_checks = (
    estimator_checks.check_transformer_data_not_an_array,
    estimator_checks.check_transformer_general,
    estimator_checks.check_transformers_unfitted,
)

general_checks = (
    pickle_load_populates_metric_registry,
    estimator_checks.check_fit2d_predict1d,
    estimator_checks.check_methods_subset_invariance,
    estimator_checks.check_fit2d_1sample,
    estimator_checks.check_fit2d_1feature,
    estimator_checks.check_fit1d,
    estimator_checks.check_get_params_invariance,
    estimator_checks.check_set_params,
    estimator_checks.check_dict_unchanged,
    estimator_checks.check_dont_overwrite_parameters,
)

nonmeta_checks = (
    estimator_checks.check_estimators_dtypes,
    estimator_checks.check_fit_score_takes_y,
    estimator_checks.check_dtype_object,
    estimator_checks.check_sample_weights_pandas_series,
    estimator_checks.check_sample_weights_list,
    estimator_checks.check_sample_weights_invariance,
    estimator_checks.check_estimators_fit_returns_self,
    estimator_checks.check_complex_data,
    estimator_checks.check_estimators_empty_data_messages,
    estimator_checks.check_pipeline_consistency,
    estimator_checks.check_estimators_nan_inf,
    estimator_checks.check_estimators_overwrite_params,
    estimator_checks.check_estimator_sparse_data,
    estimator_checks.check_estimators_pickle,
)

classifier_checks = (
    estimator_checks.check_classifier_data_not_an_array,
    estimator_checks.check_classifiers_one_label,
    estimator_checks.check_classifiers_classes,
    estimator_checks.check_estimators_partial_fit_n_features,
    estimator_checks.check_classifiers_train,
    estimator_checks.check_supervised_y_2d,
    estimator_checks.check_supervised_y_no_nan,
    estimator_checks.check_estimators_unfitted,
    estimator_checks.check_non_transformer_estimators_n_iter,
    estimator_checks.check_decision_proba_consistency,
)

regressor_checks = (
    estimator_checks.check_regressors_train,
    estimator_checks.check_regressor_data_not_an_array,
    estimator_checks.check_estimators_partial_fit_n_features,
    estimator_checks.check_regressors_no_decision_function,
    estimator_checks.check_supervised_y_2d,
    estimator_checks.check_supervised_y_no_nan,
    estimator_checks.check_regressors_int,
    estimator_checks.check_estimators_unfitted,
)

outlier_checks = (
    estimator_checks.check_outliers_fit_predict,
    estimator_checks.check_outliers_train,
    estimator_checks.check_classifier_data_not_an_array,
    estimator_checks.check_estimators_unfitted,
)


def select_tests(include, exclude=[]):
    """Return an iterable of include with all tests whose name is not in exclude"""
    for test in include:
        if test.__name__ not in exclude:
            yield test
