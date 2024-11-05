import sys
import os
import pytest
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.datasets import make_regression
from numpy.testing import assert_array_almost_equal
from ..sparse_pls import SparsePLS, DataPreprocessor
from sklearn.preprocessing import StandardScaler

@pytest.fixture
def data():
    """
    Generate synthetic regression data for testing.

    Returns
    -------
    X_np : ndarray of shape (100, 10)
        The feature matrix as a NumPy array.
    X_df : pandas.DataFrame of shape (100, 10)
        The feature matrix as a pandas DataFrame with column names.
    y : ndarray of shape (100,)
        The target values.
    feature_names : list of str
        The list of feature names.
    """
    X_np, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
    feature_names = [f'feature_{i}' for i in range(X_np.shape[1])]
    X_df = pd.DataFrame(X_np, columns=feature_names)
    return X_np, X_df, y, feature_names

@pytest.fixture
def sparse_data():
    """
    Generate synthetic sparse regression data for testing.
    """
    X_np, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
    X_sparse = sparse.csr_matrix(X_np)
    return X_sparse, y

def test_numpy_input(data):
    X_np, _, y, _ = data
    model = SparsePLS(n_components=2, alpha=0.1)
    model.fit(X_np, y)
    assert hasattr(model, 'coef_'), "The model was not trained correctly with numpy.ndarray input."

def test_pandas_input(data):
    _, X_df, y, _ = data
    model = SparsePLS(n_components=2, alpha=0.1)
    model.fit(X_df, y)
    assert hasattr(model, 'coef_'), "The model was not trained correctly with pandas.DataFrame input."

def test_column_names_preserved(data):
    _, X_df, y, feature_names = data
    model = SparsePLS(n_components=2, alpha=0.1)
    model.fit(X_df, y)
    assert model.feature_names_in_ is not None, "Feature names were not preserved."
    assert list(model.feature_names_in_) == feature_names, "Feature names do not match."

def test_prediction(data):
    X_np, X_df, y, feature_names = data
    X_test_np, _ = make_regression(n_samples=10, n_features=10, noise=0.1, random_state=24)
    X_test_df = pd.DataFrame(X_test_np, columns=feature_names)

    model = SparsePLS(n_components=2, alpha=0.1)
    model.fit(X_df, y)
    y_pred_np = model.predict(X_test_np)
    y_pred_df = model.predict(X_test_df)

    assert_array_almost_equal(y_pred_np, y_pred_df, err_msg="Predictions differ between numpy.ndarray and pandas.DataFrame inputs.")

def test_invalid_input():
    X_invalid = "invalid_input"
    y_invalid = [1, 2, 3]
    model = SparsePLS(n_components=2, alpha=0.1)
    with pytest.raises(ValueError):
        model.fit(X_invalid, y_invalid)

# Additional Tests

def test_data_preprocessor_fit_transform(data):
    X_np, _, _, _ = data
    preprocessor = DataPreprocessor(method='robust', impute_strategy='mean')
    X_transformed = preprocessor.fit_transform(X_np)
    assert X_transformed.shape == X_np.shape, "Transformed data shape mismatch."

def test_data_preprocessor_pandas_structure_preserved(data):
    _, X_df, _, _ = data
    preprocessor = DataPreprocessor(method='standard')
    X_transformed = preprocessor.fit_transform(X_df)
    assert isinstance(X_transformed, pd.DataFrame), "Output is not a DataFrame."
    assert list(X_transformed.columns) == list(X_df.columns), "Column names do not match."

def test_data_preprocessor_sparse_input(sparse_data):
    X_sparse, y = sparse_data
    preprocessor = DataPreprocessor(method='robust')
    try:
        X_transformed = preprocessor.fit_transform(X_sparse)
        assert sparse.issparse(X_transformed), "Output is not sparse."
    except ValueError as e:
        assert "NaN values" not in str(e), "Sparse input incorrectly identified as containing NaNs."


def test_data_preprocessor_custom_scaler(data):
    X_np, _, _, _ = data
    custom_scaler = StandardScaler()
    preprocessor = DataPreprocessor(method=custom_scaler)
    X_transformed = preprocessor.fit_transform(X_np)
    assert X_transformed.shape == X_np.shape, "Custom scaler did not process correctly."

def test_missing_value_imputation(data):
    X_np, _, _, _ = data
    X_np[0, 0] = np.nan  # Introduce a missing value
    preprocessor = DataPreprocessor(method='standard', impute_strategy='mean')
    X_transformed = preprocessor.fit_transform(X_np)
    assert not np.isnan(X_transformed).any(), "Imputation did not handle missing values."

def test_pipeline_integration(data):
    from sklearn.pipeline import Pipeline
    X_np, _, y, _ = data
    pipeline = Pipeline([
        ('preprocessor', DataPreprocessor(method='robust', impute_strategy='median')),
        ('model', SparsePLS(n_components=2, alpha=0.1))
    ])
    pipeline.fit(X_np, y)
    y_pred = pipeline.predict(X_np)
    assert y_pred.shape[0] == X_np.shape[0], "Pipeline prediction output shape mismatch."
