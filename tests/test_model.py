# tests/test_model.py

import os
import pickle
import pytest
import numpy as np
import pandas as pd
from typing import Tuple, List
from numpy import ndarray
from pandas import DataFrame
from scipy import sparse
from sklearn.datasets import make_regression
from numpy.testing import assert_array_almost_equal
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pathlib import Path

# Adjust import based on your package structure
from ..sparse_pls import SparsePLS, DataPreprocessor

@pytest.fixture
def data() -> Tuple[ndarray, DataFrame, ndarray, List[str]]:
    """Generate synthetic regression data for testing."""
    X_np, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
    feature_names = [f"feature_{i}" for i in range(X_np.shape[1])]
    X_df = pd.DataFrame(X_np, columns=feature_names)
    return X_np, X_df, y, feature_names

@pytest.fixture
def sparse_data():
    """Generate synthetic sparse regression data for testing."""
    X_np, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
    X_sparse = sparse.csr_matrix(X_np)
    return X_sparse, y

def test_numpy_input(data: Tuple[ndarray, DataFrame, ndarray, List[str]]) -> None:
    """
    Ensure that model can be fit on numpy.ndarray input without errors.
    """
    X_np, _, y, _ = data
    model = SparsePLS(n_components=2, alpha=0.1)
    model.fit(X_np, y)
    assert hasattr(model, "coef_"), "The model was not trained correctly with numpy.ndarray input."

def test_pandas_input(data: Tuple[ndarray, DataFrame, ndarray, List[str]]) -> None:
    """
    Ensure that model can be fit on pandas.DataFrame input without errors.
    """
    _, X_df, y, _ = data
    model = SparsePLS(n_components=2, alpha=0.1)
    model.fit(X_df, y)
    assert hasattr(model, "coef_"), "The model was not trained correctly with pandas.DataFrame input."

def test_column_names_preserved(data: Tuple[ndarray, DataFrame, ndarray, List[str]]) -> None:
    """
    Check that feature names are preserved and stored in the estimator.
    """
    _, X_df, y, feature_names = data
    model = SparsePLS(n_components=2, alpha=0.1)
    model.fit(X_df, y)
    assert model.feature_names_in_ is not None, "Feature names were not preserved."
    assert list(model.feature_names_in_) == feature_names, "Feature names do not match."

def test_prediction(data: Tuple[ndarray, DataFrame, ndarray, List[str]]) -> None:
    """
    Compare predictions using NumPy arrays vs. pandas DataFrames as input.
    They should be almost identical.
    """
    X_np, X_df, y, feature_names = data
    X_test_np, _ = make_regression(n_samples=10, n_features=10, noise=0.1, random_state=24)
    X_test_df = pd.DataFrame(X_test_np, columns=feature_names)

    model = SparsePLS(n_components=2, alpha=0.1)
    model.fit(X_df, y)
    y_pred_np = model.predict(X_test_np)
    y_pred_df = model.predict(X_test_df)

    assert_array_almost_equal(
        y_pred_np,
        y_pred_df,
        err_msg="Predictions differ between numpy.ndarray and pandas.DataFrame inputs."
    )

def test_invalid_input() -> None:
    """
    Confirm that fitting on invalid input raises ValueError.
    """
    X_invalid = "invalid_input"
    y_invalid = [1, 2, 3]
    model = SparsePLS(n_components=2, alpha=0.1)
    with pytest.raises(ValueError):
        model.fit(X_invalid, y_invalid)

def test_data_preprocessor_fit_transform(data: Tuple[ndarray, DataFrame, ndarray, List[str]]) -> None:
    """
    Test DataPreprocessor with robust scaling and mean imputation on NumPy data.
    """
    X_np, _, _, _ = data
    preprocessor = DataPreprocessor(method="robust", impute_strategy="mean")
    X_transformed = preprocessor.fit_transform(X_np)
    assert X_transformed.shape == X_np.shape, "Transformed data shape mismatch."

def test_data_preprocessor_pandas_structure_preserved(data: Tuple[ndarray, DataFrame, ndarray, List[str]]) -> None:
    """
    Verify that when the input is a DataFrame, the output remains a DataFrame 
    with the same columns (for standard scaling).
    """
    _, X_df, _, _ = data
    preprocessor = DataPreprocessor(method="standard")
    X_transformed = preprocessor.fit_transform(X_df)
    assert isinstance(X_transformed, pd.DataFrame), "Output is not a DataFrame."
    assert list(X_transformed.columns) == list(X_df.columns), "Column names do not match."

def test_data_preprocessor_sparse_input(sparse_data):
    """Test sparse input handling."""
    X_sparse, y = sparse_data
    preprocessor = DataPreprocessor(method="robust")
    try:
        X_transformed = preprocessor.fit_transform(X_sparse)
        assert sparse.issparse(X_transformed), "Output should be sparse but isn't."
    except ValueError as e:
        # Some transformations might not support sparse,
        # but ensure that it doesn't fail due to "NaN values"
        assert "NaN values" not in str(e), "Sparse input incorrectly identified as containing NaNs"

def test_data_preprocessor_custom_scaler(data: Tuple[ndarray, DataFrame, ndarray, List[str]]) -> None:
    """
    Test using a custom scaler (StandardScaler) in the DataPreprocessor.
    """
    X_np, _, _, _ = data
    custom_scaler = StandardScaler()
    preprocessor = DataPreprocessor(method=custom_scaler)
    X_transformed = preprocessor.fit_transform(X_np)
    assert X_transformed.shape == X_np.shape, "Custom scaler did not process data correctly."

def test_missing_value_imputation(data: Tuple[ndarray, DataFrame, ndarray, List[str]]) -> None:
    """
    Ensure that missing value imputation works properly.
    """
    X_np, _, _, _ = data
    X_np[0, 0] = np.nan  # Introduce a missing value
    preprocessor = DataPreprocessor(method="standard", impute_strategy="mean")
    X_transformed = preprocessor.fit_transform(X_np)
    assert not np.isnan(X_transformed).any(), "Imputation did not handle missing values."

def test_pipeline_integration(data: Tuple[ndarray, DataFrame, ndarray, List[str]]) -> None:
    """
    Confirm that SparsePLS integrates cleanly into a sklearn Pipeline.
    """
    X_np, _, y, _ = data
    pipeline = Pipeline([
        ("preprocessor", DataPreprocessor(method="robust", impute_strategy="median")),
        ("model", SparsePLS(n_components=2, alpha=0.1))
    ])
    pipeline.fit(X_np, y)
    y_pred = pipeline.predict(X_np)
    assert y_pred.shape[0] == X_np.shape[0], "Pipeline prediction output shape mismatch."

def test_use_estimator(data: Tuple[ndarray, DataFrame, ndarray, List[str]]) -> None:
    """
    Test the use_estimator method from SparsePLS.
    """
    X, _, y, _ = data
    model = SparsePLS(n_components=2, alpha=0.1)
    model.fit(X, y)
    predictions = model.use_estimator(X)
    assert predictions.shape == y.shape, "Prediction output shape mismatch."

def test_export_estimator(
    data: Tuple[ndarray, DataFrame, ndarray, List[str]],
    tmp_path: Path
) -> None:
    """
    Test exporting the trained SparsePLS model as a pickle file and reloading it.
    """
    X, _, y, _ = data
    model = SparsePLS(n_components=2, alpha=0.1)
    model.fit(X, y)

    file_path = tmp_path / "sparse_pls_model.pkl"
    model.export_estimator(str(file_path))

    assert os.path.exists(file_path), "Model was not saved successfully."

    with open(file_path, "rb") as f:
        loaded_model = pickle.load(f)

    assert isinstance(loaded_model, SparsePLS), "Loaded object is not an instance of SparsePLS."

def test_get_selected_feature_names(data: Tuple[ndarray, DataFrame, ndarray, List[str]]) -> None:
    """
    Test that get_selected_feature_names returns a list of strings or integers.
    """
    X, _, y, feature_names = data
    model = SparsePLS(n_components=2, alpha=0.1)
    model.fit(X, y)

    selected_features = model.get_selected_feature_names()
    assert isinstance(selected_features, list), "Selected features should be a list."
    assert all(isinstance(f, (str, int)) for f in selected_features), (
        f"Feature names/indices should be str or int, got: {selected_features}"
    )
