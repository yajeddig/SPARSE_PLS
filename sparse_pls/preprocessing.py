import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.impute import SimpleImputer

class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Data Preprocessor that supports multiple scaling methods and stops processing when NaN values are detected.

    This class provides a flexible interface for preprocessing data using different scaling methods.
    It implements the scikit-learn Transformer API, allowing it to be used seamlessly in pipelines.

    Parameters
    ----------
    method : str or object, default='standard'
        Scaling method to use. Options are:
        - 'standard': StandardScaler (mean=0, std=1)
        - 'robust': RobustScaler (median centering, scaling by IQR)
        - 'quantile_uniform': QuantileTransformer with uniform output
        - 'quantile_normal': QuantileTransformer with normal output
        - Custom scaler: A custom scaler instance with `fit` and `transform` methods.
    impute_strategy : str or None, default=None
        Strategy for imputing missing values. Options are:
        - 'mean': Replace missing values with the mean of the column.
        - 'median': Replace missing values with the median of the column.
        - 'most_frequent': Replace missing values with the most frequent value in the column.
        - None: Do not perform imputation and raise an error if missing values are detected.
    **kwargs : dict
        Additional keyword arguments passed to the scaler.

    Attributes
    ----------
    scaler : object
        The scaler instance used for transformation.
    imputer : SimpleImputer or None
        The imputer instance used for handling missing values, if applicable.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> preprocessor = DataPreprocessor(method='robust', impute_strategy='mean')
    >>> X_scaled = preprocessor.fit_transform(X)
    """

    def __init__(self, method='standard', impute_strategy=None, **kwargs):
        self.method = method
        self.impute_strategy = impute_strategy
        self.kwargs = {k: v for k, v in kwargs.items() if k not in ['impute_strategy']}  # Filter out unwanted kwargs
        self.scaler = None
        self.imputer = None

    def fit(self, X: np.ndarray, y: None = None) -> 'DataPreprocessor':
        """
        Fit the scaler to the data and check for NaN values.
        """
        X = self._validate_data(X)

        if self.impute_strategy:
            self.imputer = SimpleImputer(strategy=self.impute_strategy)
            X = self.imputer.fit_transform(X)
        elif np.isnan(X.data if sparse.issparse(X) else X).any():
            raise ValueError("Input data contains NaN values. Please handle missing data or set `impute_strategy`.")

        self.scaler = self._get_scaler()
        self.scaler.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the data using the fitted scaler.
        """
        X_is_dataframe = isinstance(X, pd.DataFrame)
        if X_is_dataframe:
            X_columns = X.columns
            X_index = X.index

        X = self._validate_data(X)

        if self.imputer:
            X = self.imputer.transform(X)

        if np.isnan(X.data if sparse.issparse(X) else X).any():
            raise ValueError("Input data contains NaN values. Transformation cannot proceed.")

        X_scaled = self.scaler.transform(X)

        if X_is_dataframe:
            X_scaled = pd.DataFrame(X_scaled, index=X_index, columns=X_columns)

        return X_scaled

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Undo the scaling transformation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to be inverse transformed.

        Returns
        -------
        np.ndarray or pd.DataFrame of shape (n_samples, n_features)
            Data returned to original scale.

        Raises
        ------
        ValueError
            If the data contains non-numeric values.
        """
        X_is_dataframe = isinstance(X, pd.DataFrame)
        if X_is_dataframe:
            X_columns = X.columns
            X_index = X.index

        X = self._validate_data(X)

        X_inv = self.scaler.inverse_transform(X)

        if X_is_dataframe:
            X_inv = pd.DataFrame(X_inv, index=X_index, columns=X_columns)

        return X_inv

    def _get_scaler(self):
        """
        Return the appropriate scaler instance based on the method.
        """
        if isinstance(self.method, str):
            if self.method == 'standard':
                return StandardScaler(**self.kwargs)
            elif self.method == 'robust':
                return RobustScaler(**self.kwargs)
            elif self.method == 'quantile_uniform':
                return QuantileTransformer(output_distribution='uniform', **self.kwargs)
            elif self.method == 'quantile_normal':
                return QuantileTransformer(output_distribution='normal', **self.kwargs)
            else:
                raise ValueError(f"Unknown scaling method: {self.method}.")
        elif hasattr(self.method, 'fit') and hasattr(self.method, 'transform'):
            return self.method
        else:
            raise ValueError("The provided method must be a string or a valid scaler object.")

    def _validate_data(self, X) -> np.ndarray:
        """
        Validate the input data and ensure it is numeric.
        """
        if sparse.issparse(X):
            if not np.issubdtype(X.dtype, np.number):
                raise ValueError("All elements in the sparse input array must be numeric.")
            return X

        if isinstance(X, pd.DataFrame):
            if not all([np.issubdtype(dtype, np.number) for dtype in X.dtypes]):
                raise ValueError("All columns in the input DataFrame must be numeric.")
            return X.values
        elif isinstance(X, np.ndarray):
            if not np.issubdtype(X.dtype, np.number):
                raise ValueError("All elements in the input array must be numeric.")
            return X
        else:
            X = np.array(X)
            if not np.issubdtype(X.dtype, np.number):
                raise ValueError("All elements in the input array-like structure must be numeric.")
            return X
