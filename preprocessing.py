import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer

class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Data Preprocessor that supports multiple scaling methods.

    This class provides a flexible interface for preprocessing data using different scaling methods.
    It implements the scikit-learn Transformer API, allowing it to be used seamlessly in pipelines.

    Parameters
    ----------
    method : str, default='standard'
        Scaling method to use. Options are:
        - 'standard': StandardScaler (mean=0, std=1)
        - 'robust': RobustScaler (median centering, scaling by IQR)
        - 'quantile_uniform': QuantileTransformer with uniform output
        - 'quantile_normal': QuantileTransformer with normal output

    **kwargs : dict
        Additional keyword arguments passed to the scaler.

    Attributes
    ----------
    method : str
        The scaling method used.
    scaler : object
        The scaler instance used for transformation.
    kwargs : dict
        Additional arguments for the scaler.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> preprocessor = DataPreprocessor(method='robust')
    >>> X_scaled = preprocessor.fit_transform(X)
    """

    def __init__(self, method='standard', **kwargs):
        self.method = method
        self.kwargs = kwargs
        self.scaler = None  # Will hold the scaler instance after fitting

    def fit(self, X, y=None):
        """
        Fit the scaler to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data to be scaled.

        y : None
            Ignored. This parameter exists for compatibility with the scikit-learn Transformer API.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Validate and convert input data
        X = self._validate_data(X)
        # Get the appropriate scaler based on the method
        self.scaler = self._get_scaler()
        # Fit the scaler to the data
        self.scaler.fit(X)
        return self

    def transform(self, X):
        """
        Transform the data using the fitted scaler.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to be transformed.

        Returns
        -------
        X_scaled : ndarray or DataFrame of shape (n_samples, n_features)
            Scaled data. If the input X was a pandas DataFrame, the output will also be a DataFrame
            with the same indices and column names.
        """
        # Check if the input is a pandas DataFrame to preserve its structure
        X_is_dataframe = isinstance(X, pd.DataFrame)
        if X_is_dataframe:
            X_columns = X.columns  # Preserve column names
            X_index = X.index      # Preserve index

        # Validate and convert input data
        X = self._validate_data(X)
        # Transform the data using the fitted scaler
        X_scaled = self.scaler.transform(X)

        if X_is_dataframe:
            # Convert the scaled data back to DataFrame to preserve structure
            X_scaled = pd.DataFrame(X_scaled, index=X_index, columns=X_columns)

        return X_scaled

    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to fit and transform.

        y : None
            Ignored. This parameter exists for compatibility with the scikit-learn Transformer API.

        Returns
        -------
        X_scaled : ndarray or DataFrame of shape (n_samples, n_features)
            Scaled data.
        """
        # Fit the scaler to the data, then transform it
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X):
        """
        Undo the scaling transformation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to be inverse transformed.

        Returns
        -------
        X_inv : ndarray or DataFrame of shape (n_samples, n_features)
            Data returned to original scale.
        """
        # Check if the input is a pandas DataFrame to preserve its structure
        X_is_dataframe = isinstance(X, pd.DataFrame)
        if X_is_dataframe:
            X_columns = X.columns  # Preserve column names
            X_index = X.index      # Preserve index

        # Validate and convert input data
        X = self._validate_data(X)
        # Inverse transform the data using the fitted scaler
        X_inv = self.scaler.inverse_transform(X)

        if X_is_dataframe:
            # Convert the data back to DataFrame to preserve structure
            X_inv = pd.DataFrame(X_inv, index=X_index, columns=X_columns)

        return X_inv

    def _get_scaler(self):
        """
        Return the appropriate scaler instance based on the method.

        Returns
        -------
        scaler : object
            An instance of the scaler corresponding to the specified method.

        Raises
        ------
        ValueError
            If an unknown scaling method is specified.
        """
        # Select the scaler based on the specified method
        if self.method == 'standard':
            return StandardScaler(**self.kwargs)
        elif self.method == 'robust':
            return RobustScaler(**self.kwargs)
        elif self.method == 'quantile_uniform':
            return QuantileTransformer(output_distribution='uniform', **self.kwargs)
        elif self.method == 'quantile_normal':
            return QuantileTransformer(output_distribution='normal', **self.kwargs)
        else:
            # Raise an error if the method is not recognized
            raise ValueError(f"Unknown scaling method: {self.method}")

    def _validate_data(self, X):
        """
        Validate the input data and convert it to a numpy array if necessary.

        Parameters
        ----------
        X : array-like
            Input data to validate.

        Returns
        -------
        X_validated : ndarray
            Validated data as a numpy array.

        Notes
        -----
        This method ensures that the input data is in a suitable format for processing by the scaler.
        """
        # Check the type of X and convert accordingly
        if isinstance(X, pd.DataFrame):
            # Extract values from DataFrame
            return X.values
        elif isinstance(X, np.ndarray):
            # Input is already a numpy array
            return X
        else:
            # Convert other array-like structures to numpy array
            return np.array(X)
