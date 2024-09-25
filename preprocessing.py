import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer

class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Data Preprocessor that supports multiple scaling methods.
    
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
    """
    def __init__(self, method='standard', **kwargs):
        self.method = method
        self.kwargs = kwargs
        self.scaler = None

    def fit(self, X, y=None):
        """
        Fit the scaler to the data.
        """
        X = self._validate_data(X)
        self.scaler = self._get_scaler()
        self.scaler.fit(X)
        return self

    def transform(self, X):
        """
        Transform the data using the fitted scaler.
        """
        X_is_dataframe = isinstance(X, pd.DataFrame)
        if X_is_dataframe:
            X_columns = X.columns
            X_index = X.index
        X = self._validate_data(X)
        X_scaled = self.scaler.transform(X)
        if X_is_dataframe:
            X_scaled = pd.DataFrame(X_scaled, index=X_index, columns=X_columns)
        return X_scaled

    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it.
        """
        return self.fit(X, y).transform(X)

    def _get_scaler(self):
        """
        Return the appropriate scaler based on the method.
        """
        if self.method == 'standard':
            return StandardScaler(**self.kwargs)
        elif self.method == 'robust':
            return RobustScaler(**self.kwargs)
        elif self.method == 'quantile_uniform':
            return QuantileTransformer(output_distribution='uniform', **self.kwargs)
        elif self.method == 'quantile_normal':
            return QuantileTransformer(output_distribution='normal', **self.kwargs)
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")

    def _validate_data(self, X):
        """
        Validate the input data and convert it to numpy array if necessary.
        """
        if isinstance(X, pd.DataFrame):
            return X.values
        elif isinstance(X, np.ndarray):
            return X
        else:
            return np.array(X)
