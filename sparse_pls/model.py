import numpy as np
import pandas as pd
import pickle
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.model_selection import ParameterGrid, check_cv
from scipy.linalg import svd
from joblib import Parallel, delayed
from sklearn.metrics import check_scoring
from sklearn.base import is_classifier
import logging
import matplotlib.pyplot as plt
from preprocessing import DataPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SparsePLS(BaseEstimator, RegressorMixin):
    """
    Sparse Partial Least Squares (Sparse PLS) Regression with enhanced functionality.

    The `SparsePLS` class implements a Sparse Partial Least Squares regression model,
    which is a dimensionality reduction technique that incorporates sparsity (variable selection)
    into the Partial Least Squares (PLS) regression. This class is designed to work seamlessly
    with `scikit-learn` pipelines and other machine learning tools.

    Parameters
    ----------
    n_components : int, default=2
        Number of components to keep.
    alpha : float, default=1.0
        Regularization parameter controlling sparsity. Higher values lead to sparser solutions.
    max_iter : int, default=500
        Maximum number of iterations in the iterative algorithm.
    tol : float, default=1e-6
        Tolerance for the stopping condition.
    scale : bool, default=True
        Whether to scale X and Y.
    scale_method : str, default='standard'
        Method used for scaling. Options: 'standard'.
    **kwargs : dict
        Additional keyword arguments for the scaler.

    Attributes
    ----------
    x_weights_ : np.ndarray of shape (n_features, n_components)
        Weights for X.
    y_weights_ : np.ndarray of shape (n_targets, n_components)
        Weights for Y.
    x_loadings_ : np.ndarray of shape (n_features, n_components)
        Loadings for X.
    y_loadings_ : np.ndarray of shape (n_targets, n_components)
        Loadings for Y.
    x_scores_ : np.ndarray of shape (n_samples, n_components)
        Scores for X.
    y_scores_ : np.ndarray of shape (n_samples, n_components)
        Scores for Y.
    coef_ : np.ndarray of shape (n_features, n_targets)
        Coefficients of the regression model.
    selected_variables_ : np.ndarray of shape (n_selected_variables,)
        Indices of the selected variables.
    feature_names_in_ : np.ndarray of shape (n_features,), optional
        Feature names seen during fit.
    cv_results_ : pd.DataFrame
        Cross-validation results, available after calling `optimize_parameters`.

    Notes
    -----
    This class supports multi-output regression. When fitting with multiple targets (Y),
    the model expects Y to be of shape (n_samples, n_targets). The output format of predictions
    will match the input format of Y.
    """

    def __init__(self, n_components: int = 2, alpha: float = 1.0, max_iter: int = 500,
                 tol: float = 1e-6, scale: bool = True, scale_method: str = 'standard', **kwargs):
        self.n_components = n_components
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.scale = scale
        self.scale_method = scale_method
        self.scaler_kwargs = kwargs

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R² score for the model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test data.
        y : np.ndarray of shape (n_samples,) or (n_samples, n_targets)
            True values for X.

        Returns
        -------
        float
            R² score.
        """
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

    def get_feature_importance(self) -> np.ndarray:
        """
        Return feature importance based on the absolute sum of weights.

        Returns
        -------
        np.ndarray of shape (n_features,)
            Importance of each feature.
        """
        return np.sum(np.abs(self.x_weights_), axis=1)

    def plot_weights(self) -> plt.Figure:
        """
        Plot component weights as a heatmap.

        Returns
        -------
        plt.Figure
            The figure object for the plot.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        cax = ax.imshow(self.x_weights_, aspect='auto', cmap='viridis')
        fig.colorbar(cax)
        ax.set_title('Component Weights')
        ax.set_xlabel('Components')
        ax.set_ylabel('Features')
        return fig

    def plot_selected_features(self) -> plt.Figure:
        """
        Visualize the selected features based on non-zero weights.

        Returns
        -------
        plt.Figure
            The figure object for the plot.
        """
        fig, ax = plt.subplots()
        selected_features = self.selected_variables_
        ax.bar(selected_features, self.get_feature_importance()[selected_features])
        ax.set_title("Feature Importance of Selected Features")
        ax.set_xlabel("Feature Index")
        ax.set_ylabel("Importance")
        return fig

    def fit(self, X: np.ndarray, Y: np.ndarray) -> 'SparsePLS':
        """
        Fit the Sparse PLS model to the training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        Y : np.ndarray of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        SparsePLS
            Fitted model instance.

        Raises
        ------
        ValueError
            If input data contains missing values.
        """
        if np.any(pd.isnull(X)) or np.any(pd.isnull(Y)):
            raise ValueError("Input data X and Y must not contain missing values. Please handle missing data before fitting.")

        X, Y = self._validate_data(
            X, Y,
            accept_sparse=False,
            dtype=None,
            force_all_finite=True,
            multi_output=True,
            y_numeric=True,
            reset=True
        )

        if self.scale:
            self._x_scaler = DataPreprocessor(method=self.scale_method, **self.scaler_kwargs)
            self._y_scaler = DataPreprocessor(method=self.scale_method, **self.scaler_kwargs)
            X = self._x_scaler.fit_transform(X)
            Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
            Y = self._y_scaler.fit_transform(Y)
        else:
            self._x_scaler = None
            self._y_scaler = None

        n_samples, n_features = X.shape
        n_targets = Y.shape[1] if Y.ndim > 1 else 1

        self.x_weights_ = np.zeros((n_features, self.n_components))
        self.y_weights_ = np.zeros((n_targets, self.n_components))
        self.x_scores_ = np.zeros((n_samples, self.n_components))
        self.y_scores_ = np.zeros((n_samples, self.n_components))
        self.x_loadings_ = np.zeros((n_features, self.n_components))
        self.y_loadings_ = np.zeros((n_targets, self.n_components))

        X_residual = X.copy()
        Y_residual = Y.copy()

        for k in range(self.n_components):
            w, c = self._compute_sparse_pls_component(X_residual, Y_residual)

            t = X_residual @ w
            u = Y_residual @ c

            t_norm = np.linalg.norm(t)
            if t_norm == 0:
                break
            t /= t_norm
            u /= t_norm

            p = X_residual.T @ t
            q = Y_residual.T @ t

            self.x_weights_[:, k] = w.ravel()
            self.y_weights_[:, k] = c.ravel()
            self.x_scores_[:, k] = t.ravel()
            self.y_scores_[:, k] = u.ravel()
            self.x_loadings_[:, k] = p.ravel()
            self.y_loadings_[:, k] = q.ravel()

            X_residual -= np.outer(t, p)
            Y_residual -= np.outer(t, q)

            logger.info(f"Component {k + 1}: Norm of t = {t_norm:.4f}, Deflation step completed.")

        self.coef_ = self.x_weights_ @ self._regularized_pinv(self.x_loadings_.T @ self.x_weights_) @ self.y_loadings_.T
        self.selected_variables_ = self._get_selected_variables()

        return self

    def _compute_sparse_pls_component(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute one sparse PLS component.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Residual matrix of X.
        Y : np.ndarray of shape (n_samples, n_targets)
            Residual matrix of Y.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Weight vectors for X and Y.
        """
        n_features = X.shape[1]
        n_targets = Y.shape[1] if Y.ndim > 1 else 1

        c = np.random.rand(n_targets, 1)
        c /= np.linalg.norm(c)

        for iteration in range(self.max_iter):
            w_old = np.zeros_like(c)

            z_w = X.T @ Y @ c
            w = self._soft_thresholding(z_w, self.alpha)
            if np.linalg.norm(w) == 0:
                break
            w /= np.linalg.norm(w)

            t = X @ w

            z_c = Y.T @ t
            c_new = self._soft_thresholding(z_c, self.alpha)
            if np.linalg.norm(c_new) == 0:
                break
            c_new /= np.linalg.norm(c_new)

            change_in_weights = np.linalg.norm(c_new - c)
            self._log_iteration_state(iteration, w, c, change_in_weights)

            if change_in_weights < self.tol:
                logger.info(f"Convergence reached at iteration {iteration}.")
                break

            c = c_new

        return w, c

    def _soft_thresholding(self, z: np.ndarray, alpha: float) -> np.ndarray:
        """
        Apply soft thresholding to a vector.

        Parameters
        ----------
        z : np.ndarray
            Input vector.
        alpha : float
            Thresholding parameter.

        Returns
        -------
        np.ndarray
            Thresholded vector.
        """
        return np.sign(z) * np.maximum(np.abs(z) - alpha, 0)

    def _log_iteration_state(self, iteration: int, w: np.ndarray, c: np.ndarray, change_in_weights: float) -> None:
        """
        Log the state of the current iteration for debugging.

        Parameters
        ----------
        iteration : int
            Current iteration number.
        w : np.ndarray
            Current weight vector for X.
        c : np.ndarray
            Current weight vector for Y.
        change_in_weights : float
            Change in weights for checking convergence.
        """
        logger.info(
            f"Iteration {iteration}: Norm of w = {np.linalg.norm(w):.4f}, "
            f"Norm of c = {np.linalg.norm(c):.4f}, Change in weights = {change_in_weights:.6f}"
        )

    def _regularized_pinv(self, X: np.ndarray, reg_param: float = 1e-5) -> np.ndarray:
        """
        Compute the regularized pseudo-inverse of a matrix.

        Parameters
        ----------
        X : np.ndarray
            Matrix to invert.
        reg_param : float
            Regularization parameter.

        Returns
        -------
        np.ndarray
            Regularized pseudo-inverse of X.
        """
        U, S, Vt = svd(X, full_matrices=False)
        S_inv = np.array([1/(s + reg_param) for s in S])
        return (Vt.T @ np.diag(S_inv) @ U.T)

    def _get_selected_variables(self) -> np.ndarray:
        """
        Identify the indices of variables with non-zero weights.

        Returns
        -------
        np.ndarray of shape (n_selected_variables,)
            Indices of the selected variables.
        """
        non_zero_weights = np.any(self.x_weights_ != 0, axis=1)
        return np.where(non_zero_weights)[0]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the dimensionality reduction learned on the training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            New data.

        Returns
        -------
        np.ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        check_is_fitted(self)
        X = check_array(X)
        if self.scale and self._x_scaler is not None:
            X = self._x_scaler.transform(X)
        return X @ self.x_weights_

    def fit_transform(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Fit the model to X and Y and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        Y : np.ndarray of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        np.ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        self.fit(X, Y)
        return self.x_scores_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for new data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        np.ndarray of shape (n_samples,) or (n_samples, n_targets)
            Predicted target values.
        """
        check_is_fitted(self)
        X = check_array(X)
        if self.scale and self._x_scaler is not None:
            X = self._x_scaler.transform(X)
        Y_pred = X @ self.coef_
        if self.scale and self._y_scaler is not None:
            Y_pred = Y_pred.reshape(-1, 1) if Y_pred.ndim == 1 else Y_pred
            Y_pred = self._y_scaler.inverse_transform(Y_pred)
            Y_pred = Y_pred.ravel() if Y_pred.shape[1] == 1 else Y_pred
        return Y_pred
    
    def use_estimator(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the trained estimator to new data and obtain predictions.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        np.ndarray of shape (n_samples,) or (n_samples, n_targets)
            Predicted target values.
        """
        return self.predict(X)

    def export_estimator(self, filepath: str, format: str = 'pickle') -> None:
        """
        Export the trained estimator to a file.

        Parameters
        ----------
        filepath : str
            Path where the model should be saved.
        format : str, default='pickle'
            Format to save the model. Currently supports 'pickle'.
        """
        if format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            logger.info(f"Model saved successfully at {filepath}")
        else:
            raise ValueError("Unsupported format. Only 'pickle' is currently supported.")
        
    def get_selected_feature_names(self) -> list:
        """
        Retrieve the names or indices of the most relevant selected features.

        Returns
        -------
        list
            List of selected feature names if available, otherwise feature indices.
        """
        check_is_fitted(self, "selected_variables_")

        if hasattr(self, "feature_names_in_") and self.feature_names_in_ is not None:
            # Ensure valid indices before retrieving names
            return [str(self.feature_names_in_[i]) for i in self.selected_variables_ if i < len(self.feature_names_in_)]
        else:
            # Ensure indices are returned as Python int
            return [int(i) for i in self.selected_variables_]

