import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.model_selection import ParameterGrid, check_cv
from scipy.linalg import svd
from joblib import Parallel, delayed
from sklearn.metrics import check_scoring
from sklearn.base import is_classifier
import logging
import scipy.sparse as sp
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
    x_weights_ : ndarray of shape (n_features, n_components)
        Weights for X.
    y_weights_ : ndarray of shape (n_targets, n_components)
        Weights for Y.
    x_loadings_ : ndarray of shape (n_features, n_components)
        Loadings for X.
    y_loadings_ : ndarray of shape (n_targets, n_components)
        Loadings for Y.
    x_scores_ : ndarray of shape (n_samples, n_components)
        Scores for X.
    y_scores_ : ndarray of shape (n_samples, n_components)
        Scores for Y.
    coef_ : ndarray of shape (n_features, n_targets)
        Coefficients of the regression model.
    selected_variables_ : ndarray of shape (n_selected_variables,)
        Indices of the selected variables.
    feature_names_in_ : ndarray of shape (n_features,), optional
        Feature names seen during fit.
    cv_results_ : pandas DataFrame
        Cross-validation results, available after calling `optimize_parameters`.

    Notes
    -----
    The algorithm is based on an iterative procedure that alternates between estimating
    sparse weight vectors for X and Y, and updating scores and loadings.
    """

    def __init__(self, n_components=2, alpha=1.0, max_iter=500, tol=1e-6,
                 scale=True, scale_method='standard', **kwargs):
        self.n_components = n_components
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.scale = scale
        self.scale_method = scale_method
        self.scaler_kwargs = kwargs

    def score(self, X, y):
        """
        Calculate R² score for the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            True values for X.

        Returns
        -------
        score : float
            R² score.
        """
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

    def get_feature_importance(self):
        """
        Return feature importance based on the absolute sum of weights.

        Returns
        -------
        feature_importance : ndarray of shape (n_features,)
            Importance of each feature.
        """
        return np.sum(np.abs(self.x_weights_), axis=1)

    def plot_weights(self):
        """
        Plot component weights as a heatmap.

        This method visualizes the weights of the components to give insight into
        feature contributions.
        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.imshow(self.x_weights_, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title('Component Weights')
        plt.xlabel('Components')
        plt.ylabel('Features')
        plt.show()

    def fit(self, X, Y):
        """
        Fit the Sparse PLS model to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        Y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : object
            Fitted model instance.
        """
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
            w, c = self._sparse_pls_component(X_residual, Y_residual)

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

    def _sparse_pls_component(self, X, Y):
        """
        Compute one sparse PLS component.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Residual matrix of X.
        Y : ndarray of shape (n_samples, n_targets)
            Residual matrix of Y.

        Returns
        -------
        w : ndarray of shape (n_features, 1)
            Weight vector for X.
        c : ndarray of shape (n_targets, 1)
            Weight vector for Y.
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

            if self._check_convergence(w_old, w, self.tol):
                logger.info(f"Convergence reached at iteration {iteration}")
                break

            c = c_new

        return w, c

    def _soft_thresholding(self, z, alpha):
        """
        Apply soft thresholding to a vector.

        Parameters
        ----------
        z : ndarray
            Input vector.
        alpha : float
            Thresholding parameter.

        Returns
        -------
        z_thresh : ndarray
            Thresholded vector.
        """
        return np.sign(z) * np.maximum(np.abs(z) - alpha, 0)

    def _check_convergence(self, w_old, w_new, tol):
        """
        Check if the change in weight vectors is below the specified tolerance.

        Parameters
        ----------
        w_old : ndarray
            Previous weight vector.
        w_new : ndarray
            Current weight vector.
        tol : float
            Tolerance for convergence.

        Returns
        -------
        bool
            True if converged, False otherwise.
        """
        change = np.linalg.norm(w_new - w_old)
        return change < tol

    def _regularized_pinv(self, X, reg_param=1e-5):
        """
        Compute the regularized pseudo-inverse of a matrix.

        Parameters
        ----------
        X : ndarray
            Matrix to invert.
        reg_param : float
            Regularization parameter.

        Returns
        -------
        X_pinv : ndarray
            Regularized pseudo-inverse of X.
        """
        U, S, Vt = svd(X, full_matrices=False)
        S_inv = np.array([1/(s + reg_param) for s in S])
        return (Vt.T @ np.diag(S_inv) @ U.T)

    def _get_selected_variables(self):
        """
        Identify the indices of variables with non-zero weights.

        Returns
        -------
        selected_vars : ndarray of shape (n_selected_variables,)
            Indices of the selected variables.
        """
        non_zero_weights = np.any(self.x_weights_ != 0, axis=1)
        return np.where(non_zero_weights)[0]

    def transform(self, X):
        """
        Apply the dimensionality reduction learned on the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data.

        Returns
        -------
        X_scores : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        check_is_fitted(self)
        X = check_array(X)
        if self.scale and self._x_scaler is not None:
            X = self._x_scaler.transform(X)
        return X @ self.x_weights_

    def fit_transform(self, X, Y):
        """
        Fit the model to X and Y and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        Y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        X_scores : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        self.fit(X, Y)
        return self.x_scores_

    def predict(self, X):
        """
        Predict target values for new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        Y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
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

    def optimize_parameters(self, X, Y, param_grid, cv=5, scoring='neg_mean_squared_error',
                            n_jobs=1, verbose=0, return_models=False):
        """
        Optimize hyperparameters using cross-validation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        Y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.
        param_grid : dict
            Dictionary with parameter names (`str`) as keys and lists of parameter settings to try as values.
        cv : int, cross-validation generator or an iterable, default=5
            Determines the cross-validation splitting strategy.
        scoring : str, callable or None, default='neg_mean_squared_error'
            A string (see model evaluation documentation) or a scorer callable object/function with signature
            ``scorer(estimator, X, y)``.
        n_jobs : int, default=1
            Number of jobs to run in parallel. ``-1`` means using all processors.
        verbose : int, default=0
            Controls the verbosity: the higher, the more messages.
        return_models : bool, default=False
            Whether to store the models trained for each parameter combination.

        Returns
        -------
        self : object
            Returns the instance itself with the best parameters found.

        Notes
        -----
        The best parameters are stored in the instance attributes and the model is refitted on the entire dataset.
        """
        X, Y = self._validate_data(X, Y, multi_output=True, y_numeric=True, reset=False)
        cv = check_cv(cv=cv, y=Y, classifier=is_classifier(self))
        scorer = check_scoring(self, scoring=scoring)

        param_list = list(ParameterGrid(param_grid))
        results = []

        def fit_and_score(params, train_idx, test_idx):
            if isinstance(X, pd.DataFrame):
                X_train = X.iloc[train_idx]
                X_test = X.iloc[test_idx]
            else:
                X_train = X[train_idx]
                X_test = X[test_idx]

            if isinstance(Y, (pd.Series, pd.DataFrame)):
                Y_train = Y.iloc[train_idx]
                Y_test = Y.iloc[test_idx]
            else:
                Y_train = Y[train_idx]
                Y_test = Y[test_idx]

            model = clone(self)
            model.set_params(**params)
            model.fit(X_train, Y_train)
            Y_pred = model.predict(X_test)
            score = scorer(model, X_test, Y_test)
            return score, model if return_models else None

        for params in param_list:
            splits = list(cv.split(X, Y))
            parallel = Parallel(n_jobs=n_jobs, verbose=verbose)
            scores_models = parallel(
                delayed(fit_and_score)(params, train_idx, test_idx)
                for train_idx, test_idx in splits
            )
            scores = [sm[0] for sm in scores_models]
            models = [sm[1] for sm in scores_models] if return_models else None

            mean_score = np.mean(scores)
            std_score = np.std(scores)
            result = {
                'params': params,
                'mean_score': mean_score,
                'std_score': std_score,
                'scores': scores
            }
            if return_models:
                result['models'] = models

            results.append(result)

        self.cv_results_ = pd.DataFrame(results)
        best_result = max(results, key=lambda x: x['mean_score'])
        best_params = best_result['params']
        self.set_params(**best_params)

        if verbose > 0:
            logger.info(f"Best parameters found: {best_params}")

        self.fit(X, Y)
        return self

    def plot_convergence(self, convergence_history):
        """
        Plot convergence of the weight vectors during the iterative process.

        Parameters
        ----------
        convergence_history : list of float
            History of changes in weight norms.
        """
        import matplotlib.pyplot as plt
        plt.plot(convergence_history)
        plt.title("Convergence of Weight Norms")
        plt.xlabel("Iteration")
        plt.ylabel("Change in Norm")
        plt.show()

    def plot_selected_features(self):
        """
        Visualize the selected features based on non-zero weights.
        """
        import matplotlib.pyplot as plt
        selected_features = self.selected_variables_
        plt.bar(selected_features, self.get_feature_importance()[selected_features])
        plt.title("Feature Importance of Selected Features")
        plt.xlabel("Feature Index")
        plt.ylabel("Importance")
        plt.show()
