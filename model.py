import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y
from sklearn.preprocessing import StandardScaler

class SparsePLS(BaseEstimator, TransformerMixin, RegressorMixin):
    """
    Sparse Partial Least Squares (Sparse PLS) Regression.

    Parameters
    ----------
    n_components : int, default=2
        Number of components to keep.

    alpha : float, default=1.0
        Regularization parameter controlling sparsity.

    max_iter : int, default=500
        Maximum number of iterations in the iterative algorithm.

    tol : float, default=1e-06
        Tolerance for the stopping condition.

    scale : bool, default=True
        Whether to scale X and Y.

    Attributes
    ----------
    x_weights_ : array-like of shape (n_features, n_components)
        Weights for X.

    y_weights_ : array-like of shape (n_targets, n_components)
        Weights for Y.

    x_scores_ : array-like of shape (n_samples, n_components)
        Scores for X.

    y_scores_ : array-like of shape (n_samples, n_components)
        Scores for Y.

    x_loadings_ : array-like of shape (n_features, n_components)
        Loadings for X.

    y_loadings_ : array-like of shape (n_targets, n_components)
        Loadings for Y.

    coef_ : array-like of shape (n_features, n_targets)
        Coefficients of the regression model.

    Examples
    --------
    >>> from sparse_pls import SparsePLS
    >>> model = SparsePLS(n_components=2, alpha=0.1)
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    """

    def __init__(self, n_components=2, alpha=1.0, max_iter=500, tol=1e-6, scale=True):
        self.n_components = n_components
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.scale = scale

    def fit(self, X, Y):
        """
        Fit the model to data matrix X and target(s) Y.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, Y = check_X_y(X, Y, multi_output=True, y_numeric=True)
        n_samples, n_features = X.shape
        n_targets = Y.shape[1] if Y.ndim > 1 else 1

        if self.scale:
            self._x_scaler = StandardScaler()
            self._y_scaler = StandardScaler()
            X = self._x_scaler.fit_transform(X)
            Y = self._y_scaler.fit_transform(Y)
        else:
            self._x_scaler = None
            self._y_scaler = None

        self.x_weights_ = np.zeros((n_features, self.n_components))
        self.y_weights_ = np.zeros((n_targets, self.n_components))
        self.x_scores_ = np.zeros((n_samples, self.n_components))
        self.y_scores_ = np.zeros((n_samples, self.n_components))
        self.x_loadings_ = np.zeros((n_features, self.n_components))
        self.y_loadings_ = np.zeros((n_targets, self.n_components))

        X_residual = X.copy()
        Y_residual = Y.copy()

        for k in range(self.n_components):
            w, c = self._nipals_sparsity(X_residual, Y_residual)
            t = X_residual @ w
            u = Y_residual @ c

            p = X_residual.T @ t / (t.T @ t)
            q = Y_residual.T @ t / (t.T @ t)

            X_residual -= np.outer(t, p)
            Y_residual -= np.outer(t, q)

            self.x_weights_[:, k] = w.ravel()
            self.y_weights_[:, k] = c.ravel()
            self.x_scores_[:, k] = t.ravel()
            self.y_scores_[:, k] = u.ravel()
            self.x_loadings_[:, k] = p.ravel()
            self.y_loadings_[:, k] = q.ravel()

        self.coef_ = self.x_weights_ @ np.linalg.pinv(self.x_scores_.T @ self.x_scores_) @ self.x_scores_.T @ Y

        return self

    def transform(self, X):
        """
        Apply the dimension reduction learned on the train data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data.

        Returns
        -------
        X_new : array-like of shape (n_samples, n_components)
            Transformed data.
        """
        check_is_fitted(self)
        X = check_array(X)
        if self.scale and self._x_scaler is not None:
            X = self._x_scaler.transform(X)
        X_scores = X @ self.x_weights_
        return X_scores

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
        X_scores : array-like of shape (n_samples, n_components)
            Transformed training data.
        """
        self.fit(X, Y)
        return self.x_scores_

    def predict(self, X):
        """
        Predict target values for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        Y_pred : array-like of shape (n_samples,) or (n_samples, n_targets)
            Predicted values.
        """
        check_is_fitted(self)
        X = check_array(X)
        if self.scale and self._x_scaler is not None:
            X = self._x_scaler.transform(X)
        Y_pred = X @ self.coef_
        if self.scale and self._y_scaler is not None:
            Y_pred = self._y_scaler.inverse_transform(Y_pred)
        return Y_pred

    def _nipals_sparsity(self, X, Y):
        """
        Internal method implementing the NIPALS algorithm with sparsity.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data matrix.

        Y : array-like of shape (n_samples, n_targets)
            Target matrix.

        Returns
        -------
        w : array-like of shape (n_features, 1)
            Weight vector for X.

        c : array-like of shape (n_targets, 1)
            Weight vector for Y.
        """
        n_features = X.shape[1]
        n_targets = Y.shape[1] if Y.ndim > 1 else 1

        # Initialize random weight vector for Y
        c = np.random.rand(n_targets, 1)
        c /= np.linalg.norm(c)

        for iteration in range(self.max_iter):
            w = X.T @ Y @ c
            w = self._soft_thresholding(w, self.alpha)
            if np.linalg.norm(w) == 0:
                break
            w /= np.linalg.norm(w)

            t = X @ w
            t_norm = np.linalg.norm(t)
            if t_norm == 0:
                break
            t /= t_norm

            c_new = Y.T @ t
            c_new = self._soft_thresholding(c_new, self.alpha)
            if np.linalg.norm(c_new) == 0:
                break
            c_new /= np.linalg.norm(c_new)

            # Check for convergence
            if np.linalg.norm(c_new - c) < self.tol:
                break
            c = c_new

        return w, c

    def _soft_thresholding(self, z, alpha):
        """
        Apply soft thresholding to vector z.

        Parameters
        ----------
        z : array-like
            Input vector.

        alpha : float
            Thresholding parameter.

        Returns
        -------
        z_thresh : array-like
            Thresholded vector.
        """
        return np.sign(z) * np.maximum(np.abs(z) - alpha, 0)
