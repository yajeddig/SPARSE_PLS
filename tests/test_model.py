import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sparse_pls.model import SparsePLS

def test_numpy_input():
    # Générer des données de régression synthétiques
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
    
    # Créer une instance du modèle
    model = SparsePLS(n_components=2, alpha=0.1)
    
    # Entraîner le modèle
    model.fit(X, y)
    
    # Vérifier que le modèle a été entraîné
    assert hasattr(model, 'coef_'), "Le modèle n'a pas été entraîné correctement avec numpy.ndarray"

def test_pandas_input():
    # Générer des données de régression synthétiques
    X_np, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
    
    # Convertir X en DataFrame avec des noms de colonnes
    feature_names = [f'feature_{i}' for i in range(X_np.shape[1])]
    X = pd.DataFrame(X_np, columns=feature_names)
    
    # Créer une instance du modèle
    model = SparsePLS(n_components=2, alpha=0.1)
    
    # Entraîner le modèle
    model.fit(X, y)
    
    # Vérifier que le modèle a été entraîné
    assert hasattr(model, 'coef_'), "Le modèle n'a pas été entraîné correctement avec pandas.DataFrame"

def test_column_names_preserved():
    # Générer des données de régression synthétiques
    X_np, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
    
    # Convertir X en DataFrame avec des noms de colonnes
    feature_names = [f'feature_{i}' for i in range(X_np.shape[1])]
    X = pd.DataFrame(X_np, columns=feature_names)
    
    # Créer une instance du modèle
    model = SparsePLS(n_components=2, alpha=0.1)
    
    # Entraîner le modèle
    model.fit(X, y)
    
    # Vérifier que les noms de colonnes sont conservés
    assert model.feature_names_in_ is not None, "Les noms de colonnes n'ont pas été conservés"
    assert list(model.feature_names_in_) == feature_names, "Les noms de colonnes ne correspondent pas"

def test_prediction():
    # Générer des données de régression synthétiques
    X_np, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
    X_test_np, _ = make_regression(n_samples=10, n_features=10, noise=0.1, random_state=24)
    
    # Convertir X en DataFrame avec des noms de colonnes
    feature_names = [f'feature_{i}' for i in range(X_np.shape[1])]
    X = pd.DataFrame(X_np, columns=feature_names)
    X_test = pd.DataFrame(X_test_np, columns=feature_names)
    
    # Créer une instance du modèle
    model = SparsePLS(n_components=2, alpha=0.1)
    
    # Entraîner le modèle
    model.fit(X, y)
    
    # Prédiction avec numpy.ndarray
    y_pred_np = model.predict(X_test_np)
    
    # Prédiction avec pandas.DataFrame
    y_pred_df = model.predict(X_test)
    
    # Vérifier que les prédictions sont les mêmes
    np.testing.assert_array_almost_equal(y_pred_np, y_pred_df, err_msg="Les prédictions diffèrent entre numpy.ndarray et pandas.DataFrame")

def test_invalid_input():
    # Données invalides (types incorrects)
    X_invalid = "invalid_input"
    y_invalid = [1, 2, 3]
    
    # Créer une instance du modèle
    model = SparsePLS(n_components=2, alpha=0.1)
    
    # Vérifier que ValueError est levé pour les entrées invalides
    with pytest.raises(ValueError):
        model.fit(X_invalid, y_invalid)
