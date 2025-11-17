"""
Benchmarking script for SparsePLS

This script evaluates the performance of SparsePLS across various scenarios:
- Different dataset sizes
- Varying levels of noise
- Comparison with other regression methods
- Scalability analysis
- Feature selection effectiveness
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.cross_decomposition import PLSRegression
from sparse_pls import SparsePLS


def benchmark_dataset_size():
    """Benchmark performance across different dataset sizes."""
    print("\n" + "="*60)
    print("BENCHMARK 1: Dataset Size Scalability")
    print("="*60)

    sizes = [100, 200, 500, 1000, 2000]
    n_features = 50
    n_informative = 10

    results = {
        'size': [],
        'fit_time': [],
        'predict_time': [],
        'mse': [],
        'r2': []
    }

    for size in sizes:
        print(f"\nTesting with {size} samples...")

        # Generate data
        X, y = make_regression(
            n_samples=size,
            n_features=n_features,
            n_informative=n_informative,
            noise=0.1,
            random_state=42
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Fit and measure time
        model = SparsePLS(n_components=3, alpha=0.5)

        start_time = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - start_time

        start_time = time.time()
        y_pred = model.predict(X_test)
        predict_time = time.time() - start_time

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results['size'].append(size)
        results['fit_time'].append(fit_time)
        results['predict_time'].append(predict_time)
        results['mse'].append(mse)
        results['r2'].append(r2)

        print(f"  Fit time: {fit_time:.4f}s")
        print(f"  Predict time: {predict_time:.6f}s")
        print(f"  MSE: {mse:.2f}")
        print(f"  R²: {r2:.4f}")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(results['size'], results['fit_time'], 'o-', linewidth=2)
    axes[0, 0].set_xlabel('Dataset Size')
    axes[0, 0].set_ylabel('Fit Time (s)')
    axes[0, 0].set_title('Training Time vs Dataset Size')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(results['size'], results['predict_time'], 'o-', linewidth=2, color='orange')
    axes[0, 1].set_xlabel('Dataset Size')
    axes[0, 1].set_ylabel('Prediction Time (s)')
    axes[0, 1].set_title('Prediction Time vs Dataset Size')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(results['size'], results['mse'], 'o-', linewidth=2, color='red')
    axes[1, 0].set_xlabel('Dataset Size')
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].set_title('MSE vs Dataset Size')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(results['size'], results['r2'], 'o-', linewidth=2, color='green')
    axes[1, 1].set_xlabel('Dataset Size')
    axes[1, 1].set_ylabel('R² Score')
    axes[1, 1].set_title('R² Score vs Dataset Size')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('benchmark_dataset_size.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved: benchmark_dataset_size.png")

    return pd.DataFrame(results)


def benchmark_noise_levels():
    """Benchmark performance across different noise levels."""
    print("\n" + "="*60)
    print("BENCHMARK 2: Robustness to Noise")
    print("="*60)

    noise_levels = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    n_samples = 500
    n_features = 50
    n_informative = 10

    results = {
        'noise': [],
        'mse': [],
        'r2': [],
        'selected_features': []
    }

    for noise in noise_levels:
        print(f"\nTesting with noise level {noise}...")

        # Generate data
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            noise=noise,
            random_state=42
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Fit model
        model = SparsePLS(n_components=3, alpha=0.5)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        n_selected = len(model.selected_variables_)

        results['noise'].append(noise)
        results['mse'].append(mse)
        results['r2'].append(r2)
        results['selected_features'].append(n_selected)

        print(f"  MSE: {mse:.2f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Selected features: {n_selected}/{n_features}")

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(results['noise'], results['mse'], 'o-', linewidth=2, color='red')
    axes[0].set_xlabel('Noise Level')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('MSE vs Noise Level')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(results['noise'], results['r2'], 'o-', linewidth=2, color='green')
    axes[1].set_xlabel('Noise Level')
    axes[1].set_ylabel('R² Score')
    axes[1].set_title('R² Score vs Noise Level')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(results['noise'], results['selected_features'], 'o-', linewidth=2, color='blue')
    axes[2].set_xlabel('Noise Level')
    axes[2].set_ylabel('Number of Selected Features')
    axes[2].set_title('Feature Selection vs Noise Level')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('benchmark_noise_levels.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved: benchmark_noise_levels.png")

    return pd.DataFrame(results)


def benchmark_model_comparison():
    """Compare SparsePLS with other regression models."""
    print("\n" + "="*60)
    print("BENCHMARK 3: Model Comparison")
    print("="*60)

    # Generate data
    X, y = make_regression(
        n_samples=500,
        n_features=50,
        n_informative=10,
        noise=0.5,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define models
    models = {
        'SparsePLS': SparsePLS(n_components=3, alpha=0.5),
        'Standard PLS': PLSRegression(n_components=3),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.5),
        'ElasticNet': ElasticNet(alpha=0.5, l1_ratio=0.5)
    }

    results = {
        'model': [],
        'cv_score_mean': [],
        'cv_score_std': [],
        'test_mse': [],
        'test_r2': [],
        'fit_time': [],
        'n_selected': []
    }

    print("\nRunning 5-fold cross-validation...\n")

    for name, model in models.items():
        print(f"Testing {name}...")

        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=5,
            scoring='neg_mean_squared_error'
        )

        # Fit and test
        start_time = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - start_time

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Count selected features (if applicable)
        if hasattr(model, 'selected_variables_'):
            n_selected = len(model.selected_variables_)
        elif hasattr(model, 'coef_'):
            n_selected = np.sum(np.abs(model.coef_) > 1e-5)
        else:
            n_selected = X.shape[1]

        results['model'].append(name)
        results['cv_score_mean'].append(-cv_scores.mean())
        results['cv_score_std'].append(cv_scores.std())
        results['test_mse'].append(mse)
        results['test_r2'].append(r2)
        results['fit_time'].append(fit_time)
        results['n_selected'].append(n_selected)

        print(f"  CV MSE: {-cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")
        print(f"  Test MSE: {mse:.2f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Fit time: {fit_time:.4f}s")
        print(f"  Features used: {n_selected}/{X.shape[1]}")
        print()

    df_results = pd.DataFrame(results)

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # CV MSE comparison
    axes[0, 0].barh(df_results['model'], df_results['cv_score_mean'],
                     xerr=df_results['cv_score_std'], capsize=5)
    axes[0, 0].set_xlabel('Cross-Validation MSE')
    axes[0, 0].set_title('Cross-Validation Performance')
    axes[0, 0].grid(True, alpha=0.3, axis='x')

    # Test MSE comparison
    axes[0, 1].barh(df_results['model'], df_results['test_mse'], color='orange')
    axes[0, 1].set_xlabel('Test MSE')
    axes[0, 1].set_title('Test Set Performance')
    axes[0, 1].grid(True, alpha=0.3, axis='x')

    # R² comparison
    axes[1, 0].barh(df_results['model'], df_results['test_r2'], color='green')
    axes[1, 0].set_xlabel('R² Score')
    axes[1, 0].set_title('R² Score Comparison')
    axes[1, 0].grid(True, alpha=0.3, axis='x')

    # Feature selection comparison
    axes[1, 1].barh(df_results['model'], df_results['n_selected'], color='purple')
    axes[1, 1].set_xlabel('Number of Features Used')
    axes[1, 1].set_title('Feature Selection Efficiency')
    axes[1, 1].axvline(x=X.shape[1], color='red', linestyle='--',
                       label='Total features', linewidth=1)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('benchmark_model_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Plot saved: benchmark_model_comparison.png")

    return df_results


def benchmark_feature_dimensionality():
    """Benchmark performance across different feature dimensionalities."""
    print("\n" + "="*60)
    print("BENCHMARK 4: Feature Dimensionality")
    print("="*60)

    n_samples = 500
    feature_counts = [10, 20, 50, 100, 200]

    results = {
        'n_features': [],
        'n_selected': [],
        'mse': [],
        'r2': [],
        'fit_time': []
    }

    for n_features in feature_counts:
        n_informative = min(10, n_features // 2)
        print(f"\nTesting with {n_features} features ({n_informative} informative)...")

        # Generate data
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            noise=0.5,
            random_state=42
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Fit model
        model = SparsePLS(n_components=3, alpha=0.5)

        start_time = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - start_time

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        n_selected = len(model.selected_variables_)

        results['n_features'].append(n_features)
        results['n_selected'].append(n_selected)
        results['mse'].append(mse)
        results['r2'].append(r2)
        results['fit_time'].append(fit_time)

        print(f"  Selected: {n_selected}/{n_features} features")
        print(f"  MSE: {mse:.2f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Fit time: {fit_time:.4f}s")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(results['n_features'], results['n_selected'], 'o-', linewidth=2)
    axes[0, 0].plot(results['n_features'], results['n_features'], '--',
                    color='gray', label='All features', linewidth=1)
    axes[0, 0].set_xlabel('Total Features')
    axes[0, 0].set_ylabel('Selected Features')
    axes[0, 0].set_title('Feature Selection Rate')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(results['n_features'], results['fit_time'], 'o-',
                    linewidth=2, color='orange')
    axes[0, 1].set_xlabel('Number of Features')
    axes[0, 1].set_ylabel('Fit Time (s)')
    axes[0, 1].set_title('Training Time vs Feature Count')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(results['n_features'], results['mse'], 'o-',
                    linewidth=2, color='red')
    axes[1, 0].set_xlabel('Number of Features')
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].set_title('MSE vs Feature Count')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(results['n_features'], results['r2'], 'o-',
                    linewidth=2, color='green')
    axes[1, 1].set_xlabel('Number of Features')
    axes[1, 1].set_ylabel('R² Score')
    axes[1, 1].set_title('R² Score vs Feature Count')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('benchmark_feature_dimensionality.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved: benchmark_feature_dimensionality.png")

    return pd.DataFrame(results)


def benchmark_real_data():
    """Benchmark on real-world dataset (diabetes)."""
    print("\n" + "="*60)
    print("BENCHMARK 5: Real-World Dataset (Diabetes)")
    print("="*60)

    # Load dataset
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target

    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Test different alpha values
    alphas = [0.1, 0.3, 0.5, 0.7, 1.0]

    results = {
        'alpha': [],
        'mse': [],
        'r2': [],
        'n_selected': []
    }

    print("\nTesting different alpha values...\n")

    for alpha in alphas:
        model = SparsePLS(n_components=3, alpha=alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        n_selected = len(model.selected_variables_)

        results['alpha'].append(alpha)
        results['mse'].append(mse)
        results['r2'].append(r2)
        results['n_selected'].append(n_selected)

        print(f"Alpha = {alpha:.1f}")
        print(f"  MSE: {mse:.2f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Selected: {n_selected}/{X.shape[1]} features")
        print()

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(results['alpha'], results['mse'], 'o-', linewidth=2, color='red')
    axes[0].set_xlabel('Alpha (Sparsity Parameter)')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('MSE vs Sparsity Level')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(results['alpha'], results['r2'], 'o-', linewidth=2, color='green')
    axes[1].set_xlabel('Alpha (Sparsity Parameter)')
    axes[1].set_ylabel('R² Score')
    axes[1].set_title('R² Score vs Sparsity Level')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(results['alpha'], results['n_selected'], 'o-', linewidth=2, color='blue')
    axes[2].set_xlabel('Alpha (Sparsity Parameter)')
    axes[2].set_ylabel('Number of Selected Features')
    axes[2].set_title('Feature Selection vs Sparsity Level')
    axes[2].axhline(y=X.shape[1], color='gray', linestyle='--',
                    label='Total features', linewidth=1)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('benchmark_real_data.png', dpi=300, bbox_inches='tight')
    print("✓ Plot saved: benchmark_real_data.png")

    return pd.DataFrame(results)


def main():
    """Run all benchmarks."""
    print("\n" + "="*60)
    print("SparsePLS Performance Benchmarks")
    print("="*60)
    print("\nRunning comprehensive benchmarks...")
    print("This may take a few minutes.\n")

    start_time = time.time()

    # Run benchmarks
    df1 = benchmark_dataset_size()
    df2 = benchmark_noise_levels()
    df3 = benchmark_model_comparison()
    df4 = benchmark_feature_dimensionality()
    df5 = benchmark_real_data()

    # Save results to CSV
    df1.to_csv('benchmark_dataset_size.csv', index=False)
    df2.to_csv('benchmark_noise_levels.csv', index=False)
    df3.to_csv('benchmark_model_comparison.csv', index=False)
    df4.to_csv('benchmark_feature_dimensionality.csv', index=False)
    df5.to_csv('benchmark_real_data.csv', index=False)

    total_time = time.time() - start_time

    print("\n" + "="*60)
    print("Benchmarks Complete!")
    print("="*60)
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print("\nGenerated files:")
    print("  - benchmark_dataset_size.png / .csv")
    print("  - benchmark_noise_levels.png / .csv")
    print("  - benchmark_model_comparison.png / .csv")
    print("  - benchmark_feature_dimensionality.png / .csv")
    print("  - benchmark_real_data.png / .csv")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
