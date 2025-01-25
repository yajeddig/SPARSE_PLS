from setuptools import setup, find_packages

# Lire le contenu de README.md pour la description longue
with open('readme.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='sparse_pls',  # Utilisez un underscore au lieu d'un tiret
    version='0.1.2',  # Mettez à jour le numéro de version
    author='Younes AJEDDIG',
    author_email='younes.ajeddig@live.fr',
    description='Sparse Partial Least Squares (Sparse PLS) Regression',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yajeddig/SPARSE_PLS',  # Remplacez par l'URL de votre dépôt
    packages=find_packages(),  # Trouve automatiquement les packages dans le répertoire
    install_requires=[
        'numpy>=1.20.0',
        'pandas>=1.2.0',
        'scikit-learn>=0.24.0',
        'scipy>=1.6.0',
        'joblib>=1.0.0'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',  # Version minimale de Python requise
    include_package_data=True,
)
