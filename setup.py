from setuptools import setup, find_packages

# Read the contents of README.md for long description
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='sparse-pls',  # The name of your package
    version='0.1.0',  # Start with an initial version
    author='Younes AJEDDIG',
    author_email='younes.ajeddig@live.fr',
    description='Sparse Partial Least Squares (Sparse PLS) Regression',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yajeddig/SPARSE_PLS',  # Replace with your repository URL
    packages=find_packages(),  # Automatically finds packages in the directory
    install_requires=[
        'numpy==1.26.4',
        'pandas==2.2.2',
        'scikit-learn==1.5.1',
        'scipy==1.13.1',
        'joblib==1.4.2'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',  # Minimum Python version requirement
    package_data={
        '': ['documentation/*.md'],  # Include documentation files
    },
    include_package_data=True,
)
