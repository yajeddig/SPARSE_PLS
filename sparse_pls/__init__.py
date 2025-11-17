import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from .model import SparsePLS
from .preprocessing import DataPreprocessor

__version__ = '0.1.3'
__all__ = ["SparsePLS", "DataPreprocessor"]
