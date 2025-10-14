"""
Trader Analysis Package

A research-grade framework for statistical evaluation and segmentation of crypto wallet addresses.
"""

__version__ = "1.0.0"
__author__ = "MoonCraze"

from .feature_engineering import FeatureEngineer
from .clustering import TraderSegmentation
from .prediction import HighPotentialPredictor
from .personas import PersonaAssigner
from .evaluation import ModelEvaluator

__all__ = [
    'FeatureEngineer',
    'TraderSegmentation',
    'HighPotentialPredictor',
    'PersonaAssigner',
    'ModelEvaluator'
]
