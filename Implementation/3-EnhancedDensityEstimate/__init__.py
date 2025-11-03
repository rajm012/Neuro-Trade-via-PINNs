"""
Enhanced Density Estimation Package
====================================

A comprehensive toolkit for volatility-adaptive density estimation,
regime detection, and tail risk modeling for financial returns.

Modules:
--------
- adaptive_kde: Volatility-aware kernel density estimation
- regime_detection: Market regime identification via change points
- tail_risk: Extreme Value Theory for tail risk
- validation: Testing and validation utilities
- utils: Helper functions and utilities
"""

__version__ = "1.0.0"
__author__ = "rajm012"

from .AdaptiveKDE import AdaptiveKDE
from .RegimeDetection import MarketRegimeDetector
from .TailRisk import TailRiskEstimator

__all__ = ['AdaptiveKDE', 'MarketRegimeDetector', 'TailRiskEstimator']

