"""
Phase 3: Enhanced Empirical Density Estimation
"""

__version__ = "1.0.0"
__author__ = "Phase 3 Team"
__description__ = "Non-parametric regime detection, tail risk analysis, and multi-asset correlation modeling"

from .SingleAsset.regime_detector import EnhancedRegimeDetector
from .SingleAsset.tail_risk_analyzer import EnhancedTailRiskAnalyzer
from .MultiAsset.dynamic_correlation import DynamicCorrelationEstimator
from .MultiAsset.factor_analysis import PortfolioFactorAnalysis

__all__ = [
    'EnhancedRegimeDetector',
    'EnhancedTailRiskAnalyzer',
    'DynamicCorrelationEstimator',
    'PortfolioFactorAnalysis'
]
