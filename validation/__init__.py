"""
Crypto Agent Forecaster Validation Framework

A comprehensive validation toolkit for testing crypto forecasting models.
"""

__version__ = "0.1.0"
__author__ = "Crypto Agent Forecaster Team"

from .validator import CryptoValidator, ForecastResult, ValidationMetrics
from .analytics import ValidationAnalytics
from .vps_deployment import VPSDeploymentManager, DeploymentConfig

__all__ = [
    "CryptoValidator",
    "ForecastResult", 
    "ValidationMetrics",
    "ValidationAnalytics",
    "VPSDeploymentManager",
    "DeploymentConfig",
] 