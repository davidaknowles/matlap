"""matlap: Bayesian matrix denoising via CAVI with a Matrix Laplace prior."""

from .core import CAVIResult, GridResult, LowRankCAVIResult, matlap, matlap_grid, matlap_lowrank
from .proximal import ProximalResult, proximal_cv, proximal_gradient
from .vi import VIResult, fit_vi
from .cv import cv_lambda

__all__ = [
    # CAVI
    "matlap",
    "matlap_grid",
    "matlap_lowrank",
    "CAVIResult",
    "GridResult",
    "LowRankCAVIResult",
    # Proximal gradient
    "proximal_gradient",
    "proximal_cv",
    "ProximalResult",
    # Numpyro SVI
    "fit_vi",
    "VIResult",
    # General CV
    "cv_lambda",
]
