"""matlap: Bayesian matrix denoising via CAVI with a Matrix Laplace prior."""

from .core import CAVIResult, GridResult, matlap, matlap_grid
from .proximal import ProximalResult, proximal_cv, proximal_gradient
from .vi import VIResult, fit_vi
from .cv import cv_lambda

__all__ = [
    # CAVI
    "matlap",
    "matlap_grid",
    "CAVIResult",
    "GridResult",
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
