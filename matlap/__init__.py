"""matlap: Bayesian matrix denoising via CAVI with a Matrix Laplace prior."""

from .core import CAVIResult, GridResult, matlap, matlap_grid

__all__ = ["matlap", "matlap_grid", "CAVIResult", "GridResult"]
