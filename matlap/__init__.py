"""matlap: Bayesian matrix denoising via CAVI with a Matrix Laplace prior."""

from .core import (
    CAVIResult, BatchedCAVIResult, GridResult,
    LowRankCAVIResult, LowRankGridResult,
    LowRankIsotropicResult, LowRankIsotropicGridResult,
    matlap, matlap_batched, matlap_grid,
    matlap_grid_lowrank, matlap_grid_lowrank_isotropic,
    matlap_lowrank, matlap_lowrank_isotropic,
)
from .proximal import ProximalResult, proximal_cv, proximal_gradient
from .vi import VIResult, fit_vi
from .cv import cv_lambda
from .scoring import closed_form_loo, renyi_elbo, compute_iso_prior_var

__all__ = [
    # CAVI
    "matlap",
    "matlap_batched",
    "matlap_grid",
    "matlap_grid_lowrank",
    "matlap_grid_lowrank_isotropic",
    "matlap_lowrank",
    "matlap_lowrank_isotropic",
    "CAVIResult",
    "BatchedCAVIResult",
    "GridResult",
    "LowRankCAVIResult",
    "LowRankGridResult",
    "LowRankIsotropicResult",
    "LowRankIsotropicGridResult",
    # Proximal gradient
    "proximal_gradient",
    "proximal_cv",
    "ProximalResult",
    # Numpyro SVI
    "fit_vi",
    "VIResult",
    # General CV
    "cv_lambda",
    # Post-hoc scoring
    "closed_form_loo",
    "renyi_elbo",
    "compute_iso_prior_var",
]
