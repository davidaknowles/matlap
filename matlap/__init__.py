"""matlap: Bayesian matrix denoising via CAVI with a Matrix Laplace prior."""

from .faem import FAEMResult, GradMLResult, matlap_faem, matlap_gradml
from .core import (
    CAVIResult, BatchedCAVIResult, GridResult,
    LowRankCAVIResult, LowRankGridResult,
    LowRankIsotropicResult, LowRankIsotropicGridResult,
    matlap, matlap_batched, matlap_batched_warmstart, matlap_grid,
    matlap_grid_lowrank, matlap_grid_lowrank_isotropic,
    matlap_lowrank, matlap_lowrank_isotropic,
    matlap_adaptive_lowrank_isotropic,
    matlap_adaptive_lowrank,
)
from .proximal import ProximalResult, proximal_cv, proximal_gradient
from .vi import VIResult, fit_vi
from .cv import cv_lambda, cv_score_single, make_cv_scorer
from .scoring import (
    closed_form_loo, renyi_elbo, compute_iso_prior_var,
    make_elbo_scorer, make_loo_scorer, make_renyi_scorer,
)
from .adaptive import adaptive_lambda_search, iso_warm_state, lowrank_warm_state

__all__ = [
    # FA EM and gradient marginal likelihood
    "matlap_faem",
    "matlap_gradml",
    "FAEMResult",
    "GradMLResult",
    # CAVI
    "matlap",
    "matlap_batched",
    "matlap_batched_warmstart",
    "matlap_grid",
    "matlap_grid_lowrank",
    "matlap_grid_lowrank_isotropic",
    "matlap_adaptive_lowrank_isotropic",
    "matlap_adaptive_lowrank",
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
    "cv_score_single",
    "make_cv_scorer",
    # Post-hoc scoring
    "closed_form_loo",
    "renyi_elbo",
    "compute_iso_prior_var",
    "make_elbo_scorer",
    "make_loo_scorer",
    "make_renyi_scorer",
    # Adaptive search primitives
    "adaptive_lambda_search",
    "iso_warm_state",
    "lowrank_warm_state",
]
