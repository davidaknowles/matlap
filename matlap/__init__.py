"""matlap: Bayesian matrix denoising via CAVI with a Matrix Laplace prior."""

from .faem import FAEMResult, GradMLResult, matlap_faem, matlap_gradml
from .mcmc import MCMCResult, mcmc_proximal_mala, mcmc_gsm_gibbs
from .core import (
    CAVIResult, BatchedCAVIResult, BatchedGridResult, GridResult,
    LowRankCAVIResult, LowRankGridResult,
    LowRankIsotropicResult, LowRankIsotropicGridResult,
    matlap, matlap_batched, matlap_batched_warmstart,
    matlap_iso_warmstart, matlap_iso_renyi_lambda, matlap_grid,
    matlap_grid_lowrank, matlap_grid_lowrank_isotropic, matlap_grid_batched,
    matlap_lowrank, matlap_lowrank_isotropic,
    matlap_adaptive_lowrank_isotropic,
    matlap_adaptive_lowrank,
    matlap_adaptive_batched,
)
from .proximal import ProximalResult, proximal_cv, proximal_gradient
from .taylor import TaylorResult, taylor_cv, taylor_gradient
from .vi import VIResult, fit_vi
from .cv import cv_lambda, cv_score_single, make_cv_scorer
from .scoring import (
    closed_form_loo, data_prior_var, renyi_elbo, renyi_lambda_opt,
    compute_iso_prior_var, make_elbo_scorer, make_loo_scorer, make_renyi_scorer,
)
from .adaptive import adaptive_lambda_search, batched_warm_state, iso_warm_state, lowrank_warm_state
from .simulate import sample_nnd

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
    "matlap_iso_warmstart",
    "matlap_iso_renyi_lambda",
    "matlap_grid",
    "matlap_grid_lowrank",
    "matlap_grid_lowrank_isotropic",
    "matlap_grid_batched",
    "matlap_adaptive_lowrank_isotropic",
    "matlap_adaptive_lowrank",
    "matlap_adaptive_batched",
    "matlap_lowrank",
    "matlap_lowrank_isotropic",
    "CAVIResult",
    "BatchedCAVIResult",
    "BatchedGridResult",
    "GridResult",
    "LowRankCAVIResult",
    "LowRankGridResult",
    "LowRankIsotropicResult",
    "LowRankIsotropicGridResult",
    # Proximal gradient
    "proximal_gradient",
    "proximal_cv",
    "ProximalResult",
    # Taylor-delta proximal gradient
    "taylor_gradient",
    "taylor_cv",
    "TaylorResult",
    # Numpyro SVI
    "fit_vi",
    "VIResult",
    # General CV
    "cv_lambda",
    "cv_score_single",
    "make_cv_scorer",
    # Post-hoc scoring
    "closed_form_loo",
    "data_prior_var",
    "renyi_elbo",
    "renyi_lambda_opt",
    "compute_iso_prior_var",
    "make_elbo_scorer",
    "make_loo_scorer",
    "make_renyi_scorer",
    # MCMC
    "mcmc_proximal_mala",
    "mcmc_gsm_gibbs",
    "MCMCResult",
    "adaptive_lambda_search",
    "batched_warm_state",
    "iso_warm_state",
    "lowrank_warm_state",
    # Data simulation
    "sample_nnd",
]
