"""XLA FFI-based batched LDL^T factorisation.

Loads the pre-compiled CUDA shared library and exposes ``ldlt_xla``, a
function that computes the LDL^T factorisation of a batch of symmetric
matrices inside a ``jax.jit``-compiled function with **no host-device sync
barriers**.  The kernel runs on the XLA stream managed by JAX, allowing
XLA to fuse and overlap it with surrounding computations.

Usage
-----
from matlap.xla_ext.ldlt_xla import ldlt_xla, register_ldlt_xla

register_ldlt_xla()          # call once at import time
L, d = ldlt_xla(A)           # A: (m, r, r) float32, row-major
# A ≈ L @ diag(d) @ L.T  (element-wise over the batch)
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Library registration (done once)
# ---------------------------------------------------------------------------

_SO_PATH = Path(__file__).parent / "_ldlt_kernel.so"


@cache
def register_ldlt_xla() -> None:
    """Load the compiled .so and register the XLA FFI target.

    Safe to call multiple times — the ``@cache`` decorator guarantees the
    library is only loaded and registered once per process.
    """
    import ctypes
    from jax._src import ffi

    lib = ctypes.CDLL(str(_SO_PATH))

    # Wrap the C function pointer as a PyCapsule (name=None matches what
    # JAX's own C++ registrations use).
    PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    PyCapsule_New.restype = ctypes.py_object
    PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    fn_ptr = ctypes.cast(lib.LdltBatchedFfi, ctypes.c_void_p)
    capsule = PyCapsule_New(fn_ptr, None, None)

    ffi.register_ffi_target(
        "ldlt_batched",
        capsule,
        platform="CUDA",
        api_version=1,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ldlt_xla(A: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Batched LDL^T: A ≈ L @ diag(d) @ L.T.

    Parameters
    ----------
    A : jax.Array, shape (m, r, r), float32
        Batch of symmetric matrices (only lower triangle is read, but the
        full matrix should be passed; the kernel reads whatever is there).

    Returns
    -------
    L : jax.Array, shape (m, r, r), float32
        Unit lower-triangular factor.
    d : jax.Array, shape (m, r), float32
        Diagonal factor (pivot values).

    Notes
    -----
    * Runs entirely on-device; no host-device sync is required.
    * Callable inside ``jax.jit``.
    * Small pivots (|dk| < 1e-8) are regularised to ±1e-8.
    """
    from jax._src import ffi

    m, r, r2 = A.shape
    assert r == r2, "A must be square in last two dims"

    L, d = ffi.ffi_call(
        "ldlt_batched",
        [
            jax.ShapeDtypeStruct((m, r, r), jnp.float32),
            jax.ShapeDtypeStruct((m, r), jnp.float32),
        ],
        vmap_method="sequential",
    )(A)
    return L, d


# Register automatically when this module is first imported on GPU.
def _auto_register() -> None:
    try:
        if jax.devices()[0].platform == "gpu" and _SO_PATH.exists():
            register_ldlt_xla()
    except Exception:
        pass  # gracefully degrade if no GPU or .so missing


_auto_register()
