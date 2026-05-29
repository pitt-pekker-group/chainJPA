"""rf_squid_chain.py — RF-SQUID chain construction and parametric-sweep job runner.

What's in here
--------------
- ``make_rf_squid_chain``: build a CircuitNetwork for an N-SQUID chain with
  per-SQUID parasitic inductance and a series parasitic inductor.
- ``BaseRealization``: dataclass holding the pre-multiplier parameter set
  for a single disorder realization (some fields are per-SQUID arrays,
  others are scalars).
- ``make_realizations``: draw N realizations from independent normals
  with caller-supplied means and per-field standard deviations.
- ``Job``: dataclass holding one parameter sweep point (a realization
  scaled by multipliers ``im``, ``cm``, ``lm``, ``pm``).
- ``make_job``: build a Job from a BaseRealization plus multipliers.
- ``run_job_res_freq``: compute the resonant frequency for a job.
- ``run_job``: run the pump-amplitude search for a job (takes the
  simulation settings explicitly so it can be used in parallel).

Typical usage
-------------

    from functools import partial
    from joblib import Parallel, delayed
    from rf_squid_chain import (
        make_realizations, make_job, run_job,
    )

    bases = make_realizations(mean, sigma, n_realizations=10)
    jobs  = [
        make_job(b, di, im=im, cm=cm, lm=lm, pm=pm)
        for di, b in enumerate(bases)
        for im in im_values
        for cm in cm_values
        for lm in lm_values
        for pm in pm_values
    ]

    run = partial(run_job,
                  warmup_time = warmupTime,
                  omega_p     = omega_p,
                  nc_pump     = nc_Pump,
                  phi_p       = phi_p,
                  nc_signal   = nc_Signal)

    results = Parallel(n_jobs=-1)(delayed(run)(j) for j in jobs)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from circuit_network import CircuitNetwork, CPR
from josephson_gain import find_resonant_freq, find_pump_amplitude


__all__ = [
    "make_rf_squid_chain",
    "BaseRealization",
    "make_realizations",
    "Job",
    "make_job",
    "run_job_res_freq",
    "run_job",
    "DEFAULT_GAIN_KWARGS",
]


# Default solver settings for the gain-chain ODE integrator. Override via
# the ``gain_kwargs`` argument of ``run_job``.
DEFAULT_GAIN_KWARGS = {"backend": "numba", "n_substeps": 4}


# ---------------------------------------------------------------------------
# Chain construction
# ---------------------------------------------------------------------------
def make_rf_squid_chain(L1, L1p, L2p, Ic,
                        fluxes=None,
                        node_prefix: str = "n",
                        ground_first: bool = True,
                        spacing: float = 1.0) -> CircuitNetwork:
    """Build a CircuitNetwork containing a chain of N RF-SQUIDs with
    per-SQUID parasitic inductance and a series parasitic inductor at
    the start of the chain.

    Each RF-SQUID is a [Josephson junction || parasitic inductor] in
    parallel with a linear inductor.

    Parameters
    ----------
    L1 : array_like, shape (N,)
        Linear inductance L1_i (henries) of each SQUID, in chain order.
    L1p : array_like, shape (N,)
        Parasitic linear inductance L1p_i (henries) of each SQUID.
    L2p : float
        Series parasitic linear inductance (henries) at the start of the
        chain (between ground and the first chain node).
    Ic : array_like, shape (N,)
        Josephson critical current I_c,i (amperes) of each SQUID.
    fluxes : array_like, shape (N,), optional
        External flux phi_ext at each junction (rad).
    node_prefix : str
        Prefix for generated node display names ("n0", "n1", ...).
    ground_first : bool
        If True, node 0 is created as ground (phase pinned at 0).
    spacing : float
        Horizontal spacing between consecutive nodes (for visualization).

    Returns
    -------
    CircuitNetwork
        2N+2 nodes, 3N+1 branches.
    """
    L1_arr  = list(L1)
    L1p_arr = list(L1p)
    Ic_arr  = list(Ic)
    flux_arr = None if fluxes is None else list(fluxes)

    if (len(L1_arr) != len(L1p_arr)) or (len(L1_arr) != len(Ic_arr)):
        raise ValueError(
            f"L1, L1p, and Ic must have the same length; got "
            f"len(L1)={len(L1_arr)}, len(L1p)={len(L1p_arr)}, "
            f"len(Ic)={len(Ic_arr)}."
        )
    if flux_arr is not None and len(flux_arr) != len(Ic_arr):
        raise ValueError(
            f"fluxes and Ic must have the same length; got "
            f"len(fluxes)={len(flux_arr)}, len(Ic)={len(Ic_arr)}."
        )
    N = len(L1_arr)
    if N == 0:
        raise ValueError("Need at least one SQUID (arrays are empty).")

    net = CircuitNetwork()
    for i in range(2 * N + 2):
        if i == 0:
            position = (0, 0)
        elif i % 2 == 1:
            position = (i * spacing, 0.0)
        else:
            position = (i * spacing, spacing)
        net.add_node(
            name=f"{node_prefix}{i}",
            is_ground=(ground_first and i == 0),
            position=position,
        )

    net.add_inductor(0, 1, inductance=L2p)
    for i in range(N):
        net.add_inductor(2 * i + 1, 2 * i + 3, inductance=L1_arr[i])
        net.add_inductor(2 * i + 2, 2 * i + 3, inductance=L1p_arr[i])
        if flux_arr is None:
            net.add_josephson_junction(
                2 * i + 1, 2 * i + 2, critical_current=Ic_arr[i]
            )
        else:
            net.add_josephson_junction(
                2 * i + 1, 2 * i + 2,
                critical_current=Ic_arr[i],
                phi_ext=flux_arr[i],
            )
    return net


# ---------------------------------------------------------------------------
# Disorder realizations
# ---------------------------------------------------------------------------
@dataclass
class BaseRealization:
    """Pre-multiplier parameter set for one disorder realization.

    Array fields (``baseL1``, ``baseL1p``, ``baseIc``) carry one value
    per SQUID; the scalar fields apply to the whole chain. ``basephi_ext0``
    may be either scalar (same flux at every junction) or array
    (per-SQUID flux offsets).
    """
    baseL1:       np.ndarray
    baseL1p:      np.ndarray
    baseL2p:      float
    baseIc:       np.ndarray
    basephi_ext0: np.ndarray         # may be scalar; broadcast at use time
    basekappa:    float
    baseC1:       float


def make_realizations(mean: dict, sigma: dict,
                      n_realizations: int = 10,
                      seed: int = 42) -> list[BaseRealization]:
    """Draw N independent ``BaseRealization`` samples from normals
    centered on the given means with the given standard deviations.

    Array-valued mean entries get an independent random draw per element
    (numpy broadcasting). Missing keys in ``sigma`` default to zero
    disorder for that field.

    Parameters
    ----------
    mean : dict
        Maps each field name (``'L1'``, ``'L1p'``, ``'L2p'``, ``'Ic'``,
        ``'phi_ext0'``, ``'kappa'``, ``'C1'``) to its nominal value.
        Array-valued entries (e.g. ``mean['L1']`` of length N) give
        per-SQUID nominals.
    sigma : dict
        Same keys; standard deviations of the normal draws. Missing keys
        default to zero (no disorder on that field).
    n_realizations : int
        Number of independent realizations to draw.
    seed : int
        RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    return [
        BaseRealization(
            baseL1       =        rng.normal(mean["L1"],       sigma.get("L1",       0.0)),
            baseL1p      =        rng.normal(mean["L1p"],      sigma.get("L1p",      0.0)),
            baseL2p      = float( rng.normal(mean["L2p"],      sigma.get("L2p",      0.0))),
            baseIc       =        rng.normal(mean["Ic"],       sigma.get("Ic",       0.0)),
            basephi_ext0 =        rng.normal(mean["phi_ext0"], sigma.get("phi_ext0", 0.0)),
            basekappa    = float( rng.normal(mean["kappa"],    sigma.get("kappa",    0.0))),
            baseC1       = float( rng.normal(mean["C1"],       sigma.get("C1",       0.0))),
        )
        for _ in range(n_realizations)
    ]


# ---------------------------------------------------------------------------
# Per-sweep-point job spec
# ---------------------------------------------------------------------------
@dataclass
class Job:
    """One sweep point: a realization scaled by the four multipliers.

    The multiplier fields (``im``, ``cm``, ``lm``, ``pm``) are kept in
    the dataclass so downstream analysis can reconstruct the sweep grid
    from the results.
    """
    # sweep-grid coordinates
    disorder_index: int
    im:             float   # critical-current multiplier
    cm:             float   # capacitance multiplier
    lm:             float   # inductance multiplier
    pm:             float   # phase / flux multiplier

    # scaled simulation parameters
    L1:       np.ndarray
    L1p:      np.ndarray
    L2p:      float
    Ic:       np.ndarray
    phi_ext0: np.ndarray         # scalar or per-SQUID array
    kappa:    float
    C1:       float


def make_job(base: BaseRealization,
             disorder_index: int,
             im: float = 1.0,
             cm: float = 1.0,
             lm: float = 1.0,
             pm: float = 1.0) -> Job:
    """Apply scaling multipliers to a base realization to produce a Job.

    The mapping is the one suggested by the multiplier names:
        - ``im`` (critical-current multiplier)  scales ``Ic``
        - ``cm`` (capacitance multiplier)        scales ``C1``
        - ``lm`` (inductance multiplier)         scales ``L1``, ``L1p``, ``L2p``
        - ``pm`` (phase multiplier)              scales ``phi_ext0``
        - ``kappa`` is not scaled by any multiplier

    Override this function in your notebook if you use a different
    scaling convention.
    """
    return Job(
        disorder_index = disorder_index,
        im = im, cm = cm, lm = lm, pm = pm,
        L1       = (lm / im) * base.baseL1,
        L1p      = base.baseL1p,
        L2p      = base.baseL2p,
        Ic       = im * base.baseIc,
        phi_ext0 = pm * base.basephi_ext0,
        kappa    = base.basekappa,
        C1       = cm * base.baseC1,
    )


# ---------------------------------------------------------------------------
# Per-job runners
# ---------------------------------------------------------------------------
def _broadcast_fluxes(phi_ext, n: int) -> np.ndarray:
    """Return phi_ext as a length-n numpy array, broadcasting a scalar
    to all junctions if needed. Lets ``phi_ext0`` be stored either as
    a scalar (uniform flux) or as a per-SQUID array."""
    arr = np.asarray(phi_ext, dtype=float)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    return arr


def run_job_res_freq(job: Job):
    """Compute the resonant frequency for ``job``.

    Returns
    -------
    (job, frequencies) where ``frequencies`` is a length-3 array (zeros
    if the CPR solve failed).
    """
    cpr_data = CPR(make_rf_squid_chain, {
        "L1":     job.L1,
        "L1p":    job.L1p,
        "L2p":    job.L2p,
        "Ic":     job.Ic,
        "fluxes": _broadcast_fluxes(job.phi_ext0, len(job.L1)),
    })
    if cpr_data["fail"]:
        return job, np.zeros(3)
    return job, find_resonant_freq(cpr_data["cpr"], job.C1, job.kappa)


def run_job(job: Job,
            warmup_time: float,
            omega_p: float,
            nc_pump: float,
            phi_p: float,
            nc_signal: float,
            *,
            gain_kwargs: dict | None = None,
            verbose: bool = False) -> dict:
    """Run the pump-amplitude search for ``job``.

    Returns a dict with ``'job'``, ``'result'``, and ``'status'`` keys.
    ``'result'`` is ``None`` when the CPR solve fails.

    Parameters
    ----------
    job : Job
        Sweep point to evaluate.
    warmup_time : float
        ODE warmup time (passed to ``find_pump_amplitude``).
    omega_p : float
        Pump angular frequency (rad/s).
    nc_pump, nc_signal : float
        Number of cycles for the pump and signal stages.
    phi_p : float
        Pump phase offset (rad).
    gain_kwargs : dict, optional
        Solver kwargs forwarded to the gain chain integrator. Defaults
        to ``DEFAULT_GAIN_KWARGS`` (numba backend, 4 substeps).
    verbose : bool
        Forwarded to ``find_pump_amplitude``.
    """
    if gain_kwargs is None:
        gain_kwargs = DEFAULT_GAIN_KWARGS

    cpr_data = CPR(make_rf_squid_chain, {
        "L1":     job.L1,
        "L1p":    job.L1p,
        "L2p":    job.L2p,
        "Ic":     job.Ic,
        "fluxes": _broadcast_fluxes(job.phi_ext0, len(job.L1)),
    }, verbose=verbose)
    if cpr_data["fail"]:
        return {"job": job, "result": None, "status": "cpr_failed"}

    res = find_pump_amplitude(
        warmup_time       = warmup_time,
        omega_p           = omega_p,
        nc_pump           = nc_pump,
        phi_p             = phi_p,
        nc_signal         = nc_signal,
        kappa             = job.kappa,
        cpr               = cpr_data["cpr"],
        C1                = job.C1,
        verbose           = verbose,
        gain_chain_kwargs = gain_kwargs,
    )
    return {"job": job, "result": res, "status": res["status"]}