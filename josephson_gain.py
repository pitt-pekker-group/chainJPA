"""
josephson_gain.py
=================
Simulates a Josephson parametric amplifier described by

    phi'' + K*phi' + f2*J[phi] = 2*K*phi_in'(t)

where f2 = 1 / (phi_0 * C1), with phi_0 = hbar/(2e) the reduced flux quantum
and C1 the shunt capacitance. Computes the power gain at the signal
frequency via DFT analysis.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.constants import h, e
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

PHI_0     = h / (2.0 * e)              # magnetic flux quantum, Wb
PHI_0_RED = PHI_0 / (2.0 * np.pi)      # reduced flux quantum, Phi_0 / (2*pi)


__all__ = [
    "first_cross",
    "compute_gain_chain",
    "find_resonant_freq",
    "find_pump_amplitude",
    "PHI_0",
    "PHI_0_RED",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_phi_in(
    a_s: float, omega_s: float,
    ap: float,  omega_p: float, phi_p: float,
) -> tuple[Callable, Callable]:
    """Return (phi_in, dphi_in) as vectorised callables."""
    def phi_in(t: np.ndarray) -> np.ndarray:
        return a_s * np.sin(omega_s * t) + ap * np.sin(omega_p * t + phi_p)

    def dphi_in(t: np.ndarray) -> np.ndarray:
        return (a_s * omega_s * np.cos(omega_s * t)
                + ap * omega_p * np.cos(omega_p * t + phi_p))

    return phi_in, dphi_in


def _dft_spectrum(signal: np.ndarray, N: int, ns: int) -> np.ndarray:
    """
    Mirrors Mathematica's:
        RotateRight[Sqrt[1/N] * Fourier[signal], ns][[;;2ns]]

    Mathematica Fourier uses 1/sqrt(N) normalisation, then the caller
    applies another 1/sqrt(N), giving a combined 1/N factor total.
    np.fft.fft gives the unnormalised sum, so we divide by N.
    """
    F      = np.fft.fft(signal) / N   # combined 1/N normalisation
    rolled = np.roll(F, ns)            # RotateRight[..., ns]
    return rolled[:2*ns]               # [[;; 2*ns]]


def first_cross(data, lower: float, upper: float) -> tuple[float, float]:
    """
    Walk through (x, y) data and return the (x, y) at which y first
    crosses either `upper` (from below) or `lower` (from above), located
    by interpolating through the leading subset of points.

    Mirrors Mathematica's `firstCross`:
      * 2 points  → linear
      * 3 points  → quadratic
      * 4+ points → cubic spline

    Raises ValueError if the very first point already exceeds the band,
    or if no crossing is found within the data range.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("first_cross expects an (N, 2) array of (x, y) pairs.")

    x_arr, y_arr = data[:, 0], data[:, 1]

    for i in range(len(data)):
        y_i = y_arr[i]
        if y_i >= upper:
            target = upper
        elif y_i <= lower:
            target = lower
        else:
            continue

        if i == 0:
            raise ValueError("first_cross: crossing already at first point.")

        sub_x, sub_y = x_arr[:i+1], y_arr[:i+1]
        if i == 1:                                   # linear (2 pts)
            slope = (sub_y[1] - sub_y[0]) / (sub_x[1] - sub_x[0])
            interp = lambda x: sub_y[0] + slope * (x - sub_x[0])
        elif i == 2:                                  # quadratic (3 pts)
            interp = np.poly1d(np.polyfit(sub_x, sub_y, 2))
        else:                                         # cubic spline (4+ pts)
            interp = CubicSpline(sub_x, sub_y)

        # local bracket [x_{i-1}, x_i] guarantees a sign change
        res_x = brentq(lambda x: interp(x) - target, sub_x[i-1], sub_x[i])
        return float(res_x), float(interp(res_x))

    raise ValueError("first_cross: target band not crossed within data range.")


# ---------------------------------------------------------------------------
# Fast numba-JIT integrator (optional — used when backend='auto' or 'numba')
# ---------------------------------------------------------------------------

if _HAS_NUMBA:

    @njit(cache=True, fastmath=True, inline="always")
    def _cs_eval(x, breaks, coeffs):
        """
        Evaluate scipy CubicSpline coefficients at scalar x.
        coeffs shape (4, n-1); polynomial in segment i is
            c[0,i]*dx^3 + c[1,i]*dx^2 + c[2,i]*dx + c[3,i],   dx = x - breaks[i]
        Outside the data range, scipy's default extrapolates using the
        nearest segment polynomial — we replicate that.
        """
        n = breaks.shape[0]
        if x < breaks[0]:
            i = 0
        elif x >= breaks[-1]:
            i = n - 2
        else:
            lo, hi = 0, n - 1
            while hi - lo > 1:
                mid = (lo + hi) >> 1
                if breaks[mid] <= x:
                    lo = mid
                else:
                    hi = mid
            i = lo
        dx = x - breaks[i]
        return ((coeffs[0, i] * dx + coeffs[1, i]) * dx + coeffs[2, i]) * dx + coeffs[3, i]


    @njit(cache=True, fastmath=True)
    def _rk4_drive(
        t_eval, kappa, f2, breaks, coeffs,
        a_s, omega_s, ap, omega_p, phi_p, n_substeps,
    ):
        """
        Fixed-step RK4 for  phi'' + K*phi' + f2*J(phi) = 2*K*phi_in'(t),
        with phi_in(t) = a_s*sin(omega_s*t) + ap*sin(omega_p*t + phi_p).

        Returns (2, N) array of (phi, dphi) at each t_eval point.
        Assumes uniformly-spaced t_eval (true in compute_gain_chain).
        """
        N    = t_eval.shape[0]
        out  = np.empty((2, N))
        h    = (t_eval[1] - t_eval[0]) / n_substeps   # integration step
        phi  = 0.0
        dphi = 0.0
        t    = 0.0

        # ---- warmup phase: integrate from 0 up to t_eval[0] ----
        if t_eval[0] > 0.0:
            n_warm = max(1, int(np.ceil(t_eval[0] / h)))
            h_w    = t_eval[0] / n_warm
            for _ in range(n_warm):
                # k1
                din = a_s*omega_s*np.cos(omega_s*t) + ap*omega_p*np.cos(omega_p*t + phi_p)
                k1p = dphi
                k1d = 2*kappa*din - kappa*dphi - f2*_cs_eval(phi, breaks, coeffs)
                # k2
                tm = t + 0.5*h_w
                pm = phi + 0.5*h_w*k1p; dm = dphi + 0.5*h_w*k1d
                dinm = a_s*omega_s*np.cos(omega_s*tm) + ap*omega_p*np.cos(omega_p*tm + phi_p)
                k2p = dm
                k2d = 2*kappa*dinm - kappa*dm - f2*_cs_eval(pm, breaks, coeffs)
                # k3 (same time as k2)
                pm = phi + 0.5*h_w*k2p; dm = dphi + 0.5*h_w*k2d
                k3p = dm
                k3d = 2*kappa*dinm - kappa*dm - f2*_cs_eval(pm, breaks, coeffs)
                # k4
                tn = t + h_w
                pn = phi + h_w*k3p; dn = dphi + h_w*k3d
                dinn = a_s*omega_s*np.cos(omega_s*tn) + ap*omega_p*np.cos(omega_p*tn + phi_p)
                k4p = dn
                k4d = 2*kappa*dinn - kappa*dn - f2*_cs_eval(pn, breaks, coeffs)

                phi  += h_w * (k1p + 2*k2p + 2*k3p + k4p) / 6.0
                dphi += h_w * (k1d + 2*k2d + 2*k3d + k4d) / 6.0
                t    += h_w

        out[0, 0] = phi
        out[1, 0] = dphi

        # ---- main loop: n_substeps integration steps per output point ----
        for i in range(1, N):
            for _ in range(n_substeps):
                din = a_s*omega_s*np.cos(omega_s*t) + ap*omega_p*np.cos(omega_p*t + phi_p)
                k1p = dphi
                k1d = 2*kappa*din - kappa*dphi - f2*_cs_eval(phi, breaks, coeffs)

                tm = t + 0.5*h
                pm = phi + 0.5*h*k1p; dm = dphi + 0.5*h*k1d
                dinm = a_s*omega_s*np.cos(omega_s*tm) + ap*omega_p*np.cos(omega_p*tm + phi_p)
                k2p = dm
                k2d = 2*kappa*dinm - kappa*dm - f2*_cs_eval(pm, breaks, coeffs)

                pm = phi + 0.5*h*k2p; dm = dphi + 0.5*h*k2d
                k3p = dm
                k3d = 2*kappa*dinm - kappa*dm - f2*_cs_eval(pm, breaks, coeffs)

                tn = t + h
                pn = phi + h*k3p; dn = dphi + h*k3d
                dinn = a_s*omega_s*np.cos(omega_s*tn) + ap*omega_p*np.cos(omega_p*tn + phi_p)
                k4p = dn
                k4d = 2*kappa*dinn - kappa*dn - f2*_cs_eval(pn, breaks, coeffs)

                phi  += h * (k1p + 2*k2p + 2*k3p + k4p) / 6.0
                dphi += h * (k1d + 2*k2d + 2*k3d + k4d) / 6.0
                t    += h

            out[0, i] = phi
            out[1, i] = dphi

        return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_gain_chain(
    warmup_time: float,
    omega_p:     float,
    nc_pump:     int,
    ap:          float,
    phi_p:       float,
    nc_signal:   int,
    a_s:         float,
    kappa:       float,
    cpr:         Callable[[float | np.ndarray], float | np.ndarray],
    C1:          float,
    *,
    do_analysis: bool = True,
    n_sample:    int  = 25,
    method:      str  = "RK45",
    rtol:        float = 1e-8,
    atol:        float = 1e-10,
    backend:     str  = "auto",     # 'scipy', 'numba', or 'auto'
    n_substeps:  int  = 4,           # numba: integration steps per output sample
) -> dict | object:
    """
    Solve the Josephson oscillator ODE and optionally return the gain.

    Parameters
    ----------
    warmup_time : Transient settling time; samples before this are discarded.
    omega_p     : Pump angular frequency.
    nc_pump     : Number of pump cycles used for analysis.
    ap          : Pump amplitude.
    phi_p       : Pump phase.
    nc_signal   : Number of signal cycles; sets omega_s = (nc_signal/nc_pump)*omega_p.
    a_s         : Signal amplitude.
    kappa       : Decay rate (K in the equation).
    cpr         : Current-phase-relation callable J[phi] (e.g. a CubicSpline).
    C1          : Shunt capacitance. The CPR coefficient in the EOM is
                  f2 = 1 / (phi_0 * C1), with phi_0 = hbar/(2e).
    do_analysis : If True, return a dict with gain and spectra; otherwise return the raw sol object.
    n_sample    : DFT samples per pump cycle (default 25, matches Mathematica).
    method      : solve_ivp integration method. Use "Radau" for stiff systems.
    rtol, atol  : Solver tolerances (scipy backend only).
    backend     : 'scipy' (solve_ivp), 'numba' (JIT'd RK4, ~20–50× faster, requires
                  CubicSpline cpr), or 'auto' (numba if available and applicable).
    n_substeps  : Integration substeps per output sample for the numba backend.
                  Total steps per pump cycle = n_sample * n_substeps. With defaults
                  (25 * 4 = 100 steps/cycle) RK4 accuracy is comparable to scipy
                  RK45 with rtol=1e-8.

    Returns
    -------
    If do_analysis: dict with keys 'gain', 'freqs', 'p_out', 'p_in', 'p_int'.
    Otherwise: the raw scipy OdeSolution.
    """
    # --- validate ---
    ns_max = 5 * nc_pump
    if nc_signal >= ns_max:
        raise ValueError(
            f"nc_signal={nc_signal} is outside the kept frequency range "
            f"(max {ns_max - 1} for nc_pump={nc_pump})"
        )

    # --- derived quantities ---
    omega_s   = (nc_signal / nc_pump) * omega_p
    T_pump    = 2 * np.pi / omega_p
    t_end     = warmup_time + nc_pump * T_pump

    phi_in, dphi_in = _make_phi_in(a_s, omega_s, ap, omega_p, phi_p)

    # --- ODE right-hand side ---
    # phi'' + K*phi' + f2*J[phi] = 2*K*phi_in'(t),  with f2 = 1/(phi_0 * C1)
    f2 = 1.0 / (PHI_0_RED * C1)

    def rhs(t: float, y: np.ndarray) -> list:
        phi, dphi = y
        return [dphi,
                2*kappa * dphi_in(t) - kappa * dphi - f2 * cpr(phi)]

    # --- build t_eval at the uniform analysis grid ---
    # Matches Mathematica: Table[..., {t, warmupTime, t_end - dt, dt}]
    N    = nc_pump * n_sample
    dt   = T_pump / n_sample
    t_eval_analysis = warmup_time + np.arange(N) * dt   # shape (N,)

    # --- pick backend ---
    use_numba = (
        backend == "numba"
        or (backend == "auto" and _HAS_NUMBA and isinstance(cpr, CubicSpline))
    )
    if backend == "numba" and not _HAS_NUMBA:
        raise RuntimeError("backend='numba' requires `pip install numba`.")
    if backend == "numba" and not isinstance(cpr, CubicSpline):
        raise TypeError("backend='numba' requires cpr to be a scipy CubicSpline.")

    if use_numba:
        # Extract spline internals as contiguous arrays for the JIT'd kernel
        breaks = np.ascontiguousarray(cpr.x, dtype=np.float64)
        coeffs = np.ascontiguousarray(cpr.c, dtype=np.float64)
        y = _rk4_drive(
            t_eval_analysis, kappa, f2, breaks, coeffs,
            a_s, omega_s, ap, omega_p, phi_p, n_substeps,
        )
        # Match the solve_ivp interface so downstream code is unchanged
        from types import SimpleNamespace
        sol = SimpleNamespace(t=t_eval_analysis, y=y, success=True, message="ok")
    else:
        sol = solve_ivp(
            rhs,
            t_span=(0.0, t_end),
            y0=[0.0, 0.0],
            method=method,
            t_eval=t_eval_analysis,
            rtol=rtol,
            atol=atol,
        )
        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")

    if not do_analysis:
        return sol

    return _analyze(
        phi_F       = sol.y[0],
        phi_in_arr  = phi_in(t_eval_analysis),
        nc_pump     = nc_pump,
        nc_signal   = nc_signal,
        omega_p     = omega_p,
        n_sample    = n_sample,
    )


# ---------------------------------------------------------------------------
# Spectral analysis
# ---------------------------------------------------------------------------

def _analyze(
    phi_F:      np.ndarray,
    phi_in_arr: np.ndarray,
    nc_pump:    int,
    nc_signal:  int,
    omega_p:    float,
    n_sample:   int,
) -> dict:
    """
    DFT-based power gain calculation.

    The power at frequency omega is defined as omega^2 * |F(omega)|^2,
    consistent with the Mathematica code's Abs[omega]^2 * Abs[amplitude]^2.

    Returns
    -------
    dict with keys:
        gain  : float — power gain at signal frequency
        freqs : np.ndarray — angular frequency axis, shape (2*ns,)
        p_out : np.ndarray — output power spectrum (reflected field)
        p_in  : np.ndarray — input power spectrum
        p_int : np.ndarray — internal (phi) power spectrum
    """
    N          = nc_pump * n_sample
    omega_step = omega_p / nc_pump           # DFT frequency resolution
    ns         = 5 * nc_pump                 # number of pos. freq. bins kept

    # frequency axis: [-ns, -ns+1, ..., ns-1] * omega_step
    freqs = np.arange(-ns, ns) * omega_step  # shape (2*ns,)

    # DFT of each signal
    F_out = _dft_spectrum(phi_F - phi_in_arr, N, ns)
    F_int = _dft_spectrum(phi_F,              N, ns)
    F_in  = _dft_spectrum(phi_in_arr,         N, ns)

    # Power = omega^2 * |amplitude|^2
    p_out = freqs**2 * np.abs(F_out)**2
    p_int = freqs**2 * np.abs(F_int)**2
    p_in  = freqs**2 * np.abs(F_in)**2

    # Signal bin: index ns + nc_signal corresponds to omega = nc_signal * omega_step = omega_s
    sig_idx = ns + nc_signal

    p_in_sig = p_in[sig_idx]
    if p_in_sig == 0.0:
        raise ValueError("Input power at signal frequency is zero — check a_s and nc_signal.")

    gain = p_out[sig_idx] / p_in_sig

    return {
        "gain":  gain,
        "freqs": freqs,
        "p_out": p_out,
        "p_in":  p_in,
        "p_int": p_int,
    }


# ---------------------------------------------------------------------------
# Pump amplitude search & power-added efficiency
# ---------------------------------------------------------------------------

def find_resonant_freq(cpr, C1: float, kappa: float):
    """
    Find the natural frequency of the resonator from its current-phase relation.

    The small-signal equilibrium is the (unique) zero of J(phi) inside the
    spline's domain. Linearising  phi'' + K*phi' + J(phi)/(phi_0*C1) = 0
    around that zero gives  omega_0 = sqrt( J'(x_eq) / (phi_0 * C1) ),
    and the damped frequency  omega_d = omega_0 * sqrt(1 - (kappa/(2*omega_0))^2).

    Parameters
    ----------
    cpr   : Current-phase relation as a scipy CubicSpline (needs .roots() and
            spline-evaluable derivatives via cpr(x, 1)).
    C1    : Shunt capacitance.
    kappa : Decay rate.

    Returns
    -------
    [x_eq, omega_0, omega_d]
        - x_eq    : equilibrium phase (numpy array of length 1; index [0] to scalar)
        - omega_0 : bare angular frequency
        - omega_d : kappa-shifted (damped) angular frequency
    Returns [0, 0, 0] if the CPR doesn't have exactly one root inside its domain.
    """
    x = cpr.roots(extrapolate=False)
    if len(x) != 1:
        print(f"error: number of roots not one {x}")
        return [0, 0, 0]
    omega_0 = np.sqrt(cpr(x[0], 1) / (C1 * PHI_0_RED))
    return [x, omega_0, omega_0 * np.sqrt(1 - (kappa / (2 * omega_0)) ** 2)]


def find_pump_amplitude(
    warmup_time: float,
    omega_p:     float,
    nc_pump:     int,
    phi_p:       float,
    nc_signal:   int,
    kappa:       float,
    cpr:         Callable[[float | np.ndarray], float | np.ndarray],
    C1:          float,
    *,
    target_gain_db:    float = 20.0,
    gain_band_db:      tuple[float, float] = (19.0, 21.0),
    omega_0_min:       float = 2 * np.pi * 4e9,
    omega_0_max:       float = 2 * np.pi * 9e9,
    a_s_small:         float = 0.01,
    # ----- Stage 1: pump amplitude sweep (log-spaced) -----
    ap_low:            float = 1.0,
    ap_high:           float = 80.0,
    n_pump_steps:      int   = 200,
    # ----- Binary-search refinement after Stage 1 -----
    bisect_max_iter:   int   = 15,
    bisect_gain_tol_db: float = 0.01,
    # ----- Stage 2: signal sweep -----
    n_signal_steps:    int   = 200,
    verbose:           bool  = True,
    gain_chain_kwargs: dict | None = None,
) -> dict:
    """
    Three-stage routine:

    Stage 1 (pump sweep): scan `ap` geometrically from `ap_low` to `ap_high`
        in `n_pump_steps` points (log-spaced), stopping early once the gain
        crosses `target_gain_db`.

    Stage 1b (binary search): once a bracket [a_lo, a_hi] containing the
        target-gain crossing is identified, bisect to refine to within
        `bisect_gain_tol_db` (default 0.01 dB).

    Stage 2 (compression): sweep signal amplitude squared (a_s^2) at the
        refined pump amplitude until the gain leaves the band given by
        `gain_band_db`, then use `first_cross` to locate the 1 dB
        compression point.

    Returns
    -------
    dict with keys
        pae          : power-added efficiency
                       = (nc_signal/nc_pump)^2 · a_s^2 · (G - 1) / ap2^2
        as_sq_max    : signal a_s^2 at the band-edge crossing
        ap2          : pump amplitude giving target_gain_db
        sweep_signal : list of (a_s^2, gain) sampled in stage 2
        sweep_pump   : list of (ap, gain) sampled in stage 1 + bisection
        omega_0      : kappa-shifted (damped) angular frequency from find_resonant_freq
        status       : 'ok' | 'omega_out_of_range' | 'no_pump_bias_found'
    """
    target_gain   = 10 ** (target_gain_db / 10.0)
    g_low, g_high = (10 ** (db / 10.0) for db in gain_band_db)
    extra_kwargs  = gain_chain_kwargs or {}

    # ---------- Small-signal frequency via find_resonant_freq ----------
    _, _, omega_0 = find_resonant_freq(cpr, C1, kappa)
    omega_0 = float(omega_0)
    if verbose:
        print(f"[find_pump_amplitude] omega_0 = {omega_0:.4e} rad/s "
              f"({omega_0/(2*np.pi)/1e9:.3f} GHz)")
    if not (omega_0_min <= omega_0 <= omega_0_max):
        return {"pae": 0.0, "as_sq_max": 0.0, "ap2": None,
                "sweep_signal": [], "sweep_pump": [],
                "omega_0": omega_0, "status": "omega_out_of_range"}

    # ---------- helper: one gain evaluation ----------
    def _gain_at(ap: float, a_s: float) -> float:
        return compute_gain_chain(
            warmup_time=warmup_time, omega_p=omega_p, nc_pump=nc_pump,
            ap=ap, phi_p=phi_p, nc_signal=nc_signal, a_s=a_s,
            kappa=kappa, cpr=cpr, C1=C1, do_analysis=True,
            **extra_kwargs,
        )["gain"]

    # ---------- Stage 1: pump amplitude sweep (log-spaced) ----------
    sweep_pump: list[tuple[float, float]] = []
    status    = "no_pump_bias_found"

    for ap in np.geomspace(ap_low, ap_high, n_pump_steps):
        g = _gain_at(ap, a_s_small)
        sweep_pump.append((float(ap), g))
        if g >= target_gain:
            status = "ok"
            break

    pump_arr      = np.asarray(sweep_pump)
    ap_arr, g_arr = pump_arr[:, 0], pump_arr[:, 1]

    if status != "ok" or g_arr[0] >= target_gain:
        if verbose:
            if status != "ok":
                print(f"[find_pump_amplitude] never reached {target_gain_db} dB; "
                      f"best gain {10*np.log10(g_arr.max()):.2f} dB — raise ap_high.")
            else:
                print(f"[find_pump_amplitude] ap_low already above {target_gain_db} dB "
                      f"(g(ap_low)={10*np.log10(g_arr[0]):.2f} dB) — lower ap_low.")
        return {"pae": 0.0, "as_sq_max": 0.0, "ap2": None,
                "sweep_signal": [], "sweep_pump": sweep_pump,
                "omega_0": omega_0,
                "status": "no_pump_bias_found"}

    # ---------- Stage 1b: binary search refinement ----------
    # Guaranteed bracket: g_arr[0] < target_gain and g_arr[cross] >= target_gain, cross >= 1
    cross = int(np.argmax(g_arr >= target_gain))
    a_lo, g_lo = float(ap_arr[cross-1]), float(g_arr[cross-1])
    a_hi, g_hi = float(ap_arr[cross]),   float(g_arr[cross])

    for _ in range(bisect_max_iter):
        # stop if either endpoint is within tolerance of target (compared in dB)
        err_hi_db = abs(10*np.log10(g_hi) - target_gain_db)
        err_lo_db = abs(10*np.log10(g_lo) - target_gain_db)
        if min(err_hi_db, err_lo_db) < bisect_gain_tol_db:
            break
        a_mid = 0.5 * (a_lo + a_hi)
        g_mid = _gain_at(a_mid, a_s_small)
        sweep_pump.append((a_mid, g_mid))              # record probe
        if g_mid >= target_gain:
            a_hi, g_hi = a_mid, g_mid
        else:
            a_lo, g_lo = a_mid, g_mid

    # final linear interp inside the refined bracket
    if g_hi != g_lo:
        ap2 = a_lo + (target_gain - g_lo) * (a_hi - a_lo) / (g_hi - g_lo)
    else:
        ap2 = 0.5 * (a_lo + a_hi)

    if verbose:
        g_at_ap2 = _gain_at(ap2, a_s_small)
        sweep_pump.append((ap2, g_at_ap2))
        print(f"[find_pump_amplitude] bisection: ap2={ap2:.6f}, "
              f"G(ap2)={10*np.log10(g_at_ap2):.3f} dB")

    sweep_pump = sorted(set(sweep_pump))

    # ---------- Stage 2: signal-amplitude sweep at fixed pump ----------
    sweep_signal: list[tuple[float, float]] = []
    for a_s in np.geomspace(a_s_small, ap2, n_signal_steps):
        g = _gain_at(ap2, float(a_s))
        sweep_signal.append((float(a_s) ** 2, g))
        if g < g_low or g > g_high:
            break

    if len(sweep_signal) > 1:
        as_sq_max, g_at_max = first_cross(np.array(sweep_signal), g_low, g_high)
    else:
        as_sq_max, g_at_max = 0.0, 1.0

    pae = (nc_signal ** 2 / nc_pump ** 2) * as_sq_max * (g_at_max - 1.0) / ap2 ** 2

    if verbose:
        print(f"[find_pump_amplitude] ap2={ap2:.4f}, as_sq_max={as_sq_max:.4f}, "
              f"G@max={10*np.log10(g_at_max):.2f} dB, PAE={pae:.4e}")

    return {
        "pae":          pae,
        "as_sq_max":    as_sq_max,
        "ap2":          ap2,
        "sweep_signal": sweep_signal,
        "sweep_pump":   sweep_pump,
        "omega_0":      omega_0,
        "status":       status,
    }


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Build a sinusoidal CPR (pure Josephson junction)
    phi_pts = np.linspace(-4*np.pi, 4*np.pi, 500)
    I_c     = 1.0
    cpr     = CubicSpline(phi_pts, I_c * np.sin(phi_pts), bc_type="not-a-knot")

    result = compute_gain_chain(
        warmup_time = 50e-9,
        omega_p     = 2*np.pi * 6e9,   # 6 GHz pump
        nc_pump     = 100,
        ap          = 0.3,
        phi_p       = 0.0,
        nc_signal   = 99,              # signal slightly detuned from pump
        a_s         = 1e-3,            # weak signal
        kappa       = 2*np.pi * 1e8,
        cpr         = cpr,
        C1          = 1e-12,           # 1 pF shunt capacitance
        do_analysis = True,
    )

    print(f"Power gain at signal frequency: {result['gain']:.4f}  ({10*np.log10(result['gain']):.2f} dB)")