# chainJPA — Technical Documentation

Simulation of single-port Josephson Parametric Amplifiers (JPAs) built from
chains of RF-SQUIDs, including fabrication disorder. The pipeline goes from a
circuit topology to the **power-added efficiency (PAE)** at 20 dB of gain,
evaluated at the 1 dB compression point, by direct time-domain simulation.

---

## 1. The problem

### 1.1 Physical system

The device is a chain of *N* RF-SQUIDs terminating a transmission line. Each
RF-SQUID is a Josephson junction (critical current `Ic`) shunted by a linear
inductor, with a small parasitic inductor in parallel with the junction. The
SQUIDs are joined in series, a series parasitic inductor connects the chain to
ground, and the whole chain is shunted by a capacitor `C1` and coupled to the
input line at rate `kappa`.

Driven by a strong **pump** tone and a weak **signal** tone, the junction
nonlinearity mixes the two and transfers power from pump to signal — parametric
amplification. The figure of merit is not gain alone but **power-added
efficiency**: how much signal power is added per unit of pump power consumed, a
key metric for scaling to many amplifiers where pump power is a system-level
budget.

### 1.2 Equation of motion

Every non-ground node carries a superconducting phase `phi`. Its physical flux
is `Phi = (Phi_0 / 2pi) * phi`, with `Phi_0 = h/2e` the flux quantum. Branch
physics uses the gauge-invariant phase `theta_ab = phi_a - phi_b - phi_ext_ab`:

    Josephson junction:   I = Ic * sin(theta)      U = -(Ic Phi_0 / 2pi) cos(theta)
    Linear inductor:      I = (Phi_0/2pi) theta / L  U = (Phi_0/2pi)^2 theta^2 / (2L)

Reducing the chain to its single collective mode gives a driven, damped,
nonlinear oscillator for the resonator phase `phi`:

    phi'' + kappa * phi' + J(phi) / (Phi_0_red * C1) = 2 * kappa * phi_in'(t)

where `Phi_0_red = Phi_0 / 2pi`, `J(phi)` is the **current-phase relation
(CPR)** of the full inductive network (junctions + all inductors) seen from the
port, and the drive is a two-tone field

    phi_in(t) = a_s * sin(omega_s t) + a_p * sin(omega_p t + phi_p).

`J(phi)` is *not* a simple sine: it is the effective restoring current of the
whole multi-junction, multi-inductor network, computed numerically (§3).

### 1.3 What we extract

For a given device and bias, the routine finds:

1. the small-signal resonant frequency (must land in the operating band);
2. the pump amplitude `a_p` that produces exactly 20 dB of small-signal gain;
3. the signal level at which gain compresses by 1 dB (leaves a 19–21 dB band);

and reports **PAE** at that compression point:

    PAE = (nc_signal / nc_pump)^2 * a_s^2 * (G - 1) / a_p^2.

Sweeping bias and design parameters maps out where PAE is maximized and how
robust that optimum is to fabrication scatter.

---

## 2. Method overview and data flow

The computation is a two-level nested problem:

**Inner (static):** given a set of phases pinned at the port, solve Kirchhoff's
current law for all internal node phases. Repeating this over a range of port
phase gives `J(phi)` — the CPR. → `circuit_network.py`

**Outer (dynamic):** feed `J(phi)` into the oscillator ODE, integrate in time,
Fourier-analyze the steady state to get gain, and search over pump/signal
amplitude for the 20 dB / 1 dB-compression operating point. → `josephson_gain.py`

A problem-specific layer builds the RF-SQUID network, applies disorder and
sweep multipliers, and runs one `(device, bias)` point end-to-end. →
`rf_squid_chain.py`

```
  device parameters ─► make_rf_squid_chain ─► CircuitNetwork
                                                    │
                              CPR() sweeps port phase, solving KCL at each step
                                                    │
                                                    ▼
                                            J(phi)  (CubicSpline)
                                                    │
              find_resonant_freq ◄────────┬─────────┤
                                          │         ▼
                                          │   find_pump_amplitude
                                          │   ├ Stage 1  pump sweep  ─► 20 dB crossing
                                          │   ├ Stage 1b bisection    ─► refine a_p
                                          │   └ Stage 2  signal sweep ─► 1 dB compression
                                          │         │  (each gain eval = compute_gain_chain: RK4 + DFT)
                                          ▼         ▼
                                        PAE, sweep curves, status
                                                    │
                        results list ─► pandas DataFrame ─► explorers
```

---

## 3. `circuit_network.py` — Kirchhoff solver and CPR

Generic, problem-agnostic circuit primitives and the static solver.

### 3.1 Data model

`Node`, `Branch`, and the two branch subclasses `JosephsonJunction` and
`LinearInductor` carry the per-element physics (`current`, `energy`,
`gauge_invariant_phase`). `CircuitNetwork` holds ordered lists of nodes and
branches; nodes are referenced by integer index. Ground nodes are pinned at
`phi = 0`.

### 3.2 KCL as energy minimization

Kirchhoff's current law — net current into every free node is zero — is solved
as an unconstrained minimization. The key identity: the total network energy
`U(phi)` has gradient with respect to each free-node phase equal to the net
current out of that node. So a stationary point of `U` is exactly a KCL
solution. `_compile_problem` builds a `cost_and_grad(x)` closure returning
`(U/Phi_0_red, net_current)`; rescaling by `Phi_0_red` makes the gradient come
out directly in amperes (the natural residual scale).

The branch currents and node balance are fully vectorized:

    theta    = phi[a_idx] - phi[b_idx] - phi_ext        # all branches at once
    I_branch = Ic * sin(theta) + Phi_0_red * theta / L
    outflow  = bincount(a_idx, I_branch) - bincount(b_idx, I_branch)

Two `bincount` calls accumulate every branch's contribution to its endpoints —
no Python loop over branches.

### 3.3 Two solvers

- **`solve_newton`** — damped Newton with the analytic sparse Hessian (§3.4).
  Quadratic convergence from a good guess, typically < 10 iterations. Intended
  for warm-started sweeps and refining a known approximate solution. Raises on
  non-convergence.
- **`solve_lbfgs`** — scipy L-BFGS-B with optional multi-restart. Slower but
  robust from cold starts; the fallback for hard/globally-multimodal points.

### 3.4 Performance-critical machinery

**Analytic sparse Hessian = weighted graph Laplacian.** The Hessian of `U` is
the graph Laplacian on free nodes with per-branch admittance
`Y(theta) = Ic*cos(theta) + Phi_0_red/L`. Its sparsity pattern is fixed by
topology and never changes during iteration.
`_build_sparse_hessian_machinery` computes the CSR pattern, the COO→CSR sort,
and the parallel-branch deduplication **once**; each iteration only overwrites
`H.data` in place from a fresh `Y(theta)` vector (`np.add.at` with a
precomputed group map). No sparse-matrix constructor runs in the hot loop.

**`make_sweep_solver` — the sweep fast path.** For a flux/bias sweep where only
the pinned phases change, all per-network setup (index maps, free-node
enumeration, Hessian pattern, buffers) is done once, and the returned closure is
a tight Newton loop that:
- calls SuperLU directly (`scipy.sparse.linalg._dsolve._superlu.gssv`) with
  index arrays pre-cast to `intc` and a reused `ColPerm="COLAMD"` options dict,
  skipping `spsolve`'s per-call validation and CSC conversion;
- reuses a preallocated right-hand-side buffer;
- does up to 2 steps of **iterative refinement** per solve (recomputes
  `r = grad + H·dx`, re-solves) to recover digits lost when the Hessian is
  ill-conditioned (mixed-scale inductors give condition numbers ~1e8–1e10);
- **Armijo backtracking** on `||net_current||^2` to globalize the step;
- **noise-floor-aware tolerance**: `eff_tol = max(user_tol, 1e-13*max(Ic))`, so
  passing `tol=1e-30` doesn't spin forever — it stops at the double-precision
  floor. Unlike `solve_newton`, it never raises: `info['converged']` is a flag
  and the caller decides what to do.

### 3.5 `CPR` — building the current-phase relation

`CPR(make_inductor_block, kwargs, phi_min, phi_max, phi_step)` builds the
network, pins the port node, and calls `make_sweep_solver` with
`return_injection_currents=True`. It sweeps the port phase **up then down** over
`[phi_min, phi_max]`, warm-starting each solve from the previous solution, and
records the injection current (the external current needed to hold each port
phase). It then checks two physical validity conditions:

- **no hysteresis** — up-sweep and down-sweep currents agree to < 1e-10 A;
- **monotonicity** — injection current strictly increases with phase.

Either failure sets `fail = True` (a hysteretic or non-monotonic CPR would make
the single-mode reduction invalid). On success it returns a `CubicSpline` of
injection current vs port phase — this spline **is** `J(phi)`.

---

## 4. `josephson_gain.py` — ODE integration, gain, and the operating-point search

The outer, dynamic problem. Consumes a CPR spline and returns gain / PAE.

### 4.1 `compute_gain_chain` — one gain evaluation

Given amplitudes and frequencies, integrate the oscillator ODE to steady state
and return the power gain at the signal frequency. Two backends:

- **`numba`** (default via `backend="auto"` when the CPR is a `CubicSpline` and
  numba is installed) — a JIT-compiled fixed-step RK4 kernel, `_rk4_drive`.
- **`scipy`** — `solve_ivp` (adaptive `RK45` by default), used as a fallback or
  for stiff cases (`method="Radau"`).

Both integrate the same equation and return a common interface, so downstream
analysis is backend-independent.

### 4.2 Performance-critical machinery

**JIT'd RK4 with inlined spline evaluation.** `_rk4_drive` carries
`@njit(cache=True, fastmath=True)`. The state `(phi, dphi)` is scalar; all four
RK4 stages are inlined with no NumPy temporaries. The CPR is evaluated *inside*
the kernel by `_cs_eval` (marked `inline="always"`), which reads the spline's
raw `breaks`/`coeffs` arrays — extracted once before the loop — so there is no
Python call boundary per step. `_cs_eval` finds the segment by bit-shifted
binary search and evaluates the cubic by Horner's rule. `cache=True` persists
the compiled kernel across sessions. Net effect: ~20–50× faster than the scipy
path.

**Fixed-step vs adaptive.** The numba path takes `n_substeps` (default 4) RK4
steps per output sample — `n_sample * n_substeps = 25 * 4 = 100` steps per pump
cycle — giving accuracy comparable to `solve_ivp` at `rtol=1e-8` without
per-step error-control overhead.

**Cycle-aligned DFT — no windowing.** `_analyze` and `_dft_spectrum` exploit the
fact that the analysis window is set to exactly `nc_pump` pump cycles, so both
the pump frequency and `omega_s = (nc_signal/nc_pump)*omega_p` sit **exactly on
FFT bin centers**. An unwindowed FFT then reads the amplitudes with no spectral
leakage — no Hann/flat-top window needed (which would otherwise smear the gain
by a few dB). Power at a frequency is defined `omega^2 * |F(omega)|^2`; gain is
the ratio of output to input power in the signal bin. (Conventions mirror the
original Mathematica implementation, including the `1/N` DFT normalization and
the `RotateRight` bin layout.)

### 4.3 `find_resonant_freq`

The small-signal equilibrium is the unique root of `J(phi)` inside the spline's
domain. Linearizing the (undriven) ODE around it gives

    omega_0 = sqrt( J'(x_eq) / (Phi_0_red * C1) ),
    omega_d = omega_0 * sqrt(1 - (kappa / 2 omega_0)^2)   (damped frequency).

Returns `[0, 0, 0]` if the CPR doesn't have exactly one interior root (a
degenerate or mis-biased device).

### 4.4 `find_pump_amplitude` — the three-stage operating-point search

The core routine. First calls `find_resonant_freq`; if `omega_0` is outside the
band (default 4–9 GHz) it returns immediately with `status="omega_out_of_range"`.
Otherwise:

- **Stage 1 — pump sweep.** Scan `a_p` log-spaced from `ap_low` to `ap_high`
  (default 1→80, 200 points) at a tiny fixed signal, stopping at the first
  crossing of 20 dB. If it never reaches 20 dB (or is already above it at
  `ap_low`), return `status="no_pump_bias_found"`.
- **Stage 1b — bisection.** Bisect the bracket `[a_lo, a_hi]` straddling the
  crossing until within 0.01 dB (default ≤ 15 iterations), then a final linear
  interpolation gives `ap2`, the pump amplitude for exactly 20 dB.
- **Stage 2 — signal (compression) sweep.** Hold `a_p = ap2`, sweep signal
  amplitude squared upward until the gain leaves the 19–21 dB band, then
  `first_cross` locates the band-edge crossing by interpolation (linear /
  quadratic / cubic-spline depending on how many points precede it). That gives
  the 1 dB compression signal level `as_sq_max` and the gain there.

Finally computes PAE and returns it with both sweep curves (`sweep_pump`,
`sweep_signal`), `omega_0`, and `status`.

**Status codes:** `ok`, `omega_out_of_range`, `no_pump_bias_found` (the CPR-fail
case is turned into `cpr_failed` one layer up, in `run_job`).

---

## 5. `rf_squid_chain.py` — device construction, disorder, sweep driver

The problem-specific layer that ties the two engines together.

### 5.1 `make_rf_squid_chain`

Builds a `CircuitNetwork` for an *N*-SQUID chain: per-SQUID linear inductor
`L1`, per-SQUID parasitic inductor `L1p` in parallel with the junction, a series
parasitic inductor `L2p` from ground to the first node, and per-SQUID critical
current `Ic` with optional per-junction external flux. Produces `2N+2` nodes and
`3N+1` branches.

### 5.2 Disorder model

- **`BaseRealization`** — a dataclass holding one disorder draw: per-SQUID arrays
  (`baseL1`, `baseL1p`, `baseIc`) plus scalars (`baseL2p`, `basekappa`,
  `baseC1`, `basephi_ext0`, the last of which may be scalar or per-SQUID).
- **`make_realizations(mean, sigma, n_realizations, seed)`** — draws independent
  normal samples per field. Array-valued means (e.g. `mean['Ic']` of length *N*)
  get an independent draw per SQUID; missing `sigma` keys mean zero disorder on
  that field. Seeded for reproducibility. The notebook uses `sigma={'Ic': 0.05*Ic}`
  — 5% independent scatter on each junction's critical current, everything else
  clean.

### 5.3 Sweep multipliers — `Job` and `make_job`

A parameter sweep is expressed as four scalar multipliers applied to a base
realization; `make_job` produces a `Job` (the multipliers plus the scaled
simulation arrays, so the sweep grid is reconstructable from results):

| multiplier | scales | note |
|---|---|---|
| `im` | `Ic` | overall critical-current scale |
| `cm` | `C1` | shunt capacitance |
| `lm` | `L1` as `lm/im` | see below |
| `pm` | `phi_ext0` | flux bias per plaquette |

`L1p`, `L2p`, and `kappa` are **not** scaled. The `lm/im` factor on `L1` is
deliberate: combined with `im` scaling `Ic`, the product `L1*Ic` (the SQUID
screening parameter `beta_L`) then depends only on `lm`, so `lm` tunes SQUID
nonlinearity independently of the absolute current scale set by `im`. Override
`make_job` if a different convention is wanted.

### 5.4 Per-job runners

- **`run_job_res_freq(job)`** — CPR solve + `find_resonant_freq` only (no time
  integration). Cheap; useful for pre-screening which bias points resonate in
  band.
- **`run_job(job, warmup_time, omega_p, nc_pump, phi_p, nc_signal, *, gain_kwargs, verbose)`**
  — the full pipeline for one point: build chain → `CPR` → if the CPR solve
  fails return `status="cpr_failed"`, else `find_pump_amplitude`. Returns
  `{'job', 'result', 'status'}`. Takes all simulation settings explicitly (not
  from globals) so it is safe to `functools.partial` and dispatch under
  `joblib.Parallel`. `DEFAULT_GAIN_KWARGS = {"backend": "numba", "n_substeps": 4}`.

---

## 6. `disordered_rf_squid_chain_v4.ipynb` — worked example

A two-phase study of one nominal device (10 SQUIDs, 12 GHz pump, `Q≈10` at
signal).

**Phase 1 — clean pre-scan (cells 3–11).** With `sigma={}` (no disorder,
`n_realizations=1`), sweep `im ∈ {0.23…1.0}`, `lm ∈ {0.90…0.99}`,
`pm ∈ [pi−0.55, pi+0.55]` at fixed `cm=1`. Each point runs through `run_job`
under `Parallel(n_jobs=-1, return_as="generator")` with a `tqdm` bar; the
`generator` form streams results so the bar advances in real time.
`Counter(r['status'] …)` gives a fail-mode histogram. Results are unpacked into
a DataFrame (`di, im, cm, lm, pm, PAE, sweep_pump, sweep_signal`) and explored
with `make_pae_explorer` / `make_sweep_explorer`. A raw seaborn `lineplot` shows
PAE-vs-flux for one `lm` slice — the motivation for `flux_explorer`.

**Phase 2 — disorder study (cells 13–20).** Keep only the promising points
(`df_select = df[df['PAE'] > 0.02]`), then re-run *each* of those bias points
across `n_realizations=12` disorder draws with `sigma={'Ic': 0.05*Ic}` (5%
per-junction scatter). Unpack into `df_disorder` and re-explore. This asks: of
the bias points that looked good on a clean device, which stay good once
fabrication scatter is included? — the actual design question.

---

## 7. Visualization modules

All three take the results DataFrame and share conventions: internal SI columns
(`Ic` in amps) with an optional **design-parameter mode** that relabels/rescales
for display. In `use_design_params=True` mode the multiplier columns `im`/`lm`
are swapped for the physical `designIc` (shown in μA) and `designBetaPrime`
(shown as `β'`); the DataFrame must carry those columns. A shared
significant-figure rounding helper (`_round_sig`) is used everywhere so
small-magnitude values like `Ic` in amps aren't collapsed to 1–2 digits before
display.

- **`pae_explorer.make_pae_explorer(df, ...)`** — interactive PAE heatmap. Any
  two of the five parameters go on the axes; off-axis `im`/`cm`/`lm` are fixed
  via dropdowns, off-axis `pm`/`di` can be either fixed or **aggregated**
  (pm → max, di → mean or a chosen percentile). Optional cell-boundary contour
  overlay at a chosen PAE level.
- **`sweep_explorer.make_sweep_explorer(df, ...)`** — side-by-side pump and
  signal sweep curves (gain in dB vs amplitude, log-x) for one
  `(di, im, cm, lm)` slice, one color per `pm`. Pump and signal share a hue
  normalization so equal `pm` maps to the same color in both panels. Expects the
  `sweep_pump` / `sweep_signal` n×2 arrays.
- **`flux_explorer`** — PAE-vs-flux, split into a **pure plotting function** and
  a **widget wrapper** so it can be called programmatically:
  - `plot_pae_vs_flux(df, fix=None, series=None, *, max_legend_entries=10, ...)`
    — no widgets. `fix` is a dict `column → value` (a key equal to `series` is
    ignored); `series` is the column used as hue (one curve per value), or
    `None` for a single line. When a series has more than `max_legend_entries`
    unique values, the legend is subsampled to that many evenly-spaced entries
    while all curves are still drawn. Returns the axes, so it composes into
    multi-panel figures.
  - `make_flux_explorer(df, use_design_params=False, ...)` — a thin widget shell:
    a "series" dropdown (none / di / Ic / β') plus one value dropdown per
    parameter (the one chosen as series auto-hides), calling `plot_pae_vs_flux`
    underneath.

---

## 8. Are you missing anything?

Functionally the project is **complete and internally consistent** — the two
engines, the device layer, the notebook, and the three explorers form a working
pipeline. Gaps are in packaging, testing, and a few robustness details rather
than in the science.

**Worth adding:**

1. **`requirements.txt` (or `pyproject.toml`).** Dependencies (numpy, scipy,
   networkx, matplotlib, pandas, seaborn, ipywidgets, joblib, tqdm, numba,
   jupyter) are currently undocumented. A pinned/floored file makes the repo
   reproducible. *(A drafted `requirements.txt` already exists in outputs.)*

2. **A test suite.** There is none. High-value smoke tests:
   - a pure-sine CPR through `compute_gain_chain` against a known small-signal
     gain;
   - `make_rf_squid_chain` builds `2N+2` nodes / `3N+1` branches for a few *N*;
   - `solve_newton` and `solve_lbfgs` agree on a small network;
   - numba and scipy backends agree to tolerance on the same input;
   - `make_job` scaling identities (e.g. `L1*Ic` invariant under `im`).

3. **Result persistence.** Long sweeps (the notebook runs ~14k jobs) are held
   only in memory; a crash loses everything. A cached/checkpointed runner
   (`joblib.hash` keys, atomic pickle, resume-on-restart) would help. *(A
   `job_runner.py` implementing this exists in outputs but is not yet wired into
   the notebook.)*

**Smaller robustness notes:**

4. **CPR validity uses hard-coded absolute thresholds** — hysteresis `1e-10 A`
   and strict monotonicity. For very small `Ic` (deep in the `im` sweep) these
   may want to scale with the current scale, mirroring the noise-floor logic
   already in the solver.

5. **`find_resonant_freq` prints on failure** rather than warning/returning
   quietly; under massively-parallel sweeps this clutters output. Consider the
   `warnings` module or a return-only signal.

6. **`n_sample=25` and `nc_pump=200` are convention-locked** to match the
   original Mathematica code. Fine, but worth a note in the README that they set
   the DFT resolution and bin alignment and shouldn't be changed casually.

7. **`_rk4_drive` assumes a uniform `t_eval` grid.** True for `compute_gain_chain`
   today, but it's an unchecked precondition; a guard (or a comment at the call
   site) would prevent a subtle future bug.

8. **The `pm` axis is labeled "Flux per plaquette"** in the explorers but stored
   as a bias multiplier on `phi_ext0`. The relationship is documented here; a
   one-line note in the explorer docstrings would close the loop for a new user.

---

## 9. File map

| File | Role |
|---|---|
| `circuit_network.py` | Circuit primitives, KCL Newton/L-BFGS solvers, `CPR` builder |
| `josephson_gain.py`  | ODE integration (numba/scipy), DFT gain, 3-stage operating-point search |
| `rf_squid_chain.py`  | RF-SQUID chain builder, disorder realizations, `run_job` |
| `disordered_rf_squid_chain_v4.ipynb` | Worked example: clean pre-scan → disorder study |
| `pae_explorer.py`    | Interactive PAE heatmap |
| `sweep_explorer.py`  | Side-by-side pump/signal sweep curves |
| `flux_explorer.py`   | PAE-vs-flux (pure function + widget) |
| `requirements.txt`   | Dependencies *(drafted, in outputs)* |
| `job_runner.py`      | Cached/checkpointed parallel runner *(drafted, in outputs)* |
