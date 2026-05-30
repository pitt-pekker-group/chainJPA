# chainJPA
Simulation of realistic single-port Josephson Parametric Amplifiers (JPAs)
comprised of a capacitor shunted by an inductive element composed of linear 
inductors and Josephson junctions. chainJPA is specifically designed for 
desiging JPAs while taking into account imperfections like fabrication 
disorder and parasitic inductance.

## Inductive element

The inductive element is modeled as an arbitrary, user-specified CircuitNetwork 
composed of Josephson junctions and linear inductors. Examples of circuits that
can be modeled with CircuitNetwork include 
1. chain of RF-SQUIDs
2. chain of RF-SQUIDS with intra- and inter- RF-SQUID parasitic inductance.
3. ladders of junctions and linear inductors

CircuitNetwork solves Kirchhoff's law for the circuit in order to compute the
Current-Phase Relation (CPR). In subsequent steps the CPR is then represented 
as a cubic spline.

## Amplifier dynamics
The full circuit is composed of the inductive element shunted by a capacitor 
`C1`. It is driven through a port with coupling/decay rate `kappa`. The dynamics
are governed by the equation
```
phi'' + kappa·phi' + J(phi)/(phi_0·C1) = 2·kappa·phi_{in}'(t)
```
where $J(phi)$ is the CPR from the inductive element calculation and 
where `phi_in(t) = a_s·sin(omega_s·t) + a_p·sin(omega_p·t + phi_p)` is the
combined signal + pump drive and `phi_0 = ħ/(2e)` is the reduced flux quantum. 
The ODE is integrated (numba or scipy backend) and the gain at the signal 
frequency is extracted via the Discrete Fourier Transform (DFT) (with frequences
selected to be comensurate to both signal and drive frequency).

## Amplifeir tuning
For each operating point, `find_pump_amplitude` runs a two-stage search:
1. **Pump amplitude sweep** —
    1. log-spaced scan of pump amplitude until the gain crosses 20 dB.
    2. Bisection to refine the pump amplitude to the 20 dB point within 0.01 dB.
2. **Signal sweep** — at fixed pump amplitude, sweep signal amplitude squared
   until the gain leaves the 19–21 dB band, locating the 1 dB compression
   point.

The PAE is reported as
```
PAE = (nc_signal/nc_pump)^2 · a_s^2 · (G − 1) / a_p^2
```
where $a_s$ is the signal amplitude, $a_p$ is the pump amplitude, $G$ is the gain,
evaluated at the compression point.

## Installation

Python 3.10+ recommended.

```bash
pip install numpy scipy joblib tqdm numba pandas seaborn matplotlib ipywidgets jupyter
```

Drop the eight source files into a single directory and run a notebook from
that directory. There is no package structure to install — each module
imports its neighbours directly.

Optional but recommended for speed: numba. The default `gain_chain_kwargs`
in `rf_squid_chain.run_job` use `backend="numba"` (about 10× faster than
the scipy backend for the ODE).

## Repository layout

```
chainJPA/
├── README.md
├── LICENSE                              MIT
├── circuit_network.py                   Generic circuit-network solver + CPR
├── josephson_gain.py                    Gain-chain ODE + pump amplitude search
├── rf_squid_chain.py                    RF-SQUID chain builder + Job runner
├── pae_explorer.py                      Interactive PAE heatmap widget
├── sweep_explorer.py                    Interactive sweep-curve widget
└── disordered_rf_squid_chain_v4.ipynb   Worked example
```


## Module reference

### `circuit_network`

Generic, problem-agnostic circuit-network solver.

- `Node`, `Branch`, `JosephsonJunction`, `LinearInductor` — circuit elements.
- `CircuitNetwork` — holds nodes and branches; supports `add_node`,
  `add_ground`, `add_inductor`, `add_josephson_junction`, `solve_newton`,
  `solve_lbfgs`, `make_sweep_solver` (vectorized phase-sweep solver),
  and `draw` (matplotlib visualization).
- `CPR(make_inductor_block, kwargs, ...)` — build a current-phase relation
  for an arbitrary inductive block as a CubicSpline. Sweeps phase up and
  down, checks for hysteresis and monotonicity, and returns
  `{'phis', 'currents', 'cpr', 'cn', 'fail'}`.
- `PHI_0`, `PHI_0_RED` — magnetic flux quantum and its reduced version.

### `josephson_gain`

ODE simulation of the parametric amplifier and analysis.

- `compute_gain_chain(...)` — solve the resonator ODE, return gain
  at the signal frequency via DFT. Supports `backend="numba"` (fast,
  fixed-step RK4) or `backend="scipy"` (adaptive `solve_ivp`).
- `find_resonant_freq(cpr, C1, kappa)` — small-signal natural frequency
  from the linearized resonator (returns `[x_eq, omega_0, omega_d]`).
- `find_pump_amplitude(...)` — the three-stage search described above.
  Returns `{'pae', 'as_sq_max', 'ap2', 'sweep_signal', 'sweep_pump',
  'omega_0', 'status'}` where `status` is `'ok'`,
  `'omega_out_of_range'`, or `'no_pump_bias_found'`.

### `rf_squid_chain`

Problem-specific glue: builds the RF-SQUID chain, samples disorder, and
runs one job.

- `make_rf_squid_chain(L1, L1p, L2p, Ic, fluxes=None, ...)` — build the
  CircuitNetwork. Returns a CircuitNetwork with `2N+2` nodes and
  `3N+1` branches.
- `BaseRealization` dataclass — holds the pre-multiplier parameter set
  for a single disorder realization.
- `make_realizations(mean, sigma, n_realizations, seed)` — draw N
  independent base realizations from independent normals. Array-valued
  means give per-SQUID disorder; missing `sigma` keys default to zero.
- `Job` dataclass — one sweep point: realization × multipliers
  (`im`, `cm`, `lm`, `pm`).
- `make_job(base, disorder_index, im, cm, lm, pm)` — apply the four
  multipliers to a BaseRealization to produce a Job. Current scaling
  convention:
  - `im` scales `Ic`
  - `cm` scales `C1`
  - `lm/im` scales `L1` (so that `lm` alone controls the JJ-to-linear
    inductance ratio independently of the absolute current scale `im`)
  - `pm` scales `phi_ext0`
  - `L1p`, `L2p`, and `kappa` are not scaled
- `run_job(job, warmup_time, omega_p, nc_pump, phi_p, nc_signal, *,
   gain_kwargs=None, verbose=False)` — compute the PAE for one job.
  Returns `{'job', 'result', 'status'}`.
- `run_job_res_freq(job)` — return the small-signal resonant frequency
  for one job (no time-domain integration).
- `DEFAULT_GAIN_KWARGS = {"backend": "numba", "n_substeps": 4}` — solver
  defaults; override via the `gain_kwargs` argument of `run_job`.

### `pae_explorer`

Interactive widget. After collecting results into a DataFrame `df` with
columns `di, im, cm, lm, pm, PAE`, use:

```python
from pae_explorer import make_pae_explorer
make_pae_explorer(df)
```

You can pick any two of the five parameters as heatmap axes. Off-axis `im`,
`cm`, `lm` get fixed-value dropdowns. Off-axis `pm` and `di` can be either
aggregated (pm → max, di → mean or any percentile) or fixed at a value via
checkbox toggles. Includes a cell-boundary contour overlay at a
user-defined level.

### `sweep_explorer`

Interactive widget. Expects columns `di, im, cm, lm, pm, sweep_pump,
sweep_signal` where the sweep columns hold `n×2` arrays (column 0 amplitude,
column 1 linear gain). Shows pump and signal sweeps side-by-side with
matched per-`pm` colors.

```python
from sweep_explorer import make_sweep_explorer
make_sweep_explorer(df)
```

## Workflow

End-to-end pipeline:

```
mean, sigma                         ┐
    │                               │   make_realizations
    ▼                               │
[BaseRealization × N]               │
    │                               │
    │  + multipliers (im, cm, lm, pm)│   make_job
    ▼                               │
[Job × Nsweep]                      │
    │                               │   run_job (parallel)
    ▼                               │
[{job, result, status} × Nsweep]    │
    │                               │
    │  unpack to DataFrame          │
    ▼                               │
df with PAE, sweep_pump, sweep_signal│
    │                               │
    ▼                               │
make_pae_explorer / make_sweep_explorer
```

## Quick start

```python
import numpy as np
from functools import partial
from joblib import Parallel, delayed
from tqdm import tqdm
from collections import Counter
import pandas as pd

from rf_squid_chain import make_realizations, make_job, run_job
from pae_explorer import make_pae_explorer

# ---- 1. Simulation settings (fixed for the whole sweep) -------------
omega_p     = 2 * np.pi * 12e9       # 12 GHz pump
nc_Pump     = 200                    # pump cycles for FFT
nc_Signal   = 101                    # signal cycles for FFT
warmupTime  = 10e-9                  # amount of time trimmed for transient

# ---- 2. Nominal device parameters -----------------------------------
N = 10                                     # Number RF-SQUIDs
mean = {
    'L1':       17.46e-12 * np.ones(N),    # per-SQUID linear inductance
    'L1p':      0.01e-12 * np.ones(N),     # per-SQUID parasitic L
    'L2p':      0.01e-12,                  # series parasitic L
    'Ic':       18.84e-6 * np.ones(N),     # per-SQUID critical current
    'phi_ext0': np.ones(N),                # uniform flux (same through each SQUID)
    'kappa':    2 * np.pi * 2.42e9,        # coupling rate
    'C1':       0.5e-12,                   # shunt capacitance
}

# ---- 3. Disorder (empty = nominal device, no disorder) --------------
sigma = {
     # missing keys → zero disorder
}
realizations = make_realizations(mean, sigma, n_realizations=1, seed=42)

# ---- 4. Build the job list ------------------------------------------
jobs = [
    make_job(base=realizations[di], disorder_index=di,
             im=im, cm=1.0, lm=lm, pm=pm)
    for di in range(len(realizations))
    for im in [0.23, 0.25, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0]
    for lm in [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    for pm in np.arange(np.pi-0.55, np.pi+0.5501, 0.01)
]
print(f'{len(jobs)} jobs')

# ---- 5. Run in parallel with progress bar ---------------------------
run = partial(run_job,
              warmup_time = warmupTime, omega_p = omega_p,
              nc_pump = nc_Pump, phi_p = phi_p, nc_signal = nc_Signal)

results = []
with tqdm(total=len(jobs), desc="Sweeping jobs") as pbar:
    for r in Parallel(n_jobs=-1, return_as="generator")(
            delayed(run)(j) for j in jobs):
        results.append(r)
        pbar.update(1)

print(Counter(r['status'] for r in results))  # print out how many points succeded and what failure modes were encountered

# ---- 6. Build the DataFrame -----------------------------------------
df = pd.DataFrame([
    {'di':  r['job'].disorder_index,
     'im':  r['job'].im,  'cm':  r['job'].cm,
     'lm':  r['job'].lm,  'pm':  r['job'].pm,
     'PAE': r['result']['pae'] if r['result'] is not None else np.nan,
     'sweep_pump':   np.array(r['result']['sweep_pump'])   if r['result'] else np.nan,
     'sweep_signal': np.array(r['result']['sweep_signal']) if r['result'] else np.nan}
    for r in results
])

# ---- 7. Explore --------------------------------------------------------
make_pae_explorer(df)
make_sweep_explorer(df)
```

For the second pass with fabrication disorder, see
`disordered_rf_squid_chain_v4.ipynb` — it re-runs the best operating
points from the no-disorder pass across 12 disorder realizations
with 5% spread on `Ic`.

## Status codes

`r['status']` from `run_job` is one of:

| status                | meaning                                                |
|-----------------------|--------------------------------------------------------|
| `ok`                  | full pipeline succeeded, `r['result']['pae']` is valid |
| `cpr_failed`          | the CPR sweep showed hysteresis or non-monotonicity    |
| `omega_out_of_range`  | small-signal resonance outside the 4–9 GHz window      |
| `no_pump_bias_found`  | pump sweep didn't reach 20 dB before `ap_high = 80`    |

Count them with `collections.Counter(r['status'] for r in results)` to
get a quick fail-mode histogram.

## DataFrame columns

After unpacking `results` into a DataFrame as in step 6:

| column        | type           | meaning                                      |
|---------------|----------------|----------------------------------------------|
| `di`          | int            | disorder-realization index                    |
| `im`          | float          | critical-current multiplier                   |
| `cm`          | float          | capacitance multiplier                        |
| `lm`          | float          | inductance multiplier                         |
| `pm`          | float          | phase / flux multiplier                       |
| `PAE`         | float or NaN   | power-added efficiency at 20 dB gain          |
| `sweep_pump`  | ndarray (n×2)  | pump-amplitude sweep, columns `(a_p, gain)`   |
| `sweep_signal`| ndarray (n×2)  | signal-amplitude sweep at fixed pump          |

Note: RF-SQUID linear inductance is multiplied by the factor (lm / im). All other
linear inductances are unchanged.

`PAE`, `sweep_pump`, `sweep_signal` are NaN when `r['result']` is None
(i.e., status was `cpr_failed`).

## Performance notes

- The numba backend (default) compiles the ODE on first call; subsequent
  calls reuse the compiled code. Expect ~0.5–2 s per job on a typical
  laptop core after warmup.
- `Parallel(n_jobs=-1)` uses all cores. Each worker recompiles numba
  separately, so for very short sweeps (< 50 jobs) the scipy backend
  may actually be faster — set `gain_chain_kwargs={"backend": "scipy"}`
  in `run_job`.
- For sweeps of thousands of jobs, save results to disk; a typical
  `run_job` result with default settings is ~10 KB pickled.

## License

MIT — see `LICENSE`.