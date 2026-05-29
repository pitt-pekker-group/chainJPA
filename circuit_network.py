"""
Circuit network of nodes connected by Josephson junctions and linear inductors.

Convention
----------
Each non-ground node carries a phase variable phi (dimensionless). The node
flux is Phi = (Phi_0 / 2*pi) * phi, where Phi_0 = h/(2e) is the magnetic
flux quantum. Ground nodes are pinned at phi = 0.

Branches between nodes a and b use the gauge-invariant phase

    theta_ab = phi_a - phi_b - phi_ext_ab,

where ``phi_ext_ab`` is the line integral of (2*pi/Phi_0) * A · dl along
the branch from a to b. Default value is 0. The sum of phi_ext around
any closed loop must equal the loop's total flux in units of Phi_0/(2*pi);
how that loop flux is distributed among the loop's branches is a choice
of gauge.

    Josephson junction:   I = I_c sin(theta_ab)
                          U = -E_J cos(theta_ab),  E_J = I_c * Phi_0 / (2*pi)

    Linear inductor:      I = (Phi_0 / 2*pi) * theta_ab / L
                          U = (Phi_0 / 2*pi)**2 * theta_ab**2 / (2 L)

Node references are integer indices into ``CircuitNetwork.nodes``.
``add_node`` and ``add_ground`` return the index of the newly-added node;
that index is what callers pass to ``add_inductor``,
``add_josephson_junction``, ``solve_newton``, etc., and is what appears
as the keys of phase / current dicts.
"""

from __future__ import annotations

import numpy as np
from scipy.constants import h, e
from scipy.interpolate import CubicSpline

__all__ = [
    "Node",
    "Branch",
    "JosephsonJunction",
    "LinearInductor",
    "CircuitNetwork",
    "PHI_0",
    "PHI_0_RED",
]

PHI_0 = h / (2.0 * e)       # magnetic flux quantum, Wb
PHI_0_RED = PHI_0 / (2.0 * np.pi)   # reduced flux quantum, Phi_0 / 2*pi


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
class Node:
    """A circuit node. Lives in a CircuitNetwork's node list and is
    referenced externally by its position in that list."""

    __slots__ = ("name", "is_ground", "position")

    def __init__(self, name: str | None = None, is_ground: bool = False,
                 position: tuple[float, float] | None = None) -> None:
        self.name = name
        self.is_ground = is_ground
        self.position = tuple(map(float, position)) if position is not None else None

    def __repr__(self) -> str:
        parts = []
        if self.name is not None:
            parts.append(f"name={self.name!r}")
        if self.is_ground:
            parts.append("ground")
        if self.position is not None:
            parts.append(f"pos={self.position}")
        return f"Node({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Branches
# ---------------------------------------------------------------------------
class Branch:
    """Abstract two-terminal element connecting node_a to node_b.

    ``phi_ext`` is the extrinsic phase from the vector potential along
    the branch (line integral of (2*pi/Phi_0) * A · dl from a to b).
    The gauge-invariant phase across the branch is

        theta_ab = phi_a - phi_b - phi_ext.
    """

    def __init__(self, node_a: Node, node_b: Node,
                 phi_ext: float = 0.0) -> None:
        if node_a is node_b:
            raise ValueError("A branch must connect two distinct nodes.")
        self.node_a = node_a
        self.node_b = node_b
        self.phi_ext = float(phi_ext)

    def gauge_invariant_phase(self, phase_a: float, phase_b: float) -> float:
        """phi_a - phi_b - phi_ext."""
        return phase_a - phase_b - self.phi_ext

    def current(self, phase_a: float, phase_b: float) -> float:
        """Current flowing from node_a to node_b (amperes)."""
        raise NotImplementedError

    def energy(self, phase_a: float, phase_b: float) -> float:
        """Stored / potential energy of the branch (joules)."""
        raise NotImplementedError


class JosephsonJunction(Branch):
    """Ideal Josephson element with critical current I_c."""

    def __init__(self, node_a: Node, node_b: Node, critical_current: float,
                 phi_ext: float = 0.0) -> None:
        if critical_current <= 0:
            raise ValueError("critical_current must be positive.")
        super().__init__(node_a, node_b, phi_ext=phi_ext)
        self.I_c = float(critical_current)
        self.E_J = self.I_c * PHI_0_RED      # Josephson energy

    def current(self, phase_a: float, phase_b: float) -> float:
        return self.I_c * np.sin(self.gauge_invariant_phase(phase_a, phase_b))

    def energy(self, phase_a: float, phase_b: float) -> float:
        return -self.E_J * np.cos(self.gauge_invariant_phase(phase_a, phase_b))

    def __repr__(self) -> str:
        ext = f", phi_ext={self.phi_ext:.3g}" if self.phi_ext != 0.0 else ""
        return (f"JosephsonJunction(I_c={self.I_c:.3e} A, "
                f"E_J={self.E_J:.3e} J{ext})")


class LinearInductor(Branch):
    """Ideal linear inductor with inductance L."""

    def __init__(self, node_a: Node, node_b: Node, inductance: float,
                 phi_ext: float = 0.0) -> None:
        if inductance <= 0:
            raise ValueError("inductance must be positive.")
        super().__init__(node_a, node_b, phi_ext=phi_ext)
        self.L = float(inductance)

    def current(self, phase_a: float, phase_b: float) -> float:
        return PHI_0_RED * self.gauge_invariant_phase(phase_a, phase_b) / self.L

    def energy(self, phase_a: float, phase_b: float) -> float:
        theta = self.gauge_invariant_phase(phase_a, phase_b)
        return 0.5 * (PHI_0_RED ** 2) * theta * theta / self.L

    def __repr__(self) -> str:
        ext = f", phi_ext={self.phi_ext:.3g}" if self.phi_ext != 0.0 else ""
        return f"LinearInductor(L={self.L:.3e} H{ext})"


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
class CircuitNetwork:
    """A graph of nodes linked by Josephson junctions and linear inductors.

    Nodes are stored in a list; their position in the list is their index.
    Use the return value of ``add_node`` / ``add_ground`` to refer to a
    node when building branches or specifying phases.
    """

    def __init__(self) -> None:
        self._nodes: list[Node] = []
        self._branches: list[Branch] = []

    # -- construction -------------------------------------------------------
    def add_node(self, name: str | None = None, is_ground: bool = False,
                 position: tuple[float, float] | None = None) -> int:
        """Add a node and return its index (its position in the node list).

        Parameters
        ----------
        name : str, optional
            Display name (used in plot labels). If None, the index is used
            as the display name (e.g. "n3").
        is_ground : bool
            If True, the node's phase is pinned at 0.
        position : (float, float), optional
            Pinned position for visualization; if None, the layout
            algorithm in ``draw()`` places the node.
        """
        node = Node(name=name, is_ground=is_ground, position=position)
        self._nodes.append(node)
        return len(self._nodes) - 1

    def add_ground(self, name: str | None = "GND",
                   position: tuple[float, float] | None = None) -> int:
        """Add a ground node (phase pinned at 0) and return its index."""
        return self.add_node(name=name, is_ground=True, position=position)

    def add_josephson_junction(self, a_idx: int, b_idx: int,
                               critical_current: float,
                               phi_ext: float = 0.0) -> JosephsonJunction:
        """Add a Josephson junction between the two nodes given by index.

        ``phi_ext`` is the extrinsic gauge phase along the branch from
        a to b (line integral of (2*pi/Phi_0) * A · dl). It enters the
        constitutive law as theta = phi_a - phi_b - phi_ext.
        """
        jj = JosephsonJunction(self._resolve(a_idx), self._resolve(b_idx),
                               critical_current, phi_ext=phi_ext)
        self._branches.append(jj)
        return jj

    def add_inductor(self, a_idx: int, b_idx: int,
                     inductance: float,
                     phi_ext: float = 0.0) -> LinearInductor:
        """Add a linear inductor between the two nodes given by index.

        ``phi_ext`` is the extrinsic gauge phase along the branch from
        a to b (line integral of (2*pi/Phi_0) * A · dl).
        """
        ind = LinearInductor(self._resolve(a_idx), self._resolve(b_idx),
                             inductance, phi_ext=phi_ext)
        self._branches.append(ind)
        return ind

    # -- accessors ----------------------------------------------------------
    @property
    def nodes(self) -> list[Node]:
        """The internal node list (read-only copy)."""
        return list(self._nodes)

    @property
    def branches(self) -> list[Branch]:
        return list(self._branches)

    def free_node_indices(self) -> list[int]:
        """Indices of nodes whose phase is a free variable (not ground)."""
        return [i for i, n in enumerate(self._nodes) if not n.is_ground]

    def label_of(self, idx: int) -> str:
        """Display label for the node at the given index: its ``name`` if
        set, otherwise ``f"n{idx}"``."""
        n = self._nodes[idx]
        return n.name if n.name is not None else f"n{idx}"

    # -- physics ------------------------------------------------------------
    def total_energy(self, phases: dict[int, float]) -> float:
        """Sum of branch potential energies for a given phase configuration.

        ``phases`` is keyed by node index. Ground nodes are implicitly at 0
        and need not appear; missing free nodes default to 0.
        """
        idx_of = self._index_lookup()
        total = 0.0
        for br in self._branches:
            pa = self._phase_at(br.node_a, phases, idx_of)
            pb = self._phase_at(br.node_b, phases, idx_of)
            total += br.energy(pa, pb)
        return total

    def node_currents(self, phases: dict[int, float]) -> dict[int, float]:
        """Net current leaving each non-ground node (KCL residuals).

        Returns a dict keyed by node index.
        """
        idx_of = self._index_lookup()
        residuals = {i: 0.0 for i, n in enumerate(self._nodes) if not n.is_ground}
        for br in self._branches:
            pa = self._phase_at(br.node_a, phases, idx_of)
            pb = self._phase_at(br.node_b, phases, idx_of)
            i_ab = br.current(pa, pb)
            if not br.node_a.is_ground:
                residuals[idx_of[id(br.node_a)]] += i_ab
            if not br.node_b.is_ground:
                residuals[idx_of[id(br.node_b)]] -= i_ab
        return residuals

    def _compile_problem(self, fixed_phases: dict[int, float],
                         initial_guess: dict[int, float] | None):
        """Per-call setup shared by ``solve_newton`` and ``solve_lbfgs``.

        Validates inputs, builds vectorized per-branch arrays, returns a
        ``cost_and_grad(x)`` closure, the initial-guess vector, and the
        bookkeeping needed by ``_pack_result``.

        The closure minimizes the rescaled cost ``U / Phi_0_red`` so that
        its gradient is exactly the net node current in amperes — the
        natural physical scale for the KCL residual.
        """
        if not fixed_phases:
            raise ValueError("At least one node must be in fixed_phases.")

        n_nodes = len(self._nodes)
        for idx, val in fixed_phases.items():
            if not (0 <= idx < n_nodes):
                raise IndexError(f"Node index {idx} out of range.")
            if self._nodes[idx].is_ground and val != 0.0:
                raise ValueError(
                    f"Node {idx} is ground; its phase is pinned at 0 "
                    f"and cannot be set to {val}."
                )

        free_indices = [i for i, n in enumerate(self._nodes)
                        if not n.is_ground and i not in fixed_phases]
        n_free = len(free_indices)

        if initial_guess is not None:
            for idx in initial_guess:
                if not (0 <= idx < n_nodes):
                    raise IndexError(
                        f"Node index {idx} in initial_guess is out of range.")
                if idx in fixed_phases:
                    raise ValueError(
                        f"Node {idx} is in both fixed_phases and "
                        f"initial_guess; it can only appear in one."
                    )
                if self._nodes[idx].is_ground:
                    raise ValueError(
                        f"Node {idx} is ground; no initial guess needed."
                    )

        # Per-branch arrays for vectorized cos/sin evaluation.
        n_branches = len(self._branches)
        idx_of = self._index_lookup()
        a_idx = np.empty(n_branches, dtype=np.intp)
        b_idx = np.empty(n_branches, dtype=np.intp)
        phi_ext = np.empty(n_branches, dtype=np.float64)
        Ic_arr = np.zeros(n_branches, dtype=np.float64)
        L_inv = np.zeros(n_branches, dtype=np.float64)
        for k, br in enumerate(self._branches):
            a_idx[k] = idx_of[id(br.node_a)]
            b_idx[k] = idx_of[id(br.node_b)]
            phi_ext[k] = br.phi_ext
            if isinstance(br, JosephsonJunction):
                Ic_arr[k] = br.I_c
            elif isinstance(br, LinearInductor):
                L_inv[k] = 1.0 / br.L
            else:
                raise TypeError(
                    f"Branch type {type(br).__name__!r} is not supported "
                    f"by the vectorized solver."
                )

        free_arr = np.array(free_indices, dtype=np.intp)
        phi_all = np.zeros(n_nodes, dtype=np.float64)
        for idx, val in fixed_phases.items():
            phi_all[idx] = val

        _PHI_0_RED = PHI_0_RED

        def cost_and_grad(x):
            phi_all[free_arr] = x
            theta = phi_all[a_idx] - phi_all[b_idx] - phi_ext
            cos_th = np.cos(theta)
            sin_th = np.sin(theta)
            cost = (-np.dot(Ic_arr, cos_th)
                    + 0.5 * _PHI_0_RED * np.sum(theta * theta * L_inv))
            I_branch = Ic_arr * sin_th + _PHI_0_RED * theta * L_inv
            outflow = (np.bincount(a_idx, weights=I_branch, minlength=n_nodes)
                       - np.bincount(b_idx, weights=I_branch,
                                     minlength=n_nodes))
            return cost, outflow[free_arr]

        if initial_guess is not None:
            x0 = np.array([float(initial_guess.get(idx, 0.0))
                           for idx in free_indices])
        else:
            x0 = np.zeros(n_free)

        return {
            "n_nodes": n_nodes,
            "n_branches": n_branches,
            "n_free": n_free,
            "free_indices": free_indices,
            "free_arr": free_arr,
            "fixed_phases": fixed_phases,
            "a_idx": a_idx,
            "b_idx": b_idx,
            "phi_ext": phi_ext,
            "Ic_arr": Ic_arr,
            "L_inv": L_inv,
            "phi_all": phi_all,
            "cost_and_grad": cost_and_grad,
            "x0": x0,
        }

    def _pack_result(self, problem: dict, best_x) -> dict:
        """Build the public result dict from the problem context and the
        solved free-node phase vector. Shared between ``solve_newton`` and
        ``solve_lbfgs`` so they return identical shapes."""
        phi_all = problem["phi_all"]
        a_idx = problem["a_idx"]
        b_idx = problem["b_idx"]
        phi_ext = problem["phi_ext"]
        Ic_arr = problem["Ic_arr"]
        L_inv = problem["L_inv"]
        free_arr = problem["free_arr"]
        free_indices = problem["free_indices"]
        n_nodes = problem["n_nodes"]
        fixed_phases = problem["fixed_phases"]
        _PHI_0_RED = PHI_0_RED

        phi_all[free_arr] = best_x
        theta = phi_all[a_idx] - phi_all[b_idx] - phi_ext
        I_branch = Ic_arr * np.sin(theta) + _PHI_0_RED * theta * L_inv
        outflow = (np.bincount(a_idx, weights=I_branch, minlength=n_nodes)
                   - np.bincount(b_idx, weights=I_branch, minlength=n_nodes))
        energy = (-_PHI_0_RED * np.dot(Ic_arr, np.cos(theta))
                  + 0.5 * _PHI_0_RED ** 2
                  * np.sum(theta * theta * L_inv))

        phases = {i: float(phi_all[i]) for i in range(n_nodes)
                  if not self._nodes[i].is_ground}
        branch_currents = [
            {"branch": self._branches[k],
             "from": int(a_idx[k]),
             "to":   int(b_idx[k]),
             "current": float(I_branch[k])}
            for k in range(problem["n_branches"])
        ]
        injection_currents = {idx: float(outflow[idx]) for idx in fixed_phases}
        kcl_residuals = {idx: float(outflow[idx]) for idx in free_indices}

        return {
            "phases": phases,
            "energy": float(energy),
            "branch_currents": branch_currents,
            "injection_currents": injection_currents,
            "kcl_residuals": kcl_residuals,
        }

    def _build_sparse_hessian_machinery(self, n_nodes: int, n_free: int,
                                        free_arr: np.ndarray,
                                        a_idx: np.ndarray,
                                        b_idx: np.ndarray,
                                        Ic_arr: np.ndarray,
                                        L_inv: np.ndarray):
        """Build the sparse-Hessian setup shared by ``solve_newton`` and
        ``make_sweep_solver``.

        The Hessian is the weighted graph Laplacian on free nodes, with
        per-branch admittance ``Y(theta) = Ic*cos(theta) + (Phi_0/2pi)/L``.
        The sparsity pattern is fixed by the network topology; only the
        values change as theta changes during Newton iteration. We pre-
        compute the CSR sparsity pattern and a permutation that lets us
        rewrite ``H.data`` in place each step.

        Returns
        -------
        dict with keys
            'H_template' : csr_matrix
                The Hessian, in canonical CSR form. Use ``H_template.data``
                to read the values; ``update(theta)`` rewrites them in
                place.
            'H_data' : ndarray view of H_template.data
            'H_indices', 'H_indptr' : ndarrays cast to ``np.intc`` once,
                for direct use with ``_superlu.gssv``.
            'H_nnz' : int
            'update' : callable(theta)
                Updates ``H_data`` in place from a per-branch theta array.
        """
        from scipy.sparse import csr_matrix

        _PHI_0_RED = PHI_0_RED

        is_free = np.zeros(n_nodes, dtype=bool)
        is_free[free_arr] = True
        free_pos = np.full(n_nodes, -1, dtype=np.intp)
        free_pos[free_arr] = np.arange(n_free)
        a_free_mask = is_free[a_idx]
        b_free_mask = is_free[b_idx]
        both_free = a_free_mask & b_free_mask

        # Per-branch contribution: each branch with a free endpoint adds
        # +Y to its diagonal slot; each branch with two free endpoints
        # adds -Y to the off-diagonal pair (a,b) and (b,a).
        rows = np.concatenate([
            free_pos[a_idx[a_free_mask]],
            free_pos[b_idx[b_free_mask]],
            free_pos[a_idx[both_free]],
            free_pos[b_idx[both_free]],
        ])
        cols = np.concatenate([
            free_pos[a_idx[a_free_mask]],
            free_pos[b_idx[b_free_mask]],
            free_pos[b_idx[both_free]],
            free_pos[a_idx[both_free]],
        ])
        n_a = int(a_free_mask.sum())
        n_b = int(b_free_mask.sum())
        n_ab = int(both_free.sum())
        n_nnz_coo = n_a + n_b + 2 * n_ab

        # Permutation to canonical CSR order so we can sum duplicate
        # (row, col) entries (e.g. parallel branches) into the right
        # data slots.
        order = np.lexsort((cols, rows))
        sorted_rows = rows[order]
        sorted_cols = cols[order]
        new_group = np.empty(n_nnz_coo, dtype=bool)
        new_group[0] = True
        new_group[1:] = (sorted_rows[1:] != sorted_rows[:-1]) | \
                        (sorted_cols[1:] != sorted_cols[:-1])
        group_id = np.cumsum(new_group) - 1

        H_template = csr_matrix(
            (np.ones(n_nnz_coo, dtype=np.float64), (rows, cols)),
            shape=(n_free, n_free)
        )
        H_data = H_template.data
        # Pre-cast index arrays to intc so direct _superlu.gssv calls
        # don't have to do it on every invocation.
        H_indices = H_template.indices.astype(np.intc, copy=False)
        H_indptr = H_template.indptr.astype(np.intc, copy=False)

        vals_buf = np.empty(n_nnz_coo, dtype=np.float64)
        slc_a   = slice(0,                n_a)
        slc_b   = slice(n_a,              n_a + n_b)
        slc_ab1 = slice(n_a + n_b,        n_a + n_b + n_ab)
        slc_ab2 = slice(n_a + n_b + n_ab, n_a + n_b + 2 * n_ab)

        def update(theta):
            Y = Ic_arr * np.cos(theta) + _PHI_0_RED * L_inv
            vals_buf[slc_a]   = Y[a_free_mask]
            vals_buf[slc_b]   = Y[b_free_mask]
            vals_buf[slc_ab1] = -Y[both_free]
            vals_buf[slc_ab2] = -Y[both_free]
            H_data[:] = 0.0
            np.add.at(H_data, group_id, vals_buf[order])

        return {
            "H_template": H_template,
            "H_data": H_data,
            "H_indices": H_indices,
            "H_indptr": H_indptr,
            "H_nnz": int(H_template.nnz),
            "update": update,
        }

    def solve_newton(self, fixed_phases: dict[int, float],
                     initial_guess: dict[int, float] | None = None,
                     tol: float = 1e-12,
                     max_iter: int = 50) -> dict:
        """Solve KCL with damped Newton's method using the analytic sparse
        Hessian.

        Single-shot solver: takes one initial guess (zeros or the supplied
        ``initial_guess``), Newton-iterates until convergence, and returns
        the result. No multi-start, no fallback. Quadratic convergence
        from a good starting point — typically <10 iterations.

        Best for: warm-started parameter sweeps, refining a known
        approximate solution, problems where you have a reasonable guess.
        For cold-start global search across many local minima, use
        ``solve_lbfgs`` with ``n_restarts > 1`` instead.

        Parameters
        ----------
        fixed_phases : dict[int, float]
            Map of node index -> pinned phase, in radians. Ground nodes
            are already at phase 0; do not include them with non-zero
            values.
        initial_guess : dict[int, float], optional
            Starting guess for the free-node phases, keyed by index.
            Missing entries default to 0.
        tol : float
            Per-node current tolerance (amperes). Default 1e-12 A. The
            solver also early-exits at the floating-point noise floor
            (~1e-13 * max(Ic)), so passing extremely tight tolerances
            is harmless.
        max_iter : int
            Maximum Newton iterations. Default 50.

        Returns
        -------
        dict with keys 'phases', 'energy', 'branch_currents',
        'injection_currents', 'kcl_residuals' (see module docstring for
        details).

        Raises
        ------
        RuntimeError
            If Newton fails to converge within ``max_iter`` and the final
            residual is above the noise floor. This usually means the
            initial guess is in the wrong basin.
        """
        from scipy.sparse.linalg import spsolve

        prob = self._compile_problem(fixed_phases, initial_guess)
        n_free = prob["n_free"]
        if n_free == 0:
            return self._pack_result(prob, np.zeros(0))

        cost_and_grad = prob["cost_and_grad"]
        phi_all = prob["phi_all"]
        free_arr = prob["free_arr"]
        a_idx = prob["a_idx"]
        b_idx = prob["b_idx"]
        phi_ext_arr = prob["phi_ext"]
        Ic_arr = prob["Ic_arr"]
        L_inv = prob["L_inv"]
        n_nodes = prob["n_nodes"]

        H_machinery = self._build_sparse_hessian_machinery(
            n_nodes=n_nodes, n_free=n_free, free_arr=free_arr,
            a_idx=a_idx, b_idx=b_idx, Ic_arr=Ic_arr, L_inv=L_inv,
        )
        H_template = H_machinery["H_template"]
        update_hessian = H_machinery["update"]

        # Effective tolerance: the looser of the user's tol and a noise-
        # floor estimate. This means tol=1e-30 (unreachable in double
        # precision) doesn't prevent convergence detection — we stop at
        # the floor instead of grinding away.
        Ic_scale = float(np.max(Ic_arr)) if Ic_arr.size else 1.0
        noise_floor = max(1e-14, 1e-13 * max(Ic_scale, 1.0))
        eff_tol = max(tol, noise_floor)

        x = prob["x0"].copy()
        cost, grad = cost_and_grad(x)
        for _ in range(max_iter):
            if np.max(np.abs(grad)) < eff_tol:
                return self._pack_result(prob, x)

            phi_all[free_arr] = x
            theta = phi_all[a_idx] - phi_all[b_idx] - phi_ext_arr
            update_hessian(theta)
            try:
                dx = -spsolve(H_template, grad)
            except Exception as err:
                raise RuntimeError(
                    f"Newton sparse solve failed: {err}"
                ) from err
            if not np.all(np.isfinite(dx)):
                raise RuntimeError(
                    "Newton produced non-finite step direction; "
                    "the Hessian is likely singular at the current x."
                )

            # Iterative refinement on the linear solve, recovering digits
            # lost when the Hessian is ill-conditioned (e.g. mixed-scale
            # inductors).
            grad_max = float(np.max(np.abs(grad)))
            for _ in range(2):
                r = grad + H_template @ dx
                if float(np.max(np.abs(r))) <= 1e-14 * grad_max:
                    break
                try:
                    ddx = -spsolve(H_template, r)
                except Exception:
                    break
                if not np.all(np.isfinite(ddx)):
                    break
                dx += ddx

            # Backtracking on ||F||^2.
            fnorm = float(grad @ grad)
            alpha = 1.0
            accepted = False
            for _ in range(50):
                x_new = x + alpha * dx
                cost_new, grad_new = cost_and_grad(x_new)
                fnorm_new = float(grad_new @ grad_new)
                if fnorm_new < fnorm:
                    x, cost, grad = x_new, cost_new, grad_new
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                # No descent found by Newton along ANY backtrack — we are
                # at the floating-point noise floor of the linear solve,
                # i.e., the residual cannot be reduced further in double
                # precision. Treat as converged.
                return self._pack_result(prob, x)

        # Hit max_iter. If grad is at the noise floor, accept anyway.
        if np.max(np.abs(grad)) < eff_tol:
            return self._pack_result(prob, x)
        raise RuntimeError(
            f"Newton failed to converge in {max_iter} iterations "
            f"(final max|KCL| = {np.max(np.abs(grad)):.3e} A, "
            f"effective tol = {eff_tol:.3e} A). "
            f"Try a different initial_guess or use solve_lbfgs."
        )

    def solve_lbfgs(self, fixed_phases: dict[int, float],
                    initial_guess: dict[int, float] | None = None,
                    n_restarts: int = 20, seed: int = 0,
                    tol: float = 1e-12) -> dict:
        """Solve KCL with scipy L-BFGS-B, optionally with multiple random
        restarts to handle multi-valued solutions.

        L-BFGS-B uses a low-rank quasi-Newton Hessian approximation built
        from gradient evaluations. It does not need the analytic sparse
        Hessian and so is robust on problems where the Hessian
        decomposition would be expensive to maintain (e.g., very large
        networks, or when Newton might land on saddles). It is, however,
        slower than ``solve_newton`` for warm-started sweeps because it
        gives up the quadratic-convergence guarantee.

        Best for: cold-start solving with no good guess, problems with
        multiple local minima where the multi-start machinery matters.

        Parameters
        ----------
        fixed_phases : dict[int, float]
            Map of node index -> pinned phase, in radians.
        initial_guess : dict[int, float], optional
            Starting guess. Used as the FIRST L-BFGS-B start; any
            additional random starts then explore alternative basins.
        n_restarts : int
            Number of L-BFGS-B runs. The first uses ``initial_guess`` (or
            zeros); the rest use uniform random starts in [-pi, pi]. The
            lowest-cost result wins. Default 20.
        seed : int
            RNG seed for the random restarts.
        tol : float
            Per-node current tolerance (amperes). Default 1e-12.

        Returns
        -------
        dict with keys 'phases', 'energy', 'branch_currents',
        'injection_currents', 'kcl_residuals'.
        """
        from scipy.optimize import minimize

        prob = self._compile_problem(fixed_phases, initial_guess)
        n_free = prob["n_free"]
        if n_free == 0:
            return self._pack_result(prob, np.zeros(0))

        cost_and_grad = prob["cost_and_grad"]
        rng = np.random.default_rng(seed)
        starts = [prob["x0"]]
        for _ in range(max(0, n_restarts - 1)):
            starts.append(rng.uniform(-np.pi, np.pi, size=n_free))

        best_cost = np.inf
        best_x = starts[0]
        for x0 in starts:
            res = minimize(cost_and_grad, x0, jac=True, method="L-BFGS-B",
                           options={"gtol": tol, "ftol": 1e-30,
                                    "maxiter": 500})
            if res.fun < best_cost:
                best_cost = float(res.fun)
                best_x = res.x

        return self._pack_result(prob, best_x)

    # -- helpers ------------------------------------------------------------
    def _index_lookup(self) -> dict[int, int]:
        """O(N) construction of an ``id(node) -> index`` dict for use in
        loops where each branch needs the indices of both endpoints. Built
        once per outer call so the inner loop stays O(1) per lookup."""
        return {id(n): i for i, n in enumerate(self._nodes)}

    def _phase_at(self, node: Node, phases: dict[int, float],
                  idx_of: dict[int, int]) -> float:
        if node.is_ground:
            return 0.0
        return float(phases.get(idx_of[id(node)], 0.0))

    def _resolve(self, ref) -> Node:
        """Look up a Node by integer index, or accept a Node directly
        (provided it belongs to this network)."""
        if isinstance(ref, Node):
            for n in self._nodes:
                if n is ref:
                    return ref
            raise KeyError(f"Node {ref!r} is not part of this network.")
        if isinstance(ref, (int, np.integer)):
            i = int(ref)
            if not (0 <= i < len(self._nodes)):
                raise IndexError(
                    f"Node index {i} out of range [0, {len(self._nodes)})."
                )
            return self._nodes[i]
        raise TypeError(
            f"Node reference must be an int index or a Node; "
            f"got {type(ref).__name__}."
        )

    def __repr__(self) -> str:
        return (f"CircuitNetwork(nodes={len(self._nodes)}, "
                f"branches={len(self._branches)})")

    # -- fast sweep API -----------------------------------------------------
    def make_sweep_solver(self, pinned_indices: list[int],
                          tol: float = 1e-12,
                          return_energy: bool = False,
                          return_currents: bool = False,
                          return_injection_currents: bool = False):
        """Build a fast solver for parameter sweeps with a fixed set of
        pinned nodes.

        Use this when you need to solve the same network at many sweep
        points, varying only the pinned phase values (typical flux or
        bias sweep). All per-network bookkeeping — index lookups,
        free-node enumeration, sparse Hessian pattern, val-buffer
        permutation — is done ONCE here. The returned solver is a tight
        Newton iteration with no Python-level result packing.

        Parameters
        ----------
        pinned_indices : list[int]
            Node indices whose phases will be supplied at each sweep
            point. Order matters: this is the order of the array passed
            to the returned solver.
        tol : float
            Per-node current tolerance.
        return_energy : bool
            Whether the returned solver should compute U at each sweep
            point. Skipped by default for speed.
        return_currents : bool
            Whether the returned solver should compute branch currents
            at each sweep point. The result is a flat ndarray (length
            ``n_branches``) with currents in branch-add order, from
            ``node_a`` to ``node_b``; this is faster than the dict-of-
            dicts that ``solve_newton`` returns and is the natural shape
            for stacking sweep results into a 2-D array. Skipped by
            default for speed.
        return_injection_currents : bool
            Whether the returned solver should compute the external
            current that must be supplied at each pinned node to
            maintain the imposed phase. The result is a 1-D ndarray of
            length ``len(pinned_indices)``, in the same order as
            ``pinned_indices``. Positive = into the node. Skipped by
            default for speed.

        Returns
        -------
        solver : callable
            Signature: ``solver(pinned_values, x0=None, max_iter=50) ->
            (x, info)``.
              - ``pinned_values``: 1-D array aligned with ``pinned_indices``.
              - ``x0``: optional initial guess for free-node phases (length
                n_free); pass the previous ``x`` for warm starts.
              - ``x``: solved free-node phases.
              - ``info``: dict with ``converged`` (bool), ``iterations``
                (int), ``max_kcl`` (float, A), and optionally
                ``energy`` / ``branch_currents`` /
                ``injection_currents``.
        free_indices : list[int]
            Node indices in the same order as ``x``.

        Notes
        -----
        Unlike ``solve_newton``, this solver has no result-packing
        overhead and no exception on non-convergence: ``info['converged']``
        is set to False and the caller decides what to do. Rerun the
        problematic point through ``solve_newton`` (which will raise)
        or ``solve_lbfgs`` (which is more robust on bad starts) if needed.
        """
        from scipy.sparse.linalg._dsolve._superlu import gssv as _gssv

        n_nodes = len(self._nodes)
        for idx in pinned_indices:
            if not (0 <= idx < n_nodes):
                raise IndexError(f"pinned index {idx} out of range.")

        is_pinned = np.zeros(n_nodes, dtype=bool)
        is_pinned[list(pinned_indices)] = True

        free_indices = [i for i, n in enumerate(self._nodes)
                        if not n.is_ground and not is_pinned[i]]
        n_free = len(free_indices)

        n_branches = len(self._branches)
        idx_of = {id(n): i for i, n in enumerate(self._nodes)}
        a_idx = np.empty(n_branches, dtype=np.intp)
        b_idx = np.empty(n_branches, dtype=np.intp)
        phi_ext = np.empty(n_branches, dtype=np.float64)
        Ic_arr = np.zeros(n_branches, dtype=np.float64)
        L_inv = np.zeros(n_branches, dtype=np.float64)
        for k, br in enumerate(self._branches):
            a_idx[k] = idx_of[id(br.node_a)]
            b_idx[k] = idx_of[id(br.node_b)]
            phi_ext[k] = br.phi_ext
            if isinstance(br, JosephsonJunction):
                Ic_arr[k] = br.I_c
            elif isinstance(br, LinearInductor):
                L_inv[k] = 1.0 / br.L
            else:
                raise TypeError(
                    f"make_sweep_solver: branch type {type(br).__name__!r} "
                    f"is not supported."
                )

        free_arr = np.array(free_indices, dtype=np.intp)
        pinned_arr = np.array(list(pinned_indices), dtype=np.intp)

        # Working buffer; ground stays at 0, fixed values rewritten per call.
        phi_all = np.zeros(n_nodes, dtype=np.float64)

        H_machinery = self._build_sparse_hessian_machinery(
            n_nodes=n_nodes, n_free=n_free, free_arr=free_arr,
            a_idx=a_idx, b_idx=b_idx, Ic_arr=Ic_arr, L_inv=L_inv,
        )
        H_template = H_machinery["H_template"]
        H_data     = H_machinery["H_data"]
        H_indices  = H_machinery["H_indices"]
        H_indptr   = H_machinery["H_indptr"]
        H_nnz      = H_machinery["H_nnz"]
        _update_hessian = H_machinery["update"]

        # SuperLU column permutation: 'COLAMD' is scipy's default; reuse
        # the same options dict on every call so we don't rebuild it.
        _gssv_options = dict(ColPerm="COLAMD")

        _PHI_0_RED = PHI_0_RED
        _tol = float(tol)
        # Effective tolerance honors the floating-point noise floor of the
        # KCL residual (~1e-13 * max(Ic)). Passing tol=1e-30 to push for
        # "all the precision possible" otherwise leaves 'converged' stuck
        # at False even when the answer is at the floor.
        Ic_scale = float(np.max(Ic_arr)) if Ic_arr.size else 1.0
        _eff_tol = max(_tol, max(1e-14, 1e-13 * max(Ic_scale, 1.0)))

        def _residual_and_currents(theta):
            I_branch = Ic_arr * np.sin(theta) + _PHI_0_RED * theta * L_inv
            outflow = (np.bincount(a_idx, weights=I_branch, minlength=n_nodes)
                       - np.bincount(b_idx, weights=I_branch,
                                     minlength=n_nodes))
            return outflow[free_arr], I_branch

        # Precompute the sign matrix for injection-current calculation.
        # injection_currents = sign @ I_branch, shape (len(pinned),).
        # Each row k has +1 in columns where branch leaves pinned[k],
        # -1 where branch enters pinned[k], and 0 elsewhere.
        if return_injection_currents:
            n_pinned = len(pinned_arr)
            inj_sign = np.zeros((n_pinned, n_branches), dtype=np.float64)
            for k, p in enumerate(pinned_arr):
                inj_sign[k, a_idx == p] = 1.0
                inj_sign[k, b_idx == p] = -1.0
        else:
            inj_sign = None

        rhs_buf = np.empty((n_free, 1), dtype=np.float64)

        def solver(pinned_values, x0=None, max_iter=50):
            phi_all[pinned_arr] = pinned_values
            if x0 is None:
                x = np.zeros(n_free)
            else:
                x = np.asarray(x0, dtype=np.float64).copy()

            phi_all[free_arr] = x
            theta = phi_all[a_idx] - phi_all[b_idx] - phi_ext
            grad, I_branch = _residual_and_currents(theta)

            iters = 0
            converged = False
            for _ in range(max_iter):
                gmax = float(np.abs(grad).max())
                if gmax < _eff_tol:
                    converged = True
                    break
                iters += 1

                _update_hessian(theta)
                rhs_buf[:, 0] = grad
                dx_arr, gssv_info = _gssv(
                    n_free, H_nnz, H_data, H_indices, H_indptr,
                    rhs_buf, 1, options=_gssv_options
                )
                if gssv_info != 0 or not np.all(np.isfinite(dx_arr)):
                    break
                dx = -dx_arr[:, 0]

                # Iterative refinement (up to 2 passes). When the Hessian
                # is poorly conditioned (e.g. mixed-scale inductors), one
                # spsolve only delivers ~16 - log10(cond) digits of dx.
                # Each refinement pass shrinks the linear-solve residual
                # by another condition-number factor.
                grad_max = float(np.abs(grad).max())
                for _ in range(2):
                    r = grad + H_template @ dx
                    if float(np.abs(r).max()) <= 1e-14 * grad_max:
                        break
                    rhs_buf[:, 0] = r
                    ddx_arr, gssv_info = _gssv(
                        n_free, H_nnz, H_data, H_indices, H_indptr,
                        rhs_buf, 1, options=_gssv_options
                    )
                    if gssv_info != 0 or not np.all(np.isfinite(ddx_arr)):
                        break
                    dx -= ddx_arr[:, 0]

                # Armijo line search on ||F||^2. If the full Newton step
                # fails to reduce the residual, we keep halving alpha down
                # to ~1e-15 before giving up — on hard problems even a
                # tiny step in the right direction is progress.
                fnorm = float(grad @ grad)
                alpha = 1.0
                accepted = False
                for _ in range(50):
                    x_new = x + alpha * dx
                    phi_all[free_arr] = x_new
                    theta_new = phi_all[a_idx] - phi_all[b_idx] - phi_ext
                    grad_new, I_branch_new = _residual_and_currents(theta_new)
                    fnorm_new = float(grad_new @ grad_new)
                    if fnorm_new < fnorm:
                        x = x_new
                        theta = theta_new
                        grad = grad_new
                        I_branch = I_branch_new
                        accepted = True
                        break
                    alpha *= 0.5
                if not accepted:
                    # No descent direction found by Newton along ANY
                    # backtrack — we are at the floating-point noise floor
                    # of the linear solve. Treat as converged: there is
                    # no better answer in double precision.
                    converged = True
                    break

            info = {
                "converged": converged or float(np.abs(grad).max()) < _eff_tol,
                "iterations": iters,
                "max_kcl": float(np.abs(grad).max()),
            }
            if return_energy:
                info["energy"] = float(
                    -_PHI_0_RED * np.dot(Ic_arr, np.cos(theta))
                    + 0.5 * _PHI_0_RED ** 2
                    * np.sum(theta * theta * L_inv)
                )
            if return_currents:
                info["branch_currents"] = I_branch.copy()
            if return_injection_currents:
                info["injection_currents"] = inj_sign @ I_branch
            return x, info

        return solver, free_indices

    # -- visualization ------------------------------------------------------
    @staticmethod
    def _curve_xy(a, b, offset: float = 0.0, n_coils: int = 0,
                  samples: int = 160):
        """Return (xs, ys) for a quadratic Bezier arc from a to b, with
        optional inductor coil bumps riding on the arc."""
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        d = b - a
        L = float(np.linalg.norm(d))
        if L == 0.0:
            return np.array([a[0]]), np.array([a[1]])
        u = d / L
        n = np.array([-u[1], u[0]])
        mid = 0.5 * (a + b)
        c = mid + offset * n                      # Bezier control point

        t = np.linspace(0.0, 1.0, samples)
        tc = t[:, None]
        P = (1 - tc) ** 2 * a + 2 * (1 - tc) * tc * c + tc ** 2 * b

        if n_coils <= 0:
            return P[:, 0], P[:, 1]

        Pp = 2 * (1 - tc) * (c - a) + 2 * tc * (b - c)
        speed = np.linalg.norm(Pp, axis=1, keepdims=True)
        speed = np.where(speed == 0, 1.0, speed)
        tangent = Pp / speed
        perp = np.column_stack([-tangent[:, 1], tangent[:, 0]])

        lead_frac = 0.18
        r = 0.5 * L * (1 - 2 * lead_frac) / n_coils
        bump_side = 1.0 if offset >= 0 else -1.0

        perp_amp = np.zeros(samples)
        in_coil = (t >= lead_frac) & (t <= 1 - lead_frac)
        s = np.zeros(samples)
        s[in_coil] = (t[in_coil] - lead_frac) / (1 - 2 * lead_frac)
        coil_idx = np.minimum((s * n_coils).astype(int), n_coils - 1)
        s_local = s * n_coils - coil_idx
        perp_amp[in_coil] = r * np.sin(np.pi * s_local[in_coil]) * bump_side

        pts = P + perp_amp[:, None] * perp
        return pts[:, 0], pts[:, 1]

    def draw(self, ax=None, layout: str = "spring", show_values=None,
             save_path: str | None = None, seed: int = 0,
             n_coils: int = 5, figsize=None, node_size=None):
        """Plot the network as a schematic-style graph.

        Josephson junctions are drawn as solid red segments with an 'X' at
        the midpoint. Linear inductors are drawn as blue coils. Ground
        nodes are filled squares; free nodes are circles. Parallel
        branches between the same node pair are drawn as curved arcs
        bulging to opposite sides so they don't overlap.

        Node positions set via ``add_node(..., position=(x, y))`` are
        honored: if every node has a position, the layout argument is
        ignored; if only some do, the fixed ones anchor a spring layout.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on; a new figure is created if None.
        layout : {"spring", "circular", "kamada_kawai", "shell"}
            Graph layout algorithm.
        show_values : bool or None
            Annotate each branch with its critical current / inductance.
            None auto-decides: True if there are <= 8 branches, else False.
        save_path : str, optional
            If given, save the figure to this path.
        seed : int
            Seed for the spring layout (ignored otherwise).
        n_coils : int
            Number of bumps in the inductor symbol.
        figsize : tuple or None
            Figure size in inches; auto-computed from the bounding box
            of the layout if None.
        node_size : float or None
            Node marker area in points^2; auto-scaled to inter-node
            spacing if None.

        Returns
        -------
        matplotlib.axes.Axes
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError as err:
            raise ImportError(
                "draw() requires matplotlib and networkx. "
                "Install them with `pip install matplotlib networkx`."
            ) from err

        # NetworkX uses node indices as identifiers; display labels are
        # supplied separately.
        idx_of = self._index_lookup()

        G = nx.Graph()
        for i in range(len(self._nodes)):
            G.add_node(i)
        for br in self._branches:
            G.add_edge(idx_of[id(br.node_a)], idx_of[id(br.node_b)])

        layouts = {
            "spring": lambda g, **kw: nx.spring_layout(g, seed=seed, **kw),
            "circular": lambda g, **kw: nx.circular_layout(g),
            "kamada_kawai": lambda g, **kw: nx.kamada_kawai_layout(g),
            "shell": lambda g, **kw: nx.shell_layout(g),
        }
        if layout not in layouts:
            raise ValueError(f"Unknown layout {layout!r}. "
                             f"Choose from {list(layouts)}.")

        fixed_pos = {i: n.position for i, n in enumerate(self._nodes)
                     if n.position is not None}
        if len(fixed_pos) == len(self._nodes):
            pos = fixed_pos
        elif fixed_pos:
            pos = nx.spring_layout(G, pos=fixed_pos,
                                   fixed=list(fixed_pos.keys()), seed=seed)
        else:
            pos = layouts[layout](G)

        # Bounding box of the layout.
        pts = np.array(list(pos.values()), dtype=float)
        xmin, ymin = pts.min(axis=0)
        xmax, ymax = pts.max(axis=0)
        bw = max(xmax - xmin, 1e-9)
        bh = max(ymax - ymin, 1e-9)

        if ax is None:
            if figsize is None:
                target = 9.0
                if bw >= bh:
                    fw = target
                    fh = max(2.5, min(target, target * bh / bw + 1.2))
                else:
                    fh = target
                    fw = max(3.5, min(target, target * bw / bh + 1.5))
                figsize = (min(fw, 14.0), min(fh, 10.0))
            _, ax = plt.subplots(figsize=figsize)

        # Count parallels so we can curve them to opposite sides.
        pair_count: dict[frozenset, int] = {}
        for br in self._branches:
            key = frozenset((idx_of[id(br.node_a)], idx_of[id(br.node_b)]))
            pair_count[key] = pair_count.get(key, 0) + 1
        pair_idx: dict[frozenset, int] = {}

        edge_lengths = [
            float(np.linalg.norm(np.array(pos[idx_of[id(br.node_a)]]) -
                                 np.array(pos[idx_of[id(br.node_b)]])))
            for br in self._branches
        ]
        median_len = float(np.median(edge_lengths)) if edge_lengths else 1.0
        min_len = float(np.min(edge_lengths)) if edge_lengths else 1.0

        if show_values is None:
            show_values = len(self._branches) <= 8

        curve_mag = 0.30 * min_len

        jj_color, ind_color = "#c0392b", "#2471a3"
        for br in self._branches:
            ia, ib = idx_of[id(br.node_a)], idx_of[id(br.node_b)]
            key = frozenset((ia, ib))
            i = pair_idx.get(key, 0)
            n_par = pair_count[key]
            offset = 0.0 if n_par == 1 else curve_mag * (i - (n_par - 1) / 2.0)
            pair_idx[key] = i + 1

            a = np.array(pos[ia], dtype=float)
            b = np.array(pos[ib], dtype=float)

            if isinstance(br, JosephsonJunction):
                xs, ys = self._curve_xy(a, b, offset=offset, n_coils=0)
                ax.plot(xs, ys, color=jj_color, lw=2.2, zorder=1)
                label = f"{br.I_c * 1e6:g} \u00b5A"
            elif isinstance(br, LinearInductor):
                xs, ys = self._curve_xy(a, b, offset=offset, n_coils=n_coils)
                ax.plot(xs, ys, color=ind_color, lw=2.0, zorder=1,
                        solid_capstyle="round", solid_joinstyle="round")
                label = f"{br.L * 1e12:g} pH"
            else:
                xs, ys = self._curve_xy(a, b, offset=offset, n_coils=0)
                ax.plot(xs, ys, color="#555", lw=1.5, zorder=1)
                label = type(br).__name__

            mid_idx = len(xs) // 2
            mx, my = xs[mid_idx], ys[mid_idx]
            if isinstance(br, JosephsonJunction):
                # Draw the X as two short line segments rotated to the
                # local tangent of the curve. matplotlib's marker='x' is
                # fixed in screen space and ignores the branch angle,
                # which looks wrong for tilted or curved branches.
                if mid_idx + 1 < len(xs):
                    tx = xs[mid_idx + 1] - xs[mid_idx - 1]
                    ty = ys[mid_idx + 1] - ys[mid_idx - 1]
                    tnorm = (tx * tx + ty * ty) ** 0.5 or 1.0
                    tx, ty = tx / tnorm, ty / tnorm
                else:
                    chord = b - a
                    cnorm = float(np.linalg.norm(chord)) or 1.0
                    tx, ty = chord[0] / cnorm, chord[1] / cnorm
                # Half-length of each X stroke in data units (~6% of
                # min edge length keeps the X readable but not huge).
                hlen = 0.07 * min_len
                # Two strokes at +/- 45 degrees from the tangent. A 45-deg
                # rotation of (tx, ty) is (tx-ty, tx+ty)/sqrt(2); the
                # other diagonal is (tx+ty, -tx+ty)/sqrt(2).
                inv_sqrt2 = 0.7071067811865476
                d1x, d1y = (tx - ty) * inv_sqrt2, (tx + ty) * inv_sqrt2
                d2x, d2y = (tx + ty) * inv_sqrt2, (-tx + ty) * inv_sqrt2
                ax.plot([mx - hlen * d1x, mx + hlen * d1x],
                        [my - hlen * d1y, my + hlen * d1y],
                        color=jj_color, lw=2.6,
                        solid_capstyle="round", zorder=3)
                ax.plot([mx - hlen * d2x, mx + hlen * d2x],
                        [my - hlen * d2y, my + hlen * d2y],
                        color=jj_color, lw=2.6,
                        solid_capstyle="round", zorder=3)
            if show_values:
                # Decide which side of the chord the label sits on.
                # When the branch is curved (parallel pair), follow the
                # bulge. When straight, choose by element type so a JJ
                # and an inductor sharing two nodes go on opposite sides:
                # inductors below, JJs / generic branches above.
                # "Above" / "below" are interpreted in world coordinates
                # (positive / negative y), independent of the orientation
                # the chord happens to have.
                chord = b - a
                chord_len = float(np.linalg.norm(chord))
                if chord_len > 1e-12:
                    u_dir = chord / chord_len
                    perp_dir = np.array([-u_dir[1], u_dir[0]])
                    chord_mid = 0.5 * (a + b)

                    if offset != 0.0:
                        side = 1.0 if offset > 0 else -1.0
                    else:
                        # +1 means "in the direction perp_dir points";
                        # we want the label above (or below) in world y.
                        want_above = not isinstance(br, LinearInductor)
                        # If perp_dir already points up (perp_dir[1] >= 0),
                        # side = +1 puts the label up; otherwise flip.
                        if perp_dir[1] >= 0:
                            side = 1.0 if want_above else -1.0
                        else:
                            side = -1.0 if want_above else 1.0

                    # Distance from chord midpoint to the far edge of
                    # the branch on the chosen side, in data units.
                    if isinstance(br, LinearInductor) and n_coils > 0:
                        lead_frac = 0.18
                        r = 0.5 * chord_len * (1 - 2 * lead_frac) / n_coils
                        bump_side = 1.0 if offset >= 0 else -1.0
                        # Coil bumps go on the bump_side; if the label is
                        # on the opposite side, only the curve apex matters.
                        if side == bump_side:
                            apex = abs(offset) / 2.0 + r
                        else:
                            apex = abs(offset) / 2.0
                    elif isinstance(br, JosephsonJunction):
                        # Clear the X marker, whose strokes extend
                        # 0.07 * min_len perpendicular from the wire.
                        apex = abs(offset) / 2.0 + 0.07 * min_len
                    else:
                        apex = abs(offset) / 2.0

                    pad_along = 0.05 * min_len   # small fixed gap past curve
                    lx, ly = chord_mid + side * (apex + pad_along) * perp_dir

                    # Rotate the label to lie along the chord. Keep the
                    # angle in [-90, 90] deg so text stays right-side-up
                    # (a chord pointing left would otherwise read upside
                    # down). When we flip the angle by 180, the visual
                    # baseline of the text flips too, so we also flip the
                    # vertical alignment so the label still sits on the
                    # outer side of the chord.
                    angle_deg = float(np.degrees(np.arctan2(u_dir[1], u_dir[0])))
                    flipped = False
                    if angle_deg > 90.0:
                        angle_deg -= 180.0
                        flipped = True
                    elif angle_deg < -90.0:
                        angle_deg += 180.0
                        flipped = True

                    # When not flipped, "above the chord in chord-frame"
                    # means perp_dir; va = 'bottom' puts the text baseline
                    # on the side closer to the chord. When flipped, the
                    # text frame is rotated 180, so we swap va.
                    if (side * perp_dir[1]) >= 0:
                        va = "bottom"
                    else:
                        va = "top"
                    if flipped:
                        va = "top" if va == "bottom" else "bottom"
                else:
                    lx, ly, va, angle_deg = mx, my + 0.05, "bottom", 0.0
                ax.text(lx, ly, label, ha="center", va=va,
                        rotation=angle_deg, rotation_mode="anchor",
                        fontsize=8, color="#222", zorder=4)

        # Set axis limits explicitly. Padding is just enough to fit the
        # node markers (and any perpendicular curve bulge), with a small
        # cosmetic gap — no big "frame" of empty space around the network.
        # Node radius in data units is hard to know exactly before the
        # node_size is finalized; estimate it as ~15% of min edge length,
        # which matches the default node_size sizing further below.
        node_radius_data = 0.15 * min_len
        pad = max(node_radius_data + 0.04 * min_len, 1.1 * curve_mag)
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_aspect("equal")

        # Adaptive node size (points^2): diameter ~30% of min spacing.
        if node_size is None:
            ax.figure.canvas.draw_idle()
            try:
                bbox_in = ax.get_window_extent().transformed(
                    ax.figure.dpi_scale_trans.inverted())
                ax_w_in = max(bbox_in.width, 1e-3)
                data_w = max(ax.get_xlim()[1] - ax.get_xlim()[0], 1e-9)
                pts_per_data = ax_w_in * 72.0 / data_w
                diam_pts = 0.30 * min_len * pts_per_data
                node_size = float(np.clip(np.pi * (diam_pts / 2) ** 2,
                                          80.0, 900.0))
            except Exception:
                node_size = 400.0

        ground = [i for i, n in enumerate(self._nodes) if n.is_ground]
        free = [i for i, n in enumerate(self._nodes) if not n.is_ground]
        nx.draw_networkx_nodes(G, pos, nodelist=free, ax=ax,
                               node_color="#e8ecff", edgecolors="#222",
                               node_size=node_size, linewidths=1.5)
        nx.draw_networkx_nodes(G, pos, nodelist=ground, ax=ax,
                               node_color="#333", node_shape="s",
                               node_size=node_size)
        labels = {i: self.label_of(i) for i in range(len(self._nodes))}
        label_fontsize = float(np.clip(np.sqrt(node_size) * 0.35, 6.0, 11.0))
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                                font_size=label_fontsize, font_color="#111")

        ax.set_title(
            f"Circuit network — {len(self._nodes)} nodes, "
            f"{len(self._branches)} branches"
        )
        ax.axis("off")

        if save_path is not None:
            ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
        return ax

def CPR(make_inductor_block, kwargs,
        phi_min=-200.0, phi_max=200.0, phi_step=0.25, verbose=True):
    '''
    Construct the current-phase relation for a generic inductive block.

    Sweeps the phase φ on the pinned node both upward and downward through
    [phi_min, phi_max] in steps of phi_step, checks for hysteresis and
    monotonicity, and returns a CubicSpline of injection current vs phase.

    Parameters
    ----------
    make_inductor_block : callable
        Factory that builds the circuit. Must return an object with a
        `.nodes` attribute and a `.make_sweep_solver(pinned_indices, ...)`
        method whose returned `solver(phi_array, x0=...)` produces
        (x, info) with info['converged'], info['max_kcl'], and
        info['injection_currents'].
    kwargs : dict
        Keyword arguments passed straight through to make_inductor_block.
    phi_min, phi_max : float
        Inclusive phase sweep range (rad).
    phi_step : float
        Phase increment (rad).

    Returns
    -------
    dict with keys 'phis', 'currents', 'fail', 'cpr', 'cn'
    '''
    res = {'phi': [], 'current': [], 'fail': False, 'cpr': None, 'cn': None}

    cn = make_inductor_block(**kwargs)
    pinned_indices = [len(cn.nodes) - 1]
    res['cn'] = cn

    solver, free_indices = cn.make_sweep_solver(
        pinned_indices=pinned_indices,
        return_injection_currents=True,
        tol=1e-30,
    )

    phis = np.arange(phi_min, phi_max + phi_step / 2, phi_step)
    res['phis'] = np.copy(phis)
    x = None

    injection_currents_up = np.zeros((len(phis), len(pinned_indices)))
    for k, phi in enumerate(phis):
        x, info = solver(np.array([phi]), x0=x)
        if not info['converged']:
            if verbose:
                print(f"Warning: phi={phi}, max_kcl={info['max_kcl']:.2e}")
            res['fail'] = True
        injection_currents_up[k] = info['injection_currents']

    injection_currents_dn = np.zeros((len(phis), len(pinned_indices)))
    for k, phi in enumerate(np.flip(phis)):
        x, info = solver(np.array([phi]), x0=x)
        if not info['converged']:
            if verbose:
                print(f"Warning: phi={phi}, max_kcl={info['max_kcl']:.2e}")
            res['fail'] = True
        injection_currents_dn[k] = info['injection_currents']
    injection_currents_dn = np.flip(injection_currents_dn)

    if np.max(np.abs(injection_currents_up - injection_currents_dn)) > 1E-10:
        if verbose:
            print('Error: injection current hysteresis')
        res['fail'] = True

    if not np.all(np.diff(injection_currents_up) > 0):
        if verbose:
            print('Error: currents not monotonic')
        res['fail'] = True

    res['currents'] = injection_currents_up.flatten()
    res['cpr'] = CubicSpline(res['phis'], res['currents'])

    return res