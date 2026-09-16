# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F405

"""Thermal topology optimisation on an arbitrary hexahedral mesh.

Uses FEniCS (DOLFIN 2019.1.0) + dolfin-adjoint to solve steady-state heat
conduction with SIMP material interpolation and compute the exact adjoint
gradient of the thermal compliance objective.

CRITICAL import order: dolfin_adjoint must immediately follow dolfin so that
it can monkey-patch solve/assemble and record operations on the adjoint tape.
"""

import hashlib
import os
import tempfile
from typing import Any

import meshio
import numpy as np
from dolfin import *  # noqa: F403
from dolfin_adjoint import *  # noqa: F403

# Legacy dolfin-adjoint (e.g. dolfin-adjoint==2019.1.0 from conda-forge, as
# pinned in tesseract_environment.yaml) does not propagate the forward
# solve's `solver_parameters` to the adjoint linear solve, which then
# defaults to UMFPACK and runs out of memory well before mumps would (see
# pasteurlabs/mosaic#180). Force it to mumps too. Newer dolfin-adjoint
# checkouts fixed this upstream (SolveVarFormBlock now forwards
# `linear_solver` into the adjoint solve automatically) and no longer expose
# this compat shim, so skip the patch if it's absent.
try:
    import fenics_adjoint.types.compat as _fa_compat

    def _adjoint_linalg_solve_mumps(*args: Any, **kwargs: Any) -> Any:
        return _fa_compat.backend.solve(*args, "mumps")

    _fa_compat.linalg_solve = _adjoint_linalg_solve_mumps
except ImportError:
    pass

from mosaic_shared.problems.thermal_mesh import (
    InputSchema as _CanonicalInputSchema,
)
from mosaic_shared.problems.thermal_mesh import (
    OutputSchema as _CanonicalOutputSchema,
)
from mosaic_shared.schema_types import make_differentiable
from pydantic import Field
from scipy.spatial import cKDTree
from tesseract_core.runtime import ShapeDType


class InputSchema(make_differentiable(_CanonicalInputSchema, ["rho", "source"])):
    """Inputs for FEniCS heat solver, extended with material parameters."""

    k_max: float = Field(
        default=1.0,
        description="Maximum thermal conductivity (fully solid/conducting material).",
    )
    p_exp: float = Field(
        default=3.0,
        description="SIMP penalisation exponent p (k(ρ) = k_min + (k_max−k_min)·ρ^p).",
    )


class OutputSchema(
    make_differentiable(
        _CanonicalOutputSchema, ["thermal_compliance", "identification_error"]
    )
):
    """FEniCS thermal solver output schema."""


# ---------------------------------------------------------------------------
# Mesh conversion helpers  (copied verbatim from fenics-brinkman)
# ---------------------------------------------------------------------------


def _build_fenics_mesh(pts: np.ndarray, cells: np.ndarray) -> Mesh:
    """Convert numpy hex mesh arrays to a FEniCS Mesh via meshio XDMF.

    DOLFIN XML only supports triangles/tetrahedra; XDMF supports hexahedra.

    Args:
        pts: Node coordinates, shape (n_nodes, 3), float64.
        cells: Hex cell connectivity, shape (n_cells, 8), int64.

    Returns:
        FEniCS Mesh object.
    """
    mio_mesh = meshio.Mesh(
        points=pts.astype(np.float64),
        cells=[("hexahedron", cells.astype(np.int64))],
    )
    fd, xdmf_path = tempfile.mkstemp(suffix=".xdmf")
    os.close(fd)
    h5_path = xdmf_path.replace(".xdmf", ".h5")
    try:
        meshio.write(xdmf_path, mio_mesh, file_format="xdmf")
        mesh = Mesh()
        with XDMFFile(xdmf_path) as xf:
            xf.read(mesh)
    finally:
        for p in (xdmf_path, h5_path):
            if os.path.exists(p):
                os.unlink(p)
    return mesh


def _cell_reorder_map(
    pts: np.ndarray, input_cells: np.ndarray, fenics_mesh: Mesh
) -> np.ndarray:
    """Build FEniCS-cell-index → input-cell-index permutation via centroid matching.

    FEniCS may reorder cells when loading a mesh. This function recovers the
    mapping so that rho_values[input_idx] can be assigned to the correct
    FEniCS DG0 DOF, and the adjoint gradient can be mapped back.

    Args:
        pts: Input mesh node coordinates, shape (n_nodes, 3).
        input_cells: Input cell connectivity, shape (n_input_cells, 8).
        fenics_mesh: The FEniCS Mesh built from the same data.

    Returns:
        Array of shape (n_fenics_cells,) where entry j gives the input cell
        index that corresponds to FEniCS cell j.
    """
    input_centroids = pts[input_cells].mean(axis=1)  # (n_cells, 3)

    n_cells_f = fenics_mesh.num_cells()
    fenics_centroids = np.array(
        [Cell(fenics_mesh, i).midpoint().array() for i in range(n_cells_f)]
    )  # (n_cells_f, 3)

    tree = cKDTree(input_centroids)
    _, fenics_to_input = tree.query(fenics_centroids)
    return fenics_to_input


# ---------------------------------------------------------------------------
# Neumann facet marker helper
# ---------------------------------------------------------------------------


def _mark_neumann_facets(mesh: Mesh, neumann_mask_vals: np.ndarray) -> MeshFunction:
    """Mark boundary facets by Neumann group from a per-node mask.

    A boundary facet is assigned group k if ALL of its vertices carry
    neumann_mask == k (with k > 0).  Facets on the Dirichlet boundary or
    interior facets remain unmarked (tag = 0).

    Args:
        mesh: FEniCS Mesh object.
        neumann_mask_vals: Integer array of length ≥ n_vertices.  Entry i gives
            the Neumann group (1-indexed) of vertex i; 0 means no flux.

    Returns:
        MeshFunction of size_t defined on facets, with tag k > 0 for every
        boundary facet whose vertices all belong to Neumann group k.
    """
    facet_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    facet_markers.set_all(0)
    # Build facet-to-vertex connectivity (required before iterating).
    mesh.init(mesh.topology().dim() - 1, 0)
    for facet in facets(mesh):
        if not facet.exterior():
            continue
        verts = facet.entities(0)  # vertex global indices
        groups = [
            int(neumann_mask_vals[v]) for v in verts if v < len(neumann_mask_vals)
        ]
        if groups and len(set(groups)) == 1 and groups[0] > 0:
            facet_markers[facet.index()] = groups[0]
    return facet_markers


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------
#
# `_SETUP_CACHE` holds a persistent, replayable pair of `ReducedFunctional`s
# — thermal compliance (`Chat`) and identification error (`Ihat`), each with
# controls ``[rho, source]`` — per (mesh, BC, material, target_temperature)
# combination, i.e. everything the problem depends on *except* rho/source.
# Building it does ONE real solve. Every later evaluation at a different
# rho/source — the entire point of a topology-optimisation or source-
# identification run, which calls `apply` thousands of times against the
# same mesh — calls `Chat([new_rho, new_source])` instead: dolfin-adjoint
# updates both controls' checkpoints and replays every already-recorded
# Block's `recompute()` in place (see `pyadjoint.ReducedFunctional.__call__`
# / `Tape.reset_blocks`). This reuses the compiled UFL forms and re-solves
# the *linear system* but skips reconstructing Constants/Measures/
# DirichletBCs/forms from Python and re-hitting FFC's JIT-compile cache
# lookup on every call. `Chat` and `Ihat` share one tape built from one
# solve, so replaying `Chat` also refreshes every block `Ihat` needs
# (`Tape.reset_blocks`/`get_blocks` operate on the whole tape, not per
# functional) — one replay serves both objectives. So rho/source are plain
# arguments here, never part of a cache key — the cached object is a pair of
# functionals that can be replayed at any (rho, source), not a solve at one
# point.
#
# Each `ReducedFunctional` carries *both* controls, so `.derivative()`
# returns `[d/drho, d/dsource]` from a single adjoint sweep — the adjoint
# method's cost is ~independent of the number of controls, so this is one
# backward pass instead of two.
#
# `vector_jacobian_product` is self-contained: it replays
# `Chat([rho, source])` at the point it was handed and then takes the
# adjoint(s) it needs (dolfin-adjoint's derivative is defined "around the
# last supplied value of the control" — see
# `ReducedFunctional.derivative`). It deliberately does NOT reuse a forward
# solution cached by a preceding `apply` call, even though tesseract-jax
# always invokes the two back-to-back at the same point: every other
# differentiable solver in the suite re-evaluates the forward pass inside
# its own VJP endpoint (jax-fem calls `jax.vjp` fresh each time, and
# tesseract-core's shared `jax_recipes` residual cache is disabled
# everywhere), so reusing state across the two endpoints would make this
# solver's measured VJP cost incomparable with the rest of the benchmark.
#
# The source field is always an explicit tape-tracked control (even when
# every value is zero, which leaves the solution unchanged) so a gradient
# w.r.t. source is always well-defined from the one shared solve.

_SETUP_CACHE: dict[str, dict[str, Any]] = {}


def _setup_cache_key(
    pts: np.ndarray,
    cells: np.ndarray,
    dirichlet_mask_vals: np.ndarray,
    dirichlet_values_vals: np.ndarray,
    neumann_mask_vals: np.ndarray,
    neumann_values_vals: np.ndarray,
    k_max: float,
    p_exp: float,
    target_temperature: np.ndarray,
) -> str:
    """Hash everything the Chat/Ihat graphs depend on — i.e. everything
    except rho/source, which `Chat([rho, source])` replays the graph at."""
    h = hashlib.sha256()
    for arr in (
        pts,
        cells,
        dirichlet_mask_vals,
        dirichlet_values_vals,
        neumann_mask_vals,
        neumann_values_vals,
        target_temperature,
    ):
        h.update(np.ascontiguousarray(arr).tobytes())
    h.update(np.array([k_max, p_exp], dtype=np.float64).tobytes())
    return h.hexdigest()


def _build_reduced_functionals(
    pts: np.ndarray,
    cells: np.ndarray,
    dirichlet_mask_vals: np.ndarray,
    dirichlet_values_vals: np.ndarray,
    neumann_mask_vals: np.ndarray,
    neumann_values_vals: np.ndarray,
    k_max: float,
    p_exp: float,
    target_temperature: np.ndarray,
) -> dict[str, Any]:
    """One-time setup for a (mesh, BC, material, target) combination: mesh,
    function spaces, ONE annotated forward solve with rho/source as
    controls, wrapped as two ReducedFunctionals sharing that tape.

    Solves the 3-D steady-state heat conduction topology optimisation problem:
        -∇·(k(ρ) ∇T) = 0    in Ω

    with SIMP conductivity:
        k(ρ) = k_min + (k_max − k_min) · ρ^p    (k_min = 1e-3 · k_max)

    Boundary conditions:
        T = T_prescribed                  on Γ_D  (Dirichlet groups)
        k(ρ) ∇T · n = q_n               on Γ_N  (Neumann groups)

    Thermal compliance objective:
        C = ∮_Γ_N q_n · T dΓ

    Identification-error objective (area-weighted proxy for the nodal
    ``sum((T - T_target)^2)`` computed in `apply`; see the nodal-correction
    comment below):
        I = ∫_Ω (T - T_target)² dΩ

    Returns:
        ``{"Chat", "Ihat", "rho_space", "fenics_to_input", "mesh", "T_sol",
        "nodal_correction"}`` — see module docstring above for how these get
        reused at every later (rho, source).
    """
    mesh = _build_fenics_mesh(pts, cells)
    fenics_to_input = _cell_reorder_map(pts, cells, mesh)

    # P1 (CG degree 1) for temperature; DG0 for piecewise-constant density.
    V = FunctionSpace(mesh, "CG", 1)
    DG0 = FunctionSpace(mesh, "DG", 0)
    neumann_facet_markers = _mark_neumann_facets(mesh, neumann_mask_vals)
    # DOLFIN 2019.1.0 requires a facet MeshFunction (not vertex) for
    # DirichletBC.  A boundary facet is assigned Dirichlet group k if ALL
    # of its vertices carry dirichlet_mask == k.  This is identical to the
    # Neumann facet marking logic.
    dirichlet_facet_markers = _mark_neumann_facets(mesh, dirichlet_mask_vals)

    set_working_tape(Tape())

    # ---- Density & source fields (arbitrary initial values — every `apply`
    # replays this graph at the real (rho, source) via `Chat([rho, source])`)
    rho_fn = Function(DG0, name="rho")
    rho_fn.vector()[:] = 0.5
    source_fn = Function(DG0, name="source")
    source_fn.vector()[:] = 0.0

    # ---- SIMP conductivity ------------------------------------------------
    # k(ρ) = k_min + (k_max − k_min) · ρ^p,  k_min = 1e-3 · k_max
    k_min = Constant(1e-3 * k_max)
    k_simp = k_min + (Constant(k_max) - k_min) * rho_fn**p_exp

    ds_N = Measure("ds", domain=mesh, subdomain_data=neumann_facet_markers)

    # ---- Variational problem ----------------------------------------------
    T = TrialFunction(V)
    v = TestFunction(V)

    a = inner(k_simp * grad(T), grad(v)) * dx

    # Build Neumann right-hand side: sum of q_n · v integrated over each
    # Neumann group's facets.  Starting from a zero scalar form avoids the
    # need to special-case an empty Neumann set.
    n_neumann_groups = neumann_values_vals.shape[0]
    L = Constant(0.0) * v * dx
    for k in range(n_neumann_groups):
        q_n = Constant(float(neumann_values_vals[k, 0]))
        L = L + q_n * v * ds_N(k + 1)

    # Body heat source: ∫_Ω f · v dΩ.
    L = L + source_fn * v * dx

    # ---- Dirichlet BCs ---------------------------------------------------
    bcs = []
    for k in range(dirichlet_values_vals.shape[0]):
        T_prescribed = Constant(float(dirichlet_values_vals[k, 0]))
        bc = DirichletBC(V, T_prescribed, dirichlet_facet_markers, k + 1)
        bcs.append(bc)

    # ---- Solve -------------------------------------------------------------
    # mumps: UMFPACK (the FEniCS default) runs out of memory at moderate mesh
    # sizes; the adjoint solve picks this up too (see the mumps patch above).
    # This `solver_parameters` choice is captured on the Block and reused by
    # every later replay automatically.
    T_sol = Function(V)
    solve(a == L, T_sol, bcs, solver_parameters={"linear_solver": "mumps"})

    # ---- Objective: thermal compliance ------------------------------------
    # C = ∮_Γ_N q_n · T dΓ
    # assemble is monkey-patched by dolfin_adjoint and recorded on the tape.
    J_form = Constant(0.0) * T_sol * dx
    for k in range(n_neumann_groups):
        q_n = Constant(float(neumann_values_vals[k, 0]))
        J_form = J_form + q_n * T_sol * ds_N(k + 1)
    J = assemble(J_form)

    # ---- Objective: identification error -----------------------------------
    # Built on the same tape/T_sol — replaying `Chat` also refreshes this.
    T_target_fn = Function(V)
    d2v = dof_to_vertex_map(V)
    T_tgt = np.asarray(target_temperature, dtype=np.float64)
    target_at_dofs = np.zeros(V.dim(), dtype=np.float64)
    for dof_i in range(V.dim()):
        vert_i = int(d2v[dof_i])
        if vert_i < len(T_tgt):
            target_at_dofs[dof_i] = float(T_tgt[vert_i])
    T_target_fn.vector()[:] = target_at_dofs
    diff = T_sol - T_target_fn
    I = assemble(inner(diff, diff) * dx)

    # Nodal correction: identification_error (forward) = sum(nodal diff^2),
    # while dolfin-adjoint differentiates ∫(T-T_t)² dΩ (area-weighted).
    coords = mesh.coordinates()
    domain_vol = float(
        np.prod([coords[:, i].max() - coords[:, i].min() for i in range(coords.shape[1])])
    )
    nodal_correction = float(mesh.num_vertices()) / domain_vol

    controls = [Control(rho_fn), Control(source_fn)]
    Chat = ReducedFunctional(J, controls)
    Ihat = ReducedFunctional(I, [Control(rho_fn), Control(source_fn)])

    return {
        "Chat": Chat,
        "Ihat": Ihat,
        "rho_space": DG0,
        "fenics_to_input": fenics_to_input,
        "mesh": mesh,
        "T_sol": T_sol,
        "nodal_correction": nodal_correction,
    }


def _get_reduced_functionals(
    pts: np.ndarray,
    cells: np.ndarray,
    dirichlet_mask_vals: np.ndarray,
    dirichlet_values_vals: np.ndarray,
    neumann_mask_vals: np.ndarray,
    neumann_values_vals: np.ndarray,
    k_max: float,
    p_exp: float,
    target_temperature: np.ndarray,
) -> dict[str, Any]:
    """Build (or fetch) the cached Chat/Ihat pair for this (mesh, BC,
    material, target_temperature) combination."""
    key = _setup_cache_key(
        pts,
        cells,
        dirichlet_mask_vals,
        dirichlet_values_vals,
        neumann_mask_vals,
        neumann_values_vals,
        k_max,
        p_exp,
        target_temperature,
    )
    entry = _SETUP_CACHE.get(key)
    if entry is not None:
        return entry

    entry = _build_reduced_functionals(
        pts,
        cells,
        dirichlet_mask_vals,
        dirichlet_values_vals,
        neumann_mask_vals,
        neumann_values_vals,
        k_max,
        p_exp,
        target_temperature,
    )
    _SETUP_CACHE.clear()
    _SETUP_CACHE[key] = entry
    return entry


def _gradient_pair(reduced_functional: Any, n_input_cells: int, fenics_to_input: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(d/drho, d/dsource) from one adjoint sweep, mapped to input cell order."""
    d_rho_fn, d_source_fn = reduced_functional.derivative()
    d_rho = np.zeros(n_input_cells)
    d_rho[fenics_to_input] = d_rho_fn.vector().get_local()
    d_source = np.zeros(n_input_cells)
    d_source[fenics_to_input] = d_source_fn.vector().get_local()
    return d_rho, d_source


def _solve_forward(
    rho_values: np.ndarray,
    source_values: np.ndarray,
    pts: np.ndarray,
    cells: np.ndarray,
    dirichlet_mask_vals: np.ndarray,
    dirichlet_values_vals: np.ndarray,
    neumann_mask_vals: np.ndarray,
    neumann_values_vals: np.ndarray,
    k_max: float,
    p_exp: float,
    target_temperature: np.ndarray,
) -> dict[str, Any]:
    """Evaluate thermal_compliance at this (rho, source) by replaying the
    cached `Chat` (which also refreshes `Ihat`'s shared tape state — see
    module docstring above).
    """
    entry = _get_reduced_functionals(
        pts,
        cells,
        dirichlet_mask_vals,
        dirichlet_values_vals,
        neumann_mask_vals,
        neumann_values_vals,
        k_max,
        p_exp,
        target_temperature,
    )
    fenics_to_input = entry["fenics_to_input"]
    n_cells_f = len(fenics_to_input)

    rho_fn = Function(entry["rho_space"])
    rho_fn.vector()[:] = np.clip(rho_values[fenics_to_input], 0.0, 1.0)

    source_fn = Function(entry["rho_space"])
    src_reordered = np.zeros(n_cells_f, dtype=np.float64)
    for fd_idx in range(n_cells_f):
        inp_idx = int(fenics_to_input[fd_idx])
        if inp_idx < len(source_values):
            src_reordered[fd_idx] = float(source_values[inp_idx])
    source_fn.vector()[:] = src_reordered

    J = entry["Chat"]([rho_fn, source_fn])
    # `entry["T_sol"]` is the Python object from the ONE-TIME setup solve;
    # dolfin-adjoint's replay creates a fresh Function each time (see
    # `GenericSolveBlock._create_initial_guess`) and stores the result on the
    # ORIGINAL block_variable's checkpoint rather than mutating that first
    # object in place, so the current value must be read back through it.
    T_current = entry["T_sol"].block_variable.saved_output
    T_vertices = T_current.compute_vertex_values(entry["mesh"])

    return {
        "Chat": entry["Chat"],
        "Ihat": entry["Ihat"],
        "J": J,
        "T_vertices": T_vertices,
        "fenics_to_input": fenics_to_input,
        "n_input_cells": len(rho_values),
        "nodal_correction": entry["nodal_correction"],
    }


# ---------------------------------------------------------------------------
# Tesseract endpoints
# ---------------------------------------------------------------------------


def apply(inputs: InputSchema) -> OutputSchema:
    """Forward pass: solve heat conduction and return compliance + temperature.

    Args:
        inputs: Validated InputSchema containing the density field, mesh,
                boundary conditions, and material parameters.

    Returns:
        OutputSchema with thermal_compliance (scalar), temperature (n_vertices,),
        and identification_error (scalar).
    """
    hm = inputs.hex_mesh
    pts = np.asarray(hm.points[: hm.n_points], dtype=np.float64)
    cells = np.asarray(hm.faces[: hm.n_faces], dtype=np.int64)
    rho_values = np.asarray(inputs.rho[: hm.n_faces], dtype=np.float64)
    source_values = np.asarray(inputs.source[: hm.n_faces], dtype=np.float64)
    target_temp = np.asarray(inputs.target_temperature, dtype=np.float32)
    bc = inputs.boundary_conditions
    dm = np.asarray(bc.dirichlet.mask if bc.dirichlet else [])
    dv = np.asarray(
        bc.dirichlet.values
        if bc.dirichlet and bc.dirichlet.values is not None
        else np.zeros((0, 1)),
        dtype=np.float64,
    )
    vm = np.asarray(bc.neumann.mask if bc.neumann else [])
    vv = np.asarray(
        bc.neumann.values if bc.neumann else np.zeros((0, 1)), dtype=np.float64
    )

    state = _solve_forward(
        rho_values,
        source_values,
        pts,
        cells,
        dm,
        dv,
        vm,
        vv,
        inputs.k_max,
        inputs.p_exp,
        np.asarray(inputs.target_temperature, dtype=np.float64),
    )

    T_f32 = state["T_vertices"].astype(np.float32)
    n = min(len(T_f32), len(target_temp))
    id_error = np.float32(np.sum((T_f32[:n] - target_temp[:n]) ** 2))

    return OutputSchema(
        thermal_compliance=np.float32(float(state["J"])),
        identification_error=id_error,
    )


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """VJP via dolfin-adjoint: ∂C/∂ρ, ∂C/∂source, ∂id_err/∂ρ, ∂id_err/∂source.

    Self-contained, matching every other differentiable solver in the suite:
    replays the forward problem at the (rho, source) it is given and then
    computes the adjoint(s), rather than reusing a solution cached by a
    preceding ``apply`` call — see the module docstring above for why. The
    replay reuses the cached mesh/function spaces and the compiled UFL
    forms, so no mesh rebuild or form reconstruction happens. Each
    functional carries both controls, so one ``.derivative()`` call yields
    both ``d/drho`` and ``d/dsource`` from a single adjoint sweep.

    Supports:
        rho    → thermal_compliance  (SIMP adjoint via dolfin-adjoint)
        rho    → identification_error (SIMP adjoint on ||T-T_target||² functional)
        source → thermal_compliance
        source → identification_error (nodal L2 adjoint with area correction)

    Args:
        inputs: Validated InputSchema.
        vjp_inputs: Names of inputs for which gradients are requested.
        vjp_outputs: Names of outputs whose cotangents are provided.
        cotangent_vector: Dict of output-name → cotangent scalar/array.

    Returns:
        Dict mapping requested input names to gradient arrays.
    """
    want_rho = "rho" in vjp_inputs
    want_source = "source" in vjp_inputs

    if not want_rho and not want_source:
        return {}

    hm = inputs.hex_mesh
    pts = np.asarray(hm.points[: hm.n_points], dtype=np.float64)
    cells = np.asarray(hm.faces[: hm.n_faces], dtype=np.int64)
    rho_values = np.asarray(inputs.rho[: hm.n_faces], dtype=np.float64)
    source_values = np.asarray(inputs.source[: hm.n_faces], dtype=np.float64)
    bc = inputs.boundary_conditions
    dm = np.asarray(bc.dirichlet.mask if bc.dirichlet else [])
    dv = np.asarray(
        bc.dirichlet.values
        if bc.dirichlet and bc.dirichlet.values is not None
        else np.zeros((0, 1)),
        dtype=np.float64,
    )
    vm = np.asarray(bc.neumann.mask if bc.neumann else [])
    vv = np.asarray(
        bc.neumann.values if bc.neumann else np.zeros((0, 1)), dtype=np.float64
    )
    state = _solve_forward(
        rho_values,
        source_values,
        pts,
        cells,
        dm,
        dv,
        vm,
        vv,
        inputs.k_max,
        inputs.p_exp,
        np.asarray(inputs.target_temperature, dtype=np.float64),
    )

    n_input_cells = state["n_input_cells"]
    fenics_to_input = state["fenics_to_input"]
    result = {}
    grad_rho = np.zeros(len(np.asarray(inputs.rho)), dtype=np.float32) if want_rho else None
    grad_source = (
        np.zeros(len(np.asarray(inputs.source)), dtype=np.float32) if want_source else None
    )

    # One adjoint sweep per objective (not per control): each ReducedFunctional
    # carries both [rho, source] controls, so `.derivative()` yields both
    # gradients at once.
    cot_compliance = float(cotangent_vector.get("thermal_compliance", 0.0))
    if cot_compliance != 0.0:
        dC_drho, dC_dsource = _gradient_pair(state["Chat"], n_input_cells, fenics_to_input)
        if want_rho:
            grad_rho[: hm.n_faces] += (dC_drho * cot_compliance).astype(np.float32)
        if want_source:
            grad_source[: hm.n_faces] += (dC_dsource * cot_compliance).astype(np.float32)

    cot_id_error = float(cotangent_vector.get("identification_error", 0.0))
    if cot_id_error != 0.0:
        dI_drho, dI_dsource = _gradient_pair(state["Ihat"], n_input_cells, fenics_to_input)
        nodal_correction = state["nodal_correction"]
        if want_rho:
            grad_rho[: hm.n_faces] += (
                dI_drho * nodal_correction * cot_id_error
            ).astype(np.float32)
        if want_source:
            grad_source[: hm.n_faces] += (
                dI_dsource * nodal_correction * cot_id_error
            ).astype(np.float32)

    if want_rho:
        result["rho"] = grad_rho
    if want_source:
        result["source"] = grad_source

    return result


def abstract_eval(abstract_inputs: InputSchema) -> dict:
    """Shape inference without running the solver."""
    return {
        "thermal_compliance": ShapeDType(shape=(), dtype="float32"),
        "identification_error": ShapeDType(shape=(), dtype="float32"),
    }
