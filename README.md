# JuliaGMG

A Julia geometric-multigrid-preconditioned Krylov solver. Solves 2D variable-coefficient PDEs on the unit square with Dirichlet boundary conditions using a Krylov outer loop (GMRES or CG) preconditioned by one multigrid V-cycle per iteration. The implementation runs on CPU (plain Julia loops); GPU kernels via `KernelAbstractions.jl` and `Adapt.jl` are planned, and the types are already parameterised for it (see below).

The built-in operator is the variable-coefficient Helmholtz equation:

```
(α(x,y) − ∇²) u = f    on [0,1]²,  u = 0 on ∂Ω
```

Any other 5-point finite-difference operator can be plugged in by supplying a stencil function and a coarsening function — no changes to the library are required.

---

## Installation

JuliaGMG is distributed through the BSLRegistry (registration pending until v0.1.0 is registered):

```julia
pkg> registry add https://gitlab.mpcdf.mpg.de/bsl6d/BSLRegistry.git
pkg> add JuliaGMG
```

Run the tests from a checkout:

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

---

## Package Structure

The repository is split into a **generic library** (`src/`) and a **Helmholtz application layer** (`helmholtz/`). The library knows nothing about the specific PDE; the application layer supplies the operator constructor, coefficient fields, and the top-level runner.

```
JuliaGMG.jl/
├── Project.toml          # JuliaGMG 0.1.0; deps: KernelAbstractions, Krylov, LinearOperators, ...
├── bin/
│   ├── Project.toml      # CLI environment: JuliaGMG (path ".."), LinearAlgebra, Printf, Random
│   └── julia_gmg.jl      # CLI: loads library + includes helmholtz/ files, parses args, calls run_solver
├── src/                  # JuliaGMG library — generic multigrid, agnostic to the specific PDE
│   ├── JuliaGMG.jl        # Module root: includes all files and re-exports public API
│   ├── types.jl           # FDOp, Level, MGHierarchy, build_hierarchy
│   ├── operators.jl       # Matrix-free apply!, BCs, residual, vec_to_padded!/padded_to_vec!
│   ├── smoother.jl        # Weighted Jacobi and Red-Black Gauss-Seidel smoothers
│   ├── transfer.jl        # Full-weight restriction and piecewise-constant prolongation
│   ├── vcycle.jl          # Recursive V-cycle and coarse-grid direct solve (LU)
│   └── solver.jl          # GMRES/CG dispatcher types, MGPreconditioner, solve()
└── helmholtz/            # Helmholtz application layer — included via include() in bin/julia_gmg.jl
    ├── helmholtz_op.jl    # HelmholtzOp constructor (returns an FDOp)
    ├── alpha_fields.jl    # Six generators for the spatially varying coefficient α(x,y)
    └── runner.jl          # make_rhs_and_exact, run_solver (manufactures RHS, calls solve(), reports error)
└── test/                 # Pkg.test(): MMS convergence, h-independent iterations, apply! vs dense matrix
```

### Data layout convention

All field arrays are **padded**: interior cell `(i, j)` maps to array index `[i+1, j+1]`, so the surrounding ring `[1, :]`, `[end, :]`, `[:, 1]`, `[:, end]` holds Dirichlet ghost values. The Krylov library works with flat vectors; `vec_to_padded!` / `padded_to_vec!` convert between the two.

---

## Type Hierarchy

```
FDOp{T, A, StencilFn, CoarsenFn}   generic matrix-free 5-point FD operator
HelmholtzOp(alpha, ...)             Helmholtz convenience constructor returning an FDOp (helmholtz/ layer)
Level{T, A, Op}                     one multigrid level; owns x, b, r arrays + an operator
MGHierarchy{T, L, F}                full level stack, smoothing counts, pre-factorised coarse-grid LU
MGPreconditioner{T, H}              thin wrapper that exposes MGHierarchy as a LinearAlgebra preconditioner
GMRES / CG                          empty structs used as solver selector tokens for solve()
```

Type parameter `A <: AbstractMatrix{T}` propagates from the finest operator down through every level, and `Adapt.@adapt_structure` is applied to `FDOp` and `Level`, in preparation for GPU arrays (e.g. `CUDA.CuMatrix{Float32}`). The loops in `apply!`, the smoothers and the transfer operators are plain CPU loops today; GPU kernels are planned.

---

## The Generic Operator: `FDOp`

`FDOp` is parameterised by two callables that encode the operator and its multigrid coarsening rule:

```julia
struct FDOp{T, A<:AbstractMatrix{T}, StencilFn, CoarsenFn}
    coeff::A            # padded (nx+2)×(ny+2) coefficient array
    nx::Int
    ny::Int
    hx::T
    hy::T
    stencil_fn::StencilFn   # (i, j, coeff, ihx2, ihy2) -> (diag, wx_neg, wx_pos, wy_neg, wy_pos)
    coarsen_fn::CoarsenFn   # (fine_coeff, cnx, cny) -> coarse_coeff
end
```

**`stencil_fn(i, j, coeff, ihx2, ihy2)`** returns five stencil weights for padded interior cell `(i, j)`:

- `diag` — center (diagonal) weight
- `wx_neg`, `wx_pos` — weights for the left and right neighbours
- `wy_neg`, `wy_pos` — weights for the bottom and top neighbours

`ihx2 = 1/hx²`, `ihy2 = 1/hy²` are pre-computed and passed in. `coeff` is the operator's full padded coefficient array at the current multigrid level.

**`coarsen_fn(fine_coeff, cnx, cny)`** returns a `(cnx+2)×(cny+2)` coefficient array for the next coarser level. Only the coefficient array is coarsened; the `stencil_fn` is reused unchanged at every level.

`apply!`, the smoothers, and `build_hierarchy` are all generic over `FDOp` — they call `stencil_fn` at each cell rather than hard-coding any particular operator.

---

## Solving Algorithm

The following traces the full call chain from `run_solver` to a converged solution.

### Step 1 — Build the α field

```julia
alpha = make_alpha(T, nx, ny, hx, hy, :LogUniform; alpha_min=1.0, alpha_max=150.0)
```

Returns a padded `(nx+2) × (ny+2)` array with interior values filled by one of six generators (`:Constant`, `:SineWave`, `:Random`, `:LogUniform`, `:TurbulentNoise`, `:GaussianSpiral`). Ghost cells are left zero; the operator populates them at runtime.

### Step 2 — Construct the fine-grid operator

```julia
op = HelmholtzOp(alpha, nx, ny, hx, hy)
```

Returns an `FDOp` whose `stencil_fn` implements the 5-point Helmholtz stencil:

```
y[i,j] = (α[i,j] + 2/hx² + 2/hy²) * x[i,j]
        - (x[i-1,j] + x[i+1,j]) / hx²
        - (x[i,j-1] + x[i,j+1]) / hy²
```

and whose `coarsen_fn` averages each 2×2 block of the fine α field (arithmetic mean).

`fill_bc_dirichlet!` is called first to set ghost values: `x_ghost = −x_interior`, which enforces a zero Dirichlet condition at the cell face.

### Step 3 — Build the multigrid hierarchy

```julia
H = build_hierarchy(op; npre=2, npost=2, min_coarse=4)
```

Halves the grid until `min(nx, ny) / 2 < min_coarse`. For each coarser level:

- `op.coarsen_fn` produces the coarsened coefficient array.
- A new `FDOp` is created with the same `stencil_fn` and `coarsen_fn` but the coarser grid and coefficient array.
- A `Level` is allocated with zero-initialised `x`, `b`, `r` arrays of the correct padded size.

At the coarsest level, `build_hierarchy` assembles the dense matrix (by applying the operator to each canonical basis vector) and pre-factorises it with `lu()`. This factorisation is reused throughout the solve.

### Step 4 — Krylov outer loop

The library's `solve()` function is the main entry point. It is dispatched on a solver type token — pass `GMRES()` or `CG()`:

```julia
x_sol, stats, residuals, solve_time =
    solve(op, b_vec, H, GMRES(); tol_rel=1e-10, tol_abs=1e-16, max_iters=300)
# or
x_sol, stats, residuals, solve_time =
    solve(op, b_vec, H, CG();   tol_rel=1e-10, tol_abs=1e-16, max_iters=300)
```

`b_vec` is a flat `nx*ny` right-hand-side vector. `solve()` wraps the operator and preconditioner for `Krylov.jl` without any Helmholtz-specific assumptions:

In the Helmholtz application layer, `run_solver` manufactures `b_vec` from the known exact solution `u_exact(x,y) = sin(πx)sin(πy)` via `make_rhs_and_exact`, calls `solve()`, and reports the L2 error.

- The operator is wrapped in a `LinearOperators.LinearOperator` that calls `apply!` without forming a matrix.
- The hierarchy is wrapped in an `MGPreconditioner`; calling `mul!(z, P, r)` applies one V-cycle (step 5) setting `z ← M⁻¹ r`.
- **GMRES** uses right preconditioning (`N=P`), so `stats.residuals` contains true absolute residuals. The solver stops when `‖r‖ ≤ max(tol_abs, tol_rel · ‖b‖)`.
- **CG** uses left preconditioning (`M=P`). Krylov.jl's internal stopping is bypassed; a callback monitors `‖r_k‖` directly and stops below the same target.

### Step 5 — Multigrid V-cycle  `z = M⁻¹ r`

`vcycle!(H, lvl=1)` is recursive and handles both the preconditioner call and the coarse-grid recursion.

```
vcycle!(H, lvl):
  smooth!(lvl, npre)                     # npre Red-Black Gauss-Seidel sweeps

  if lvl == coarsest:
    coarse_solve!(H, lvl)                # ldiv! with pre-factorised LU; RETURN

  residual!(lvl.r, lvl)                  # r ← b − A x
  restrict!(next.b, lvl.r, lvl.op)      # full-weight: b_coarse[I,J] = ¼ Σ r_fine children
  fill!(next.x, 0)
  vcycle!(H, lvl + 1)                   # recurse on coarser level
  prolong_add!(lvl.x, next.x, next.op) # piecewise-const: x_fine children += x_coarse[I,J]

  smooth!(lvl, npost)                    # npost RBGS sweeps
```

**Smoother details.** `rbgs_sweep!` alternates a red pass (cells with `(i+j)` even) and a black pass (`(i+j)` odd), updating each cell in-place with a Gauss-Seidel solve using the diagonal and off-diagonal weights from `stencil_fn`. `jacobi_sweep!` is also available (default weight `ω = 2/3`) but is not used by default.

**Transfer operators.** `restrict!` uses full weighting (average of 4 fine children). `prolong_add!` uses piecewise-constant injection (same value broadcast to all 4 fine children, added to the current fine value).

**Coarse solve.** `coarse_solve!` copies `b` to a CPU flat vector, calls `ldiv!(H.coarse_lu, b_vec)`, and scatters the result back into the padded `x` array.

---

## Running

The CLI has its own environment in `bin/` (it needs `Printf` and `Random`, which the library does not):

```bash
# From the JuliaGMG.jl/ directory
julia --project=bin -e 'using Pkg; Pkg.instantiate()'   # once
julia --project=bin bin/julia_gmg.jl --nx 128 --ny 128 \
      --alpha_min 1.0 --alpha_max 150.0 \
      --alpha_field LogUniform \
      --solver GMRES --tol_rel 1e-10 --max_iters 300 \
      --npre 2 --npost 2 --min_coarse 4
```

`--solver CG` selects CG (any other value runs GMRES). The right-hand side is manufactured by applying the discrete operator to `sin(πx)sin(πy)`, so the reported L2 error measures the algebraic (solver) error, not the discretisation error.

---

## Adding a New Operator

Construct an `FDOp` directly with your own `StencilSpec`, `stencil_fn`, and `coarsen_fn`. No library changes are needed.

### StencilSpec

`StencilSpec{N}` describes the topology of an N-point stencil as a compile-time constant:

```julia
struct StencilSpec{N}
    offsets::NTuple{N, NTuple{2,Int}}   # (di, dj) offsets in padded-index space
    center_idx::Int                      # index of the (0,0) entry
end
```

The library provides `STENCIL_5PT` for the canonical 5-point layout (west, east, south, north, center):

```julia
STENCIL_5PT = StencilSpec(((-1,0),(1,0),(0,-1),(0,1),(0,0)), 5)
```

`stencil_fn` must return an `NTuple{N,T}` with weights ordered to match `offsets`.

### Pure Laplacian (−∇²u = f, no coefficient field)

```julia
dummy = zeros(T, nx+2, ny+2)   # coeff array unused
op = FDOp(dummy, nx, ny, hx, hy,
    STENCIL_5PT,
    (i, j, _, ihx2, ihy2) -> (-ihx2, -ihx2, -ihy2, -ihy2, 2*ihx2 + 2*ihy2),
    (fa, cnx, cny) -> similar(fa, cnx+2, cny+2))
```

### Variable-coefficient diffusion −∇·(α ∇φ) = f

The divergence-form diffusion operator with a spatially varying coefficient α(x,y) > 0.
Coefficient values are needed at cell faces, not cell centers. Using the harmonic mean at
each face is the standard choice for strongly heterogeneous α because it preserves local
flux balance:

```text
α_{i+½,j} = 2 α[i,j] α[i+1,j] / (α[i,j] + α[i+1,j])
```

The resulting 5-point stencil weights, ordered west/east/south/north/center to match
`STENCIL_5PT`, are:

```text
west   = -α_{i-½,j} / hx²
east   = -α_{i+½,j} / hx²
south  = -α_{i,j-½} / hy²
north  = -α_{i,j+½} / hy²
center =  (α_{i+½,j} + α_{i-½,j}) / hx²  +  (α_{i,j+½} + α_{i,j-½}) / hy²
```

```julia
stencil_fn = (i, j, coeff, ihx2, ihy2) -> begin
    a  = coeff[i,   j  ]
    ae = coeff[i+1, j  ]            # east neighbour (padded index)
    aw = coeff[i-1, j  ]            # west
    an = coeff[i,   j+1]            # north
    as = coeff[i,   j-1]            # south
    # harmonic-mean face conductivities; on boundary faces (ghost coeff = 0) use the cell value
    face(a, b) = b > 0 ? 2*a*b / (a + b) : a
    wxn = -face(a, aw) * ihx2   # west
    wxp = -face(a, ae) * ihx2   # east
    wyn = -face(a, as) * ihy2   # south
    wyp = -face(a, an) * ihy2   # north
    diag = -(wxn + wxp + wyn + wyp)   # positive (M-matrix)
    (wxn, wxp, wyn, wyp, diag)        # ordered: west, east, south, north, center
end

# Harmonic coarsening: preserves the divergence-form structure at coarser levels
coarsen_fn = (fa, cnx, cny) -> begin
    ca = similar(fa, cnx+2, cny+2); fill!(ca, zero(eltype(fa)))
    @views ca[2:cnx+1, 2:cny+1] .= 4 ./ (
        1 ./ fa[2:2:2*cnx+1, 2:2:2*cny+1] .+
        1 ./ fa[3:2:2*cnx+2, 2:2:2*cny+1] .+
        1 ./ fa[2:2:2*cnx+1, 3:2:2*cny+2] .+
        1 ./ fa[3:2:2*cnx+2, 3:2:2*cny+2])
    ca
end

op = FDOp(alpha, nx, ny, hx, hy, STENCIL_5PT, stencil_fn, coarsen_fn)
```

Note: `coeff` at the ghost ring `[1,:]`, `[end,:]`, `[:,1]`, `[:,end]` is zero by convention
(the α field is only defined for interior cells, and `coarsen_fn` leaves it zero on coarser
levels). A plain harmonic mean would then give zero boundary-face conductivity, i.e. a
singular pure-Neumann operator instead of the Dirichlet one; `face` above falls back to the
cell value there. This variant is tested in `test/runtests.jl` (2nd-order convergence).

---

### Variable-coefficient Helmholtz with harmonic coarsening

```julia
op = FDOp(alpha, nx, ny, hx, hy,
    STENCIL_5PT,
    (i, j, coeff, ihx2, ihy2) ->
        (-ihx2, -ihx2, -ihy2, -ihy2, coeff[i,j] + 2*ihx2 + 2*ihy2),
    (fa, cnx, cny) -> begin
        ca = similar(fa, cnx+2, cny+2); fill!(ca, zero(eltype(fa)))
        @views ca[2:cnx+1, 2:cny+1] .= 4 ./ (
            1 ./ fa[2:2:2*cnx+1, 2:2:2*cny+1] .+
            1 ./ fa[3:2:2*cnx+2, 2:2:2*cny+1] .+
            1 ./ fa[2:2:2*cnx+1, 3:2:2*cny+2] .+
            1 ./ fa[3:2:2*cnx+2, 3:2:2*cny+2])
        ca
    end)
```

### 9-point isotropic Laplacian

For a 9-point stencil, define the offsets explicitly and order the `stencil_fn` return
tuple to match:

```julia
STENCIL_9PT = StencilSpec(
    ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1),(0,0)), 9
)

# Isotropic 9-point Laplacian weights (Mehrstellen / Collatz scheme):
#   cardinal neighbors: weight = 1/(6 h²),  diagonal neighbors: 1/(12 h²),
#   center: -(4/(6h²) + 4/(12h²)) = -10/(12h²)   [assumes hx == hy == h]
stencil_fn_9pt = (i, j, _, ihx2, ihy2) -> begin
    wc = ihx2 / 6         # cardinal weight (assumes hx == hy)
    wd = ihx2 / 12        # diagonal weight
    (-wc, -wc, -wc, -wc, -wd, -wd, -wd, -wd, 4*wc + 4*wd)
end

op = FDOp(dummy, nx, ny, hx, hy, STENCIL_9PT, stencil_fn_9pt,
          (fa, cnx, cny) -> similar(fa, cnx+2, cny+2))
```

> **RBGS coloring note:** The `(i+j) % 2` coloring used by the RBGS smoother does not
> guarantee independent simultaneous updates for 9-point stencils — diagonal neighbors
> `(i±1,j±1)` share the same color as `(i,j)`. The smoother remains valid (Gauss-Seidel-like
> ordering) but is no longer a true simultaneous update. Use Jacobi (`jacobi_sweep!`) if
> strict independence is required.

### Stencil function signature reference

```text
stencil_fn(i, j, coeff, ihx2, ihy2) -> NTuple{N, T}
```

| Argument | Meaning |
| --- | --- |
| `i`, `j` | Padded array indices of the current interior cell |
| `coeff` | Full padded coefficient array at this multigrid level |
| `ihx2`, `ihy2` | Precomputed `1/hx²`, `1/hy²` |
| return value | `N` weights, one per entry of `stencil.offsets` |

The stencil function is called identically inside `apply!` and both smoothers.

```text
coarsen_fn(fine_coeff, cnx, cny) -> coarse_coeff
```

Returns a `(cnx+2)×(cny+2)` padded array. The ghost ring should be left zero (it is filled at runtime by `fill_bc_dirichlet!`). The same `stencil_fn` is reused at every coarser level — only `coeff` and the grid dimensions change.
