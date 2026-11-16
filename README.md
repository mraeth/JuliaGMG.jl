# JuliaGMG

A Julia geometric-multigrid-preconditioned Krylov solver. Solves 2D (or 1D, `ny = 1`) variable-coefficient PDEs on rectangular cell-centred grids of any size, optionally for a batch of independent planes at once, with Dirichlet, Neumann or periodic boundary conditions (per side) using a Krylov outer loop (GMRES or CG) preconditioned by one multigrid V-cycle per iteration. The implementation runs on CPU (plain Julia loops); GPU kernels via `KernelAbstractions.jl` and `Adapt.jl` are planned, and the types are already parameterised for it (see below).

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
│   ├── types.jl           # BoundaryCondition types, FDOp, Level, MGHierarchy, build_hierarchy
│   ├── operators.jl       # Matrix-free apply!, fill_bc!, residual, vec_to_padded!/padded_to_vec!
│   ├── smoother.jl        # Weighted Jacobi and Red-Black Gauss-Seidel smoothers
│   ├── transfer.jl        # Averaging restriction and piecewise-constant prolongation (full or semi-coarsening)
│   ├── vcycle.jl          # Recursive V-cycle and coarse-grid direct solve (LU)
│   └── solver.jl          # GMRES/CG dispatcher types, MGPreconditioner, solve()
└── helmholtz/            # Helmholtz application layer — included via include() in bin/julia_gmg.jl
    ├── helmholtz_op.jl    # HelmholtzOp constructor (returns an FDOp)
    ├── alpha_fields.jl    # Six generators for the spatially varying coefficient α(x,y)
    └── runner.jl          # make_rhs_and_exact, run_solver (manufactures RHS, calls solve(), reports error)
└── test/                 # Pkg.test(): MMS convergence (Dirichlet/Neumann/periodic, singular, 1D, non-power-of-two, strips, hx ≠ hy), h-independent iterations, batch vs separate solves, apply! vs dense matrix
```

### Data layout convention

All field arrays are **padded**: interior cell `(i, j)` maps to array index `[i+1, j+1]`, so the surrounding ring `[1, :]`, `[end, :]`, `[:, 1]`, `[:, end]` holds ghost values set by `fill_bc!` from the operator's boundary conditions (see [Boundary conditions](#boundary-conditions)). The Krylov library works with flat vectors; `vec_to_padded!` / `padded_to_vec!` convert between the two.

**Batch.** Arrays may have a third dimension `nb`: `(nx+2, ny+2, nb)` holds `nb` independent planes (e.g. the perpendicular planes of a 3D run stacked along the field), each with its own coefficients, sharing `nx`, `ny`, `hx`, `hy`, stencil and `bc`. Flat vectors then have length `nx*ny*nb`, plane-major: interior cell `(i, j)` of plane `k` is entry `((k-1)*ny + j-1)*nx + i`. A 2D `Matrix` is the case `nb = 1` (all loops run over `k in axes(x, 3)` with `x[i,j,k]`, which Julia allows for a `Matrix`). The planes are solved together by one Krylov iteration (one stopping test on the norm over all planes) with a block-diagonal V-cycle: smoothing and transfers act plane by plane, and the coarse direct solve factorises each plane's matrix separately.

---

## Type Hierarchy

```
FDOp{T, A, N, StencilFn, CoarsenFn, B}  generic matrix-free N-point FD operator
BoundaryCondition                   abstract; Dirichlet(), Neumann(), Periodic(), one per side
HelmholtzOp(alpha, ...)             Helmholtz convenience constructor returning an FDOp (helmholtz/ layer)
Level{T, A, Op}                     one multigrid level; owns x, b, r arrays + an operator
MGHierarchy{T, L, F}                full level stack, smoothing counts, pre-factorised coarse-grid LU
MGPreconditioner{T, H}              thin wrapper that exposes MGHierarchy as a LinearAlgebra preconditioner
GMRES / CG                          empty structs used as solver selector tokens for solve()
```

Type parameter `A <: AbstractArray{T}` (2D, or 3D for a batch) propagates from the finest operator down through every level, and `Adapt.@adapt_structure` is applied to `FDOp` and `Level`, in preparation for GPU arrays (e.g. `CUDA.CuMatrix{Float32}`). The loops in `apply!`, the smoothers and the transfer operators are plain CPU loops today; GPU kernels are planned.

---

## The Generic Operator: `FDOp`

`FDOp` is parameterised by two callables that encode the operator and its multigrid coarsening rule:

```julia
struct FDOp{T, A<:AbstractArray{T}, N, StencilFn, CoarsenFn, B}
    coeff::A                # padded (nx+2)×(ny+2) or (nx+2)×(ny+2)×nb coefficient array
    nx::Int
    ny::Int
    hx::T
    hy::T
    stencil::StencilSpec{N} # stencil topology, e.g. STENCIL_5PT
    stencil_fn::StencilFn   # (i, j, coeff, ihx2, ihy2) -> NTuple{N,T} weights
    coarsen_fn::CoarsenFn   # (fine_coeff, cnx, cny) -> coarse_coeff
    bc::B                   # (west, east, south, north) boundary conditions
end

FDOp(coeff, nx, ny, hx, hy, stencil, stencil_fn, coarsen_fn)       # all sides Dirichlet()
FDOp(coeff, nx, ny, hx, hy, stencil, stencil_fn, coarsen_fn, bc)
```

**`stencil_fn(i, j, coeff, ihx2, ihy2)`** returns the stencil weights for padded interior cell `(i, j)`, ordered like `stencil.offsets` (for `STENCIL_5PT`: west, east, south, north, center).

`ihx2 = 1/hx²`, `ihy2 = 1/hy²` are pre-computed and passed in. `coeff` is the operator's padded coefficient array at the current multigrid level: for a 2D `Matrix` the array itself, for a batch a view `@view coeff[:, :, k]` of the current plane, so the same 2D `stencil_fn` serves both.

**`coarsen_fn(fine_coeff, cnx, cny)`** returns a `(cnx+2)×(cny+2)` coefficient array for the next coarser level, from one 2D plane `fine_coeff` (for a batch it is called per plane). With semi-coarsening a direction is either halved or kept, so the factor per direction is inferred from the sizes: `fx = (size(fine_coeff, 1) - 2) ÷ cnx ∈ {1, 2}`, likewise `fy`; average over the `fx×fy` fine children (examples below). Only the coefficient array is coarsened; the `stencil_fn` is reused unchanged at every level.

`apply!`, the smoothers, and `build_hierarchy` are all generic over `FDOp` — they call `stencil_fn` at each cell rather than hard-coding any particular operator.

### Boundary conditions

`bc = (west, east, south, north)`, each a `BoundaryCondition`. Ghost rules on the cell-centred grid (the boundary is the cell face), applied by `fill_bc!(x, op)` before every stencil evaluation in `apply!` and the smoothers:

| Type | Condition at the face | Ghost value |
| --- | --- | --- |
| `Dirichlet()` (default) | `u = 0` | `−` adjacent interior |
| `Neumann()` | `∂u/∂n = 0` | `+` adjacent interior |
| `Periodic()` | wrap | opposite interior row/column |

`Periodic()` must be set on both opposite sides (`ArgumentError` otherwise). The ghost corners are filled consistently (x sides first, then y sides over the full width), so 9-point stencils see wrapped corners for doubly periodic grids. Only homogeneous conditions are supported; put inhomogeneous boundary data into the right-hand side.

For periodic sides the `FDOp` constructor also wraps the **ghost ring of `coeff` in place** (and so on every coarse level, since `build_hierarchy` constructs those through the same constructor and passes `bc` on), so stencils reading `coeff[i±1, j±1]`, e.g. harmonic face means, are correct across the periodic boundary. On Dirichlet/Neumann sides the coefficient ghosts are left as given (zero by convention).

**Singular operators.** If the operator annihilates constants (pure Poisson/diffusion with all sides `Periodic()`/`Neumann()`), build the hierarchy with `build_hierarchy(op; singular=true)`. Then `solve` projects the mean out of `b` (a non-mean-free `b` is inconsistent; its constant part is silently discarded, i.e. the least-squares problem is solved), the preconditioner works on the mean-free subspace (input and output projected), the coarse direct solve factorises `A_c + 11ᵀ/n_c` (non-singular, returns the mean-free solution for mean-free data), and the returned solution has zero mean.

```julia
op = FDOp(zeros(nx+2, ny+2), nx, ny, hx, hy, STENCIL_5PT,
          (i, j, _, ihx2, ihy2) -> (-ihx2, -ihx2, -ihy2, -ihy2, 2ihx2 + 2ihy2),
          (fa, cnx, cny) -> zeros(cnx+2, cny+2),
          (Periodic(), Periodic(), Periodic(), Periodic()))
H = build_hierarchy(op; singular=true)
x, stats, _, _ = solve(op, b, H, CG())      # mean-free x
```

---

## Grid shapes

**Arbitrary sizes and spacings.** `nx`, `ny`, `hx`, `hy` are independent (non-square domains, `hx ≠ hy`). Semi-coarsening (see [Step 3](#step-3--build-the-multigrid-hierarchy)) handles long strips such as `128×8` with `h`-independent iterations. It does not cure mesh anisotropy: with point RBGS, iterations grow with `hx/hy` (e.g. `hx = 2hy`: 15–18 GMRES iterations; `[0,1]×[0,1/16]` with `ny = 8` fixed and `nx = 128/256/512`, i.e. `hy/hx` up to 16: 8/13/28).

**1D.** Use `ny = 1` (never coarsened), a `stencil_fn` with zero y-weights and `Periodic()` in y; `hy` is arbitrary (e.g. 1):

```julia
op = FDOp(alpha, nx, 1, 1/nx, 1.0, STENCIL_5PT,
          (i, j, c, ihx2, _) -> (-ihx2, -ihx2, 0.0, 0.0, c[i,j] + 2ihx2),
          coarsen_fn, (Periodic(), Periodic(), Periodic(), Periodic()))   # or (Neumann(), Neumann(), Periodic(), Periodic()), ...
```

`alpha` is `(nx+2)×3`. With zero y-weights the y ghosts never enter. `Periodic()` (or `Neumann()`) in y also makes a standard y-stencil, whose y-part annihilates constants, cancel exactly at `ny = 1`, so a 2D stencil gives the 1D operator too, but its y-terms still enter the smoother diagonal and weaken it; zero y-weights are preferred. `Dirichlet()` in y with nonzero y-weights adds a spurious `4/hy²` shift.

**Batch.** `coeff` of size `(nx+2, ny+2, nb)` (see [Data layout](#data-layout-convention)); `stencil_fn` and `coarsen_fn` stay 2D and see one plane at a time:

```julia
coeff = cat(alpha_1, alpha_2, alpha_3; dims=3)              # each (nx+2)×(ny+2)
op = FDOp(coeff, nx, ny, hx, hy, STENCIL_5PT, stencil_fn, coarsen_fn, bc)
H  = build_hierarchy(op; singular)                         # nb coarse LUs
x, stats, _, _ = solve(op, b, H, CG())                      # b, x: length nx*ny*nb
```

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

and whose `coarsen_fn` averages the 2×2 (or, for semi-coarsening, 2×1 / 1×2) block of fine α values (arithmetic mean).

`apply!` calls `fill_bc!` first to set the ghost values from `op.bc`; with the default `Dirichlet()` sides `x_ghost = −x_interior`, which enforces `u = 0` at the cell face.

### Step 3 — Build the multigrid hierarchy

```julia
H = build_hierarchy(op; npre=2, npost=2, min_coarse=4, singular=false)
```

Coarsens x and y independently (semi-coarsening): a direction is halved (and its spacing doubled) while its size is even and the halved size is `≥ min_coarse`; otherwise it keeps size and spacing on the next level. Coarsening stops when neither direction can be halved. Examples with `min_coarse = 4`: `128×128 → … → 4×4`; `128×8 → 64×4 → 32×4 → … → 4×4`; `48 → 24 → 12 → 6`; `45×64 → 45×32 → … → 45×4` (odd sizes are never coarsened); `n×1` (1D) coarsens x only. For each coarser level:

- `op.coarsen_fn` produces the coarsened coefficient array (per plane for a batch).
- A new `FDOp` is created with the same `stencil`, `stencil_fn`, `coarsen_fn` and `bc` but the coarser grid and coefficient array (periodic coefficient ghosts are wrapped by the constructor).
- A `Level` is allocated with zero-initialised `x`, `b`, `r` arrays of the correct padded size.

At the coarsest level, `build_hierarchy` assembles the dense matrix of each plane (by applying the operator to each canonical basis vector) and pre-factorises it with `lu()` (with `singular=true`, of `A_c + 11ᵀ/n_c`), giving a vector of `nb` LUs reused throughout the solve. The coarse system gets whatever size remains, so sizes with a large odd factor give a large dense system; `build_hierarchy` emits a `@warn` when it has more than 4096 unknowns per plane (e.g. `65×65`). Choose sizes `m·2^k` with small `m` (e.g. 48, 96, 192) to avoid this.

### Step 4 — Krylov outer loop

The library's `solve()` function is the main entry point. It is dispatched on a solver type token — pass `GMRES()` or `CG()`:

```julia
x_sol, stats, residuals, solve_time =
    solve(op, b_vec, H, GMRES(); tol_rel=1e-10, tol_abs=1e-16, max_iters=300)
# or
x_sol, stats, residuals, solve_time =
    solve(op, b_vec, H, CG();   tol_rel=1e-10, tol_abs=1e-16, max_iters=300)
```

`b_vec` is a flat `nx*ny*nb` right-hand-side vector (plane-major, see [Data layout](#data-layout-convention)). `solve()` wraps the operator and preconditioner for `Krylov.jl` without any Helmholtz-specific assumptions:

In the Helmholtz application layer, `run_solver` manufactures `b_vec` from the known exact solution `u_exact(x,y) = sin(πx)sin(πy)` via `make_rhs_and_exact`, calls `solve()`, and reports the L2 error.

- The operator is wrapped in a `LinearOperators.LinearOperator` that calls `apply!` without forming a matrix.
- The hierarchy is wrapped in an `MGPreconditioner`; calling `mul!(z, P, r)` applies one V-cycle (step 5) setting `z ← M⁻¹ r`.
- **GMRES** uses right preconditioning (`N=P`), so `stats.residuals` contains true absolute residuals. The solver stops when `‖r‖ ≤ max(tol_abs, tol_rel · ‖b‖)`.
- With `singular=true` (see [Boundary conditions](#boundary-conditions)), `b_vec` and the solution are projected onto the mean-free subspace, plane by plane for a batch.
- **CG** uses left preconditioning (`M=P`), which requires the symmetric V-cycle described below. Krylov.jl's internal stopping is bypassed; a callback monitors `‖r_k‖` directly and stops below the same target.

### Step 5 — Multigrid V-cycle  `z = M⁻¹ r`

`vcycle!(H, lvl=1)` is recursive and handles both the preconditioner call and the coarse-grid recursion.

```
vcycle!(H, lvl):
  smooth!(lvl, npre)                     # npre Red-Black Gauss-Seidel sweeps

  if lvl == coarsest:
    coarse_solve!(H, lvl)                # ldiv! with pre-factorised LU; RETURN

  residual!(lvl.r, lvl)                  # r ← b − A x
  restrict!(next.b, lvl.r, lvl.op)      # average: b_coarse[I,J] = mean of the fx×fy r_fine children
  fill!(next.x, 0)
  vcycle!(H, lvl + 1)                   # recurse on coarser level
  prolong_add!(lvl.x, next.x, next.op) # piecewise-const: x_fine children += x_coarse[I,J]

  smooth!(lvl, npost; reverse=true)      # npost RBGS sweeps, black then red
```

**Smoother details.** `rbgs_sweep!` alternates a red pass (cells with `(i+j)` even) and a black pass (`(i+j)` odd), updating each cell in-place with a Gauss-Seidel solve using the diagonal and off-diagonal weights from `stencil_fn`. `reverse=true` runs black before red, the adjoint sweep: post-smoothing in reverse colour order makes the V-cycle a symmetric preconditioner for symmetric operators (restriction is `1/(fx·fy)` × the transpose of prolongation), so CG is valid. With it, CG needs about as many iterations as GMRES (10 vs. 16–25 before, see `test/runtests.jl`). `jacobi_sweep!` is also available (default weight `ω = 2/3`) but is not used by default.

**Transfer operators.** `restrict!` averages the `fx×fy` fine children (4 for full coarsening, 2 for semi-coarsening; the factors are inferred from the array sizes). `prolong_add!` uses piecewise-constant injection (same value added to all `fx×fy` fine children). Restriction is `1/(fx·fy)` × the transpose of prolongation, so the V-cycle stays symmetric.

**Coarse solve.** `coarse_solve!` copies `b` to the CPU, and for each plane `k` gathers a flat vector, calls `ldiv!(H.coarse_lu[k], b_vec)`, and scatters the result back into the padded `x` array.

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
# fx, fy ∈ {1, 2}: semi-coarsening keeps a direction (see coarsen_fn contract)
coarsen_fn = (fa, cnx, cny) -> begin
    fx, fy = (size(fa, 1) - 2) ÷ cnx, (size(fa, 2) - 2) ÷ cny
    ca = similar(fa, cnx+2, cny+2); fill!(ca, zero(eltype(fa)))
    for J in 1:cny, I in 1:cnx
        s = zero(eltype(fa))
        for dj in 1:fy, di in 1:fx
            s += 1 / fa[fx*(I-1)+di+1, fy*(J-1)+dj+1]
        end
        ca[I+1, J+1] = fx * fy / s
    end
    ca
end

op = FDOp(alpha, nx, ny, hx, hy, STENCIL_5PT, stencil_fn, coarsen_fn)
```

Note: `coeff` at the ghost ring `[1,:]`, `[end,:]`, `[:,1]`, `[:,end]` is zero by convention on
Dirichlet/Neumann sides (the α field is only defined for interior cells, and `coarsen_fn` leaves
it zero on coarser levels). A plain harmonic mean would then give zero boundary-face
conductivity; `face` above falls back to the cell value there, which together with the
Dirichlet/Neumann ghost of the solution gives the intended boundary flux. On `Periodic()` sides
the constructor wraps the coefficient ghosts, so the face mean across the periodic boundary is
the true harmonic mean. This variant is tested in `test/runtests.jl` with Dirichlet sides and
with Neumann in x / periodic in y (`singular=true`), both 2nd order.

---

### Variable-coefficient Helmholtz with harmonic coarsening

```julia
op = FDOp(alpha, nx, ny, hx, hy,
    STENCIL_5PT,
    (i, j, coeff, ihx2, ihy2) ->
        (-ihx2, -ihx2, -ihy2, -ihy2, coeff[i,j] + 2*ihx2 + 2*ihy2),
    coarsen_fn)        # harmonic coarsening as in the diffusion example above
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
| `coeff` | Padded 2D coefficient array at this multigrid level (for a batch: view of the current plane) |
| `ihx2`, `ihy2` | Precomputed `1/hx²`, `1/hy²` |
| return value | `N` weights, one per entry of `stencil.offsets` |

The stencil function is called identically inside `apply!` and both smoothers.

```text
coarsen_fn(fine_coeff, cnx, cny) -> coarse_coeff
```

Returns a `(cnx+2)×(cny+2)` padded array from one 2D plane `fine_coeff`. Per direction the coarsening factor is `(size(fine_coeff, 1) - 2) ÷ cnx` (x) and `(size(fine_coeff, 2) - 2) ÷ cny` (y), each 1 or 2. The ghost ring should be left zero; on `Periodic()` sides the `FDOp` constructor wraps it. (The solution ghosts are filled at runtime by `fill_bc!`.) The same `stencil_fn` is reused at every coarser level — only `coeff` and the grid dimensions change.
