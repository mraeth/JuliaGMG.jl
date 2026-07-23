# JuliaGMG

A GPU-capable Julia version of a geometric multigrid preconditioned Krylov solver. Solves the 2D variable-coefficient Helmholtz equation

```
(α(x,y) − ∇²) u = f    on [0,1]²,  u = 0 on ∂Ω
```

using a Krylov outer loop (GMRES or CG) preconditioned by one multigrid V-cycle per iteration. The implementation is designed to run on CPU today and is GPU-portable via `KernelAbstractions.jl` and `Adapt.jl`.

---

## Package Structure

```
julia_gmg/
├── Project.toml          # JuliaGMG 0.1.0; deps: KernelAbstractions, Krylov, LinearOperators, ...
├── Manifest.toml
├── bin/
│   └── julia_gmg.jl      # CLI entry point with hand-rolled argument parser
└── src/
    ├── JuliaGMG.jl        # Module root: includes all files and re-exports public API
    ├── types.jl           # Core structs (HelmholtzOp, Level, MGHierarchy) + hierarchy builder
    ├── alpha_fields.jl    # Six generators for the spatially varying coefficient α(x,y)
    ├── operators.jl       # Matrix-free operator application, BCs, residual, helper utilities
    ├── smoother.jl        # Weighted Jacobi and Red-Black Gauss-Seidel smoothers
    ├── transfer.jl        # Full-weight restriction and piecewise-constant prolongation
    ├── vcycle.jl          # Recursive V-cycle and coarse-grid direct solve (LU)
    └── solver.jl          # MGPreconditioner wrapper + run_solver() top-level function
```

### Data layout convention

All field arrays are **padded**: interior cell `(i, j)` maps to array index `[i+1, j+1]`, so the surrounding ring `[1, :]`, `[end, :]`, `[:, 1]`, `[:, end]` holds Dirichlet ghost values. The Krylov library works with flat vectors; `vec_to_padded!` / `padded_to_vec!` convert between the two.

---

## Type Hierarchy

```
HelmholtzOp{T, A}      matrix-free operator; stores α grid and mesh spacings
Level{T, A}            one multigrid level; owns x, b, r arrays + an HelmholtzOp
MGHierarchy{T, L, F}   full level stack, smoothing counts, pre-factorised coarse-grid LU
MGPreconditioner{T, H} thin wrapper that exposes MGHierarchy as a LinearAlgebra preconditioner
```

Type parameter `A <: AbstractMatrix{T}` propagates from the finest operator down through every level, so swapping `Matrix{Float64}` for a `CUDA.CuMatrix{Float32}` changes the backend without touching any algorithm code. The `Adapt.@adapt_structure` macro is applied to all three structs for this reason.

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

Stores the α field and mesh spacings. No matrix is assembled; `apply!` evaluates the 5-point stencil on the fly:

```
y[i,j] = (α[i,j] + 2/hx² + 2/hy²) * x[i,j]
        - (x[i-1,j] + x[i+1,j]) / hx²
        - (x[i,j-1] + x[i,j+1]) / hy²
```

`fill_bc_dirichlet!` is called first to set ghost values: `x_ghost = −x_interior`, which enforces a zero Dirichlet condition at the cell face.

### Step 3 — Build the multigrid hierarchy

```julia
H = build_hierarchy(op; npre=2, npost=2, min_coarse=4)
```

Halves the grid until `min(nx, ny) / 2 < min_coarse`. For each coarser level:

- A `HelmholtzOp` is created with `nx/2, ny/2` and the coarsened α field, obtained by averaging each 2×2 block of fine-grid interior cells.
- A `Level` is allocated with zero-initialised `x`, `b`, `r` arrays of the correct padded size.

At the coarsest level, `build_hierarchy` assembles the dense matrix (by applying the operator to each canonical basis vector) and pre-factorises it with `lu()`. This factorisation is reused throughout the solve.

### Step 4 — Krylov outer loop

```julia
x_sol, stats, ... = run_solver(op; solver_type=:gmres, tol_rel=1e-10, max_iters=300)
```

`run_solver` manufactures the RHS from the known exact solution `u_exact(x,y) = sin(πx)sin(πy)` and wraps the operator and preconditioner for `Krylov.jl`:

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

**Smoother details.** `rbgs_sweep!` alternates a red pass (cells with `(i+j)` even) and a black pass (`(i+j)` odd), updating each cell in-place with a Gauss-Seidel solve against the 5-point stencil diagonal `α[i,j] + 2/hx² + 2/hy²`. `fill_bc_dirichlet!` is called after each half-sweep. `jacobi_sweep!` is also available (default weight `ω = 2/3`) but is not used by default.

**Transfer operators.** `restrict!` uses full weighting (average of 4 fine children). `prolong_add!` uses piecewise-constant injection (same value broadcast to all 4 fine children, added to the current fine value).

**Coarse solve.** `coarse_solve!` copies `b` to a CPU flat vector, calls `ldiv!(H.coarse_lu, b_vec)`, and scatters the result back into the padded `x` array.

---

## Running

```bash
# From the julia_gmg/ directory
julia --project bin/julia_gmg.jl --nx 128 --ny 128 \
      --alpha_min 1.0 --alpha_max 150.0 \
      --alpha_field LogUniform \
      --solver gmres --tol_rel 1e-10 --max_iters 300 \
      --npre 2 --npost 2 --min_coarse 4
```

Output format matches the C++ `mfes` `--verbose 2` output for direct comparison by `../run_comparison.py`.

---

## Blueprint: Adding a New Linear Operator

The current code is written for `HelmholtzOp` concretely rather than against an abstract interface. Adding a new operator means providing the same surface that `HelmholtzOp` provides and wiring it into the hierarchy builder and cycle.

### What you need to implement

| Function | Signature | Purpose |
|---|---|---|
| `apply!` | `apply!(y::A, op::MyOp, x::A)` | Matrix-free operator application `y ← A x` |
| `fill_bc_*!` | `fill_bc!(x::A, op::MyOp)` | Fill ghost ring to enforce BCs |
| `residual!` | `residual!(r::A, lvl::Level{T,A,MyOp})` | In-place `r ← b − A x` at a level |
| `smooth!` | `smooth!(lvl::Level{T,A,MyOp}, nsweeps)` | One or more smoother sweeps |
| Coarsening | called inside `build_hierarchy` | Restrict operator data (e.g. coefficients) to coarser grids |
| Dense assembly | called for coarsest level | Build dense matrix for LU factorisation |

### Step 1 — Define the operator struct

```julia
# src/operators.jl (or a new file)
struct MyOp{T, A <: AbstractMatrix{T}}
    coeff::A          # spatially varying coefficient, padded (nx+2)×(ny+2)
    nx::Int
    ny::Int
    hx::T
    hy::T
end
Adapt.@adapt_structure MyOp   # enables GPU portability
```

Store one such struct per multigrid level (pass it as the `op` field of `Level`). Coarsen the coefficient field in `build_hierarchy` by whichever averaging rule is appropriate for your operator (arithmetic, harmonic, etc.).

### Step 2 — Boundary conditions

```julia
function fill_bc_myop!(x::AbstractMatrix, op::MyOp)
    # Fill the ghost ring of x to enforce your BC.
    # For zero Dirichlet: ghost = -interior (same as fill_bc_dirichlet!).
    # For Neumann: ghost = +interior.
    # For periodic: ghost = opposite interior cell.
end
```

### Step 3 — Operator application

```julia
function apply!(y::AbstractMatrix, op::MyOp, x::AbstractMatrix)
    fill_bc_myop!(x, op)
    nx, ny, hx, hy = op.nx, op.ny, op.hx, op.hy
    @inbounds for j in 2:ny+1, i in 2:nx+1
        # 5-point (or 9-point, or anisotropic, etc.) stencil
        # Use op.coeff[i,j] for any spatially varying coefficient
        y[i,j] = ...
    end
end
```

### Step 4 — Residual

```julia
function residual!(r::AbstractMatrix, lvl::Level{T,A,MyOp{T,A}}) where {T,A}
    fill!(r, zero(T))
    apply!(r, lvl.op, lvl.x)   # r ← A x
    @inbounds for j in 2:lvl.op.ny+1, i in 2:lvl.op.nx+1
        r[i,j] = lvl.b[i,j] - r[i,j]
    end
end
```

### Step 5 — Smoother

```julia
function smooth!(lvl::Level{T,A,MyOp{T,A}}, nsweeps::Int) where {T,A}
    op = lvl.op
    hx2, hy2 = op.hx^2, op.hy^2
    for _ in 1:nsweeps
        for colour in (0, 1)           # red, then black
            fill_bc_myop!(lvl.x, op)
            @inbounds for j in 2:op.ny+1, i in 2:op.nx+1
                (i + j) % 2 == colour || continue
                diag = op.coeff[i,j] + 2/hx2 + 2/hy2   # adjust for your operator
                off  = (lvl.x[i-1,j] + lvl.x[i+1,j]) / hx2 +
                       (lvl.x[i,j-1] + lvl.x[i,j+1]) / hy2
                lvl.x[i,j] = (lvl.b[i,j] + off) / diag
            end
        end
    end
end
```

Override `smooth!` whenever your operator has a different (possibly spatially varying) diagonal from the Helmholtz case.

### Step 6 — Hierarchy builder

In `types.jl`, `build_hierarchy` is currently specific to `HelmholtzOp`. To support `MyOp`, either:

- **Extend the existing function** by dispatching on the fine-level operator type, or
- **Write a new `build_hierarchy` method** with the same signature but a different first argument type.

Inside the builder:
1. Create `MyOp` structs for each coarse level with coarsened coefficient data.
2. Allocate `Level` structs with zero-filled arrays.
3. At the coarsest level, assemble the dense matrix (apply `MyOp` to each basis vector) and call `lu()`.

```julia
function build_hierarchy(fine_op::MyOp{T,A}; npre=2, npost=2, min_coarse=4) where {T,A}
    levels = Level{T,A,MyOp{T,A}}[]
    op = fine_op
    while min(op.nx, op.ny) ÷ 2 >= min_coarse
        push!(levels, Level(op))     # allocate x, b, r at this resolution
        op = coarsen(op)             # halve nx/ny, restrict coeff
    end
    push!(levels, Level(op))
    coarse_mat = assemble_dense(levels[end].op)
    coarse_lu  = lu(coarse_mat)
    MGHierarchy(levels, npre, npost, coarse_lu)
end
```

### Step 7 — Wire into `run_solver` and the CLI

`run_solver` (in `solver.jl`) calls `make_rhs_and_exact` which is also Helmholtz-specific. Provide a corresponding function for your operator:

```julia
function make_rhs_and_exact(op::MyOp{T}) where T
    # return (b, u_exact) as padded arrays
    # b = A * u_exact  for a chosen manufactured solution
end
```

Then add a dispatch branch in `run_solver`, or write a thin wrapper that calls the same Krylov + V-cycle machinery with the new operator and preconditioner. The `MGPreconditioner` and `vcycle!` are generic over the level type, so they require no changes as long as `smooth!`, `residual!`, `restrict!`, and `prolong_add!` dispatch correctly on the new operator type.

In `bin/julia_gmg.jl`, add a command-line flag (e.g. `--op myop`) and branch to the new constructor.

### Summary checklist

| Item | Required | Notes |
|---|---|---|
| Operator struct with `Adapt.@adapt_structure` | Yes | Enables GPU portability |
| `fill_bc_*!(x, op)` | Yes | Called by `apply!` and `smooth!` |
| `apply!(y, op, x)` | Yes | Core stencil evaluation |
| `residual!(r, lvl)` | Yes | `r ← b − A x` |
| `smooth!(lvl, nsweeps)` | Yes | Override if diagonal differs from Helmholtz |
| `build_hierarchy(op; ...)` | Yes | Coarsen coefficients, pre-factorise coarsest level |
| `make_rhs_and_exact(op)` | Yes | Manufactured solution for testing |
| `restrict!` / `prolong_add!` | Only if non-standard | Default full-weight / piecewise-const work for most FD operators |
| CLI flag in `bin/julia_gmg.jl` | If CLI access needed | |
