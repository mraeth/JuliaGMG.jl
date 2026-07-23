#!/usr/bin/env julia
# CLI entry point for JuliaGMG Helmholtz solver.
# Output format matches MFES --verbose 2 so run_comparison.py's parse_mfes() works directly.
#
# Usage:
#   julia --project=<julia_gmg_dir> bin/julia_gmg.jl \
#       --op HELMHOLTZ --alpha_field LogUniform \
#       --alpha_min 1 --alpha_max 150 \
#       --nx 128 --ny 128 --bc DIRICHLET \
#       --solver GMRES --tol_rel 1e-10 --max_iters 300 --verbose 2

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JuliaGMG
using Printf

# ---------------------------------------------------------------------------
# Argument parsing (hand-rolled to avoid extra dependency)
# ---------------------------------------------------------------------------
function parse_args(argv)
    opts = Dict{String,Any}(
        "op"         => "HELMHOLTZ",
        "alpha_field"=> "LogUniform",
        "alpha_min"  => 1.0,
        "alpha_max"  => 150.0,
        "n_periods_x"=> 1,
        "n_periods_y"=> 1,
        "nx"         => 128,
        "ny"         => 128,
        "bc"         => "DIRICHLET",
        "solver"     => "GMRES",
        "tol_rel"    => 1e-10,
        "tol_abs"    => 1e-16,
        "max_iters"  => 300,
        "verbose"    => 2,
        "npre"       => 2,
        "npost"      => 2,
        "min_coarse" => 4,
    )
    i = 1
    while i <= length(argv)
        key = lstrip(argv[i], '-')
        if i + 1 <= length(argv) && !startswith(argv[i+1], "--")
            val = argv[i+1]
            i += 2
        else
            i += 1; continue
        end
        if key in ("alpha_min","alpha_max","tol_rel","tol_abs")
            opts[key] = parse(Float64, val)
        elseif key in ("nx","ny","n_periods_x","n_periods_y","max_iters","verbose","npre","npost","min_coarse")
            opts[key] = parse(Int, val)
        else
            opts[key] = val
        end
    end
    opts
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function main(argv=ARGS)
    opts = parse_args(argv)

    nx   = opts["nx"]
    ny   = opts["ny"]
    T    = Float64
    hx   = T(1.0 / nx)
    hy   = T(1.0 / ny)

    alpha_sym = Symbol(opts["alpha_field"])
    n_periods = max(opts["n_periods_x"], opts["n_periods_y"])

    # Print banner
    @printf("=========================================================\n")
    @printf(" Julia GMG (geometric multigrid, Krylov.jl)\n")
    @printf(" Operator: %s\n", opts["op"])
    @printf(" Alpha   : %s  [%.4g, %.4g]\n", opts["alpha_field"], opts["alpha_min"], opts["alpha_max"])
    @printf(" Grid    : %d x %d  (hx/hy = %.4e/%.4e)\n", nx, ny, hx, hy)
    @printf(" Solver  : %s  tol_rel=%.1e  max_iters=%d\n",
            opts["solver"], opts["tol_rel"], opts["max_iters"])
    @printf("=========================================================\n")

    alpha = make_alpha(T, nx, ny, hx, hy, alpha_sym;
                       alpha_min=opts["alpha_min"],
                       alpha_max=opts["alpha_max"],
                       n_periods=n_periods)

    op = HelmholtzOp{T, Matrix{T}}(alpha, nx, ny, hx, hy)

    solver_sym = opts["solver"] == "CG" ? :CG : :GMRES

    if solver_sym == :CG && opts["op"] == "HELMHOLTZ"
        # CG is valid for Helmholtz (SPD when alpha > 0); warn but proceed
        @printf("NOTE: CG with HELMHOLTZ operator (alpha > 0 required for SPD).\n")
    end

    run_solver(op;
               solver_type = solver_sym,
               tol_rel     = T(opts["tol_rel"]),
               tol_abs     = T(opts["tol_abs"]),
               max_iters   = opts["max_iters"],
               npre        = opts["npre"],
               npost       = opts["npost"],
               verbose     = opts["verbose"])
end

main(ARGS)
