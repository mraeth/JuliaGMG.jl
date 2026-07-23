module JuliaGMG

using Printf

include("types.jl")
include("alpha_fields.jl")
include("operators.jl")
include("smoother.jl")
include("transfer.jl")
include("vcycle.jl")
include("solver.jl")

export HelmholtzOp, Level, MGHierarchy
export build_hierarchy
export make_alpha
export apply!, fill_bc_dirichlet!, residual!, interior_norm
export make_rhs_and_exact, vec_to_padded!, padded_to_vec!
export smooth!, jacobi_sweep!, rbgs_sweep!
export restrict!, prolong_add!
export vcycle!, coarse_solve!
export MGPreconditioner, run_solver

end
