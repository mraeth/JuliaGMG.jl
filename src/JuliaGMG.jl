module JuliaGMG

include("types.jl")
include("operators.jl")
include("smoother.jl")
include("transfer.jl")
include("vcycle.jl")
include("solver.jl")

export FDOp, Level, MGHierarchy, build_hierarchy
export apply!, fill_bc_dirichlet!, residual!, interior_norm
export vec_to_padded!, padded_to_vec!
export smooth!, jacobi_sweep!, rbgs_sweep!
export restrict!, prolong_add!
export vcycle!, coarse_solve!
export MGPreconditioner
export GMRES, CG, solve

end
