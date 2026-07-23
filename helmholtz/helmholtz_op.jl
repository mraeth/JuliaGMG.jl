# Helmholtz operator constructor: (α - ∇²)u = f on (0,1)², Dirichlet BCs.
# Returns an FDOp whose stencil_fn implements the 5-point Helmholtz stencil and
# whose coarsen_fn averages each 2×2 block of α arithmetically.
# Requires FDOp and STENCIL_5PT to be in scope (load via `using JuliaGMG`).
function HelmholtzOp(alpha::A, nx, ny, hx::T, hy::T) where {T, A<:AbstractMatrix{T}}
    # Weights ordered to match STENCIL_5PT offsets: west, east, south, north, center.
    stencil_fn = (i, j, coeff, ihx2, ihy2) ->
        (-ihx2, -ihx2, -ihy2, -ihy2, coeff[i,j] + 2*ihx2 + 2*ihy2)
    coarsen_fn = (fa, cnx, cny) -> begin
        ca = similar(fa, cnx+2, cny+2)
        fill!(ca, zero(eltype(fa)))
        @views ca[2:cnx+1, 2:cny+1] .= eltype(fa)(0.25) .* (
            fa[2:2:2*cnx+1, 2:2:2*cny+1] .+
            fa[3:2:2*cnx+2, 2:2:2*cny+1] .+
            fa[2:2:2*cnx+1, 3:2:2*cny+2] .+
            fa[3:2:2*cnx+2, 3:2:2*cny+2])
        ca
    end
    FDOp(alpha, nx, ny, hx, hy, STENCIL_5PT, stencil_fn, coarsen_fn)
end
