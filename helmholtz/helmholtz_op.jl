# Helmholtz operator constructor: (α - ∇²)u = f on (0,1)², Dirichlet BCs.
# Returns an FDOp whose stencil_fn implements the 5-point Helmholtz stencil and
# whose coarsen_fn averages each 2×2 (or 2×1/1×2) block of α arithmetically.
# Requires FDOp and STENCIL_5PT to be in scope (load via `using JuliaGMG`).
function HelmholtzOp(alpha::A, nx, ny, hx::T, hy::T) where {T, A<:AbstractMatrix{T}}
    # Weights ordered to match STENCIL_5PT offsets: west, east, south, north, center.
    stencil_fn = (i, j, coeff, ihx2, ihy2) ->
        (-ihx2, -ihx2, -ihy2, -ihy2, coeff[i,j] + 2*ihx2 + 2*ihy2)
    # Arithmetic mean over the fx×fy fine children (factor 1 or 2 per direction).
    coarsen_fn = (fa, cnx, cny) -> begin
        fx, fy = (size(fa, 1) - 2) ÷ cnx, (size(fa, 2) - 2) ÷ cny
        ca = similar(fa, cnx+2, cny+2)
        fill!(ca, zero(eltype(fa)))
        for J in 1:cny, I in 1:cnx
            s = zero(eltype(fa))
            for dj in 1:fy, di in 1:fx
                s += fa[fx*(I-1)+di+1, fy*(J-1)+dj+1]
            end
            ca[I+1, J+1] = s / (fx * fy)
        end
        ca
    end
    FDOp(alpha, nx, ny, hx, hy, STENCIL_5PT, stencil_fn, coarsen_fn)
end
