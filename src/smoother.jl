# Smoothers for generic FDOp.  Weights come from op.stencil_fn and the stencil
# topology from op.stencil, so the same smoother code works for any N-point operator.

# ---------------------------------------------------------------------------
# Weighted Jacobi: x += ω * D^{-1} * (b - Ax)
# ---------------------------------------------------------------------------
function jacobi_sweep!(lvl::Level, nsweeps::Int; omega=2/3)
    op = lvl.op
    nx, ny = op.nx, op.ny
    T = eltype(lvl.x)
    ω = T(omega)
    ihx2 = 1 / op.hx^2
    ihy2 = 1 / op.hy^2

    cidx = op.stencil.center_idx
    for _ in 1:nsweeps
        residual!(lvl.r, lvl)
        @inbounds for k in axes(lvl.x, 3)
            coeff = _plane(op.coeff, k)
            for j in 2:ny+1, i in 2:nx+1
                d = op.stencil_fn(i, j, coeff, ihx2, ihy2)[cidx]
                lvl.x[i,j,k] += ω * lvl.r[i,j,k] / d
            end
        end
    end
end


# ---------------------------------------------------------------------------
# Red-Black Gauss-Seidel
# Red cells: (i+j) even;  Black cells: (i+j) odd
# reverse=true sweeps black then red: the adjoint of the forward sweep, so a
# V-cycle with forward pre- and reverse post-smoothing is symmetric (valid for CG).
# ---------------------------------------------------------------------------
function rbgs_sweep!(lvl::Level, nsweeps::Int; reverse::Bool=false)
    first, second = reverse ? (1, 0) : (0, 1)
    for _ in 1:nsweeps
        _rbgs_colour!(lvl, first)
        _rbgs_colour!(lvl, second)
    end
end

# One Gauss-Seidel pass over the cells with (i+j) % 2 == colour.
function _rbgs_colour!(lvl::Level, colour::Int)
    op = lvl.op
    nx, ny = op.nx, op.ny
    ihx2 = 1 / op.hx^2
    ihy2 = 1 / op.hy^2
    cidx    = op.stencil.center_idx
    offsets = op.stencil.offsets

    fill_bc!(lvl.x, op)
    @inbounds for k in axes(lvl.x, 3)
        coeff = _plane(op.coeff, k)
        for j in 2:ny+1, i in 2:nx+1
            (i + j) % 2 == colour || continue
            weights = op.stencil_fn(i, j, coeff, ihx2, ihy2)
            d   = weights[cidx]
            off = zero(eltype(lvl.x))
            for l in eachindex(weights)
                l == cidx && continue
                di, dj = offsets[l]
                off += weights[l] * lvl.x[i+di, j+dj, k]
            end
            lvl.x[i,j,k] = (lvl.b[i,j,k] - off) / d
        end
    end
end


# Default smoother used by the V-cycle: RBGS
smooth!(lvl::Level, nsweeps::Int; reverse::Bool=false) = rbgs_sweep!(lvl, nsweeps; reverse)
