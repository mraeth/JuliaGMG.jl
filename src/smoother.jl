# Smoothers for the Helmholtz operator.

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

    for _ in 1:nsweeps
        residual!(lvl.r, lvl)
        @inbounds for j in 2:ny+1, i in 2:nx+1
            diag = op.alpha[i,j] + 2*ihx2 + 2*ihy2
            lvl.x[i,j] += ω * lvl.r[i,j] / diag
        end
    end
end


# ---------------------------------------------------------------------------
# Red-Black Gauss-Seidel (same smoother as MFES)
# Red cells: (i+j) even;  Black cells: (i+j) odd
# ---------------------------------------------------------------------------
function rbgs_sweep!(lvl::Level, nsweeps::Int)
    op = lvl.op
    nx, ny = op.nx, op.ny
    ihx2 = 1 / op.hx^2
    ihy2 = 1 / op.hy^2

    for _ in 1:nsweeps
        fill_bc_dirichlet!(lvl.x, op)

        # Red pass (i+j even)
        @inbounds for j in 2:ny+1, i in 2:nx+1
            (i + j) % 2 == 0 || continue
            diag = op.alpha[i,j] + 2*ihx2 + 2*ihy2
            off  = (lvl.x[i-1,j] + lvl.x[i+1,j]) * ihx2 +
                   (lvl.x[i,j-1] + lvl.x[i,j+1]) * ihy2
            lvl.x[i,j] = (lvl.b[i,j] + off) / diag
        end

        fill_bc_dirichlet!(lvl.x, op)

        # Black pass (i+j odd)
        @inbounds for j in 2:ny+1, i in 2:nx+1
            (i + j) % 2 == 1 || continue
            diag = op.alpha[i,j] + 2*ihx2 + 2*ihy2
            off  = (lvl.x[i-1,j] + lvl.x[i+1,j]) * ihx2 +
                   (lvl.x[i,j-1] + lvl.x[i,j+1]) * ihy2
            lvl.x[i,j] = (lvl.b[i,j] + off) / diag
        end
    end
end


# Default smoother used by the V-cycle: RBGS
smooth!(lvl::Level, nsweeps::Int) = rbgs_sweep!(lvl, nsweeps)
