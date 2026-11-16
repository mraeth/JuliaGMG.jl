# Matrix-free finite-difference operator.
# Padded arrays: interior at [2:nx+1, 2:ny+1], ghost ring at rows/cols 1 and nx+2/ny+2.
# An optional third dimension holds nb independent planes; a Matrix is the case nb = 1
# (x[i,j,1] and axes(x, 3) == 1:1 are valid for a Matrix).

# Ghost value from the adjacent interior value v and the opposite-side interior value p.
_ghost(::Dirichlet, old, v, p) = -v     # u = 0 on the face
_ghost(::Neumann,   old, v, p) = v      # ∂u/∂n = 0 on the face
_ghost(::Periodic,  old, v, p) = p
# Coefficient ghosts: wrapped for Periodic, left as they are otherwise.
_ghost_coeff(bc, old, v, p) = bc isa Periodic ? p : old

# Fill the ghost ring side by side; the y pass runs over the x ghosts too, so corners
# are consistent (e.g. wrapped twice for doubly periodic).
function _fill_ghosts!(x, nx, ny, (w, e, s, n), g)
    @inbounds for k in axes(x, 3)
        for j in 2:ny+1
            x[1,    j, k] = g(w, x[1,    j, k], x[2,    j, k], x[nx+1, j, k])
            x[nx+2, j, k] = g(e, x[nx+2, j, k], x[nx+1, j, k], x[2,    j, k])
        end
        for i in 1:nx+2
            x[i, 1,    k] = g(s, x[i, 1,    k], x[i, 2,    k], x[i, ny+1, k])
            x[i, ny+2, k] = g(n, x[i, ny+2, k], x[i, ny+1, k], x[i, 2,    k])
        end
    end
    x
end

fill_bc!(x, op) = _fill_ghosts!(x, op.nx, op.ny, op.bc, _ghost)

# y ← A*x  (interior only; ghost ring of y is left undefined)
function apply!(y, op::FDOp, x)
    fill_bc!(x, op)
    ihx2 = 1 / op.hx^2
    ihy2 = 1 / op.hy^2
    offsets = op.stencil.offsets
    @inbounds for k in axes(x, 3)
        coeff = _plane(op.coeff, k)
        for j in 2:op.ny+1, i in 2:op.nx+1
            weights = op.stencil_fn(i, j, coeff, ihx2, ihy2)
            acc = zero(eltype(y))
            for l in eachindex(weights)
                di, dj = offsets[l]
                acc += weights[l] * x[i+di, j+dj, k]
            end
            y[i,j,k] = acc
        end
    end
    y
end

# r ← b - A*x
function residual!(r, lvl::Level)
    apply!(r, lvl.op, lvl.x)
    nx, ny = lvl.op.nx, lvl.op.ny
    @views r[2:nx+1, 2:ny+1, :] .= lvl.b[2:nx+1, 2:ny+1, :] .- r[2:nx+1, 2:ny+1, :]
    r
end

# Subtract the mean of each plane (singular operators work on the mean-free subspace):
# flat vector with planes of n entries, or padded array (interior only).
function _remove_mean!(v::AbstractVector, n)
    for k in 1:length(v)÷n
        p = @view v[(k-1)*n+1:k*n]
        p .-= sum(p) / n
    end
    v
end
function _remove_mean!(pad::AbstractArray, nx, ny)
    for k in axes(pad, 3)
        p = @view pad[2:nx+1, 2:ny+1, k]
        p .-= sum(p) / length(p)
    end
    pad
end

# Euclidean norm of interior DOFs over all planes (unweighted, matching MFES convention)
function interior_norm(x, nx, ny)
    sqrt(sum(abs2, @view(x[2:nx+1, 2:ny+1, :])))
end

# Flat (length nx*ny*nb, plane-major) ↔ padded (nx+2)×(ny+2)[×nb] helpers
function vec_to_padded!(pad, v, nx, ny)
    @inbounds for k in axes(pad, 3), j in 1:ny, i in 1:nx
        pad[i+1, j+1, k] = v[((k-1)*ny + j-1)*nx + i]
    end
end

function padded_to_vec!(v, pad, nx, ny)
    @inbounds for k in axes(pad, 3), j in 1:ny, i in 1:nx
        v[((k-1)*ny + j-1)*nx + i] = pad[i+1, j+1, k]
    end
end
