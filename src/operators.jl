# Matrix-free finite-difference operator.
# Padded arrays: interior at [2:nx+1, 2:ny+1], ghost ring at rows/cols 1 and nx+2/ny+2.
# Dirichlet BCs: u_ghost = -u_interior (linear reflection; boundary midpoint = 0).

function fill_bc_dirichlet!(x, op)
    nx, ny = op.nx, op.ny
    @views x[1,      2:ny+1] .= .-x[2,    2:ny+1]
    @views x[nx+2,   2:ny+1] .= .-x[nx+1, 2:ny+1]
    @views x[2:nx+1, 1     ] .= .-x[2:nx+1, 2   ]
    @views x[2:nx+1, ny+2  ] .= .-x[2:nx+1, ny+1]
    # corners not used by stencil
    x[1, 1] = x[nx+2, 1] = x[1, ny+2] = x[nx+2, ny+2] = zero(eltype(x))
end

# y ← A*x  (interior only; ghost ring of y is left undefined)
function apply!(y, op::FDOp, x)
    fill_bc_dirichlet!(x, op)
    ihx2 = 1 / op.hx^2
    ihy2 = 1 / op.hy^2
    offsets = op.stencil.offsets
    @inbounds for j in 2:op.ny+1, i in 2:op.nx+1
        weights = op.stencil_fn(i, j, op.coeff, ihx2, ihy2)
        acc = zero(eltype(y))
        for k in eachindex(weights)
            di, dj = offsets[k]
            acc += weights[k] * x[i+di, j+dj]
        end
        y[i,j] = acc
    end
    y
end

# r ← b - A*x
function residual!(r, lvl::Level)
    apply!(r, lvl.op, lvl.x)
    nx, ny = lvl.op.nx, lvl.op.ny
    @views r[2:nx+1, 2:ny+1] .= lvl.b[2:nx+1, 2:ny+1] .- r[2:nx+1, 2:ny+1]
    r
end

# Euclidean norm of interior DOFs (unweighted, matching MFES convention)
function interior_norm(x, nx, ny)
    sqrt(sum(abs2, @view(x[2:nx+1, 2:ny+1])))
end

# Flat (length nx*ny) ↔ padded (nx+2)×(ny+2) helpers
function vec_to_padded!(pad, v, nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        pad[i+1, j+1] = v[(j-1)*nx + i]
    end
end

function padded_to_vec!(v, pad, nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        v[(j-1)*nx + i] = pad[i+1, j+1]
    end
end
