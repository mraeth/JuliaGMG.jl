# Matrix-free Helmholtz operator: (α - ∇²)u = f
# Padded arrays: interior at [2:nx+1, 2:ny+1], ghost ring at rows/cols 1 and nx+2/ny+2.
# Dirichlet BCs: u_ghost = -u_interior (linear reflection; boundary midpoint = 0).

function fill_bc_dirichlet!(x, op::HelmholtzOp)
    nx, ny = op.nx, op.ny
    @views x[1,      2:ny+1] .= .-x[2,    2:ny+1]
    @views x[nx+2,   2:ny+1] .= .-x[nx+1, 2:ny+1]
    @views x[2:nx+1, 1     ] .= .-x[2:nx+1, 2   ]
    @views x[2:nx+1, ny+2  ] .= .-x[2:nx+1, ny+1]
    # corners not used by stencil
    x[1, 1] = x[nx+2, 1] = x[1, ny+2] = x[nx+2, ny+2] = zero(eltype(x))
end

# y ← A*x  (interior only; ghost ring of y is left undefined)
function apply!(y, op::HelmholtzOp, x)
    fill_bc_dirichlet!(x, op)
    nx, ny = op.nx, op.ny
    ihx2 = 1 / op.hx^2
    ihy2 = 1 / op.hy^2
    @inbounds for j in 2:ny+1, i in 2:nx+1
        y[i,j] = (op.alpha[i,j] + 2*ihx2 + 2*ihy2) * x[i,j] -
                 (x[i-1,j] + x[i+1,j]) * ihx2 -
                 (x[i,j-1] + x[i,j+1]) * ihy2
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

# Build RHS f by applying A to the manufactured solution u_exact = sin(πx)sin(πy)
function make_rhs_and_exact(op::HelmholtzOp{T}) where T
    nx, ny = op.nx, op.ny
    u = zeros(T, nx+2, ny+2)
    @inbounds for j in 1:ny, i in 1:nx
        x = (i - T(0.5)) * op.hx
        y = (j - T(0.5)) * op.hy
        u[i+1, j+1] = sin(T(π)*x) * sin(T(π)*y)
    end
    b = zeros(T, nx+2, ny+2)
    apply!(b, op, u)
    b, u
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
