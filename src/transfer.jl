# Transfer operators between multigrid levels.
#
# Grid convention: interior index I (1-based) maps to padded index I+1.
# Fine level: nx×ny interior cells.
# Coarse level: (nx÷2)×(ny÷2) interior cells.
#
# Restriction: full-weighting (4-cell average), matching MFES.
#   coarse[I,J] = 0.25 * sum(fine[2I-1..2I, 2J-1..2J])  (interior-based)
#   In padded indices: coarse[I+1, J+1] uses fine[2I, 2I+1] × fine[2J, 2J+1]
#
# Prolongation: piecewise-constant (inject coarse value to all 4 fine children), matching MFES.
#   fine[2I-1+di, 2J-1+dj] += coarse[I,J]  for di,dj ∈ {0,1}
#   In padded indices: fine[2I, 2I+1] × fine[2J, 2J+1] += coarse[I+1, J+1]

# b_coarse ← R * r_fine  (writes interior of b_coarse; ghost ring is left zero)
function restrict!(b_coarse, r_fine, fine_op)
    nx, ny = fine_op.nx, fine_op.ny
    cnx, cny = nx ÷ 2, ny ÷ 2
    T = eltype(b_coarse)
    @inbounds for J in 1:cny, I in 1:cnx
        fi = 2*I; fj = 2*J
        b_coarse[I+1, J+1] = T(0.25) * (r_fine[fi,   fj  ] +
                                          r_fine[fi+1, fj  ] +
                                          r_fine[fi,   fj+1] +
                                          r_fine[fi+1, fj+1])
    end
end

# x_fine += P * x_coarse  (adds to interior of x_fine; ghost ring not touched here)
function prolong_add!(x_fine, x_coarse, coarse_op)
    cnx, cny = coarse_op.nx, coarse_op.ny
    @inbounds for J in 1:cny, I in 1:cnx
        fi = 2*I; fj = 2*J
        v = x_coarse[I+1, J+1]
        x_fine[fi,   fj  ] += v
        x_fine[fi+1, fj  ] += v
        x_fine[fi,   fj+1] += v
        x_fine[fi+1, fj+1] += v
    end
end
