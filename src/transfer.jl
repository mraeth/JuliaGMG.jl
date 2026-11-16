# Transfer operators between multigrid levels.
#
# Grid convention: interior index I (1-based) maps to padded index I+1.
# Per direction the coarse level either halves the fine size (factor f = 2) or keeps it
# (f = 1, semi-coarsening); the factors are inferred from the array sizes.
#
# Restriction: average of the fx×fy fine children (4-cell full weighting for full
# coarsening, matching MFES; 2-cell average for semi-coarsening).
#   coarse[I,J] = mean(fine[fx(I-1)+1 .. fx*I, fy(J-1)+1 .. fy*J])  (interior-based)
#
# Prolongation: piecewise-constant (inject coarse value to all fx×fy fine children), matching MFES.
# Restriction = prolongationᵀ / (fx fy), so the V-cycle stays symmetric.

# b_coarse ← R * r_fine  (writes interior of b_coarse; ghost ring is left zero)
function restrict!(b_coarse, r_fine, fine_op)
    cnx, cny = size(b_coarse, 1) - 2, size(b_coarse, 2) - 2
    fx, fy = fine_op.nx ÷ cnx, fine_op.ny ÷ cny
    T = eltype(b_coarse)
    w = T(1 / (fx * fy))
    @inbounds for k in axes(b_coarse, 3), J in 1:cny, I in 1:cnx
        s = zero(T)
        for dj in 1:fy, di in 1:fx
            s += r_fine[fx*(I-1)+di+1, fy*(J-1)+dj+1, k]
        end
        b_coarse[I+1, J+1, k] = w * s
    end
end

# x_fine += P * x_coarse  (adds to interior of x_fine; ghost ring not touched here)
function prolong_add!(x_fine, x_coarse, coarse_op)
    cnx, cny = coarse_op.nx, coarse_op.ny
    fx, fy = (size(x_fine, 1) - 2) ÷ cnx, (size(x_fine, 2) - 2) ÷ cny
    @inbounds for k in axes(x_fine, 3), J in 1:cny, I in 1:cnx
        v = x_coarse[I+1, J+1, k]
        for dj in 1:fy, di in 1:fx
            x_fine[fx*(I-1)+di+1, fy*(J-1)+dj+1, k] += v
        end
    end
end
