using Random

# All functions return a padded (nx+2)×(ny+2) Matrix{T}.
# Interior cells are at [i+1, j+1] for i=1..nx, j=1..ny.
# Ghost ring is left zero (BCs will fill it during apply!).
# Cell centres: x_i = (i - 0.5)*hx, y_j = (j - 0.5)*hy  (1-based interior index i)

function make_alpha(::Type{T}, nx::Int, ny::Int, hx::T, hy::T, field::Symbol;
                    alpha_min::Real=1.0, alpha_max::Real=1.0, n_periods::Int=1) where T
    alpha = zeros(T, nx+2, ny+2)
    amin = T(alpha_min)
    amax = T(alpha_max)

    if field == :Constant
        fill_constant!(alpha, nx, ny, amin, amax)

    elseif field == :SineWave
        fill_sinewave!(alpha, nx, ny, hx, hy, amin, amax, n_periods)

    elseif field == :Random
        fill_random!(alpha, nx, ny, amin, amax)

    elseif field == :LogUniform
        fill_loguniform!(alpha, nx, ny, amin, amax)

    elseif field == :TurbulentNoise
        fill_turbulent_noise!(alpha, nx, ny, hx, hy, amin, amax)

    elseif field == :GaussianSpiral
        fill_gaussian_spiral!(alpha, nx, ny, hx, hy, amin, amax)

    else
        error("Unknown alpha field type: $field")
    end

    alpha
end


# ---------------------------------------------------------------------------
# Constant
# ---------------------------------------------------------------------------
function fill_constant!(alpha, nx, ny, amin, amax)
    v = (amin + amax) / 2
    @inbounds for j in 2:ny+1, i in 2:nx+1
        alpha[i,j] = v
    end
end


# ---------------------------------------------------------------------------
# SineWave: α = mean + amp*sin(2π n x)*sin(2π n y)
# ---------------------------------------------------------------------------
function fill_sinewave!(alpha, nx, ny, hx, hy, amin, amax, n_periods)
    mean = (amin + amax) / 2
    amp  = (amax - amin) / 2
    kx   = 2π * n_periods
    ky   = 2π * n_periods
    @inbounds for j in 1:ny, i in 1:nx
        x = (i - eltype(alpha)(0.5)) * hx
        y = (j - eltype(alpha)(0.5)) * hy
        alpha[i+1, j+1] = mean + amp * sin(kx*x) * sin(ky*y)
    end
end


# ---------------------------------------------------------------------------
# Random: Uniform(amin, amax), non-reproducible seed
# ---------------------------------------------------------------------------
function fill_random!(alpha, nx, ny, amin, amax)
    T = eltype(alpha)
    rng = MersenneTwister()
    @inbounds for j in 1:ny, i in 1:nx
        alpha[i+1, j+1] = amin + (amax - amin) * rand(rng, T)
    end
end


# ---------------------------------------------------------------------------
# LogUniform: α = amin * (amax/amin)^U, U ~ Uniform(0,1), seed=42
# Uses Julia's MersenneTwister(42); same distribution as MFES mt19937(42).
# ---------------------------------------------------------------------------
function fill_loguniform!(alpha, nx, ny, amin, amax)
    T = eltype(alpha)
    rng = MersenneTwister(42)
    log_ratio = log(amax / amin)
    # Row-major sweep (j outer, i inner) to match MFES mt19937 draw order
    @inbounds for j in 1:ny, i in 1:nx
        u = rand(rng, T)
        alpha[i+1, j+1] = amin * exp(log_ratio * u)
    end
end


# ---------------------------------------------------------------------------
# TurbulentNoise: band-limited noise, M=15 modes per dim, seed=5
# Exact port of turbulence_noise_raw() from Helmholtz.cpp using UInt32 hashes.
# ---------------------------------------------------------------------------
function _hash_rng(ix::UInt32, iy::UInt32, k::UInt32, seed::UInt32)::Float64
    val = (ix * UInt32(73856093)) ⊻ (iy * UInt32(19349663)) ⊻ (k * UInt32(83492791)) ⊻ seed
    val = (val >> UInt32(13)) ⊻ val
    val = val * (val * val * UInt32(15731) + UInt32(789221)) + UInt32(1376312589)
    Float64(val & UInt32(0x7fffffff)) / 10000.0 % 1.0
end

function fill_turbulent_noise!(alpha, nx, ny, hx, hy, amin, amax)
    T = eltype(alpha)
    M = 15   # N=32, M = N÷2 - 1 = 15
    seed = UInt32(5)

    noise = zeros(Float64, nx, ny)

    for ix in 1:M, iy in 1:M
        ixu = UInt32(ix); iyu = UInt32(iy)
        r1 = _hash_rng(ixu, iyu, UInt32(1), seed)
        r2 = _hash_rng(ixu, iyu, UInt32(2), seed)
        r3 = _hash_rng(ixu, iyu, UInt32(3), seed)
        r4 = _hash_rng(ixu, iyu, UInt32(4), seed)
        denom = 4π^2 * (ix^2 + iy^2)
        @inbounds for j in 1:ny, i in 1:nx
            x = (i - 0.5) * hx
            y = (j - 0.5) * hy
            noise[i,j] += r1 * sin(2π*ix*(x + r2)) * r3 * sin(2π*iy*(y + r4)) / denom
        end
    end

    # Window by sin(πx)sin(πy) to enforce boundary zeros
    @inbounds for j in 1:ny, i in 1:nx
        x = (i - 0.5) * hx
        y = (j - 0.5) * hy
        noise[i,j] *= sin(π*x) * sin(π*y)
    end

    vmin, vmax = extrema(noise)
    span = vmax - vmin + 1e-300
    log_ratio = log(amax / amin)
    @inbounds for j in 1:ny, i in 1:nx
        t = (noise[i,j] - vmin) / span
        alpha[i+1, j+1] = T(amin * exp(log_ratio * t))
    end
end


# ---------------------------------------------------------------------------
# GaussianSpiral: 4-arm Archimedean spiral, a=4π, σ=0.05
# Exact port of gaussian_spiral_raw() from Helmholtz.cpp.
# ---------------------------------------------------------------------------
function fill_gaussian_spiral!(alpha, nx, ny, hx, hy, amin, amax)
    T = eltype(alpha)
    a     = 4.0π
    sigma = 0.05
    log_ratio = log(amax / amin)

    @inbounds for j in 1:ny, i in 1:nx
        x = (i - 0.5) * hx
        y = (j - 0.5) * hy
        dx = x - 0.5; dy = y - 0.5
        r  = sqrt(dx^2 + dy^2)
        theta = atan(dy, dx)
        theta < 0 && (theta += 2π)

        value = 0.0
        for arm in 0:3
            theta += π / 2
            in_band = (r > 0.05 * (arm % 2)) & (r < (0.5 - 0.03 * ((arm + 1) % 2)))
            if in_band
                theta_spiral = a * r
                dtheta = rem(theta - theta_spiral + π, 2π) - π
                dtheta < -π && (dtheta += 2π)
                dtheta >  π && (dtheta -= 2π)
                arc = r * dtheta
                value += exp(-arc^2 / (2 * sigma^2 * (r + 0.1)^2))
            end
        end
        value += exp(-500.0 * (r - 0.5)^2 / (2 * sigma^2))
        t = clamp(value, 0.0, 1.0)
        alpha[i+1, j+1] = T(amin * exp(log_ratio * t))
    end
end
