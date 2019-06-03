using Distances
using NearestNeighbors
using StatsBase

using Printf
using LinearAlgebra
using SparseArrays
using Statistics

import Base: show
import StatsBase: IntegerVector, RealVector, RealMatrix, counts

using Clustering

# Affinity propagation
#
#   Reference:
#       Clustering by Passing Messages Between Data Points.
#       Brendan J. Frey and Delbert Dueck
#       Science, vol 315, pages 972-976, 2007.
#

#### Interface

"""
    AffinityPropResult <: ClusteringResult
The output of affinity propagation clustering ([`affinityprop`](@ref)).
# Fields
 * `exemplars::Vector{Int}`: indices of *exemplars* (cluster centers)
 * `assignments::Vector{Int}`: cluster assignments for each data point
 * `iterations::Int`: number of iterations executed
 * `converged::Bool`: converged or not
 * `energy::Flaot64`: computes configuration energy
"""
mutable struct AffinityPropResult <: ClusteringResult
    exemplars::Vector{Int}      # indexes of exemplars (centers)
    assignments::Vector{Int}    # assignments for each point
    counts::Vector{Int}         # number of data points in each cluster
    iterations::Int             # number of iterations executed
    converged::Bool             # converged or not
    energy::Float64               # energy of the configuration
end

const _afp_default_maxiter = 200
const _afp_default_damp = 0.5
const _afp_default_y = 10.0
const _afp_default_γ = 0.01
const _afp_default_γfact = 1.0
const _afp_default_tol = 1.0e-6
const _afp_default_display = :none

"""
    affinityprop(S::DenseMatrix; [maxiter=200], [tol=1e-6], [damp=0.5],
                 [display=:none]) -> AffinityPropResult
Perform affinity propagation clustering based on a similarity matrix `S`.
``S_{ij}`` (``i ≠ j``) is the similarity (or the negated distance) between
the ``i``-th and ``j``-th points, ``S_{ii}`` defines the *availability*
of the ``i``-th point as an *exemplar*.
# Arguments
 - `damp::Real`: the dampening coefficient, ``0 ≤ \\mathrm{damp} < 1``.
   Larger values indicate slower (and probably more stable) update.
   ``\\mathrm{damp} = 0`` disables dampening.
 - `maxiter`, `tol`, `display`: see [common options](@ref common_options)
# References
> Brendan J. Frey and Delbert Dueck. *Clustering by Passing Messages
> Between Data Points.* Science, vol 315, pages 972-976, 2007.
"""
function affinityprop_mod(S::DenseMatrix{T};
                      maxiter::Integer=_afp_default_maxiter,
                      tol::Real=_afp_default_tol,
                      damp::Real=_afp_default_damp,
                      y::Float64=_afp_default_y,
                      γ::Float64=_afp_default_γ,
                      γfact::Float64=_afp_default_γfact,
                      display::Symbol=_afp_default_display,
                      run_vanilla::Bool=false) where T<:AbstractFloat

    # check arguments
    n = size(S, 1)
    size(S, 2) == n || throw(ArgumentError("S must be a square matrix ($(size(S)) given)."))
    n >= 2 || throw(ArgumentError("At least two data points are required ($n given)."))
    tol > 0 || throw(ArgumentError("tol must be a positive value ($tol given)."))
    0 <= damp < 1 || throw(ArgumentError("damp must be a non-negative real value below 1 ($damp given)."))

    # invoke core implementation
    _affinityprop_mod(S, round(Int, maxiter), tol, convert(T, damp), display_level(display), y, γ, γfact, run_vanilla)
end


#### Implementation

function _affinityprop_mod(S::DenseMatrix{T},
                       maxiter::Int,
                       tol::Real,
                       damp::T,
                       displevel::Int,
                       y::Float64,
                       γ::Float64,
                       γfact::Float64,
                       run_vanilla::Bool) where T<:AbstractFloat
    n = size(S, 1)
    n2 = n * n

    # initialize messages
    R = .-rand(T, n, n)./100  # responsibilities
    A = .-rand(T, n, n)./100  # availabilities

    # initialize messages reference
    R_ref = .-rand(T, n, n)./100  # responsibilities
    A_ref = .-rand(T, n, n)./100  # availabilities

    if run_vanilla
        # initialize interaction messages
        A_up = zeros(T, n)  # from replica to reference
        A_down = zeros(T, n)  # from reference to replica
    else
        # initialize interaction messages
        A_up = randn(T, n)./100  # from replica to reference
        A_down = randn(T, n)./100  # from reference to replica
    end

    # prepare storages
    Rt = Matrix{T}(undef, n, n)
    At = Matrix{T}(undef, n, n)

    # prepare storages reference
    Rt_ref = Matrix{T}(undef, n, n)
    At_ref = Matrix{T}(undef, n, n)

    # prepare interaction storage
    At_up = Vector{T}(undef, n)  # from replica to reference
    At_down = Vector{T}(undef, n)  # from reference to replica



    if displevel >= 2
        @printf "%7s %12s | %8s \n" "Iters" "objv-change" "exemplars"
        println("-----------------------------------------------------")
    end

    t = 0
    converged = false
    while !converged && t < maxiter
        t += 1
        γ *= γfact
        # compute new messages

        _afp_compute_r!(Rt, S, A, A_down)
        _afp_dampen_update!(R, Rt, damp)

        _afp_compute_a!(At, R)
        _afp_dampen_update!(A, At, damp)

        if !run_vanilla
            _afp_compute_r_ref!(Rt_ref, A_ref, A_up, y)
            _afp_dampen_update!(R_ref, Rt_ref, damp)

            _afp_compute_a!(At_ref, R_ref)
            _afp_dampen_update!(A_ref, At_ref, damp)

            _afp_compute_a_down!(At_down, A_ref, A_up, γ, y)
            _afp_dampen_update!(A_down, At_down, damp)

            _afp_compute_a_up!(At_up, A, S, γ)
            _afp_dampen_update!(A_up, At_up, damp)
        end

            # normalize_message!(A)
            # normalize_message!(R)
            # normalize_message!(A_ref)
            # normalize_message!(R_ref)
            # normalize_message!(A_up)
            # normalize_message!(A_down)

        if t % 50 == 0 || t in [1,5,10,25]
            @printf("step = %4d, a: %.3f, r: %.3f, a*: %.3f, r*: %.3f, a_up: %.3f, a_down: %.3f\n", t, mean(A), mean(R), mean(A_ref), mean(R_ref), mean(A_up), mean(A_down))
        end

        # determine convergence
        ch = max(Linfdist(A, At), Linfdist(R, Rt)) / (one(T) - damp)
        converged = (ch < tol)

        if displevel >= 2
            # count the number of exemplars
            ne = _afp_count_exemplars(A, R)
            @printf("%7d %12.4e | %8d\n", t, ch, ne)
        end
    end

    # extract exemplars and assignments
    exemplars = _afp_extract_exemplars(A, R)
    assignments, counts = _afp_get_assignments(S, exemplars)
    energy = _afp_compute_energy(S, exemplars, assignments)

    if displevel >= 1
        if converged
            println("Affinity propagation converged with $t iterations: $(length(exemplars)) exemplars.")
        else
            println("Affinity propagation terminated without convergence after $t iterations: $(length(exemplars)) exemplars.")
        end
    end

    # produce output struct
    return AffinityPropResult(exemplars, assignments, counts, t, converged, energy)
end

function normalize_message!(A::Matrix{T}) where T
    A .-= maximum(A, dims=2)
end

function normalize_message!(A::Array{T}) where T
    A .-= maximum(A)
end


# compute responsibilities
function _afp_compute_r!(R::Matrix{T}, S::DenseMatrix{T}, A::Matrix{T}, A_down::Vector{T}) where T
    n = size(S, 1)

    I1 = Vector{Int}(undef, n)  # I1[i] is the column index of the maximum element in (A+S)[i,:]
    Y1 = Vector{T}(undef, n)    # Y1[i] is the maximum element in (A+S)[i,:]
    Y2 = Vector{T}(undef, n)    # Y2[i] is the second maximum element in (A+S)[i,:]

    # Find the first and second maximum elements along each row
    @inbounds for i = 1:n
        v1 = A[i,1] + S[i,1] + (i==1) * A_down[i]
        v2 = A[i,2] + S[i,2] + (i==2) * A_down[i]
        if v1 > v2
            I1[i] = 1
            Y1[i] = v1
            Y2[i] = v2
        else
            I1[i] = 2
            Y1[i] = v2
            Y2[i] = v1
        end
    end
    @inbounds for j = 3:n, i = 1:n
        v = A[i,j] + S[i,j] + (i==j) * A_down[i]
        if v > Y2[i]
            if v > Y1[i]
                Y2[i] = Y1[i]
                I1[i] = j
                Y1[i] = v
            else
                Y2[i] = v
            end
        end
    end

    # compute R values
    @inbounds for j = 1:n, i = 1:n
        mv = (j == I1[i] ? Y2[i] : Y1[i])
        R[i,j] = S[i,j] + (i==j) * A_down[i] - mv
    end

    return R
end

# compute auxiliary replica responsibilities
function _afp_compute_r_ref!(R::Matrix{T}, A::Matrix{T}, A_up::Vector{T}, y::T) where T
    n = size(S, 1)

    I1 = Vector{Int}(undef, n)  # I1[i] is the column index of the maximum element in (A+S)[i,:]
    Y1 = Vector{T}(undef, n)    # Y1[i] is the maximum element in (A+S)[i,:]
    Y2 = Vector{T}(undef, n)    # Y2[i] is the second maximum element in (A+S)[i,:]

    # Find the first and second maximum elements along each row
    @inbounds for i = 1:n
        v1 = A[i,1] + (i==1) * y * A_up[i]
        v2 = A[i,2] + (i==2) * y * A_up[i]
        if v1 > v2
            I1[i] = 1
            Y1[i] = v1
            Y2[i] = v2
        else
            I1[i] = 2
            Y1[i] = v2
            Y2[i] = v1
        end
    end
    @inbounds for j = 3:n, i = 1:n
        v = A[i,j] + (i==j) * y * A_up[i]
        if v > Y2[i]
            if v > Y1[i]
                Y2[i] = Y1[i]
                I1[i] = j
                Y1[i] = v
            else
                Y2[i] = v
            end
        end
    end

    # compute R values
    @inbounds for j = 1:n, i = 1:n
        mv = (j == I1[i] ? Y2[i] : Y1[i])
        R[i,j] = - mv + (i==j) * y * A_up[i]
    end

    return R
end


# compute availabilities
function _afp_compute_a!(A::Matrix{T}, R::Matrix{T}) where T
    n = size(R, 1)
    z = zero(T)
    for j = 1:n
        @inbounds rjj = R[j,j]

        # compute s <- sum_{i \ne j} max(0, R(i,j))
        s = z
        for i = 1:n
            if i != j
                @inbounds r = R[i,j]
                if r > z
                    s += r
                end
            end
        end

        for i = 1:n
            if i == j
                @inbounds A[i,j] = s
            else
                @inbounds r = R[i,j]
                u = rjj + s
                if r > z
                    u -= r
                end
                A[i,j] = ifelse(u < z, u, z)
            end
        end
    end
    return A
end

# compute interactions
function _afp_compute_a_up!(A_up::Vector{T}, A::Matrix{T}, S::Matrix{T}, γ::T) where T

    n = size(S, 1)

    A_up = maximum(S .+ A + γ*I, dims=2) .- maximum(S .+ A .+ γ .* (ones(n,n)-I), dims=2)

    return A_up
end

function _afp_compute_a_down!(A_down::Vector{T}, A_ref::Matrix{T}, A_up::Vector{T}, γ::T, y::T) where T
    n = size(A_ref, 1)

    A_down = maximum(A_ref + γ*I .+ (y-1).*Diagonal(A_up), dims=2) .- maximum(A_ref .+ (y-1).*Diagonal(A_up)  .+ γ .* (ones(n,n)-I), dims=2)

    return A_down
end


# dampen update
function _afp_dampen_update!(x::Array{T}, xt::Array{T}, damp::T) where T
    ct = one(T) - damp
    for i = 1:length(x)
        @inbounds x[i] = ct * xt[i] + damp * x[i]
    end
    return x
end

# count the number of exemplars
function _afp_count_exemplars(A::Matrix, R::Matrix)
    n = size(A,1)
    c = 0
    for i = 1:n
        @inbounds if A[i,i] + R[i,i] > 0
            c += 1
        end
    end
    return c
end

# extract all exemplars
function _afp_extract_exemplars(A::Matrix, R::Matrix)
    n = size(A,1)
    r = Int[]
    for i = 1:n
        @inbounds if A[i,i] + R[i,i] > 0
            push!(r, i)
        end
    end
    return r
end

# get assignments
function _afp_get_assignments(S::DenseMatrix, exemplars::Vector{Int})
    n = size(S, 1)
    k = length(exemplars)
    Se = S[:, exemplars]
    a = Vector{Int}(undef, n)
    cnts = zeros(Int, k)
    for i = 1:n
        p = 1
        v = Se[i,1]
        for j = 2:k
            s = Se[i,j]
            if s > v
                v = s
                p = j
            end
        end
        a[i] = p
    end
    for i = 1:k
        a[exemplars[i]] = i
    end
    for i = 1:n
        @inbounds cnts[a[i]] += 1
    end
    return (a, cnts)
end

const DisplayLevels = Dict(:none => 0, :final => 1, :iter => 2)

display_level(s::Symbol) = get(DisplayLevels, s) do
    throw(ArgumentError("Invalid value for the 'display' option: $s."))
end

function _afp_compute_energy(S::Matrix{Float64}, exemplars::Vector{Int}, assignments::Vector{Int})
    E = 0
    N = size(S,2)

    for i in 1:N
        E -= S[i,exemplars[assignments[i]]]
    end

    return E
end
