module APBP

using Random
using Statistics
using ExtractMacro
using LinearAlgebra

using Clustering

mutable struct FMessages
    N::Integer
    # original graph messages
    ψ1::Matrix{Float64}
    ψ2::Matrix{Float64}
    ϕ1::Matrix{Float64}
    ϕ2::Matrix{Float64}
    # auxiliary graph messages
    ψs1::Matrix{Float64}
    ψs2::Matrix{Float64}
    ϕs1::Matrix{Float64}
    ϕs2::Matrix{Float64}
    # exchange messages
    # from/to auxiliary graph and R
    ψ̂::Vector{Float64}
    ϕ̂::Vector{Float64}
    # from/to original graph and R
    ψ̃::Vector{Float64}
    ϕ̃::Vector{Float64}
    function FMessages(N, x = 0.0)
        # the messages are initialized to zero
        ψ1 = x .* randn(N, N)
        ψ2 = x .* randn(N, N)
        ϕ1 = x .* randn(N, N)
        ϕ2 = x .* randn(N, N)
        ψs1 = x .* randn(N, N)
        ψs2 = x .* randn(N, N)
        ϕs1 = x .* randn(N, N)
        ϕs2 = x .* randn(N, N)
        ψ̂ = x .* randn(N)
        ϕ̂ = x .* randn(N)
        ψ̃ = x .* randn(N)
        ϕ̃ = x .* randn(N)
        return new(N, ψ1, ψ2, ϕ1, ϕ2, ψs1, ψs2, ϕs1, ϕs2, ψ̂, ϕ̂, ψ̃, ϕ̃)
    end
end

# mutable struct Messages
#     N::Integer
#     # original graph messages
#     ψ::Array{Float64, 3}
#     ϕ::Array{Float64, 3}
#     function Messages(N)
#         # the messages are initialized to zero
#         ψ = zeros(Float64, 3, N, N)
#         ϕ = zeros(Float64, 3, N, N)
#         return new(N, ψ, ϕ)
#     end
# end

function gen_data(k = 3, σ = 0.1, nk = 30, m = 2)
    x0 = [rand(m) for j = 1:k]
    if σ isa Number
        σ = [σ for j = 1:k]
    end
    @assert σ isa Vector
    @assert length(σ) == k
    n = nk * k
    data = zeros(m, n)
    for j = 1:k
        data[:,(j-1)*nk+1:j*nk] = randn(m, nk) .* σ[j] .+ x0[j]
    end
    return data, x0
end

function compute_similarities(data::Matrix{Float64})
    # compute (anti)similarities from data using euclidean distances
    N = size(data, 2)
    s = zeros(N, N)
    [s[i,j] = sum((data[:,i] .- data[:,j]).^2) for i = 1:N for j = 1:N]
    return s
end

function runFBP(data::Matrix{Float64}, γ::Float64, y::Float64;
                seed::Int = 123,
                δ::Float64 = 0.0, # damping parameter (δ = 0 -> no damping)
                γfact::Float64 = 1.0,
                yfact::Float64 = 1.0,
                λ::Union{Float64,Nothing} = nothing, # self-similarity
                initrand::Float64 = 0.0,
                max_iter::Integer = 1_000, t_stop::Integer = 10)

    seed > 0 && Random.seed!(seed)
    N = size(data, 2)
    s = compute_similarities(data)
    λ == nothing && (λ = median(s))

    mess = FMessages(N, initrand) # mess are initialized to gaussians with σ=initrand
    d = zeros(Int, N)
    p = zeros(Int, N)
    t = 1
    t_stab = 0
    E = 0.0
    while t < max_iter && t_stab < t_stop
        oneFBPstep!(mess, s, λ, γ, y, δ)
        d, p = assign_variables(mess, s, λ)
        E = 0.0
        if is_good(d, p)
            t_stab += 1
            E = energy(d, p, s, λ)
        else
            t_stab = 0
        end
        d2, p2 = assign_variables_best(mess, s, λ)
        E2 = energy(d2, p2, s, λ)
        println("t = $t \t t_stab = $t_stab \t γ = $γ \t y = $y \t E = $E E2 = $E2 \t num_exemplars = $(sum(d.==1))")
        t += 1
        γ *= γfact
        y *= yfact
    end
    println("energy AP: \t $(energy_AP(s, λ))")

    return is_good(d, p), t, E, d, p
end

macro damp(δ, ex)
    @assert Meta.isexpr(ex, :(=))
    δ = esc(δ)
    dst = ex.args[1]
    src = ex.args[2]
    if Meta.isexpr(src, :tuple)
        rx = Expr(:tuple, [:($(esc(dst))[$i] * $δ + $(esc(src.args[i])) * (1 - $δ)) for i = 1:length(src.args)]...)
        return :($(esc(dst)) = $rx)
    end
    dst = esc(dst)
    src = esc(src)
    return :($dst = $dst * $δ + $src * (1 - $δ))
end

# function runRBP(data::Matrix{Float64}, ρ::Float64;
#                 seed::Int = 123,
#                 ρfact::Float64 = 1.0,
#                 λ = nothing, # self-similarity
#                 max_iter::Integer = 1_000, t_stop::Integer = 10)
#
#     seed > 0 && Random.seed!(seed)
#     N = size(data, 2)
#     s = compute_similarities(data)
#     if λ == nothing
#         λ = median(s)
#     end
#
#     mess = Messages(N) # mess are initialized to zero
#     d = zeros(Int, N)
#     p = zeros(Int, N)
#     t = 1
#     t_stab = 0
#     E = 0.0
#     ψ¹ = zeros(Float64, N)
#     ψ² = zeros(Float64, N)
#     while t < max_iter && t_stab < t_stop
#         oneRBPstep!(mess, ψ¹, ψ², s, λ, ρ)
#         #oneRBPstep!(mess, s, λ, ρ)
#         normalize_messages!(mess)
#         d, p = assign_variables(mess, s, λ)
#         E = 0.0
#         if is_good(d, p)
#             t_stab += 1
#             E = energy(d, p, s, λ)
#         else
#             t_stab = 0
#         end
#         println("t = $t \t t_stab = $t_stab \t ρ = $ρ \t E = $E \t num_exemplars = $(sum(d.==1))")
#         t += 1
#         ρ *= ρfact
#     end
#     return is_good(d, p), t, E, d, p
# end

function oneFBPstep!(mess::FMessages, s::Matrix{Float64}, λ::Float64, γ::Float64, y::Float64, δ::Float64)
    # perform a single asynchronous step of f-BP update equations
    @extract mess : N ψ1 ψ2 ϕ1 ϕ2 ψs1 ψs2 ϕs1 ϕs2 ψ̂ ϕ̂ ψ̃ ϕ̃

    sumϕ1v = zeros(N)
    sumϕs1v = zeros(N)

    @inbounds for i in randperm(N)
        sumϕ1 = 0.0
        sumϕs1 = 0.0
        for j in 1:N
            j == i && continue
            ψ1ji = ψ1[j,i]
            ψ2ji = ψ2[j,i]
            ϕ3 = max(ψ1ji, 0.0)
            @damp δ  ϕ1[j,i] = max(0.0, ψ2ji - ϕ3)
            @damp δ  ϕ2[j,i] = ψ1ji - ϕ3
            sumϕ1 += ϕ1[j,i]

            ψs1ji = ψs1[j,i]
            ψs2ji = ψs2[j,i]
            ϕs3 = max(ψs1ji, 0.0)
            @damp δ  ϕs1[j,i] = max(0.0, ψs2ji - ϕs3)
            @damp δ  ϕs2[j,i] = ψs1ji - ϕs3
            sumϕs1 += ϕs1[j,i]
        end
        sumϕ1v[i], sumϕs1v[i] = sumϕ1, sumϕs1

    end
    @inbounds for i in randperm(N)
        sumϕ1, sumϕs1 = sumϕ1v[i], sumϕs1v[i]

        ϕ2ᴹ, ϕ2ᵐ, jᴹ = -Inf, -Inf, -1
        for j = 1:N
            j == i && continue
            ϕ2ⱼ = ϕ2[j,i] - s[j,i]
            if ϕ2ⱼ > ϕ2ᴹ
                ϕ2ᴹ, ϕ2ᵐ, jᴹ = ϕ2ⱼ, ϕ2ᴹ, j
            elseif ϕ2ⱼ > ϕ2ᵐ
                ϕ2ᵐ = ϕ2ⱼ
            end
        end
        @assert jᴹ ≠ -1

        ϕs2ᴹ, ϕs2ᵐ, jsᴹ = -Inf, -Inf, -1
        for j = 1:N
            j == i && continue
            ϕs2ⱼ = ϕs2[j,i]
            if ϕs2ⱼ > ϕs2ᴹ
                ϕs2ᴹ, ϕs2ᵐ, jsᴹ = ϕs2ⱼ, ϕs2ᴹ, j
            elseif ϕs2ⱼ > ϕs2ᵐ
                ϕs2ᵐ = ϕs2ⱼ
            end
        end
        @assert jsᴹ ≠ -1

        # original graph cavity fields
        for j = 1:N
            j == i && continue

            ψ3 = jᴹ==j ? ϕ2ᵐ : ϕ2ᴹ

            @damp δ  ψ1[i,j] = -λ + sumϕ1 - ϕ1[j,i] + ϕ̂[i] - ψ3
            @damp δ  ψ2[i,j] = -s[i,j] - ψ3
        end

        # auxiliary graph cavity fields
        for j = 1:N
            j == i && continue

            ψs3 = jsᴹ==j ? ϕs2ᵐ : ϕs2ᴹ

            @damp δ  ψs1[i,j] = sumϕs1 - ϕs1[j,i] + y * ϕ̃[i] - ψs3
            @damp δ  ψs2[i,j] = -ψs3
        end

        # ϕ̃[i] = γ + max(ψ̂[i], -γ) - max(ψ̂[i], γ)
        @damp δ  ϕ̃[i] = abs(ψ̂[i]) ≥ γ ? sign(ψ̂[i]) * γ : ψ̂[i] # assumes γ > 0
        @damp δ  ϕ̂[i] = abs(ψ̃[i]) ≥ γ ? sign(ψ̃[i]) * γ : ψ̃[i] # assumes γ > 0

        @damp δ  ψ̂[i] = sumϕ1 - λ - ϕ2ᴹ
        @damp δ  ψ̃[i] = (y-1) * ϕ̃[i] + sumϕs1 - ϕs2ᴹ
    end
end

# function oneRBPstep!(mess::Messages, ψ¹::Array{Float64,1}, ψ²::Array{Float64,1}, s::Array{Float64, 2}, λ::Float64, ρ::Float64)
#     # perform a single asynchronous step of R-BP update equations
#     @extract mess : N ψ ϕ
#
#     @inbounds for i in randperm(N)
#
#         S1 = sum(ϕ[1,:,i])
#         S3 = sum(ϕ[3,:,i])
#         # compute local fields for reinforcement
#         new_loc1 = -λ + S1 + ρ * ψ¹[i]
#         new_loc2 = -Inf
#         ψᴮ = 0.0
#         @inbounds for j in 1:N
#             j == i && continue
#             ψᴮ = -s[i,j] + S3 - ϕ[3,j,i] + ϕ[2,j,i] + ρ * ψ²[i]
#             if ψᴮ > new_loc2
#                 new_loc2 = ψᴮ
#             end
#         end
#
#         # update all the ϕs. In general ϕ(t) = f(ψ(t-1))
#         ϕ[1,i,:] = max.(ψ[1,i,:], ψ[2,i,:], ψ[3,i,:])
#         ϕ[2,i,:] = ψ[1,i,:]
#         ϕ[3,i,:] = max.(ψ[1,i,:], ψ[3,i,:])
#
#         # update all the ψs.
#         S1 = sum(ϕ[1,:,i])
#         S3 = sum(ϕ[3,:,i])
#
#         @inbounds for j = 1:N
#             ψ[1,i,j] = -λ + S1 - ϕ[1,j,i] + ρ * ψ¹[i]
#             ψ[2,i,j] = -s[i,j] + S3 - ϕ[3,j,i] + ρ * ψ²[i]
#             r = filter(x->!(x in [i,j]), 1:N)
#             ψ[3,i,j] = S3 - ϕ[3,j,i] + maximum(ϕ[2,r,i] - ϕ[3,r,i] - s[r,i]) + ρ * ψ²[i]
#         end
#         ψ[:,i,i] .= 0.0 # phihat riempie la diagonale anche se ϕ[i,i] = 0
#
#         ψ¹[i] = new_loc1
#         ψ²[i] = new_loc2
#
#     end
#
# end

# function oneRBPstep!(mess::Messages, s::Array{Float64, 2}, λ::Float64, ρ::Float64)
#     # perform a single asynchronous step of R-BP update equations
#     @extract mess : N ψ ϕ
#
#     @inbounds for i in randperm(N)
#
#         # update all the ϕs. In general ϕ(t) = f(ψ(t-1))
#         ϕ[1,i,:] = max.(ψ[1,i,:], ψ[2,i,:], ψ[3,i,:])
#         ϕ[2,i,:] = ψ[1,i,:]
#         ϕ[3,i,:] = max.(ψ[1,i,:], ψ[3,i,:])
#
#         # update all the ψs.
#         S1 = sum(ϕ[1,:,i])
#         S3 = sum(ϕ[3,:,i])
#
#         # compute local fields for reinforcement
#         # nothing changes if I update these with ϕ(t) instead of ϕ(t-1)
#         ψ¹ = -λ + S1
#         ψ² = -Inf
#         @inbounds for j in 1:N
#             j == i && continue
#             ψᴮ = -s[i,j] + S3 - ϕ[3,j,i] + ϕ[2,j,i]
#             if ψᴮ > ψ²
#                 ψ² = ψᴮ
#             end
#         end
#
#         # NOT SURE ABOUT TIME INDICES
#         @inbounds for j = 1:N
#             ψ[1,i,j] = -λ + S1 - ϕ[1,j,i] + ρ * ψ¹
#             ψ[2,i,j] = -s[i,j] + S3 - ϕ[3,j,i] + ρ * ψ²
#             r = filter(x->!(x in [i,j]), 1:N)
#             ψ[3,i,j] = S3 - ϕ[3,j,i] + maximum(ϕ[2,r,i] - ϕ[3,r,i] - s[r,i]) + ρ * ψ²
#         end
#         ψ[:,i,i] .= 0.0 # phihat riempie la diagonale anche se ϕ[i,i] = 0
#
#     end
#
# end

function assign_variables(mess::FMessages, s::Matrix{Float64}, λ::Float64)
    @extract mess : N ϕ1 ϕ2 ϕ̂

    p = zeros(Int, N) # pointers
    d = fill(1, N)    # depths

    for i = 1:N
        sumϕ1 = sum(ϕ1[k,i] for k = 1:N if k ≠ i)
        ψᴬ = -λ + sumϕ1 + ϕ̂[i]
        ψᴹ = -Inf
        jᴹ = -1
        for j in 1:N
            j == i && continue
            ψᴮ = -s[i,j] + ϕ2[j,i] - ϕ̂[i]
            if ψᴮ > ψᴹ
                ψᴹ = ψᴮ
                jᴹ = j
            end
        end
        if ψᴹ > ψᴬ
            @assert jᴹ ≠ -1
            d[i] = 2
            p[i] = jᴹ
        end
    end
    return d, p
end

function assign_variables_best(mess::FMessages, s::Matrix{Float64}, λ::Float64)
    @extract mess : N ϕ1 ϕ2 ϕ̂

    p = zeros(Int, N) # pointers
    d = fill(1, N)    # depths
    ex = Int[]

    for i = 1:N
        sumϕ1 = sum(ϕ1[k,i] for k = 1:N if k ≠ i)
        ψ1 = -λ + sumϕ1 + ϕ̂[i]
        ψ2 = maximum(-s[i,j] + ϕ2[j,i] - ϕ̂[i] for j = 1:N if j ≠ i)
        if ψ1 > ψ2
            push!(ex, i)
        else
            d[i] = 2
        end
    end
    length(ex) == 0 && (push!(ex, 1); d[1] = 1)
    for i = 1:N
        d[i] == 1 && continue
        p[i] = ex[findmin(s[i,ex])[2]]
        # @assert p[i] ∈ ex
    end
    # @show d, p
    @assert is_good(d, p)
    # @info "BP ex = $ex"

    return d, p
end

# function assign_variables(mess::Messages, s::Matrix{Float64}, λ::Float64)
#     @extract mess : N ψ ϕ
#
#     p = zeros(Int, N) # pointers
#     d = zeros(Int, N) # depths
#     ψᴬ = 0.0
#     ψᴮ = 0.0
#     ψᴹ = 0.0
#
#     @inbounds for i = 1:N
#         ψᴬ = -λ + sum(ϕ[1,:,i])
#         ψᴹ = -Inf
#         jᴹ = -1
#         S = sum(ϕ[3,:,i])
#         @inbounds for j in 1:N
#             j == i && continue
#             ψᴮ = -s[i,j] + S - ϕ[3,j,i] + ϕ[2,j,i]
#             if ψᴮ > ψᴹ
#                 ψᴹ = ψᴮ
#                 jᴹ = j
#             end
#         end
#         if ψᴬ > ψᴹ
#             d[i] = 1
#             p[i] = 0 # is pointing to the root
#         else
#             @assert jᴹ ≠ -1
#             d[i] = 2
#             p[i] = jᴹ
#         end
#     end
#     return d, p
# end

function is_good(d::Vector{Int}, p::Vector{Int})
    N = size(d, 1)
    @inbounds for i = 1:N
        p[i] ≠ 0 && d[p[i]] == 2 && return false
    end
    return true
end

function energy_AP(s::Matrix{Float64}, λ::Float64)
    N = size(s, 2)
    s_ap = -s
    s_ap[diagind(s_ap)] .= -λ

    @time R = affinityprop(s_ap, display=:none)
    R.converged || return Inf

    as = R.assignments
    ex = R.exemplars
    p = ex[as]
    d = fill(2, N)
    for j in ex
        p[j], d[j] = 0, 1
    end
    @assert is_good(d, p)
    # @info "AP ex = $ex"
    return energy(d, p, s, λ)
end

function energy(d::Vector{Int}, p::Vector{Int}, s::Matrix{Float64}, λ::Float64)
    N = length(d)
    return sum(d[i] == 1 ? λ : s[i, p[i]] for i = 1:N)
end

end # module
