module APBP

using Random
using Statistics
using ExtractMacro

mutable struct FMessages
    N::Integer
    # original graph messages
    ψ::Array{Float64, 3}
    ϕ::Array{Float64, 3}
    # auxiliary graph messages
    ψs::Array{Float64, 3}
    ϕs::Array{Float64, 3}
    # exchange messages
    # from/to auxiliary graph and R
    ψ̂::Array{Float64, 2}
    ϕ̂::Array{Float64, 2}
    # from/to original graph and R
    ψ̃::Array{Float64, 2}
    ϕ̃::Array{Float64, 2}
    function FMessages(N)
        # the messages are initialized to zero
        ψ = zeros(Float64, 3, N, N)
        ϕ = zeros(Float64, 3, N, N)
        ψs = zeros(Float64, 3, N, N)
        ϕs = zeros(Float64, 3, N, N)
        ψ̂ = zeros(Float64, 2, N)
        ϕ̂ = zeros(Float64, 2, N)
        ψ̃ = zeros(Float64, 2, N)
        ϕ̃ = zeros(Float64, 2, N)
        return new(N, ψ, ϕ, ψs, ϕs, ψ̂, ϕ̂, ψ̃, ϕ̃)
    end
end

mutable struct Messages
    N::Integer
    # original graph messages
    ψ::Array{Float64, 3}
    ϕ::Array{Float64, 3}
    function Messages(N)
        # the messages are initialized to zero
        ψ = zeros(Float64, 3, N, N)
        ϕ = zeros(Float64, 3, N, N)
        return new(N, ψ, ϕ)
    end
end

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
                γfact::Float64 = 1.0,
                yfact::Float64 = 1.0,
                λ = nothing, # self-similarity
                max_iter::Integer = 1_000, t_stop::Integer = 10)

    seed > 0 && Random.seed!(seed)
    N = size(data, 2)
    s = compute_similarities(data)
    if λ == nothing
        λ = median(s)
    end

    mess = FMessages(N) # mess are initialized to zero
    d = zeros(Int, N)
    p = zeros(Int, N)
    t = 1
    t_stab = 0
    E = 0.0
    while t < max_iter && t_stab < t_stop
        oneFBPstep!(mess, s, λ, γ, y)
        normalize_messages!(mess)
        d, p = assign_variables(mess, s, λ)
        E = 0.0
        if is_good(d, p)
            t_stab += 1
            E = energy(d, p, s, λ)
        else
            t_stab = 0
        end
        println("t = $t \t t_stab = $t_stab \t γ = $γ \t y = $y \t E = $E \t num_exemplars = $(sum(d.==1))")
        t += 1
        γ *= γfact
        y *= yfact
    end

    #
    # exemplars = unique(p[d.==2])
    # the exemplars are data[:,exemplars]
    #

    return is_good(d, p), t, E, d, p
end

function runRBP(data::Matrix{Float64}, ρ::Float64;
                seed::Int = 123,
                ρfact::Float64 = 1.0,
                λ = nothing, # self-similarity
                max_iter::Integer = 1_000, t_stop::Integer = 10)

    seed > 0 && Random.seed!(seed)
    N = size(data, 2)
    s = compute_similarities(data)
    if λ == nothing
        λ = median(s)
    end

    mess = Messages(N) # mess are initialized to zero
    d = zeros(Int, N)
    p = zeros(Int, N)
    t = 1
    t_stab = 0
    E = 0.0
    ψ¹ = zeros(Float64, N)
    ψ² = zeros(Float64, N)
    while t < max_iter && t_stab < t_stop
        oneRBPstep!(mess, ψ¹, ψ², s, λ, ρ)
        #oneRBPstep!(mess, s, λ, ρ)
        normalize_messages!(mess)
        d, p = assign_variables(mess, s, λ)
        E = 0.0
        if is_good(d, p)
            t_stab += 1
            E = energy(d, p, s, λ)
        else
            t_stab = 0
        end
        println("t = $t \t t_stab = $t_stab \t ρ = $ρ \t E = $E \t num_exemplars = $(sum(d.==1))")
        t += 1
        ρ *= ρfact
    end
    return is_good(d, p), t, E, d, p
end


function normalize_messages!(mess::FMessages)
    @extract mess : ψ ϕ ψs ϕs ψ̂ ϕ̂ ψ̃ ϕ̃
    ψ .-= maximum(ψ, dims=1)
    ϕ .-= maximum(ϕ, dims=1)
    ψs .-= maximum(ψs, dims=1)
    ϕs .-= maximum(ϕs, dims=1)
    ψ̃ .-= maximum(ψ̃, dims=1)
    ϕ̃ .-= maximum(ϕ̃, dims=1)
    ψ̂ .-= maximum(ψ̂, dims=1)
    ϕ̂ .-= maximum(ϕ̂, dims=1)
end

function normalize_messages!(mess::Messages)
    @extract mess : ψ ϕ
    ψ .-= maximum(ψ, dims=1)
    ϕ .-= maximum(ϕ, dims=1)
end

function oneFBPstep!(mess::FMessages, s::Array{Float64, 2}, λ::Float64, γ::Float64, y::Float64)
    # perform a single asynchronous step of f-BP update equations
    @extract mess : N ψ ϕ ψs ϕs ψ̂ ϕ̂ ψ̃ ϕ̃

    @inbounds for i in randperm(N)
        # update all the ϕs. In general ϕ(t) = f(ψ(t-1))
        ϕ[1,i,:] = max.(ψ[1,i,:], ψ[2,i,:], ψ[3,i,:])
        ϕ[2,i,:] = ψ[1,i,:]
        ϕ[3,i,:] = max.(ψ[1,i,:], ψ[3,i,:])

        ϕs[1,i,:] = max.(ψs[1,i,:], ψs[2,i,:], ψs[3,i,:])
        ϕs[2,i,:] = ψs[1,i,:]
        ϕs[3,i,:] = max.(ψs[1,i,:], ψs[3,i,:])

        ϕ̃[1,i] = max(ψ̂[1,i] + γ, ψ̂[2,i])
        ϕ̃[2,i] = max(ψ̂[1,i], ψ̂[2,i] + γ)

        ϕ̂[1,i] = max(ψ̃[1,i] + γ, ψ̃[2,i])
        ϕ̂[2,i] = max(ψ̃[1,i], ψ̃[2,i] + γ)

        # update all the ψs.
        ψ̂[1,i] = sum(ϕ[1,:,i]) - λ
        ψ̂[2,i] = sum(ϕ[3,:,i]) + maximum(ϕ[2,:,i] - ϕ[3,:,i] - s[:,i])

        ψ̃[1,i] = (y-1) * ϕ̃[1,i] + sum(ϕs[1,:,i])
        ψ̃[2,i] = (y-1) * ϕ̃[2,i] + sum(ϕs[3,:,i]) + maximum(ϕs[2,:,i] - ϕs[3,:,i])

        # original graph cavity fields
        S1 = sum(ϕ[1,:,i])
        S3 = sum(ϕ[3,:,i])
        @inbounds for j = 1:N
            ψ[1,i,j] = -λ + S1 - ϕ[1,j,i] + ϕ̂[1,i]
            ψ[2,i,j] = -s[i,j] + S3 - ϕ[3,j,i] + ϕ̂[2,i]
            r = filter(x->!(x in [i,j]), 1:N)
            ψ[3,i,j] = S3 - ϕ[3,j,i] + ϕ̂[2,i] + maximum(ϕ[2,r,i] - ϕ[3,r,i] - s[r,i])
        end
        ψ[:,i,i] .= 0.0 # phihat riempie la diagonale anche se ϕ[i,i] = 0

        # auxiliary graph cavity fields
        S1 = sum(ϕs[1,:,i])
        S3 = sum(ϕs[3,:,i])
        @inbounds for j = 1:N
            ψs[1,i,j] = S1 - ϕs[1,j,i] + y * ϕ̃[1,i]
            ψs[2,i,j] = S3 - ϕs[3,j,i] + y * ϕ̃[2,i]
            r = filter(x->!(x in [i,j]), 1:N)
            ψs[3,i,j] = S3 - ϕs[3,j,i] + y * ϕ̃[2,i] + maximum(ϕs[2,r,i] - ϕs[3,r,i])
        end
        ψs[:,i,i] .= 0.0
    end

end

function oneRBPstep!(mess::Messages, ψ¹::Array{Float64,1}, ψ²::Array{Float64,1}, s::Array{Float64, 2}, λ::Float64, ρ::Float64)
    # perform a single asynchronous step of R-BP update equations
    @extract mess : N ψ ϕ

    @inbounds for i in randperm(N)

        S1 = sum(ϕ[1,:,i])
        S3 = sum(ϕ[3,:,i])
        # compute local fields for reinforcement
        new_loc1 = -λ + S1 + ρ * ψ¹[i]
        new_loc2 = -Inf
        ψᴮ = 0.0
        @inbounds for j in 1:N
            j == i && continue
            ψᴮ = -s[i,j] + S3 - ϕ[3,j,i] + ϕ[2,j,i] + ρ * ψ²[i]
            if ψᴮ > new_loc2
                new_loc2 = ψᴮ
            end
        end

        # update all the ϕs. In general ϕ(t) = f(ψ(t-1))
        ϕ[1,i,:] = max.(ψ[1,i,:], ψ[2,i,:], ψ[3,i,:])
        ϕ[2,i,:] = ψ[1,i,:]
        ϕ[3,i,:] = max.(ψ[1,i,:], ψ[3,i,:])

        # update all the ψs.
        S1 = sum(ϕ[1,:,i])
        S3 = sum(ϕ[3,:,i])

        @inbounds for j = 1:N
            ψ[1,i,j] = -λ + S1 - ϕ[1,j,i] + ρ * ψ¹[i]
            ψ[2,i,j] = -s[i,j] + S3 - ϕ[3,j,i] + ρ * ψ²[i]
            r = filter(x->!(x in [i,j]), 1:N)
            ψ[3,i,j] = S3 - ϕ[3,j,i] + maximum(ϕ[2,r,i] - ϕ[3,r,i] - s[r,i]) + ρ * ψ²[i]
        end
        ψ[:,i,i] .= 0.0 # phihat riempie la diagonale anche se ϕ[i,i] = 0

        ψ¹[i] = new_loc1
        ψ²[i] = new_loc2

    end

end

function oneRBPstep!(mess::Messages, s::Array{Float64, 2}, λ::Float64, ρ::Float64)
    # perform a single asynchronous step of R-BP update equations
    @extract mess : N ψ ϕ

    @inbounds for i in randperm(N)

        # update all the ϕs. In general ϕ(t) = f(ψ(t-1))
        ϕ[1,i,:] = max.(ψ[1,i,:], ψ[2,i,:], ψ[3,i,:])
        ϕ[2,i,:] = ψ[1,i,:]
        ϕ[3,i,:] = max.(ψ[1,i,:], ψ[3,i,:])

        # update all the ψs.
        S1 = sum(ϕ[1,:,i])
        S3 = sum(ϕ[3,:,i])

        # compute local fields for reinforcement
        # nothing changes if I update these with ϕ(t) instead of ϕ(t-1)
        ψ¹ = -λ + S1
        ψ² = -Inf
        @inbounds for j in 1:N
            j == i && continue
            ψᴮ = -s[i,j] + S3 - ϕ[3,j,i] + ϕ[2,j,i]
            if ψᴮ > ψ²
                ψ² = ψᴮ
            end
        end

        # NOT SURE ABOUT TIME INDICES
        @inbounds for j = 1:N
            ψ[1,i,j] = -λ + S1 - ϕ[1,j,i] + ρ * ψ¹
            ψ[2,i,j] = -s[i,j] + S3 - ϕ[3,j,i] + ρ * ψ²
            r = filter(x->!(x in [i,j]), 1:N)
            ψ[3,i,j] = S3 - ϕ[3,j,i] + maximum(ϕ[2,r,i] - ϕ[3,r,i] - s[r,i]) + ρ * ψ²
        end
        ψ[:,i,i] .= 0.0 # phihat riempie la diagonale anche se ϕ[i,i] = 0

    end

end

function assign_variables(mess::FMessages, s::Matrix{Float64}, λ::Float64)
    @extract mess : N ψ ϕ ϕ̃

    p = zeros(Int, N) # pointers
    d = zeros(Int, N) # depths
    ψᴬ = 0.0
    ψᴮ = 0.0
    ψᴹ = 0.0

    @inbounds for i = 1:N
        ψᴬ = -λ + sum(ϕ[1,:,i]) + ϕ̃[1,i]
        ψᴹ = -Inf
        jᴹ = -1
        S = sum(ϕ[3,:,i])
        @inbounds for j in 1:N
            j == i && continue
            ψᴮ = -s[i,j] + S - ϕ[3,j,i] + ϕ[2,j,i] + ϕ̃[2,i]
            if ψᴮ > ψᴹ
                ψᴹ = ψᴮ
                jᴹ = j
            end
        end
        if ψᴬ > ψᴹ
            d[i] = 1
            p[i] = 0 # is pointing to the root
        else
            @assert jᴹ ≠ -1
            d[i] = 2
            p[i] = jᴹ
        end
    end
    return d, p
end

function assign_variables(mess::Messages, s::Matrix{Float64}, λ::Float64)
    @extract mess : N ψ ϕ

    p = zeros(Int, N) # pointers
    d = zeros(Int, N) # depths
    ψᴬ = 0.0
    ψᴮ = 0.0
    ψᴹ = 0.0

    @inbounds for i = 1:N
        ψᴬ = -λ + sum(ϕ[1,:,i])
        ψᴹ = -Inf
        jᴹ = -1
        S = sum(ϕ[3,:,i])
        @inbounds for j in 1:N
            j == i && continue
            ψᴮ = -s[i,j] + S - ϕ[3,j,i] + ϕ[2,j,i]
            if ψᴮ > ψᴹ
                ψᴹ = ψᴮ
                jᴹ = j
            end
        end
        if ψᴬ > ψᴹ
            d[i] = 1
            p[i] = 0 # is pointing to the root
        else
            @assert jᴹ ≠ -1
            d[i] = 2
            p[i] = jᴹ
        end
    end
    return d, p
end

function is_good(d::Vector{Int}, p::Vector{Int})
    N = size(d, 1)
    @inbounds for i = 1:N
        p[i] ≠ 0 && d[p[i]] == 2 && return false
    end
    return true
end

function energy(d::Vector{Int}, p::Vector{Int}, s::Matrix{Float64}, λ::Float64)
    N = size(s, 1)
    E = 0
    @inbounds for i = 1:N
        if d[i] == 1
            E += λ
        else
            E += s[i,p[i]]
        end
    end
    return E
end

end # module
