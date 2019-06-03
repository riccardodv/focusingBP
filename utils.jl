using Distances
using NearestNeighbors
using StatsBase

using Printf
using LinearAlgebra
using SparseArrays
using Statistics
using Distances

import Base: show
import StatsBase: IntegerVector, RealVector, RealMatrix, counts

using DataFrames
using Gadfly
using CSV
import Cairo, Fontconfig
using Formatting: format

using DelimitedFiles
using LinearAlgebra
using Printf

using Random
using LaTeXStrings


function list2matrix(similarities::Array{Float64,2}, lambda::Array{Float64,1})
    N = Int(maximum(similarities))
    S = -1e20 .* (ones(N,N) - I)
    [S[Int(similarities[i,1]), Int(similarities[i,2])] = similarities[i,3] for i in 1:size(similarities)[1]]
    S += diagm(0 => ones(N).* lambda)
    S ./= median(lambda)
    return S
end

function eucledian_space2similarities(main_folder::String,
        folder::String, filename::String; subset_size::Int=0,
        lambda::Float64=0.0, partitioning_flag::Bool=false)

    points = readdlm(main_folder * folder * filename * ".txt")
    N = size(points)[1]

    if subset_size==0
        subset_size = N
    end

    permutation = Random.randperm(N)[1:subset_size]
    points = points[permutation,:]

    S = .-pairwise(Euclidean(), points, dims=1);
    S ./= -median(S);
    if lambda==0
        lambda = median(S)
    end

    S += diagm(0 => lambda .* ones(subset_size));

    if partitioning_flag
        true_assignments = readdlm(main_folder * folder * "/partitioning" * filename * "-gt.pa", skipstart=4,)
        true_assignments = convert(Array{Int, 2}, true_assignments)[permutation,1];
        return S, true_assignments, size(points)[2]
    end

    return S, size(points)[2]

end

function add_legend_column_to_df!(df::DataFrame)
    df[:,:Legend] = map((x,y) -> format("rAP with γ = {:.1f}, γ_fact = {:.2f}", x ,y), df[:γ], df[:γfact])
    df[df.Algo .== "AP",:Legend] = repeat(["AP"], sum(df.Algo .== "AP"));
end
