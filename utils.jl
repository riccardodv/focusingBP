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
#import Cairo, Fontconfig
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
        lambda::Float64=0.0)

    points = readdlm(main_folder * folder * filename * ".txt")
    N = size(points)[1]

    if subset_size==0
        subset_size = N
    end

    permutation = Random.randperm(N)[1:subset_size]
    points = points[permutation,:]

    S = .-pairwise(Euclidean(), points, dims=1);
    S ./= -median(S);
    # S = (S .- mean(S)) ./ std(S)
    if lambda==0
        lambda = median(S)
    end
    #   lambda=median(S)
    S += diagm(0 => lambda .* ones(subset_size));

    if occursin("g2", filename)
        true_assignments = readdlm(main_folder * folder * "/partitioning" * filename * "-gt.pa", skipstart=4,)
        true_assignments = convert(Array{Int, 2}, true_assignments)[permutation,1];
        return S, true_assignments, size(points)[2]
    end

    return S, size(points)[2]

end

function add_legend_column_to_df!(df::DataFrame)
    df[:,:Legend] = map((x,y,z) -> format("rAP with γ = {:.2f}, γ_fact = {:.2f}, y = {:.1f}", x ,y, z), df[:γ], df[:γfact], df[:y])
    df[df.Algo .== "AP",:Legend] = repeat(["AP"], sum(df.Algo .== "AP"));
end

function experiment(S::Array{Float64,2},
                    damp_list::Array{Float64,1},
                    γ_list::Array{Float64,1},
                    γfact_list::Array{Float64,1},
                    y_list::Array{Float64,1},
                    repetitions::Int,
                    filename::String,
                    experiment_ID::String)

    N = size(S)[1]

    partitions_flag = occursin("g2", filename)

    dataframe_path = "./results_dataframes_" * experiment_ID * filename * ".csv"


    if isfile(dataframe_path)
        df = CSV.read(dataframe_path)
    else
        if !isdir("./results_dataframes_" * experiment_ID)
            mkdir("./results_dataframes_" * experiment_ID)
        end
        df = DataFrame(Algo = String[], Energy = Float64[], Damp = Float64[], y = Float64[], γ = Float64[], γfact = Float64[],
                D = Int[], N = Int[], Assignments = Float64[])
    end

    for γ in γ_list, γfact in γfact_list, damp in damp_list, y in y_list

        for rep in 1:repetitions
            println("repetition ", rep, "/", repetitions)
            @time sol_AP = affinityprop_mod(S, damp=damp, maxiter=200, run_vanilla=true)
            @time sol_AP_mod = affinityprop_mod(S, damp=damp, maxiter=200, y=y, γ=γ, γfact=γfact, run_vanilla=false)

            if partitions_flag
                temp1 = mean(sol_AP.assignments .== true_assignments)
                temp1 = max.(1 .- temp1, temp1)

                temp2 = mean(sol_AP_mod.assignments .== true_assignments)
                temp2 = max.(1 .- temp2, temp2)
            else
                temp1, temp2 = nothing, nothing
            end

            push!(df, ["AP", sol_AP.energy / N, damp, -1.0, -1.0, -1.0, D, N, temp1])
            push!(df, ["rAP", sol_AP_mod.energy / N, damp, y, γ, γfact, D, N, temp2])

            CSV.write(dataframe_path, df);
        end

    end
end

function experiment_plot(column::Symbol, filename::String,
    df::DataFrame, subset_size::Int; density::Bool=false,
    x_min::Float64=0.5, x_max::Float64=1.0,
    bandwidth::Float64=0.0033, legend::Symbol=:Legend)
    add_legend_column_to_df!(df);

    plot_type = density ? Geom.density(bandwidth=bandwidth) : Geom.histogram(density=true, position=:dodge)
    df = df[(df.N .== subset_size) ,:]

    Gadfly.plot(df, x=column,
        Coord.Cartesian(xmin=x_min,xmax=x_max),
        color=legend, plot_type,
        Guide.xlabel(string(column)), Guide.ylabel("pdf"))
end
