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
using Clustering
using Main.Iterators: flatten

# NOTE: needs to include one of these before this file
# include("replicated_AP__ci_cistar_interaction.jl")
# include("replicated_AP__s_interaction.jl")

"""
Utilities and support functions for Replicated Affinity Propagation.

 * list2matrix:  takes a list of pairwise similarities and returns a matrix;

 * eucledian_space2similarities: reads txt file with dataset points coordinates
    and returns similarity matrix;

 * df2data: convert datapoints dataframe into data + label vectors;

 * add_legend_column_to_df!: takes results_dataframe as input and adds the legend
    column for rAP parameters (γ, γfact and y);

 * compute_two_clusters_assignments: maps predictions and labels to compute accuracy
    for 2-clusters datasets;

 * experiments: run grid search for AP and rAP for "repetitions" times with
    different initializations. Also a run with zero initializations is done
    for each algorithm. Results are stored in a dataframe located in

        dataframe_path = "./results_dataframes_" * experiment_ID * "/" * filename * "_" * string(subset_size) * ".csv"

        e.g. ./results_dataframes_0000/g2-2-40_1024.csv

    DataFrame structure:

    │ Row │ Algo    │ Energy   │ Damp     │ y        │ γ        │ γfact    │ D      │ N      │ Accuracy │ Exemplars │ Assignments │ Zero_init │
    │     │ String  │ Float64  │ Float64  │ Float64  │ Float64  │ Float64  │ Int64  │ Int64  │ Float64  │ String    │   String    │ Bool      │
    ├─────┼─────────┼──────────┼──────────┼──────────┼──────────┼──────────┼────────┼────────┼──────────┼───────────┼─────────────┼───────────┤
    │ 1   │ rAP     │ 0.948999 │ 0.8      │ 5.0      │ 0.1      │ 1.05     │ 2      │ 1024   │ 0.499702 │ [302, 629]│ [1,...,1]   │ true      │

 * Experiment plot: takes DataFrame and plots one density curve (or histogram)
    for each value in the column specified in legend::Symbol and saves it in pdf;

 * dataset_scatter_hits: takes the data-points dataframe and the exemplars list
    and returns a 2D plot where each point size and colours depends on its
    frequency in being an exemplar;

 * find_knn: find the KNNs of a given exemplar;

 * parse_numbers: to parse a string into and array - it is employed when reading
    a dataframe from csv in which arrays are wrongly saved as strings;

 * split_df_by_algo: is used after grid search to choose a set of parameters for
    rAP and split the dataframe with all exepriments into AP and rAP dataframe
    in order to perform further analysis;

 * boxplot: draw boxplots for energy, energy profile and accuracy around
    the given exemplars.
"""


triumedian(S::Matrix) = median(S[i,j] for i = 1:size(S,1) for j = (i+1):size(S,2))

function list2matrix(similarities::Matrix{Float64}, λ::Vector{Float64})
    N = max(Int(maximum(similarities[:,1:2])), length(λ))
    S = fill(-1e20, (N,N))
    for k in 1:size(similarities, 1)
        i, j, s = similarities[k, :]
        S[Int(i), Int(j)] = s
    end
    S[diagind(S)] = λ
    S ./= triumedian(S)
    return S
end

function coordinates2similarities(points_coordinates::Matrix{Float64}; λ::Union{Float64,Nothing} = nothing)
    N = size(points_coordinates, 1)
    S = .-pairwise(Euclidean(), points_coordinates, dims=1)
    S ./= -triumedian(S);
    # S = (S .- mean(S)) ./ std(S)
    λ ≡ nothing && (λ = triumedian(S))
    S[diagind(S)] .= λ
    return S
end

function load_euclidean_data(main_dir::String, dir::String, filename::String;
                             subset_size::Union{Int,Nothing} = nothing,
                             λ::Union{Float64,Nothing} = nothing)

    points = readdlm(joinpath(main_dir, dir, filename * ".txt"))
    N = size(points, 1)

    subset_size ≡ nothing && (subset_size = N)

    permutation = randperm(N)[1:subset_size]
    points = points[permutation,:]

    S = coordinates2similarities(points, λ=λ)

    if occursin("g2", filename)
        true_assignments = readdlm(joinpath(main_dir, dir, "partitioning", filename * "-gt.pa"), skipstart=4)
        true_assignments = convert(Matrix{Int}, true_assignments)[permutation,1]
        return S, true_assignments, size(points, 2), points
    end
    if occursin("s4", filename)
        true_assignments = readdlm(joinpath(main_dir, dir, "partitioning", filename * "-label.pa"), skipstart=5)
        true_assignments = convert(Matrix{Int}, true_assignments)[permutation,1]
        return S, true_assignments, size(points, 2), points
    end

    return S, size(points, 2), points
end

function df2data(points_df::DataFrame)
    data = zeros(length(points_df.x), 2)
    data[:,1] = points_df.x
    data[:,2] = points_df.y

    true_assignments = points_df.class
    true_assignments = convert(Vector{Int64}, true_assignments)

    return data, true_assignments
end

function add_legend_column_to_df!(df::DataFrame)
    df[!,:Legend] = map((x,y,z) -> format("rAP with γ = {:.2f}, γ_fact = {:.2f}, y = {:.1f}", x ,y, z), df[!,:γ], df[!,:γfact], df[!,:y])
    df[df.Algo .== "AP",:Legend] = repeat(["AP"], sum(df.Algo .== "AP"));
end

function compute_two_clusters_assignments(predicted::Vector{Int64}, real::Vector{Int64})
    assignments = mean(predicted .== real)
    assignments = max.(1 .- assignments, assignments)
    return assignments
end

function experiment(S::Matrix{Float64},
                    true_assignments::Vector{Int64},
                    damp_list::Vector{Float64},
                    γ_list::Vector{Float64},
                    γfact_list::Vector{Float64},
                    y_list::Vector{Float64},
                    repetitions::Int,
                    filename::String,
                    experiment_ID::String)

    N = size(S, 1)
    @assert size(S, 2) == N

    g2_flag = occursin("g2", filename)
    cities_flag = occursin("citi", filename)

    ##### ATTENTION #####
    # g2_flag = false

    # display = :iter
    display = :final

    dn = "results_dataframes_$(experiment_ID)"
    fn = "$(filename)_$(N).csv"
    dataframe_path = joinpath(dn, fn)


    if isfile(dataframe_path)
        df = CSV.read(dataframe_path)
        df.Assignments = parse_numbers.(df.Assignments)
        df.Exemplars = parse_numbers.(df.Exemplars)
    else
        mkpath(dn)
        df = DataFrame(Algo = String[], Energy = Float64[], Damp = Float64[],
                       y = Float64[], γ = Float64[], γfact = Float64[],
                       N = Int[], Accuracy = Float64[], Exemplars = Vector{Int64}[],
                       Assignments = Vector{Int64}[], Zero_init = Bool[])
    end

    println("plain AP, zero init")
    @time sol_AP = AP.affinitypropR(S, damp=0.8, maxiter=200, run_vanilla=true, zero_init=true, display=:iter)

    if cities_flag
        accuracy_AP = -1.0
    elseif g2_flag
        accuracy_AP = compute_two_clusters_assignments(sol_AP.assignments, true_assignments)
    else
        accuracy_AP = randindex(sol_AP.assignments, true_assignments)[2]
    end
    push!(df, ("AP", sol_AP.energy / N, 0.8, -2.0, -2.0, -2.0, N, accuracy_AP, sol_AP.exemplars, sol_AP.assignments, true))
    CSV.write(dataframe_path, df)

    for γ in γ_list, γfact in γfact_list, damp in damp_list, y in y_list

        println("repl  AP, zero init, γ=$γ γf=$γfact damp=$damp y=$y")
        @time sol_rAP = AP.affinitypropR(S, damp=damp, maxiter=200, y=y, γ=γ, γfact=γfact, run_vanilla=false, zero_init=true, display=:iter)
        if cities_flag
            accuracy_rAP = -1.0
        elseif g2_flag
            accuracy_rAP = compute_two_clusters_assignments(sol_rAP.assignments, true_assignments)
        else
            accuracy_rAP = randindex(sol_rAP.assignments, true_assignments)[2]
        end
        push!(df, ["rAP", sol_rAP.energy / N, damp, y, γ, γfact, N, accuracy_rAP, sol_rAP.exemplars, sol_rAP.assignments, true])
        CSV.write(dataframe_path, df);

        for rep in 1:repetitions
            # println("repetition ", rep, "/", repetitions)
            println("plain AP, rand init [$rep/$repetitions]")
            @time sol_AP = AP.affinitypropR(S, damp=damp, maxiter=200, run_vanilla=true)
            println("repl  AP, rand init, γ=$γ γf=$γfact damp=$damp y=$y [$rep/$repetitions]")
            @time sol_rAP = AP.affinitypropR(S, damp=damp, maxiter=200, y=y, γ=γ, γfact=γfact, run_vanilla=false)

            if cities_flag
                accuracy_AP = -1.0
                accuracy_rAP = -1.0
            elseif g2_flag
                accuracy_AP = compute_two_clusters_assignments(sol_AP.assignments, true_assignments)
                accuracy_rAP = compute_two_clusters_assignments(sol_rAP.assignments, true_assignments)
            else
                accuracy_AP = randindex(sol_AP.assignments, true_assignments)[2]
                accuracy_rAP = randindex(sol_rAP.assignments, true_assignments)[2]
            end

            push!(df, ["AP", sol_AP.energy / N, damp, -1.0, -1.0, -1.0, N, accuracy_AP, sol_AP.exemplars, sol_AP.assignments, false])
            push!(df, ["rAP", sol_rAP.energy / N, damp, y, γ, γfact, N, accuracy_rAP, sol_rAP.exemplars, sol_rAP.assignments, false])

            CSV.write(dataframe_path, df);
        end
    end
    println("done")
end

function experiment_plot(column::Symbol, filename::String, df::DataFrame, subset_size::Int;
                         density::Bool = false,
                         x_min::Float64 = 0.5,
                         x_max::Float64 = 1.0,
                         bandwidth::Float64 = 0.0033,
                         legend::Symbol = :Legend,
                         plot_name::String = "_",
                         plot2pdf::Bool = false)

    add_legend_column_to_df!(df)

    plot_type = density ? Geom.density(bandwidth=bandwidth) :
                          Geom.histogram(density=true, position=:dodge, bincount=20)
    df = df[(df.N .== subset_size),:]

    xintercept = df[(df.Zero_init .== true),:][!,column]

    plot_ = Gadfly.plot(df,
                        x = column, xintercept = xintercept,
                        Coord.Cartesian(xmin = x_min, xmax = x_max),
                        Scale.color_discrete_manual("royalblue", "tomato"),
                        color = legend, plot_type,
                        Geom.vline(color = ["royalblue", "tomato"],
                                   size = 0.5mm, style = :dash),
                        Guide.colorkey(title = ""),
                        Guide.xlabel(string(column)),
                        Guide.ylabel("pdf"),
                        style(key_position = :top),
                       )

    if plot2pdf
        draw(PDF(joinpath("figures", plot_name * ".pdf"), 600px, 300px), plot_)
    else
        return plot_
    end
end

function dataset_scatter_hits(points_df::DataFrame, exemplars::Vector{Vector{Int64}}, filename::String, algo::String)
    N = length(points_df.x)

    cmap = algo == "AP" ?
           Scale.lab_gradient("skyblue", "darkblue", "darkblue", "darkblue") :
           Scale.lab_gradient("lightsalmon", "brown")
    color_scale = Scale.color_continuous(colormap = cmap)

    exemplars = collect(flatten(exemplars))
    h = fit(Histogram, exemplars, range(-0.5, N+0.5, length = N+1))

    n_ex = length(exemplars)
    increase = 60 / n_ex  # 50 3

    points_df.Hits = h.weights

    P = sortperm(points_df.Hits)
    points_df = points_df[P,:]

    dataset_scatter_plot = plot(points_df, x = :x, y = :y,
                                color = h.weights[P] ./ sum(h.weights) .* 100,
                                size = h.weights[P] .* increase .+ 0.9,
                                Geom.point, color_scale, # +0.9 0.033
                                Theme(highlight_width = 0pt),
                                # Coord.cartesian(fixed=true, xmin=-7.5, xmax=7.5, ymin=-7.5, ymax=7.5,),
                                Guide.colorkey(title = "Hits"),
                                Coord.cartesian(fixed = true, xmin = 400, xmax = 700, ymin = 400, ymax = 700),
                                Guide.xlabel(""), Guide.ylabel(""),
                                # Guide.title("hits per exemplar"))
                               )

    draw(PDF(joinpath("figures", filename * ".pdf"), 400px, 400px), dataset_scatter_plot)
    return dataset_scatter_plot
end

function find_knn(data::Matrix{Float64}, exemplar::Int64; k::Int64 = 5)
    point = data[exemplar,:]
    kdtree = KDTree(convert(Matrix{Float64}, transpose(data)))
    idxs, dists = knn(kdtree, point, k, true)

    return idxs, dists
end

function parse_numbers(s)
    s = s[2:end-1]
    pieces = split(s, ',', keepempty = false)
    return map(p->parse(Int64, p), pieces)
end

function split_df_by_algo(df::DataFrame, γ::Float64, γfact::Float64, y::Float64)

    df_AP = df[df.y .< 0, :];
    df_AP.Assignments = parse_numbers.(df_AP.Assignments)
    df_AP.Exemplars = parse_numbers.(df_AP.Exemplars)

    df_rAP = df[(df.γ .== γ) .& (df.γfact .== γfact) .& (df.y .== y) , :]
    df_rAP.Assignments = parse_numbers.(df_rAP.Assignments)
    df_rAP.Exemplars = parse_numbers.(df_rAP.Exemplars)

    return df_AP, df_rAP
end

function boxplots(data::Matrix{Float64}, S::Matrix{Float64},
                  exemplars_AP::Vector{Vector{Int64}},
                  exemplars_rAP::Vector{Vector{Int64}},
                  filename::String;
                  k_neigs::Int64 = 10)

    num_points = size(data, 2)
    cumulate_energy = zeros(2, k_neigs)

    df_neigh = DataFrame(Algo=String[], Neigh_Num=Int64[], Energy=Float64[], Energy_profile=Float64[], Accuracy=Float64[])

    for j in findall((length.(exemplars_AP) .== 2))
        exemplars_cluster_1, _ = find_knn(data, exemplars_AP[j][1]; k = k_neigs)
        exemplars_cluster_2, _ = find_knn(data, exemplars_AP[j][2]; k = k_neigs)
        exemplars = hcat(exemplars_cluster_1, exemplars_cluster_2)

        i = 1
        baseline_assignments = assignments, _ = AP._afp_get_assignments(S, exemplars[i,:])
        baseline_energy = AP._afp_compute_energy(S, exemplars[i,:], assignments) / num_points

        for i in 1:k_neigs
            assignments, _ = AP._afp_get_assignments(S, exemplars[i,:])
            energy = AP._afp_compute_energy(S, exemplars[i,:], assignments) / num_points
            accuracy = compute_two_clusters_assignments(assignments, class)
            # accuracy = randindex(assignments, class,)[2]
            push!(df_neigh, ["AP", i, energy, energy - baseline_energy, accuracy])
        end
    end

    for j in findall((length.(exemplars_rAP) .== 2))
        exemplars_cluster_1, _ = find_knn(data, exemplars_rAP[j][1]; k = k_neigs)
        exemplars_cluster_2, _ = find_knn(data, exemplars_rAP[j][2]; k = k_neigs)
        exemplars = hcat(exemplars_cluster_1, exemplars_cluster_2)

        i = 1
        baseline_assignments = assignments, _ = AP._afp_get_assignments(S, exemplars[i,:])
        baseline_energy = AP._afp_compute_energy(S, exemplars[i,:], assignments) / num_points

        for i in 1:k_neigs
            assignments, _ = AP._afp_get_assignments(S, exemplars[i,:])
            energy = AP._afp_compute_energy(S, exemplars[i,:], assignments) / num_points
            accuracy = compute_two_clusters_assignments(assignments, class)
            # accuracy = randindex(assignments, class,)[2]
            push!(df_neigh, ["rAP", i, energy, energy - baseline_energy, accuracy])
        end
    end
    df_neigh.strNN = string.(df_neigh.Neigh_Num .- 1)

    plot_ = plot(df_neigh, x = :strNN, y = :Energy,
                 Geom.boxplot(suppress_outliers = true),
                 color = :Algo,
                 Scale.color_discrete_manual("royalblue", "tomato"),
                 Theme(boxplot_spacing = 0.35cm),
                 # Coord.Cartesian(ymin = -0.025, ymax = 0.025),
                 # Coord.Cartesian(ymin = 0.86, ymax = 0.95),
                 Guide.xlabel("k-nearest neighbor"), Guide.ylabel("Energy"),
                 Guide.title("Energy of exemplars knn")
                 # Guide.title("Energy difference of exemplars knn to original exemplar - BOXPLOT")
                )
    draw(PDF(joinpath("figures" * filename * "_neighbors_energy_boxplot.pdf"), 800px, 400px), plot_)

    plot_ = plot(df_neigh, x = :strNN, y = :Accuracy,
                 Geom.boxplot(suppress_outliers = true),
                 color = :Algo,
                 Theme(boxplot_spacing = 0.35cm),
                 Scale.color_discrete_manual("royalblue", "tomato"),
                 # Coord.Cartesian(ymin = 0.88, ymax = 0.95),
                 Guide.xlabel("k-nearest neighbor"), Guide.ylabel("Accuracy"),
                 Guide.title("Accuracy of exemplars knn")
                )
    draw(PDF(joinpath("figures", filename * "_neighbors_accuracy_boxplot.pdf"), 800px, 400px), plot_)

    plot_ = plot(df_neigh, x = :strNN, y = :Energy_profile,
                 Geom.boxplot(suppress_outliers = true),
                 color = :Algo,
                 Scale.color_discrete_manual("royalblue", "tomato"),
                 Theme(boxplot_spacing = 0.35cm),
                 # Coord.Cartesian(ymin = -0.025, ymax = 0.025),
                 Guide.xlabel("k-nearest neighbor"), Guide.ylabel("Energy profile around exemplars"),
                 Guide.title("Energy difference of exemplars knn to original exemplar")
                )
    draw(PDF(joinpath("figures", filename * "_neighbors_energy_profile_boxplot.pdf"), 800px, 400px), plot_)
end
