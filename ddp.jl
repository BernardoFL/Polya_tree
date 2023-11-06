using Turing, Distributions, Plots, StatsPlots, Random, CategoricalArrays, StatsModels, DataFrames, LinearAlgebra
import CSV
using StatsFuns: logistic
using Turing.RandomMeasures
using RCall



pfsr = DataFrame(CSV.File("/Users/bernardo/Desktop/pfsr.txt", ignorerepeated=true, delim=" ", missingstring="NA"))
data = Matrix(pfsr)
df = data[[1], [1, 2, 4, 5, 6]]
df = vcat(df, data[[1], [1, 3, 7, 8, 9]])

for i in 2:size(data, 1)
    df = vcat(df, data[[i], [1, 2, 4, 5, 6]])
    df = vcat(df, data[[i], [1, 3, 7, 8, 9]])
end

pfsr = DataFrame(df, ["k", "n", "median", "lo", "hi"])

pfsr[!, :bio] = repeat(["+", "-"], size(data, 1))

covariates = DataFrame(CSV.File("/Users/bernardo/Desktop/Z.txt", ignorerepeated=true, delim=" "))
leftjoin!(pfsr, covariates, on=:k)
DataFrames.transform!(pfsr, :phase => CategoricalArray, :agent => CategoricalArray, :tumor => CategoricalArray, :mono => CategoricalArray, :firstline => CategoricalArray; renamecols=false)
DataFrames.transform!(pfsr, :bio => CategoricalArray, renamecols=false)

#y doesn't matter all i want is the dummy variables
dropmissing!(pfsr)
replace!(pfsr.k, Pair.(unique(pfsr.k), collect(1:35))...)
X = ModelFrame(@formula(k ~ 0 + phase + agent + tumor + mono + firstline), pfsr, contrasts=Dict(:x => DummyCoding())) |> modelmatrix
X = hcat(pfsr.k, X)

DataFrames.transform!(pfsr, :k => CategoricalArray)
X = ModelFrame(@formula(mono ~  bio  + tumor + k), pfsr, contrasts=Dict(:x => DummyCoding())) |> modelmatrix
X = hcat(pfsr.n, pfsr.k, X)
CSV.write("/Users/bernardo/Documents/X.csv", pfsr)
##################
@model function anovaddp(n_obs, n_tot, x)
    # Hyper-parameters, i.e. concentration parameter and parameters of H.
    # N is the number of trials, p the number of variables
    k = convert.(Int, x[:,1])
    x = x[:, Not(1)]

    N, p = size(x)

    α = 1.0
    μ0 = zeros(p)
    σ0 = 1.0

    # Define random measure, e.g. Dirichlet process.
    rpm = DirichletProcess(α)

    # Define the base distribution, i.e. expected value of the Dirichlet process.
    H = MvNormal(μ0, σ0 * I)

    # Latent assignment.
    z = zeros(Int, N)

    # Locations of the infinitely many clusters.
    β = zeros(Float64, p)
    v = zeros(N)

    # Random effects
   # w ~ MvNormal(zeros(maximum(k)), I)

    for i in 1:N

        # Number of clusters.
        K = maximum(z)
        nk = Vector{Int}(map(k -> sum(z .== k), 1:K))

        # Draw the latent assignment.
        z[i] ~ ChineseRestaurantProcess(rpm, nk)

        # Create a new cluster?
        if z[i] > K
            β = [β tzeros(Float64, p)]

            # Draw location of new cluster.
            β[:,z[i]] ~ H
        end

        # Draw observation.
        v[i] = logistic(β[:,z[i]] ⋅ x[i, :] + w[k[i]])
        n_obs[i] ~ Binomial(n_tot[i], v[i])
    end
end

m = anovaddp(pfsr.n .÷ 2, pfsr.n, X)
chain = sample(m, Gibbs(), 100)
k = map(
    t -> length(unique(vec(chain[t, MCMCChains.namesingroup(chain, :z), :].value))),
    1:20000,
);
plot(k; xlabel="Iteration", ylabel="Number of clusters", label="Chain 1")