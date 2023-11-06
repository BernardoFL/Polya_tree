#### Polya tree
using DataFrames, Distributions, LinearAlgebra, Plots, ProgressBars
import CSV
include("tree_struc.jl")

#Read data and filter
X = CSV.File("/Users/bernardo/Documents/X.csv") |> DataFrame |> Matrix
X = X[:,Not(1)]


probs = DataFrame(CSV.File("/Users/bernardo/Documents/probs.csv")) |> Matrix

#Note: the partition sequence is indeed fixed, rn centered around the Lebesgue measure
# to do: fix the probs = thing, get matrix of probabilities. Figure out how to trace back the leaves to get the final probabilities.
##use tree_struc.jl to store the trees
function grow_trees(nlevels, splits, probs, c)
    nlevels ≥ 3 || throw(BoundsError("nlevels must be at least 3!"))
    #instansiate trees

    ## Create trees for each thing
    forest = Matrix{Tree}(undef, 200, size(splits,1))
    for i in 1:size(forest, 2)
        for j in 1:size(forest,1)
            forest[j,i] =  Tree(probs[i,:], splits[i,:])
        end
    end
   # forest = [Tree(inter -> area_prob(curvas[i], points, α₀, α₁, inter), collect(select(x[i,:], :low, :median, :hi))) for i in 1:nrow(x)]
   for i in ProgressBar(2:nlevels)
        #levels are horizontal, trees vertical
        #probs = [area_prob(xi, j, i, α₀, α₁, αₘ, sigma) for j in nodes_in_level, xi in x]
        #probs = hcat(probs...)
        for k in 1:200
            for tree in forest[k,:]
                nodes_in_level = findall(x -> x.level == i-1, tree.nodes)  #find all the nodes to split
                for j in nodes_in_level 
                    #y = area_prob(curvas[tree], points, α₀, α₁, [inter[1], mean(inter)]) # <- add the interval here
                    split_node!(tree, j, rand(Beta(c * 2.0^i, c * 2.0^i)), Normal(0, 10))
                end
            end
        end
    end 

    return forest
end

using Distributions
splits = X[:, [4,5,6,6]]
splits[:,end] .+= 3 #add log(3) to the last quantile
splits = cdf.(Normal(0, 10), log.(splits))
bosque = grow_trees(7, splits, probs, 2)

##check intervals
X[1, [3,4,5,2]]
get_quantile(bosque[2, 2], 0.25)
get_quantile(bosque[2, 1], 0.5)
get_quantile(bosque[2, 2], 0.75)
#do tapply(medians, bio, mean)
#X[:,7] is bio
using StatsPlots
import Statistics.median
medians_est = map(x -> get_quantile(x, 0.5), bosque)
medians_est = mapslices(median, medians_est, dims = 1)'
#medians_est = exp.(medians_est)
bio = string.(X[:,1])
bio[bio .== "2.0"] .= "Bio +"
bio[bio .== "1.0"] .= "Bio -"

boxplot(bio, exp.(medians_est), fillalpha=0.75, label ="", outliers=false)
savefig("~/Documents/Code/Polya_tree/medians_boxplots.png")

using KernelDensity
plot(kde(medians_est[bio .== "Bio +"]))
plot!(kde(medians_est[bio.=="Bio -"]))

##Now get the estimated probability that + > -
using StatsBase
# Get the distribution of the difference for each trial
samples_forest = map(k -> begin
        bio_pos = vcat(simulate_from_tree.(50, bosque[:, X[:, 2].==k.&&X[:, 1].==2.0])...)
        bio_neg = vcat(simulate_from_tree.(50, bosque[:, X[:, 2].==k.&&X[:, 1].==1.0])...)
        if length(bio_neg) == 0
            return [NaN, NaN, NaN]
        else
            return quantile(bio_pos .- bio_neg, [0.25, 0.5, 0.75])
        end
    end,
    unique(X[:, 2]))
samples_forest = exp.(hcat(samples_forest...)'[Not(10), :])
CSV.write("/Users/bernardo/Documents/Code/Polya_tree/inter_res.csv", DataFrame(samples_forest, :auto))
samples_forest = map(k -> begin
        bio_pos = vcat(simulate_from_tree.(50, bosque[:, X[:, 2].==k.&&X[:, 1].==2.0])...)
        bio_neg = vcat(simulate_from_tree.(50, bosque[:, X[:, 2].==k.&&X[:, 1].==1.0])...)
        if length(bio_neg) == 0
            return NaN
        else
            return bio_pos .≥ bio_neg
        end
    end,
    unique(X[:, 2]))

  
    #samples_forest[i] = mean(bio_pos .≥ bio_neg)
mean(filter(!isnan, vcat(samples_forest...)) )
plot_tree(bosque[ 7,9],false)
plot(get_bins(bosque[1,1])[Not(end)], median_bosque[:,9], linetype=:steppre, fill = (0,:auto))
plot([exp(n.height) for n in bosque[7, 9].nodes[findall(x -> x.level == 8, bosque[7, 9].nodes)]], linetype=:steppre)
# at the end for the summary get the initial partition back. Probably add the partition as an attribute
## 1. Sacar el promedio por columnas de los arboles.
## 2. Agregar las particiones.
## 3. Graficar todos.
## 4. Cambiar el color por + y -.
## 5. Simular datos para ver si sí jala]

## prior centering
##
X = X[[1,2,5],:]
bosque = grow_trees(9, X[:, [3, 4, 5, 2]], fill(0.5, (3,3)), 1)
import Statistics.median
median_bosque = mapslices(median_tree, bosque, dims=1)
median_bosque = mapslices(x -> x ./ sum(x), median_bosque, dims=1)
cdf_bosque = mapslices(cumsum, median_bosque, dims=1)

plot(get_leaves(bosque[1, 1])[:bins], cdf_bosque[:,1], label ="")
savefig("~/Documents/Code/Polya_tree/prior_1.png")
plot(get_leaves(bosque[1, 2])[:bins], cdf_bosque[:, 2], label="")
savefig("~/Documents/Code/Polya_tree/prior_2.png")
plot(get_leaves(bosque[1, 3])[:bins], cdf_bosque[:, 3], label="")
savefig("~/Documents/Code/Polya_tree/prior_3.png")



