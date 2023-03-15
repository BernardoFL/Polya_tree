#### Polya tree
using DataFrames, Distributions, LinearAlgebra, Plots
import CSV
include("tree_struc.jl")

#Read data and 
data = DataFrame(CSV.File("/Users/bernardo/Desktop/latents.csv"))
select!(data, Not(:Column1))


### Plot a 3d density 
function plot_mvpdf(d)
    l_lim = mean(d) .- 3 .* sqrt.(var(d))
    u_lim = mean(d) .+ 3 .* sqrt.(var(d))
    x1 = l_lim[1]:0.1:u_lim[1]
    x2 = l_lim[2]:0.1:u_lim[2]

    z = [pdf(d, [i, j]) for i in x1, j in x2]
    display(plot!(x1, x2, z, st=:surface,  c=:blues, alpha=0.4))
end



function get_curves(x, sigma)
    [y -> pdf(MultivariateNormal(row, sigma), y) for row in eachrow(x)]
end

function eval_funcs(funcs, x)
    res = zeros(length(funcs))
    for i in 1:length(funcs)
        res[i] = funcs[i](x[i,:])
    end
    return res
end

function inside_areas(attempt_h, attempt_v, curvas, α₀, α₁)
    for i in eachindex(curvas)
       if -α₁ * curvas[i](attempt_h) ≤ attempt_v ≤ α₀ * curvas[i](attempt_h)
        return true
       end
    end
    return false
end

#= function get_probs()
    count(i -> 0<th<1 && -α₁ * curvas[i](attempt_h) ≤ attempt_v ≤ α₀ * curvas[i](attempt_h))
end =#
# The first row is the first coordinate
function get_window(x)
    mins = minimum(x, dims = 1)
    maxes = maximum(x, dims = 1)

    return [[mins[1] maxes[1]] ; [mins[2] maxes[2]]]
end
## x ∈ R^2. One x per time
## @param: 
##  - n: number of samples
##  - x: data point to center at (n x 2 array)
##  - alpha0: dilating parameter for 
##  - alpha1: blah
##  - sigma: variance term for the normal kernel
##  - num: Boolean, sample only the top part or the top and the bottom part
function rarea(n, x, alpha0, alpha1, sigma)
    ##### Sample uniformly in rectangle. Get a Dirichlet process on it and see for each x the mass it puts on the area
    curvas = get_curves(x, sigma)
    mx_wind =  alpha0 * curvas[1](x[1,:])
    min_wind = -alpha1 * curvas[1](x[1,:])

# the problem is in the horizontal coordinates.
    #sample both coordinates
    #base_m = Product([Uniform(0,1), Uniform(minimum(x)-2, maximum(x)+2), Uniform(minimum(x)-2, maximum(x)+2), Uniform(min_wind, max_wind)])

    hor = Array{Float64}(undef, n, 2)
    ver = repeat([0.0],n)

    i = 1
    hw = get_window(x)
    #accept reject
    while i <= n
        attempt_h = [rand(Uniform(hw[1,1]-2*det(sigma), hw[1,2]+2*det(sigma))), rand(Uniform(hw[2,1]-2*det(sigma), hw[2,2]+2*det(sigma)))]
        attempt_v = rand(Uniform(min_wind, mx_wind))
        if inside_areas(attempt_h, attempt_v, curvas, alpha0, alpha1)
                ver[i] = attempt_v
                hor[i,:] = attempt_h
                i += 1
        end
    end
    return DataFrame(th = rand(), h=hor[:,1], v=hor[:,2], z=ver)
end


#@enter rarea(2, [1,1], 1, 1, [1 0; 0 1], true)
## samples Dirichlet process from 
## @param alpha - the alpha parameter of a Dirichlet process 
function polya_urn(n, x, αₘ, alpha0, alpha1, sigma)
    res = DataFrame(th = zeros(n), h=zeros(n), v=zeros(n), z=zeros(n))
    res[1,:] = rarea(1, x, alpha0, alpha1, sigma)[1,:]
    for i in 2:n
        if rand() < αₘ/(αₘ + i) # if the Ber(alpha/alpha+n) == 1
            res[i,:] = res[sample(1:(i-1)),:]
        else
            res[i,:] = rarea(1, x, alpha0, alpha1, sigma)[1,:]
        end
    end
    return res
end


"""
    area_prob(x, j, m, α₀, α₁, α, sigma)

This does these steps:
    1. Get vectors of bivariate normals
    2. Sample 5000 points from the DP on the union of the inside_areas
    3. The output is a vector of size nrow(x), so we initialize it with zeros
    4. Iterating through each clinical trial:
     4.1 Store horizontal coordinates, then count how many of the points fall into a the single area
     4.2 Divide it by the total number of points and store the probability. 

j is the node and m is the level on the tree
"""
function area_prob(curva, points::DataFrame, α₀, α₁, inter::Vector{Float64})
    length(inter) == 2 || throw(ArgumentError("The interval needs to have two elements"))

    num1 = count(p -> -α₁ * curva([p.h, p.v]) ≤ p.z ≤ α₀ * curva([p.h, p.v]) && inter[1] ≤ p.th ≤ inter[2], eachrow(points))
    num2 = count(p -> 0 ≤ p.z ≤ α₀ * curva([p.h, p.v]) && inter[1] ≤ p.th ≤ inter[2], eachrow(points))
    return num1 / (num1 + num2)
end


function area_prob(curva, x, αₘ::Integer, α₀, α₁, inter::Vector{Float64}, sigma)
    length(inter) == 2 || throw(ArgumentError("The interval needs to have two elements"))

    points = polya_urn(5000, select(x, :Lat1, :Lat2), αₘ, α₀, α₁, sigma)
    num1 = count(p -> -α₁ * curva([p.h, p.v]) ≤ p.z ≤ α₀ * curva([p.h, p.v]) && inter[1] ≤ p.th ≤ inter[2], eachrow(points))
    num2 = count(p -> 0 ≤ p.z ≤ α₀ * curva([p.h, p.v]) && inter[1] ≤ p.th ≤ inter[2], eachrow(points))
    return num1 / (num1 + num2)
end
#@enter area_prob([1 1; 2 2], 1, 3, 1,1,1,1)
intento = polya_urn(5, x, 1, 1, 1, [1 0; 0 1]) 

#Note: the partition sequence is indeed fixed, rn centered around the Lebesgue measure
# to do: fix the probs = thing, get matrix of probabilities. Figure out how to trace back the leaves to get the final probabilities.
##use tree_struc.jl to store the trees
function grow_trees(nlevels, x, αₘ::Function, α₀, α₁, sigma)
    #instansiate trees
    curvas = get_curves(x, sigma)
    
    ## Create trees for each thing
    forest = Vector{Tree}(undef, nrow(x))
    for i in 1:length(forest)
        function get_probs(inter, m) 
            area_prob(curvas[i], x, αₘ(m), α₀, α₁, inter, sigma)
        end
        forest[i] = Tree(get_probs, collect(select(x[i,:], :low, :median, :hi)))
    end
    forest = [Tree(inter -> area_prob(curvas[i], points, α₀, α₁, inter), collect(select(x[i,:], :low, :median, :hi))) for i in 1:nrow(x)]
    for i in 2:nlevels
        points = polya_urn(5000, select(x, :Lat1, :Lat2), αₘ, α₀, α₁, sigma) # 1 DP for each level

        #levels are horizontal, trees vertical
        #probs = [area_prob(xi, j, i, α₀, α₁, αₘ, sigma) for j in nodes_in_level, xi in x]
        #probs = hcat(probs...)
            for tree in eachindex(forest)
                nodes_in_level = findall(x -> x.level == i-1, forest[tree])  #find all the nodes to split
                for j in nodes_in_level 
                    inter = forest[tree][j].inter
                    y = area_prob(curvas[tree], points, α₀, α₁, [inter[1], mean(inter)]) # <- add the interval here
                    split_node!(tree, j, y)
            end
        end
    end 
    return forest
end

##### Testing
plot_mvpdf(MvNormal([ 1,1],I))
plot_mvpdf(MvNormal([2, 2], I))
puntos = polya_urn(1000, [1 1; 2 2], 1, 1, 1, [1 0; 0 1])
scatter(puntos.h, puntos.v, puntos.z)


 #Plots.CURRENT_PLOT.nullableplot = nothing
plot((x,y) -> pdf(MultivariateNormal([0,0],I), [x,y]))
# at the end for the summary get the initial partition back. Probably add the partition as an attribute

