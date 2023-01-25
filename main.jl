#### Polya tree
using DataFrames, Distributions, LinearAlgebra, Plots
import CSV

data = DataFrame(CSV.File("/Users/bernardo/Downloads/osr.csv"))


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

function inside_area(attempt_h, attempt_v, curvas, α₀, α₁)
    for i in eachindex(curvas)
       if -α₁ * curvas[i](attempt_h) ≤ attempt_v ≤ α₀ * curvas[i](attempt_h)
        return true
       end
    end
    return false
end

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
        if inside_area(attempt_h, attempt_v, curvas, alpha0, alpha1)
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
function polya_urn(n, x, α, alpha0, alpha1, sigma)
    res = DataFrame(th = zeros(n), h=zeros(n), v=zeros(n), z=zeros(n))
    res[1,:] = rarea(1, x, alpha0, alpha1, sigma)[1,:]
    for i in 2:n
        if rand() < α/(α + i) # if the Ber(alpha/alpha+n) == 1
            res[i,:] = res[sample(1:(i-1)),:]
        else
            res[i,:] = rarea(1, x, alpha0, alpha1, sigma)[1,:]
        end
    end
    return res
end


function area_prob(A)
    points = rarea(1000, x, alpha0, alpha1, sigma)
    points = 1
end
polya_urn(5, x, 1, 1, 1, [1 0; 0 1]) 


plot_mvpdf(MvNormal([1,1],I))
plot_mvpdf(MvNormal([2, 2], I))
puntos = polya_urn(1000, [1 1; 2 2], 1, 1, 1, [1 0; 0 1])
scatter(puntos.h, puntos.v, puntos.z)


 #Plots.CURRENT_PLOT.nullableplot = nothing
plot((x,y) -> pdf(MultivariateNormal([0,0],I), [x,y]))
function sample_space 