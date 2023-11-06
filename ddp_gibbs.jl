using Distributions, Plots, Random, Statistics, DataFrames, StatsBase, LinearAlgebra, CSV, RCall, BlockDiagonals
using StatsFuns: logistic, logit
using ProgressBars, MCMCDiagnosticTools, MCMCChains
Random.seed!(32)


function stick_breaking(mk, α)
    V = zeros(Float64, length(mk))
    for i in 1:length(mk) 
        V[i] = rand(Beta(1 + mk[i], α + sum(mk[(i+1):end]) ))
    end
    W = V
    for i in 2:length(W)
        W[i] = prod(1 .- V[1:(i-1)]) * V[i]
    end
    return W
end


#N is the number of data points on the father node. Returns unnormalized weights
# since we're clustering studies, i need to change the index as iterating thorugh k
function update_π(y, x, mk, N, β, trial, m, α)
    trial_u = unique(trial)
    pis = zeros(length(trial_u), m)
    sb_weights =  stick_breaking(mk, α)

    for i ∈ trial_u #for each trial
        # find which cohorts belong to each trial
        ix = findall(trial_i -> trial_i == i, trial)
        for k in 1:m #for each cluster
            pis[i, k] = sb_weights[k] * prod([pdf(Binomial(N[j], logistic( x[j,:] ⋅ β[:,k])), y[j]) for j in ix])
        end
    end
    return pis
end

function rpg(n, h, z)
    R"library(BayesLogit)"
    return reval("BayesLogit::rpg($(n), h=$(h), z=$(z))") |> rcopy
end

# beta is the current matrix of coeffs together with the vector of psis. Trial is the indicator for trial
function update_omegas(β, x, sn, n, trial)
    ω = zeros(n)
    for k in 1:n
        ω[k] = rpg(1, 1, x[k,:] ⋅ β[:,sn[trial[k]]] )
    end
    return ω
end

function _doubles_vector(x)
    return collect.([(xi, xi) for xi in x]) |> vcat
end

#= function get_pg_params(y, )
    μ_0 = repeat([μ_0], new_p + length(ψ))
    Σ_0 = Diagonal(vcat(repeat([σ_0], new_p), ψ)) # this should correct for p*m
    Ω = Diagonal(Ω)
    κ = y[sn] .- N[sn] ./ 2

    #el pedo esta aqui (7 de agosto). La sigma_0 tiene el tamaño incorrecto
    V = Matrix(Hermitian(inv(new_X' * Ω * new_X + Σ_0)))
end
 =#
#Note: ψ must be the duplicated version
function update_betas(y, X, sn, N, Ω, μ_0, σ_0, ψ, m, p_r, trial)
    p = size(X, 2) - p_r - size(X,1)#remove the offsets
    #μ_0 = repeat([μ_0], p-p_r)
    # for the offset we can use a Gaussian with a tiny tiny variance
    Σ_0 = Diagonal(vcat(repeat([0.00001], size(X, 1)), repeat([σ_0], p ), repeat([ψ], p_r))) |> Matrix# this should correct for p*m
    new_betas = zeros(size(X, 2), m)
    for k in 1:m
        ki = findall(i -> i == k, sn) #which trials belong to said cluster
        if isempty(ki) #if no observations in the cluster just sample from the prior
            new_betas[:,k] = rand(MvNormal(μ_0, Σ_0))
            continue 
        end
        
        #get which cohorts to keep
        coh = trial .∈ [ki]
        Ωk = Diagonal(Ω[coh])
        κ = y[coh] .- N[coh] ./ 2 #N is the vector on X.csv, so already has the appropiate repetiions
        V = Matrix(Hermitian(inv(X[coh,:]' * Ωk * X[coh,:] + Σ_0)))

        new_betas[:,k] = rand(MvNormal(V * (X[coh,:]' * κ + Σ_0 * μ_0), V)) ##el problema está aquí en alguna part
        if any(isnan.(new_betas[:,k])) throw(DomainError(new_betas[:,k], ":(")) end
    end

return new_betas

#=     p = size(x,2) - p_r

    
    new_X = _get_upd_matrix(x, sn, m, p_r) #add the contrastes
#this is okay

   # new_X = cat(new_X, x[:,(end-p_r+1):end], dims=2)
    new_betas = zeros(size(new_X,2))

    new_p = size(new_X,2) - p_r


   # μ_0 = repeat([μ_0], size(new_X,2))
    Σ_0 = Diagonal(vcat(repeat([σ_0], new_p), repeat([ψ], p_r))) |> Matrix# this should correct for p*m

    # i need to permute all of the vectors
    reff_ix = vcat([findall(sn .== mi) for mi in 1:m]...) #vector of indices per cluster
    Ω = Diagonal(Ω[reff_ix])
    κ = y[reff_ix] .- N[reff_ix] ./ 2 #N is the vector on X.csv, so already has the appropiate repetiions


    V = Matrix(Hermitian(inv(new_X' * Ω * new_X + Σ_0)))

#here i assume i center it at zero, change later
    new_betas = rand(MvNormal(V * (new_X' * κ + Σ_0 * zeros(size(Σ_0,1))), V)) ##el problema está aquí en alguna parte

    β = reshape(new_betas[1:(end-p_r)], (p,m))

    γ = new_betas[(end-p_r+1):end]

    return vcat(β, repeat(γ, 1, size(β,2))) =#
end

#= 
function gauss_conditional(i, μ, Σ, γ) #i is the end of the first block
    Σ11 = Σ[1:i, 1:i]
    Σ12 = Σ[1:i, (i+1):end]
    Σ21 = Σ[(i+1):end, 1:i]
    Σ22 = Σ[(i+1):end, (i+1):end]

    μ1 = μ[1:i]
    μ2 = μ[(i+1):end]

    μ_cond = μ1 + Σ12 * inv(Σ22) * (γ - μ2)
    Σ_cond = Σ11 - Σ12 * inv(Σ22) * Σ21

    return Dict(:β|γ => MvNormal(μ_cond, Σ_cond), :γ => MvNormal(μ2, Σ22) )
end =#

function update_sn(πi, m)
    mapslices(x -> sample(1:m, Weights(x)), πi, dims=2)
end

function get_γ(β, sn, p)
    γ = zeros(length(sn))
    for m in 1:size(β,2)
        #it's p+1 because the first ramdom effect is the referewnce category
        γ[sn .== m] = β[Not(1:p), m][sn .== m] 
    end
    return γ
end

function update_ψ(γ, a, b)
    return rand(InverseGamma(a + length(γ)/2, b + 0.5*γ'*γ)) 
end


#This gets the sparse matrix needed for the grouping of the betas. 
#For the random effects to work they need to be repeated in each vector of betas.
function _get_upd_matrix(x, sn, m, p_r)
    # create a tensor storing the repeated fixed effects
    x_new = [x[:, 1:(end-p_r)] for i in 1:m] #array of repeaed xs. 
    #now delete from each block the rows that don't belong to each cluster.
    for mi in 1:m
        #sn .== mi is the indices for cluster mi
        x_new[mi] = x_new[mi][sn .== mi, :] #remove the rows not in cluster
    end
    #construct the block diagonal matrix
    x_new = BlockDiagonal(x_new)

    #Now attach the random effects at the end
    reff_ix = vcat([findall(sn .== mi) for mi in 1:m]...) #vector of indices per cluster
    #the second arg is the random effects matrix with rows permuted to match the reordering done on the blocks
    return hcat(x_new, x[reff_ix, (end-p_r+1):end])
end


#= function plot_mixture(result, y, titulo)
    histogram(y, normalize = true, bins = 22)
    x = collect(5:0.1:38)
    plot!(x, result[2, :], linewidth = 3, title = titulo, legend = false, fillalpha = 0.3, ribbon = (result[1, :], result[3, :]))
end =#
#p_r number of random effects
#note N is a vector :(). Trial is the indicator for each trial
#μ_0 is a vector
function gibbs_sampler(niter, y, x, p_r, m, N, μ_0, σ_0, α, a,b, trial)
    n = length(y)
    p = size(x,2)

    ψ = ones(niter) #hypervariance for random neffects
    pis =fill(1 / m, length(unique(trial)), m, niter)
    sn = fill(1, (length(unique(trial)), niter))
    sn[:, 1] = update_sn(pis[:, :, 1], m)
    mk = mapslices(x -> x |> values |> collect, pis[:, :, 1], dims=2)

    # p_r = 34 #how many random effects thereare
    β = zeros(p, m, niter)
    Ω = zeros(n, niter)


    for i in ProgressBar(2:niter)

        pis[:, :, i] = update_π(y, x, mk, N, β[:,:,i-1], trial, m, α)
        sn[:, i] = update_sn(pis[:,:,i], m)
        mk = mapslices(x -> x |> values |> collect, pis[:, :, i], dims=2)

        Ω[:, i] = update_omegas(β[:,:,i-1], x, sn[:,i], n, trial)
       
    
        β[:, :, i] = update_betas(y, x, sn[:, i], N, Ω[:, i], μ_0, σ_0, ψ[i-1], m, p_r, trial)
        #34 is the number of random effects
        
        ψ[i] = update_ψ(get_γ(β[:, :, i], sn[Not(1), i], p - p_r), a, b)
    end
    return Dict("β" => β, "ψ" => ψ,"sn" => sn)
end
#note: las full conditional de las psis solo dependen de la prior
X = CSV.File("/Users/bernardo/Documents/X.csv") |> DataFrame |> Matrix
X = X[:,Not([1,2])]
bio = X[:,1]

X[iszero.(X[:, 3]), 3] .= 0.01


# #niter, y, x, trial, m, N, μ_0, σ_0, α, a, b
# we need a different hyperparameter mean for each observation
μ_0 = vcat(logit.( cdf.(Normal.(0,5), log.(X[:,4]) )),  zeros(size(X,2)-size(X,1)-5))
ch_med = gibbs_sampler(50000, X[:,2] .÷ 2, X[:,Not(1,2,3,4,5)], 27, 5, X[:,2], μ_0, 1*0.5, 1, 1,1, Int.(X[:,1]))
ch_med = get_diagnostics_obj(ch_med, 28)
 scatter(DataFrame(rhat(ch_med; sections=:parameters)).rhat, label="", title = "r̂ for Median")
savefig("~/Documents/Code/Polya_tree/r_hat_med.png")

# now for the left quartile
#cheating

μ_0 = vcat(logit.( cdf.(Normal.(0,5), log.(X[:,3]))),  zeros(size(X,2)-size(X,1)-5)) 
ch_q25 = gibbs_sampler(50000, X[:, 2] .÷ 4, X[:, Not(1, 2, 3, 4, 5)], 27, 5, X[:, 2] .÷ 2, μ_0, 1 * 2/9, 1, 1, 1, Int.(X[:, 1]))
  
ch_q25 = get_diagnostics_obj(ch_q25)
scatter(DataFrame(rhat(ch_q25; sections=:parameters)).rhat, label="", title="r̂ for q25 and q75")
savefig("~/Documents/Code/Polya_tree/r_hat_q25.png")

μ_0 = vcat(logit.(cdf.(Normal.(0, 5), log.(X[:, 5]) .- log.(X[:, 4]))), zeros(size(X, 2) - size(X, 1) - 5))
ch_q75 = gibbs_sampler(50000, X[:, 2] .÷ 4, X[:, Not(1, 2, 3, 4, 5)], 27, 5, X[:, 2] .÷ 2, μ_0, 1 * 2 / 9, 1, 1, 1, Int.(X[:, 1]))


function get_diagnostics_obj(chains, p_r)
    p, m, n = size(chains["β"])

    γs = chains["β"][(end-p_r):end, chains["sn"], :] #they're the same accross m
    βs = chains["β"][1:(end-p_r), :, :]

    γs = permutedims(γs, (2, 1))
    βs = reshape(permutedims(βs, (3,1,2)), n, (p-p_r)*m)
    
    nombres = Symbol.(vcat(["β" * ".$i" for i in 1:size(βs, 2)], ["γ" * ".$i" for i in 1:size(γs,2)],
        ["ψ" ], ["sn" * ".$i" for i in 1:size(chains["sn"], 1)]))

    chs = Chains(cat(βs, γs, chains["ψ"], chains["sn"]', dims=2), nombres; start=90000)
    return set_section(chs, Dict(:sn => Symbol.(["sn" * ".$i" for i in 1:size(chains["sn"], 1)])))
end

#This is without the objs, get the separate matrices 
function estimate_probs(chains, X, trial, burnin)
    #for each row
    probs = zeros(size(X,1))
    for i in 1:size(X,1)
        #for each iteration of the chain
        probs_i = zeros(length(burnin:size(chains["β"], 3)))
        for j in 1:length(probs_i)
            #β = (p,m,niter), sn=(n,niter)
            #Get the betas from the cluster that at each iteration the obs belongs to
            probs_i[j] = chains["β"][:,chains["sn"][trial[i],j],j] ⋅ X[i,:]
        end
        probs[i] =  logistic(mean(probs_i))
    end
    return probs
end


mean_probs = hcat(estimate_probs(ch_q25, X[:, Not(1, 2, 3, 4, 5)], Int.(X[:, 1]), 49000),
    estimate_probs(ch_med, X[:, Not(1, 2, 3, 4, 5)], Int.(X[:, 1]), 49000),
    estimate_probs(ch_q75, X[:, Not(1, 2, 3, 4, 5)], Int.(X[:, 1]), 49000))

CSV.write("/Users/bernardo/Documents/probs.csv", DataFrame(mean_probs, :auto))   