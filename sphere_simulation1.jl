using Distributed
addprocs(2)
@everywhere using Distributions
@everywhere using GLM
@everywhere using StatsBase
#@everywhere using DataFrames
@everywhere using Statistics
@everywhere using Manifolds
@everywhere using LinearAlgebra
@everywhere using CSV
@everywhere using DelimitedFiles
#@everywhere using OSQP
#@everywhere using SparseArrays
@everywhere using MLBase
@everywhere using Optim

###Utility functions
@everywhere function l2norm(x)
  sqrt(sum(x.^2))
end

@everywhere function SpheGeoDist(y1,y2) 
  y1 = y1 ./ l2norm(y1)
  y2 = y2 ./ l2norm(y2)
  if sum(y1 .* y2) > 1
    return(0)
  elseif (sum(y1.*y2) < -1)
    return(pi)
  else 
    return(acos(sum(y1 .* y2)))
  end
end

@everywhere function SpheGeoGrad(y1,y2) 
  tmp = 1 - sum(y1 .* y2)^ 2
  return(-(tmp)^(-0.5) .* y1)
end

@everywhere function SpheGeoHess(y1,y2) #,tol = 1e-10){
  return(- sum(y1 .* y2) * 
           (1 - sum(y1 .* y2) ^ 2) ^ (-1.5) .* (y1 * y2') ) ##'
end

###kernels for the local Fr reg
@everywhere function ker_gauss(x)
  exp(-x^2 / 2) / sqrt(2*pi)
end

@everywhere function ker_unif(x)
  Int64((x<=1) & (x>=-1))
end

@everywhere function ker_epan(x, n=1)
  (2*n+1) / (4*n) * (1-x^(2*n)) * (abs(x)<=1)
end

#### estimation performance calculator for estimates of the parameter, which are unit vectors
@everywhere function bias_calc2(esti, true_beta,reps)
  d = length(true_beta)
  extrinsic_mean = mean([esti[i] for i in 1:length(reps)])
  if sum([x==0 for x in extrinsic_mean])==d
    println("Too few points")
    return hcat([200, 200, 200])
  else
    S = Manifolds.Sphere(d-1)
    intrinsic_mean = mean(S,esti,GradientDescentEstimation())
    #intrinsic_var  = var(S, esti, intrinsic_mean; corrected=true)
    bias = (sum(intrinsic_mean.* true_beta, dims = 1))
    bias_angle = acos.(bias)
    dev= mean([sum(intrinsic_mean.* esti[i]) for i in 1:reps])
    angle= var([acos.(sum(intrinsic_mean.* esti[i])) for i in 1:reps])
  return (bias,bias_angle,dev,angle)
  end
end


####### Local fr regression given any data, direction along which to compute projection, and bandw choice
@everywhere function LocLin(dat, direc, xout, bw; ker = ker_gauss)
  xin = dat.xin
  yin = dat.yin
  d1, d2 = size(xin)
  #direc = randn(d2)
  projec = generate_proj(xin, direc)
  xin_eff = projec
  yin_eff = yin
  #xout = mean(xin_eff)
  ## computing weights
  mu0 = mean(ker.(((xin_eff .-  xout) ./ bw)))
  mu1 = mean(ker.(((xin_eff .-  xout) ./ bw)) .* (xin_eff .-  xout))
  mu2 = mean(ker.(((xin_eff .-  xout) ./ bw)) .* ((xin_eff .- xout).^2))
  s = ker.(((xin_eff .-  xout) ./ bw)) .* (mu2 .- mu1 .*(xin_eff .- xout)) ./
    (mu0 .* mu2 .- mu1^2) 
  ## Implements optimization in the local fr regression
  # initial guess
  y0 = vcat(mean(yin .*s, dims = 1)...) #'
  y0 = y0 ./ l2norm(y0)
  #kk = findall(ker.((xout .- xin_eff) ./ bw).>0)
  if sum([sum(yin[i,:] .* y0) for i in 1:d1][findall(ker.((xout .- xin_eff) ./ bw).>0)] .> 1-1e-8) !=0
    y0 = y0 .+ randn(m) .* 1e-3
    y0 = y0 ./ l2norm(y0)
  end
  ##compute value
  function f(y)
    y <- y / l2norm(y)
    if all(x->l2norm(x)==1,y)
      return((value = Inf))
    end
    return (mean(s .* [SpheGeoDist(yin[i,:], y)^2 for i in 1:d1]))
  end
  ##compute gradient
  function g!(G, y)
    G = vcat(2 * mean([SpheGeoDist(yin[i,:], y) .*
                       SpheGeoGrad(yin[i,:], y) for i in 1:d1] .* s, dims = 1)...)
  end
  ##compute Hessian
  function h!(H, y)
    res = zeros(m,m,d1)
    for i in 1:d1
      grad_i = SpheGeoGrad(yin[i,:], y)
      res[:,:,i] = (grad_i * grad_i' .+ #'
                  SpheGeoDist(yin[i,:], y) .* SpheGeoHess(yin[i,:], y)) .* s[i] ##'
    end
    H = reshape(2 * mean(res, dims = m),m,m)
  end
  ##NewtonTrustRegion
  res = optimize(f, g!, h!, y0, NewtonTrustRegion(initial_delta = 0.1,
                                                  delta_hat = 1e5))
  # res = trust::trust(objFctn, y0, 0.1, 1)
  return(res.minimizer./ l2norm(res.minimizer))
end

### Bandwidth selection 5-fold CV function for the local Frechet regression along any projected direction
@everywhere function bwCV(dat, direc, bw; ker = ker_gauss)
  xin = dat.xin
  yin = dat.yin
  d1, d2 = size(xin)
  projec = generate_proj(xin, direc)
  m = size(yin,2)
  ind_cv = collect(Kfold(d1,5))
  cv_err = 0.
  for i in 1: 5 #size(ind_cv,1)
    xin_eff = xin[ind_cv[i],:]
    yin_eff = yin[ind_cv[i],:]
    dat_cv = (xin = xin_eff, yin = yin_eff)
    for k in 1:(d1 - size(ind_cv[i],1))
      res = LocLin(dat_cv, direc, projec[setdiff((1:d1), ind_cv[i])[k]], bw;ker = ker_gauss)
      cv_err += SpheGeoDist(res, yin[setdiff((1:d1), ind_cv[i])[k],:])
    end
    cv_err = cv_err/(d1 - size(ind_cv[i],1))
  end
  return (cv_err/5)
end


#### Data generation mechanisms
@everywhere function generate_data(n,true_beta,link)
  d = length(true_beta)
  xin = rand(n,d)
  link_proj = link.([sum(true_beta.*xin[i,:]) for i in 1:n])
  link_proj = (link_proj .- minimum(link_proj))/(maximum(link_proj) -minimum(link_proj))
  err_sd = 0.2
  phi_true = acos.(link_proj)
  theta_true = pi .* link_proj
  #m = 3
  ytrue = zeros(n,m)
  ytrue = [[sin.(phi_true[i]) .* cos.(theta_true[i]),
            sin.(phi_true[i]) .* sin.(theta_true[i]),
            cos.(phi_true[i])] for i in 1:n]
  basis =(b1 = [[cos.(phi_true[i]) .* cos.(theta_true[i]),
                 cos.(phi_true[i]) .* sin.(theta_true[i]),
                 -sin.(phi_true[i])] for i in 1:n],
          b2 = [[sin.(theta_true[i]), -cos.(theta_true[i]), 0] for i in 1:n])
  yin_tg = basis.b1 .* rand(Normal(0,err_sd), n) .+
    basis.b2 .* rand(Normal(0,err_sd), n)
  yin = zeros(n,m)
  for i in 1:n
    tgNorm = sqrt(sum(yin_tg[i].^2))
    if tgNorm < 1e-10
      yin[i,:] = ytrue[i]
    else
      yin[i,:] = sin.(tgNorm) .* yin_tg[i] ./ tgNorm + cos.(tgNorm) .* ytrue[i]
    end
  end
  return (xin = xin, yin = yin)
end
### Given a multivariate predictor and a given direction, calculates the projection 
@everywhere function generate_proj(xin, direc)
  projec = Vector{Float64}(undef, size(xin, 1))
  for ind1 in 1:size(xin, 1)
    total = 0.
    for ind2 in 1:size(xin, 2)
      total += xin[ind1, ind2] * direc[ind2]
    end
    projec[ind1] = total
  end
  return (projec.- minimum(projec))/(maximum(projec)-minimum(projec))
end


### Binning step: given data and direction bins the support of the projection and returns a representative point for the data (xin and yin)
#### Depends on the number of bins: M
@everywhere function binned_data(dat, direc, bw, M)
  if ismissing(M)
    hh = tuning(dat,direc, bw, M)
    bw = hh[1]
    M = ceil(Int, hh[2])
  end
  xin = dat.xin
  yin = dat.yin
  d1, d2 = size(xin)
  m = size(yin,2)
  #direc = normalize(randn(d2))
  projec = generate_proj(xin, direc)
  #M = ceil(Int, d1^(1/3))
  range_of_projec = collect(range(minimum(projec),stop = maximum(projec), length = M))
  binned_xmean = zeros(M,d2)
  binned_xmean[1,:] =  dat.xin[findall(projec .<= range_of_projec[1]),:][1,:]
  for l in 2:(M-1)
    binned_xmean[l,:] = dat.xin[findall((projec .> range_of_projec[l]) .* (projec .<= range_of_projec[l+1])),:][1,:]
  end
  binned_xmean[M,:] = dat.xin[findall(projec .>= range_of_projec[M]),:][1,:]
  binned_ymean = zeros(M,m)
  binned_ymean[1,:] =  dat.yin[findall(projec .<= range_of_projec[1]),:][1,:]
  for l in 2:(M-1)
    binned_ymean[l,:] = dat.yin[findall((projec .> range_of_projec[l]) .* (projec .<= range_of_projec[l+1])),:][1,:]
  end
  binned_ymean[M,:] = dat.yin[findall(projec .>= range_of_projec[M]),:][1,:]
  return (projec = projec, binned_xmean = binned_xmean, binned_ymean = binned_ymean)
end


### CV criterion to select M 
@everywhere function bwCV_M(dat, direc, M, bw; ker = ker_gauss)
  binned_dat = binned_data(dat, direc, bw, M)
  xin_binned = binned_dat.binned_xmean
  yin_binned = binned_dat.binned_ymean
  proj_binned = generate_proj(xin_binned, direc)
  d, m = size(yin_binned)
  cv_err = 0.
  for i in 1:d
    xin_eff = xin_binned[ 1:end .!= i,:]
    yin_eff = yin_binned[ 1:end .!= i,:]
    dat_cv = (xin = xin_eff, yin = yin_eff)
    res = LocLin(dat_cv, direc, proj_binned[i], bw; ker = ker_gauss)
    cv_err += SpheGeoDist(res , yin_binned[i,:])
  end
  return (cv_err/d)
end

### Implements the selcetion of bandw for the local fr reg and, for the  optimal choice of bandw, selects the optimal bin size
@everywhere function tuning(dat, direc, bw, M; ker = ker_gauss)
  xin = dat.xin
  yin = dat.yin
  d1, d2 = size(xin)
  projec = generate_proj(xin, direc)
  #direc = direc_curr_i
  #M = missing
  #bw = missing
  if ismissing(bw)
    xinSt = unique(sort!(projec))
    bw_min = maximum(maximum.([diff(xinSt), xinSt[2] .- minimum.(projec), 
                               maximum.(projec) .- xinSt]))*1.1 / (ker == ker_gauss ? 3 : 1)
    bw_max = (maximum(projec) - minimum(projec))/3
    if bw_max < bw_min 
      if bw_min > bw_max*3/2
        #warning("Data is too sparse.")
        bw_max = bw_min*1.01
      else 
        bw_max = bw_max*3/2
      end
    end
    #bw_range = [bw_min, bw_max]
    #bw_init = mean(bw_range)
    bw = optimize(x -> bwCV(dat, direc, x), bw_min, bw_max).minimizer
  end
  if ismissing(M)
    M_range = [ceil(Int, d1^(1/p)) for p in 3:7]
    #M_range = collect(range(minimum(binned_dat.binned_xmean),stop = maximum(binned_dat.binned_xmean), length = 30))
    M_curr = M_range[1]
    cv_err_curr = bwCV_M(dat, direc, M_curr,bw)
    for M in M_range
      cv_err_test = bwCV_M(dat, direc, M, bw)
      if cv_err_test < cv_err_curr
        cv_err_curr, M_curr = cv_err_test, M
      end
    end
    M = ceil(Int, M_curr)
  end
  return([bw,M])
end


#n = 100
# reps = 4
#b          = [4,0,0,0] #[4, 17.3,5,7.1]
#b0         = normalize(b)
#d          = length(b0)
#link = x  -> x
#dat = generate_data(n, b0, link)
#xin = dat.xin; yin = dat.yin; d1, d2 = size(xin)
#ker = ker_gauss; bw = .4; direc = randn(d2)

## Main implementation function that returns the estimated direction parameter (unit vector) 
@everywhere function estimate_ichimura(dat, bw, M;ker = ker_gauss)
  xin = dat.xin
  yin = dat.yin
  d1, d2 = size(xin)
  m = size(yin,2)
  direc_curr_i = normalize(randn(d2))
  bw1 = bw
  #M = missing
  #if ismissing(bw)
  hh = tuning(dat,direc_curr_i, bw, M)
  bw = hh[1]
  M = ceil(Int, hh[2])
  #end
  binned_dat = binned_data(dat, direc_curr_i,bw, M)
  d = size(binned_dat.binned_xmean,1)
  err = 0.
  for l in 1:d
    res = LocLin(dat, direc_curr_i, 
                 binned_dat.binned_xmean[l], bw;ker = ker_gauss)
    err = (err + SpheGeoDist(res, binned_dat.binned_ymean[l,:]))
  end
  fdi_curr  = err/d
  for i in 2:500
    #global fdi_curr, direc_curr_i , direc_curr_t, fdt_curr,
    direc_test = normalize(randn(d2))
    binned_dat = binned_data(dat, direc_test,bw, M)
    d = size(binned_dat.binned_xmean,1)
    if ismissing(bw1)
      bw = tuning(dat,direc_curr_i,bw1,M)[1]
    end
    err = 0.
    for l in 1:d
      res = LocLin(dat, direc_test, 
                   binned_dat.binned_xmean[l], bw;ker = ker_gauss)
      err = (err + SpheGeoDist(res,binned_dat.binned_ymean[l,:]) )
    end
    fdi_test   = err/d
    if fdi_test < fdi_curr
      direc_curr_i,  fdi_curr = direc_test, fdi_test
    end
  end
  normalize(direc_curr_i)
end

### generates data; implements the method, calculates the bias and deviance using parallel computing over multiple replication
function estimate_2pred(n; reps = 100,bw, M, link)
  b  = [4, 1.3,-2.5,1.7]
  b0 = normalize(b)
  d  = length(b0)
  #link = x  -> x
  foo  = _ -> estimate_ichimura(generate_data(n, b0, link), bw, M)
  cwp  = CachingPool(workers())
  est = pmap(foo, cwp, 1:reps)
  est_signed =  [est[i].* sign(sum(b0.*est[i])) for i in 1:reps]
  #return [est est_signed]
  #  output = [est est_signed]
  # writedlm( "/home/titli123/SIM_June16/sim_final_june17/$link.csv",  output, ',')
  return [bias_calc2(est, b0, reps);
          bias_calc2(est_signed, b0, reps)]
end
@everywhere m = 3
link = [x ->x; x -> x^2; x -> exp(x)]
for ll in 1:3
ddd = estimate_2pred(100; reps = 2,bw = 1, M = 5, link = link[ll])
#ddd = estimate_2pred(100; reps = 2,bw = missing, M = 5, link = link[ll])
#ddd = estimate_2pred(100; reps = 2,bw = missing, M = missing, link = link[ll])
#ddd = estimate_2pred(100; reps = 2,bw = 1, M = missing, link = link[ll])
#println(ddd)
end
