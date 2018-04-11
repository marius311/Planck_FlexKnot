module CMBForecast

using PyCall
using Interpolations
using Fisher
using Base.Threads
using NamedArrays

export get_Cℓ, get_Nℓ, get_dCℓs, get_fish, get_cp, named_Cℓ_array, noise_add, get_dτ_dmi


# fiducial parameters (table 8 column 2 of http://arxiv.org/abs/1605.02985)
p0_lcdm = Dict(
    :As => 1.885*exp(2*0.0581),
    :ns => 0.9624,
    :tau => 0.0581,
    :cosmomc_theta => 0.0104075,
    :ombh2 => 0.02214,
    :omch2 => 0.1207,
    :mnu => 0.060,
    :nnu=>3.046,
    :nrun=>0,
    :omk=>0,
    :w=>-1
)

function __init__()
    global camb
    if myid()==1
        @pyimport camb as _camb
        camb = _camb
        camb.CAMBparams()[:set_accuracy](AccuracyBoost=1,lAccuracyBoost=1)
        camb.reionization[:include_helium_fullreion][:value]=false
    end
end


"""
    load_mh_modes(file)

Load the reionization eigenmodes from Mortonson&Hu, smoothly transition them to
zero at the end points, and interpolate them to higher resolution ensuring the
derivatives at the end points are zero as well. 

`file` should point to the `xefid.dat` file from Mortonson&Hu
"""
function load_mh_modes(file)

    #read initial modes and smoothly transition to zero at endpoints 
    z = readdlm(file)[1,:][:]
    dl = 8
    w = [(cos(pi*(dl:-1:0)/dl)+1)/2; ones(length(z)-2*dl-2); (cos(pi*(0:dl)/dl)+1)/2]
    modes = (readdlm(file)[2:end,:]'.*w)'

    #interpolate and ensure derivative at end points is zero
    itp = [interpolate(modes[i,:][:],BSpline(Cubic(Natural())),OnGrid()) for i in 1:95];
    modes = [Float64(itp[i][jp]) for i=1:95, jp=linspace(1,95,1024)]
    z = [linspace(z[1],z[end],1024)...]
    
    return (z, modes)

end

doc"""
    get_cp(lmax=2998; params...)
    
Get a CAMBparams object for the given set of parameters which can then be passed to 
camb.get_results or any other such function.
"""
function get_cp(;lmax=2998, sigmoid=nothing, params...)
    
    params = Dict{Any,Any}(params)
    
    @assert (:tau in keys(params)) ⊻ (:fidxe in keys(params)) "Must provide one and only one_ of tau or fiducial reionization history"
    
    #some basic derived parameters
    if ~(:H0 in keys(params))
        params[:H0] = nothing
    end
    if (:clamp in keys(params))
        params[:As] = pop!(params,:clamp)*exp(2*params[:tau])*1e-9
    else
        params[:As] *= 1e-9
    end
    
    #get reionization modes
    z, modes = collect(pop!(params,:z,[])), pop!(params,:modes,[])
    xems = Float64[pop!(params,Symbol("m_$i"),0) for i in 1:size(modes,1)]
    if get(params,:tau,nothing)==0
        pop!(params,:tau)
    end
    
    #set camb parameters
    cp = camb.CAMBparams()
    cp[:DoLensing] = pop!(params,:DoLensing,true)
    cp[:NonLinear] = pop!(params,:NonLinear,0)
    cp[:k_eta_max_scalar] = pop!(params,:k_eta_max_scalar,10000)
    
    fidxe = pop!(params,:fidxe,nothing)
    
    #set twice (workaround for camb.set_params bug when setting w/theta simultaneously)
    camb.set_params(cp,lmax=lmax; params...)
    camb.set_params(cp,lmax=lmax; params...) 
    
    #set reioinzation modes if we have any
    if fidxe != nothing || any(xems.!=0)
        if fidxe == nothing
            fidxe = cp[:Reion][:get_xe](z=z)
        end
        cp[:tau] = 0
        
        if any(xems.!=0)
            if sigmoid==nothing
                xe = fidxe + modes'*xems
            else
                S,Sinv = sigmoid
                xe = S.(@view (Sinv.(fidxe) .+ modes'*xems)[:])
            end
        else
            xe = fidxe
        end
        cp[:Reion][:set_xe](z=z,xe=xe,smooth=1e-3)
    end
    
    return cp

end


named_Cℓ_array(arr) = NamedArray(arr, ([:T,:E,:d], [:T,:E,:d], collect(1:size(arr)[end])), ("x₁","x₂","ℓ"));


doc"""
    get_Cℓ(; lmax=2998, use_lensed_cls=false, params...)
    
Get the the signal covariace matrix, Cℓ[i,j,l] where i,j run over T,E,d and l is
the multipole moment.
"""
function get_Cℓ(; lmax=2998, use_lensed_cls=false, params...)

    cp = get_cp(lmax=lmax; params...)
      
    #call camb
    r = camb.get_results(cp)
    
    #output results into our array
    scls = r[Symbol("get_"*(use_lensed_cls ? "" : "un")*"lensed_scalar_cls")](lmax)[2:end,:]
    lcls = r[:get_lens_potential_cls](lmax)[2:end,:]
    
    Cℓ = named_Cℓ_array(Array{Float64}((3,3,lmax)))
    for l in 1:lmax
        scale = 1e12 * cp[:TCMB]^2 * 2pi/l/(l+1)
        tt,ee,te    = [scale*scls[l,i] for i in [1,2,4]]
        dd,td,ed    = [scale*lcls[l,i] for i in [1,2,3]]
        Cℓ[:,:,l]   = [[tt te td];
                       [te ee ed];
                       [td ed dd]]
    end
    
    return Cℓ
                    
end

doc"""
    get_Nℓ(beam, noise_T, noise_P=(sqrt(2) * noise_T); noise_dd=Inf, lmax=2998)

Get the the noise covariace matrix, C[i,j,l] where i,j run over T,E,d and l is
the multipole moment.

# Arguments

* `beam`: beam FWHM in arcmin
* `noise_T`: temperature noise in μK-arcmin
* `noise_P`: polariation noise in μK-arcmin (default: $\sqrt{2}$ * `noise_T`)
* `noise_dd`: deflection angle noise $N_\ell^{dd}$ (if string, load from file)
"""
function get_Nℓ(beam, noise_T, noise_P=(sqrt(2)*noise_T); noise_dd=Inf, lmax=2998)
        
    Nℓ = named_Cℓ_array(zeros(3,3,lmax))
    
    noise_T = deg2rad(noise_T/60)^2
    noise_P = deg2rad(noise_P/60)^2
    if typeof(noise_dd)<:String
        noise_dd = (2.725e6^2)*readdlm(noise_dd)[2:end]
    elseif noise_dd == Inf
        noise_dd = fill(Inf,lmax)
    end
    
    for l in 1:lmax
        Blm² = exp(l*(l+1)*deg2rad(beam/60)^2/(8*log(2)))
        Nℓ[:,:,l] = diagm([noise_T*Blm²,noise_P*Blm²,noise_dd[l]])
    end
    
    return Nℓ
    
end

doc"""
Combine several noise spectra according to optimal weighting.
"""
function noise_add{T<:NamedArray}(Nℓs::T...)
    names = [(n.dicts,n.dimnames) for n in Nℓs]
    # @assert all(names[1].==names) "Can only call noise_add on same-shape noise spectra"
    NamedArray(mapslices(inv,sum([mapslices(inv,n.array,(1,2)) for n in Nℓs]),(1,2)),names[1]...)
end
noise_add{T<:Union{Base.Generator,AbstractArray}}(Nℓs::T) = noise_add(Nℓs...)


function get_fish{T<:NamedArray}(Cℓ::T, 
                                 Nℓ::T,        
                                 dCℓ::Dict{Symbol,T};
                                 fsky=1, 
                                 lranges=nothing, 
                                 ps=collect(keys(dCℓ)))

    
    if isa(lranges,UnitRange) lranges = Dict(x=>lranges for x in [:TT,:TE,:EE,:Td,:dd]) end
    lranges = Dict(map(Symbol,(string(k)...))=>v for (k,v)=lranges)
    lmin, lmax = (m([m(r) for r=values(lranges)]) for m=[minimum,maximum])
    
    Tℓ = Cℓ + Nℓ            
    fish = zeros(Float64,(length(ps),length(ps)))

    for l in lmin:lmax
        use = [k for (k,v)=lranges if l in v]
        cv = [1/(2l+1)/fsky*(Tℓ[a,c,l]*Tℓ[b,d,l] + Tℓ[a,d,l]*Tℓ[b,c,l]) for (a,b)=use, (c,d)=use]
        d = [dCℓ[k][a,b,l] for (a,b)=use, k=ps]
        fish += d' * inv(cv) * d
    end

    return FisherMatrix(fish,ps)
end


function get_dCℓs(p0::Dict, dp::Dict, pvary=keys(p0))
    Dict(k => -((get_Cℓ(;merge(p0,Dict(k=>p0[k]+x*dp[k]/2))...) for x=[1,-1])...)/dp[k] for k in pvary)
end

get_τ(;p...) = camb.get_background(get_cp(;p...))[:get_tau]()

function get_dτ_dmi(p0,nmodes=10)
    p = merge(p0,Dict(:lmax=>100, :DoLensing=>false))
    dp = 0.1
    dτs = [1; [(get_τ(;merge(p,Dict(k=>dp/2))...) - get_τ(;merge(p,Dict(k=>-dp/2))...))/dp
            for k in [Symbol("m_$i") for i=1:nmodes]]]
end



end
