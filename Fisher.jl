"""

A simple Julia type for holding Fisher matrices along with parameter names so
that they can be added together and the rows and columns are automatically
aligned. 

To create a Fisher matrix:

    > f = FisherMatrix([1 0; 0 2],["param1","param2"])
    Fisher.FisherMatrix{Int64,ASCIIString}, std-dev of 2 parameters: 
      param2 => 0.7071067811865476 
      param1 => 1.0
    
The array itself and the names are stored under `f.fish` and `f.names`. We can
create a second  fisher matrix and add them together. Note the rows/columns are
automatically aligned. 

    > f2 = FisherMatrix([3 0; 0 4],["param2","param3"])
    Fisher.FisherMatrix{Int64,ASCIIString}, std-dev of 2 parameters: 
      param2 => 0.5773502691896257 
      param3 => 0.5 
      
    > f + f2
    Fisher.FisherMatrix{Float64,ASCIIString}, std-dev of 3 parameters: 
      param2 => 0.4472135954999579 
      param1 => 1.0 
      param3 => 0.5 
      
 The standard deviations (the sqrt's of the diagonal of the inverse) can be
 extracted with the `stds` method
 
    > stds(f + f2)
    Dict{ASCIIString,Float64} with 3 entries:
      "param2" => 0.4472135954999579
      "param1" => 1.0
      "param3" => 0.5

"""
module Fisher

export FisherMatrix, stds, corr, with_fixed, like1d, like2d, likegrid

using NamedArrays, PyCall, PyPlot, SpecialFunctions
@pyimport matplotlib.patches as patches
import Base: +, show

type FisherMatrix{Tvals,Tnames}
    fish::AbstractArray{Tvals,2}
    names::Array{Tnames,1}
end

"""
    stds(f::FisherMatrix)
    
Get the standard deviations of each parameter, i.e. the sqrts of the diagonal of
the inverse of the Fisher matrix.
"""
stds(f::FisherMatrix) = Dict(zip(f.names,sqrt.(diag(inv(f.fish)))))

function +(f1::FisherMatrix, f2::FisherMatrix)
    names = collect(union(Set(f1.names),Set(f2.names)))
    fish = zeros(length(names),length(names))
    for f in (f1,f2)
        ix = indexin(f.names,names)
        fish[ix,ix] += f.fish
    end
    FisherMatrix(fish,names)
end

function +(f::FisherMatrix, a::AbstractArray)
    @assert size(f.fish)==size(a) "When adding Fisher matrix to array, must be same size"
    FisherMatrix(a+f.fish, f.names)
end    

+(a::AbstractArray, f::FisherMatrix) = +(f,a)

function show(io::IO, f::FisherMatrix) 
    write(io,"$(typeof(f)), std-dev of $(length(f.names)) parameters: \n")
    for (k,v) in sort(collect(stds(f)))
        write(io,"  $k => $v \n")
    end
end


doc"""
    cov2corr{T}(f::AbstractArray{T<:Real,2})
    
Get covariance matrix from a correlation matrix.
"""
function cov2corr{T<:Real}(f::AbstractArray{T,2})
    m = copy(f)
    n,n = size(m)
    for i in 1:n
        sm = sqrt(m[i,i])
        m[i,:] /= sm
        m[:,i] /= sm
    end
    m
end


doc"""
    corr(f::FisherMatrix, params=nothing)
    
Get the correlation matrix for the given FisherMatrix. If params is given,
return the correlation for only a particular subset of parameters. Note this
means the Fisher matrix is first inverted and then the correlation is computed. 
"""
function corr(f::FisherMatrix, params=f.names)
    ii = indexin(params,f.names)
    NamedArray(cov2corr(inv(f.fish)[ii,ii]),(params,params),("p₁","p₂"))
end


doc"""
    with_fixed(params,f::FisherMatrix)

Get a Fisher matrix with some of the parameters held fixed.
"""
function with_fixed(params, f::FisherMatrix)
    ii = deleteat!(collect(1:length(f.names)),sort(indexin(params,f.names)))
    FisherMatrix(f.fish[ii,ii],f.names[ii])
end


function with_marged(params, f::FisherMatrix)
    ii = deleteat!(collect(1:length(f.names)),sort(indexin(params,f.names)))
    FisherMatrix(inv(inv(f.fish)[ii,ii]),f.names[ii])
end


prior(param,std::Real) = FisherMatrix(eye(1)/std^2,[param])


# Plotting
# --------

function like1d{Tnames,Tvals}(fish::FisherMatrix{Tvals,Tnames}, param::Tnames, center=0; kwargs...)
    like1d(center, stds(fish)[param]; kwargs...)
end
    
function like2d{Tnames,Tvals}(fish::FisherMatrix{Tvals,Tnames}, param1::Tnames, param2::Tnames, center=(0,0); kwargs...)
    ii = indexin([param1,param2],fish.names)
    @assert !any(ii.==0) "Fisher doesn't contain both parameters $param1 and $param2"
    like2d(center, inv(fish.fish)[ii,ii]; kwargs...)
end

function like1d(μ,σ; σlim=4, color=nothing, maxed=false, kwargs...)
    x = linspace(μ-4σ, μ+4σ,100)
    A = maxed ? 1 : 1/√(2π)/σ
    plot(x, @.(A*exp(-(x-μ)^2/2/σ^2)); color=color, kwargs...)
end

function like2d(μ,Σ; σs=1:2, σlim=1.618max(σs...), color=nothing, kwargs...)
    λ, v = eig(Σ)
    θ = rad2deg(atan2(v[2,1],v[1,1]))
    getc(σ) = imag(√2*√(complex(log(erfc(σ/√2)))))
    if (color == nothing) color = pybuiltin("next")(gca()[:_get_lines][:prop_cycler])["color"] end
    lw = get(plt[:rcParams],"lines.linewidth",1)
    for c in map(getc,σs)
        gca()[:add_artist](patches.Ellipse(;
            xy=μ, width=2c*√λ[1], height=2c*√λ[2], angle=θ, lw=lw, 
            facecolor="none", edgecolor=color, fill=false, kwargs...
        ))
    end
    if σlim != nothing
        xlim((μ[1]+x*getc(σlim)*√λ[1] for x=[-1,1])...)
        ylim((μ[2]+x*getc(σlim)*√λ[2] for x=[-1,1])...)
    end
end

function likegrid{Tnames,Tvals}(fish::FisherMatrix{Tvals,Tnames}, params::AbstractArray{Tnames}=fish.names; centers=Dict(), kwargs...)  
    fig=gcf()
    n=length(params)
    for (i,p1) in enumerate(params)
        for (j,p2) in enumerate(params)
            if (i<=j)
                ax=fig[:add_subplot](n,n,(j-1)*n+i)
                if i==j 
                    like1d(fish,p1,get(centers,p1,0); kwargs...)
                else
                    like2d(fish,p1,p2,(get(centers,p1,0),get(centers,p2,0)); kwargs...)
                end
            end
        end
    end
end


end
