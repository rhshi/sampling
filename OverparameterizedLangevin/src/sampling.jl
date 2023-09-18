export simGradFlow, plotHist, simLangevin, plotHistChange, make_f, g

function make_f(Vgrad; T=1)    
    function f(u, p, t)
        return -Vgrad(u) ./ T
    end;
    return f
end;

function g(u, p, t)
    return sqrt(2)
end;

function simGradFlow(f, u0, tspan)
    prob = ODEProblem(f, u0, tspan)
    sol = solve(prob)
    return sol
end;

function plotHist(data, d; D::Union{MvNormal, Nothing}=nothing, nbins=50)
    h = fit(Histogram, Tuple(cord for cord in data), nbins=nbins)
    h = normalize(h, mode=:pdf);
    if d==1
        display(plot(h))
#         gui()
    elseif d==2
        x, y = histParams(h)

        x1 = first(x)
        x2 = last(x)
        y1 = first(y)
        y2 = last(y)
        
        lim_min = min(x1, x2, y1, y2)
        lim_max = max(x1, x2, y1, y2)
        
        display(plot(h, xlims=(lim_min, lim_max), ylims=(lim_min, lim_max)))
        x, y = histParams(h)
        p = plot(x, y, h.weights, st=:surface, xlims=(lim_min, lim_max), ylims=(lim_min, lim_max));
        if !isnothing(D)
            Dz = map(z -> pdf(D, collect(z)), Iterators.product(x, y))
            wireframe!(x, y, Dz)
        end
        display(p)
#         gui()
    end;
end;

function simLangevin(f, g, u0, tspan; trajectories=10000, nbinsbins=50, D::Union{MvNormal, Nothing}=nothing)
    d = length(u0)
    t = tspan[2]
    
    prob = SDEProblem(f, g, u0, tspan);
    function prob_func(prob, i, repeat)
        remake(prob, u0 = randn(d) + prob.u0)
    end
    ensembleprob = EnsembleProblem(prob, prob_func=prob_func);
    sol = solve(ensembleprob, SRIW1(), trajectories=trajectories);
    
    data = componentwise_vectors_timepoint(sol, tspan[2]);
    
    plotHist(data, d, D=D)
    
    return sol    
end;

function histParams(h)
    gap1 = (h.edges[1][2] - h.edges[1][1])/2
    gap2 = (h.edges[2][2] - h.edges[2][1])/2

    xlen, ylen = size(h.weights)

    x = LinRange(first(h.edges[1])+gap1, last(h.edges[1])-gap1, xlen)
    y = LinRange(first(h.edges[2])+gap2, last(h.edges[2])-gap2, ylen);
   return x, y 
end;

function plotHistChange(sol, ts, nbins=50)
    hists = []
    lim_min = Inf
    lim_max = -Inf
    for t in ts
        data = componentwise_vectors_timepoint(sol, t)
        h = fit(Histogram, Tuple(cord for cord in data), nbins=nbins)
        h = normalize(h, mode=:pdf);
        x, y = histParams(h);
        
        x1 = first(x)
        x2 = last(x)
        y1 = first(y)
        y2 = last(y)
        c_min = min(x1, x2, y1, y2)
        c_max = max(x1, x2, y1, y2)
        if c_min < lim_min
            lim_min = c_min
        end
        if c_max > lim_max
            lim_max = c_max
        end
        
        push!(hists, (x, y, h.weights))
    end
    t = ts[1]
    p = wireframe(hists[1][1], hists[1][2], hists[1][3], color=1, label="t=$t$(@sprintf("%d", t))", xlims=(lim_min, lim_max), ylims=(lim_min, lim_max))
    c = 2
    for hist in hists[2:end]
        t = ts[c]
        wireframe!(hist[1], hist[2], hist[3], color=c, label="t=$t$(@sprintf("%d", t))", xlims=xlims(p), ylims=ylims(p))
        c += 1
    end
    display(p)
    return
end;