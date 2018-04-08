__precompile__(true)
module ParallelPOMCPOW

importall POMDPs
using Parameters
using POMCPOW
using POMDPToolbox
using StaticArrays

export
    ParallelPOMCPOWSolver

struct ParallelPOMCPOWSolver
    powsolver::POMCPOWSolver
    batch_size::Int
end

struct ParallelPOMCPOWPlanner{P<:POMCPOWPlanner}
    powplanner::P
    batch_size::Int
end

function solve(s::ParallelPOMCPOWSolver, m::POMDP)
    powplanner = solve(s.powsolver, m)
    return ParallelPOMCPOWPlanner(powplanner, s.batch_size)
end

include("belief.jl")
include("message.jl")
include("worker.jl")

action(p::ParallelPOMCPOWPlanner, b) = first(action_info(p, b))

function POMDPToolbox.action_info(p::ParallelPOMCPOWPlanner, b)
    pp = p.powplanner
    ps = pp.solver
    @assert ps.eps == 0.01 # not supported; should be default
    @assert !ps.enable_action_pw # action progressive widening not supported
    m = pp.problem
    info = Dict{Symbol, Any}()

    acts = SVector(collect(iterator(actions(m, b)))...)
    na = length(acts)

    report_channel = RemoteChannel(()->Channel{WorkReport{na}}(10*nprocs()))
    if nprocs() > 1
        workers = 2:nprocs()
    else
        workers = [1]
    end
    # should these channels be on 1 or on the remote? I think on 1 since it will be idle the most?
    update_channels = Dict(w => RemoteChannel(()->Channel{GlobalUpdate{na}}(10*nprocs()), 1) for w in workers)

    futures = Dict(w => remotecall(run_worker,
                                   w, p, b, acts,
                                   update_channels[w],
                                   report_channel) for w in workers)

    qs = @SVector(zeros(Float64, na)) # XXX should be init_q
    adjusted_ns = @SVector(zeros(Int, na))
    worker_ns = Dict(w=>@SVector(zeros(Int, na)) for w in workers) # number of simulations for that action 
    worker_n_obs = Dict(w=>@SVector(ones(Int, na)) for w in workers)
    worker_qs = Dict(w=>@SVector(zeros(Float64, na)) for w in workers)

    max_ns_per_obs = MVector{na, Int}()
    min_ns_per_obs = MVector{na, Int}()
    w_max_ns_per_obs = MVector{na, Int}()

    new_branches = Dict{Int, Nullable{Int}}()

    start_ns = time_ns()
    end_ns = start_ns + 1e9*ps.max_time
    for (w,c) in update_channels
        put!(c, GlobalUpdate(adjusted_ns, qs, Nullable{Int}(), false))
    end
    while true
        # process all waiting reports
        wait(report_channel)
        while isready(report_channel)
            r = take!(report_channel)
            if r.error
                error("error on worker")
            end
            worker_ns[r.id] = r.n_sims
            worker_n_obs[r.id] = r.n_obs
        end

        if time_ns() > end_ns
            break
        end

        # calculate global qs: each observation branch gets equal weight
        q_sums = zeros(Float64, na)
        n_obs = zeros(Int, na)
        fill!(max_ns_per_obs, typemin(Int))
        fill!(min_ns_per_obs, typemax(Int))
        for w in workers
            wno = worker_n_obs[w]
            n_obs += wno
            qs += worker_qs[w].*wno
            ns_per_obs = div.(worker_ns[w], wno)
            newmax = ns_per_obs .> max_ns_per_obs
            w_max_ns_per_obs[newmax] = w
            max_ns_per_obs[newmax] = ns_per_obs[newmax]
            min_ns_per_obs[:] = min.(ns_per_obs, min_ns_per_obs)
        end
        qs = convert(SVector{na, Float64}, q_sums./n_obs)

        # calculate adjusted global ns
        # find minimum ns per obs and act as if all obs get the same
        adjusted_ns = convert(SVector{na, Int}, min_ns_per_obs.*n_obs)

        if sum(adjusted_ns) >= ps.tree_queries
            break
        end

        # add new branches
        empty!(new_branches)
        for ai in 1:na
            if n_obs[ai] <= ps.k_observation*adjusted_ns[ai]^ps.alpha_observation
                # find worker with the maximum sims per obs for that action
                w = w_max_ns_per_obs[ai]
                new_branches[w] = ai
            end
        end

        # send instructions
        for (w, c) in update_channels
            put!(c, GlobalUpdate(adjusted_ns, qs, get(new_branches, w, Nullable{Int}()), false))
        end
    end

    info[:search_time_us] = (time_ns() - start_ns)/1000
    # terminate all worker tasks
    for (w, c) in update_channels
        put!(c, GlobalUpdate(adjusted_ns, qs, Nullable{Int}(), true))
    end
    info[:tree_queries] = sum(sum(ns for (w, ns) in worker_ns))
    info[:adjusted_queries] = sum(adjusted_ns)

    for f in futures
        fetch(f)
    end

    # choose the best
    return acts[indmax(qs)], info
end

function normalized_ns
    
end

end # module
