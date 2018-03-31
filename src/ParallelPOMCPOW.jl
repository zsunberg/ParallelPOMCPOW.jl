__precompile__(true)
module ParallelPOMCPOW

importall POMDPs
using Parameters
using POMCPOW
using POMDPToolbox

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

action(p::ParallelPOMCPOWPlanner, b) = first(action_info(p, b))

function action_info(p::ParallelPOMCPOWPlanner, b)
    pp = p.powplanner
    ps = pp.solver
    @assert ps.eps == 0.01 # not supported; should be default
    m = pp.problem
    bs = p.batch_size
    info = Dict{Symbol, Any}()

    S = state_type(m)
    A = action_type(m)
    O = obs_type(m)
    NBU = typeof(pp.node_sr_belief_updater)
    NB = POMCPOW.belief_type(NBU, typeof(m))
    TreeType = POMCPOWTree{NB, A, O, FilteredSRBelief{typeof(b),A,NBU,NB}}

    acts = collect(iterator(actions(m, b)))
    na = length(acts)
    ns = zeros(Int, na) # XXX should be init_n
    pending_ns = zeros(Int, na)
    qs = zeros(Float64, na) # XXX should be init_q
    no = ceil(Int, ps.k_observation)
    ao_ns = [zeros(Int, no) for i in 1:na]
    subtrees = [Vector{TreeType}(no) for i in 1:na]
    locks = [falses(no) for i in 1:na]
    
    np = nprocs()  # determine the number of processes available

    start_ns = time_ns()
    end_ns = start_ns + 1e9*ps.max_time
    @sync begin
        for proc in 1:np
            if proc != myid() || np == 1
                @async begin
                    while sum(pending_ns) < ps.tree_queries
                        if time_ns() > end_ns
                            break
                        end
                        ai, oi = next_inds(pp.criterion,
                                           locks,
                                           qs,
                                           pending_ns,
                                           ao_ns)

                        locks[ai][oi] = true
                        pending_ns[ai] += bs
                        if !isassigned(subtrees[ai], oi)
                            s = rand(ps.rng, b)
                            sp, o, r = generate_sor(m, s, acts[ai], ps.rng)
                            nb = POMCPOW.init_node_sr_belief(pp.node_sr_belief_updater,
                                                             m, s, acts[ai], sp, o, r
                                                            )
                            rb = FilteredSRBelief(b, acts[ai], pp.node_sr_belief_updater, nb)
                            subtrees[ai][oi] = POMCPOWTree{NB,A,O,typeof(rb)}(rb, bs)
                        end
                        t = subtrees[ai][oi]
                        result = remotecall_fetch(run_batch, proc, t, pp, bs)
                        if time_ns() > end_ns
                            break
                            locks[ai][oi] = false
                        end
                        
                        # integrate results
                        subtrees[ai][oi] = result.subtree
                        ns[ai] += bs
                        ao_ns[ai][oi] += bs
                        qs[ai] += (result.qsum/bs - qs[ai])/ns[ai]

                        locks[ai][oi] = false
                    end
                    # end of while loop
                    search_time_ns = time_ns() - start_ns
                    if !haskey(info, :search_time_us)
                        info[:search_time_us] = search_time_ns/1000
                    end
                end
            end
        end
    end

    info[:tree_queries] = sum(ns)

    # choose the best
    return acts[indmax(qs)], info
end

function next_inds(crit::MaxUCB, locks, qs, ns, ao_ns)
    total_n = sum(ns)
    if total_n == 0
        return (1, 1)
    else
        a_locks = [all(l) for l in locks]
        ltn = log(total_n)
        vals = qs + crit.c*sqrt.(ltn./ns)
        ai = indmax(vals-Inf.*a_locks)
        @assert a_locks[ai] == false
        oi = indmin(ao_ns[ai]+Inf.*locks[ai])
        @assert locks[ai][oi] == false
        return (ai, oi)
    end
end

struct BatchResult{T<:POMCPOWTree}
    subtree::T
    qsum::Float64
end

struct FilteredSRBelief{B, A, NF, NB}
    b0::B
    a::A
    nf::NF
    srb::NB
end

function Base.rand(rng::AbstractRNG, b::FilteredSRBelief)
    s = rand(rng, b.b0)
    sp, r = generate_sr(b.srb.model, s, b.a, rng)
    POMCPOW.push_weighted!(b.srb, b.nf, s, sp, r)
    return rand(rng, b.srb)
end

function run_batch(subtree::POMCPOWTree, p::POMCPOWPlanner, bsize::Int)
    m = p.problem
    qsum = 0.0
    for i in 1:bsize
        s, r = rand(p.solver.rng, subtree.root_belief)
        if isterminal(m, s)
            qsum += r
        else
            node = POWTreeObsNode(subtree, 1)
            qsum += r + discount(m)*POMCPOW.simulate(p, node, s, p.solver.max_depth-1)
        end
    end

    return BatchResult(subtree, qsum)
end

end # module
