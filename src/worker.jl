# the code that will run on a worker process

function run_worker(p::ParallelPOMCPOWPlanner, b0, acts::AbstractVector, input::RemoteChannel, output::RemoteChannel)
    na = length(acts)
    try
        pp = p.powplanner
        srand(pp, rand(UInt32))
        m = pp.problem
        rng = pp.solver.rng
        c = pp.solver.criterion.c

        trees = [[new_tree(pp, b0, a, p.batch_size)] for a in acts]
        n_sims = @MVector(zeros(Int, na))
        q_sums = @MVector(zeros(Float64, na))

        while true
            # don't go unless at least one input has been given
            up = take!(input)
            # get latest update
            while isready(input)
                up = take!(input)
            end
            if up.terminate
                break
            end
            global_ns = convert(MVector, up.global_ns)
            global_qs = convert(MVector, up.global_qs)
            if !isnull(up.new_branch)
                nbi = get(up.new_branch)
                push!(trees[nbi], new_tree(pp, b0, acts[nbi], p.batch_size))
            end

            # run a batch
            for k in 1:p.batch_size
                # choose which to run on 
                ltn = log(sum(global_ns))
                ai = indmax(begin
                    q = global_qs[i]
                    n = global_ns[i]
                    ucb(q, c, n, ltn)
                end for i in 1:na)

                # run sims for the number of trees
                for i in 1:length(trees[ai])
                    # choose the tree with the least sims
                    t = trees[ai][indmin(t.total_n[1] for t in trees[ai])]
                    s, r = rand(rng, t.root_belief)
                    if isterminal(m, s)
                        q = r
                    else
                        node = POWTreeObsNode(t, 1)
                        q = r + discount(m)*POMCPOW.simulate(pp, node, s, pp.solver.max_depth-1)
                    end
                    n_sims[ai] += 1
                    q_sums[ai] += q
                    global_ns[ai] += 1
                    global_qs[ai] += (q-global_qs[ai])/global_ns[ai]
                end
            end

            # report results
            put!(output, WorkReport(myid(), false,
                                    SVector{na, Int}([length(tv) for tv in trees]...),
                                    convert(SVector, n_sims),
                                    convert(SVector, q_sums./n_sims)))
        end
    catch ex
        println("caught")
        showerror(STDERR, ex)
        put!(output, WorkReport(myid(), true,
                                @SVector(zeros(Int, na)),
                                @SVector(zeros(Int, na)),
                                @SVector(zeros(Float64, na))))
    end
end

function new_tree(pp::POMCPOWPlanner, b, a, sz)
    m = pp.problem
    ps = pp.solver
    s = rand(ps.rng, b)
    sp, o, r = generate_sor(m, s, a, ps.rng)
    nb = POMCPOW.init_node_sr_belief(pp.node_sr_belief_updater,
                                     m, s, a, sp, o, r
                                    )
    rb = FilteredSRBelief(b, a, pp.node_sr_belief_updater, nb)
    return POMCPOWTree{typeof(nb), typeof(a), typeof(o), typeof(rb)}(rb, sz)
end

function ucb(q, c, n, ltn)
    if n == 0 && ltn <= 0.0
        return q
    elseif n == 0 && q == -Inf
        return Inf
    else
        return q + c*sqrt(ltn/n)
    end
end
