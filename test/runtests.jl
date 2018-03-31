using ParallelPOMCPOW
using Base.Test

using POMCPOW
using POMDPs
using POMDPModels

m = BabyPOMDP()

ps = POMCPOWSolver(tree_queries=10_000, max_depth=10)
solver = ParallelPOMCPOWSolver(ps, 50)

planner = solve(solver, m)

action(planner, initial_state_distribution(m))
action(planner, initial_state_distribution(m))
@time action(planner, initial_state_distribution(m))

splanner = solve(ps, m)
action(splanner, initial_state_distribution(m))
@time action(splanner, initial_state_distribution(m))
