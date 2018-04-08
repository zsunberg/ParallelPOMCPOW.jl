using ParallelPOMCPOW
using Base.Test

using POMCPOW
using POMDPs
using POMDPModels

m = BabyPOMDP()

ps = POMCPOWSolver(tree_queries=100_000, max_depth=10, enable_action_pw=false)
solver = ParallelPOMCPOWSolver(ps, 50)

planner = solve(solver, m)

action(planner, initial_state_distribution(m))
action(planner, initial_state_distribution(m))
@time action(planner, initial_state_distribution(m))

splanner = solve(ps, m)
action(splanner, initial_state_distribution(m))
@time action(splanner, initial_state_distribution(m))
