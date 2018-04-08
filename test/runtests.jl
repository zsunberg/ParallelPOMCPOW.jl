using ParallelPOMCPOW
using Base.Test

using POMCPOW
using POMDPs
using POMDPModels
using POMDPToolbox

m = BabyPOMDP()

ps = POMCPOWSolver(tree_queries=100_000, max_depth=10, enable_action_pw=false)
solver = ParallelPOMCPOWSolver(ps, 50)

planner = solve(solver, m)

action(planner, initial_state_distribution(m))
action(planner, initial_state_distribution(m))
@time a, info = action_info(planner, initial_state_distribution(m))
@show info[:search_time_us]/1e6
@show info[:tree_queries]

splanner = solve(ps, m)
action(splanner, initial_state_distribution(m))
@time a, info = action_info(splanner, initial_state_distribution(m))
@show info[:search_time_us]/1e6
