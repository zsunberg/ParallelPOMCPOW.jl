# passed from a worker to the global report channel
struct WorkReport{N} # each vector has one entry for each action
    id::Int
    error::Bool
    n_obs::SVector{N, Int}
    n_sims::SVector{N, Int}
    qs::SVector{N, Float64}
end

# passed to a single worker channel
struct GlobalUpdate{N}
    global_ns::SVector{N, Int}
    global_qs::SVector{N, Float64}
    new_branch::Nullable{Int}
    terminate::Bool
end
