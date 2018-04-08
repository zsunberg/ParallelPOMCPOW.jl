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
