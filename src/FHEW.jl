module FHEW

using Einsum
using EllipsisNotation
using FFTW
using LinearAlgebra
using Parameters
using Random


@with_kw struct Context
    stddev::Float64 = 3.2
    logQ::Int = 27
    logQks::Int = 14

    N::Int = 1024
    Q::Int = (1 << logQ)
    Qks::Int = (1 << logQks)

    n::Int = 512
    q::Int = 2 * N


    root_powers::Vector{ComplexF64} = @. exp(pi / N * (0:(N÷2-1)) * im)
    root_powers_inv::Vector{ComplexF64} = @. exp(pi / N * (0:-1:(-N÷2+1)) * im)

    # Gadget decomp
    d::Int = 3
    logB::Int = 8
    decomp_shift::Vector{Int} = collect(logQ .- logB .* range(d, 1, step=-1))
    gvector::Vector{Int} = 1 .<< decomp_shift
    msbmask::Int = reduce(+, (1 << (i + logB - 2) for i in 1:3))

    # Gadget decomp for keyswitching
    dks::Int = 2
    logBks::Int = 7
    decomp_shift_ks::Vector{Int} = collect(logQks .- logBks .* range(dks, 1, step=-1))
    gvector_ks::Vector{Int} = 1 .<< decomp_shift_ks
    Bks::Int = 1 << logBks

    f_nand::Vector{Int} = zeros(Int, N)
end

keygen(dim) = rand(0:1, dim)
errgen(std) = round(Int, std .* randn())
errgen(std, N) = round.(Int, std .* randn(N))
uniform(dim, modulus) = rand(0:modulus, dim)
errpolygen(dim, stddev) = round.(Int, stddev .* randn(dim))


function encryptLWE(message, dim, modulus, key, stddev)
    ct = uniform(dim + 1, modulus)
    ct[1] = 0
    ct[1] = message * (modulus ÷ 4) - dot(ct[2:end], key)
    ct[1] += errgen(stddev)
    ct .= mod.(ct, modulus - 1)
    return ct
end
encryptLWE(message, key, c::Context) = encryptLWE(message, c.n, c.q, key, c.stddev)

function decryptLWE(ct, key, modulus::Int)
    beta, alpha = ct[1], ct[2:end]
    m_dec = beta + dot(alpha, key)
    m_dec = mod(m_dec, modulus)
    m_dec = round(Int, m_dec / (modulus ÷ 4))
    return mod(m_dec, 4)
end

decryptLWE(ct, key, c::Context) = decryptLWE(ct, key, c.q)

function negacyclic_fft(a, N, root_powers)
    a_precomp = (a[1:(N÷2), ..] .+ a[(N÷2+1):end, ..] .* im) .* root_powers

    return fft(a_precomp, 1)  # along the first dimension
end

function negacyclic_ifft(A, Q, root_powers_inv)
    b = ifft(A, 1)  # along the first dimension
    b .*= root_powers_inv
    a = vcat(real.(b), imag.(b))

    aint = Int.(round.(a))
    aint .&= Q - 1

    return aint
end

function encryptRLWEfft(m, sfft, context)
    N = context.N
    Q = context.Q

    alpha = negacyclic_fft(uniform(N, Q), N, context.root_powers)
    beta = negacyclic_fft(errgen(context.stddev, N), N, context.root_powers)

    beta .+= -alpha .* sfft .+ negacyclic_fft(m, N, context.root_powers)

    return hcat(beta, alpha)
end

function extract(ctRLWE)
    beta = ctRLWE[1, 1]
    alpha = ctRLWE[:, 2]
    alpha[2:end] .= reverse(-alpha[2:end])

    return (beta, alpha)
end

function normalize(v, logQ)
    Q = (1 << logQ)
    v .&= (Q - 1)
    msb = (v .& (Q ÷ 2)) .>> (logQ - 1)
    v .-= Q .* msb
    return v
end


function decryptRLWEfft(ctfft, sfft, context)
    Q = context.Q
    logQ = context.logQ

    beta, alpha = ctfft[:, 1], ctfft[:, 2]
    ifft_res = negacyclic_ifft(beta .+ alpha .* sfft, Q, context.root_powers_inv)
    return normalize(ifft_res, logQ)
end

function decompose(a, context)
    d = context.d
    logB = context.logB
    decomp_shift = context.decomp_shift
    mask = (1 << logB) - 1

    if ndims(a) == 1
        res = (reshape(a, (size(a)..., 1)) .>> reshape(decomp_shift, (1, d))) .& mask
        return res
    elseif ndims(a) == 2
        res = (reshape(a, (size(a)..., 1)) .>> reshape(decomp_shift, (1, 1, d))) .& mask
        return res
    else
        @error "unsupported dims"
    end
end

function signed_decompose(a, context)
    msbmask = context.msbmask
    da = decompose(a .+ (a .& msbmask), context)
    da .-= decompose((a .& msbmask), context)

    return da
end

function encryptRGSWfft(z, skfft, context)
    N, Q, d = context.N, context.Q, context.d
    logQ = context.logQ
    rgsw = zeros(Int, (N, 2, 2, d))

    rgsw[:, 2, :, :] .= rand(0:(Q-1), (N, 2, d))
    rgsw[:, 1, :, :] .= round.(Int, randn((N, 2, d)))

    rgsw .&= (Q - 1)

    rgsw = normalize(rgsw, logQ)

    root_powers = ComplexF64.(0:(N÷2-1))
    root_powers = @. exp(pi / N * root_powers * im)

    root_powers_inv = ComplexF64.(range(0, -N ÷ 2 + 1, step=-1))
    root_powers_inv = @. exp(pi / N * root_powers_inv * im)
    rgswfft = negacyclic_fft(rgsw, N, context.root_powers)

    rgswfft[:, 1, :, :] .-= rgswfft[:, 2, :, :] .* reshape(skfft, (N ÷ 2, 1, 1))

    gvector = context.gvector
    @einsum gs1[j, i] := gvector[i] * z[j]
    gzfft = negacyclic_fft(gs1, N, context.root_powers)

    rgswfft[:, 1, 1, :] .+= gzfft
    rgswfft[:, 2, 2, :] .+= gzfft

    return rgswfft
end

function rgswmult(ctfft, rgswfft, context)
    ct = negacyclic_ifft(ctfft, context.Q, context.root_powers_inv)

    dct = signed_decompose(ct, context)
    dctfft = negacyclic_fft(dct, context.N, context.root_powers)

    @einsum gs1[i, j] := rgswfft[i, j, k, l] * dctfft[i, k, l]

    return gs1
end

function brkgen(sk, skNfft, context)
    N = context.N
    n = context.n
    zero_poly = zeros(Int, N)
    one_poly = zeros(Int, N)
    one_poly[1] = 1

    dummy = encryptRGSWfft(zero_poly, skNfft, context)
    brk = typeof(dummy)[dummy for i in 1:context.n]

    for i in 1:n
        if sk[i] == 0
            brk[i] = encryptRGSWfft(zero_poly, skNfft, context)
        else
            brk[i] = encryptRGSWfft(one_poly, skNfft, context)
        end
    end
    return brk
end

function precompute_alpha(context)
    N = context.N
    q = context.q
    poly = zeros(Int, N)
    alphapoly = typeof(negacyclic_fft(poly, N, context.root_powers))[]

    for i in 1:q
        poly = zeros(Int, N)
        poly[1] = -1
        if i - 1 < N
            poly[i] += 1
        else
            poly[i-N] += 1
        end
        push!(alphapoly, negacyclic_fft(poly, N, context.root_powers))
    end

    return alphapoly
end


function nand_map(i, context)
    Q, q, N, = context.Q, context.q, context.N
    i += 2 * N
    i = mod(i, 2 * N)

    if 3 * (q >> 3) <= i < 7 * (q >> 3)
        return -(Q >> 3)
    else
        return Q >> 3
    end
end

function decompose_ks(a, context)
    @assert ndims(a) == 1

    n, N, Bks, dks = context.n, context.N, context.Bks, context.dks
    Qks = context.Qks
    logBks = context.logBks
    logQks = context.logQks
    decomp_shift_ks = logQks .- logBks .* range(dks, 1, step=-1)
    mask_ks = (1 << logBks) - 1

    @einsum res[i, j] := (a[i] >> decomp_shift_ks[j]) & mask_ks

    return res
end

function kskGenLWE(sk, skN, context)
    n, N, Bks, dks = context.n, context.N, context.Bks, context.dks
    d = context.d
    Qks = context.Qks
    gvector_ks = context.gvector_ks

    ksk = rand(0:Qks-1, (n + 1, N, Bks, dks))
    ksk[1, ..] .= round.(Int, context.stddev .* randn((N, Bks, dks)))
    ksk[1:1, ..] .-= sum(ksk[2:end, :, :, :] .* sk, dims=1)

    temp0 = collect(0:Bks-1)
    @einsum temp2[i, j, k] := skN[i] * temp0[j] * gvector_ks[k]

    ksk[1, ..] .+= temp2
    ksk .&= Qks - 1
    return ksk
end


function keyswitchLWE(ctLWE, kskLWE, context)
    beta, alpha = ctLWE[1], ctLWE[2:end]
    dalpha = decompose_ks(alpha, context)
    switched = zeros(Int, context.n + 1)
    switched[1] = beta

    for r in 1:context.dks
        for i in 1:context.N
            switched .+= kskLWE[:, i, dalpha[i, r]+1, r]
        end
    end

    switched .&= context.Qks - 1
    return switched
end


function nand_bootstrapping(ct0, ct1, eval_keys, context)
    ksk, brk, alphapoly = eval_keys

    N = context.N
    Q = context.Q
    q = context.q
    ctsum = @. (ct0 + ct1) & (q - 1)

    acc = zeros(Int, (N, 2))

    acc[:, 1] .= copy(context.f_nand)

    accfft = negacyclic_fft(acc, N, context.root_powers)
    beta = ctsum[1]
    xbeta = zeros(Int, N)

    if beta < N
        xbeta[beta] = 1
    else
        xbeta[beta-N+1] = -1
    end

    accfft .*= negacyclic_fft(xbeta, N, context.root_powers)


    alpha = ctsum[2:end]

    n = context.n
    for i in 1:n
        ai = alpha[i]
        partial = rgswmult(accfft, brk[i], context)
        accfft .+= alphapoly[ai] .* partial
    end

    acc = negacyclic_ifft(accfft, N, context.root_powers_inv)

    beta, alpha = extract(acc)

    beta += (Q >> 3)
    beta &= (Q - 1)
    alpha .&= (Q - 1)

    accLWE = vcat(beta, alpha)
    accLWE_ms = round.(Int, accLWE .* (context.Qks / context.Q))
    accLWE_ks = keyswitchLWE(accLWE_ms, ksk, context)

    return round.(Int, accLWE_ks .* (context.q / context.Qks))
end


function generate_context(params=(;))
    c = Context(params...)
    f_nand = c.f_nand
    for i in 1:c.N
        f_nand[i] = nand_map(-(i - 1), c)
    end
    return c
end

function generate_keys(c::Context)
    sk = keygen(c.n)
    skN = keygen(c.N)
    skNfft = negacyclic_fft(skN, c.N, c.root_powers)

    alphapoly = precompute_alpha(c)
    brk = brkgen(sk, skNfft, c)
    ksk = kskGenLWE(sk, skN, c)
    eval_keys = (ksk, brk, alphapoly)

    return sk, eval_keys
end

export generate_context, generate_keys, encryptLWE, nand_bootstrapping, decryptLWE

end
