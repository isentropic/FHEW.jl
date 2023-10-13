using FHEW
using Test, BenchmarkTools

@testset "LWE keyswitching" begin
    c = generate_context()
    sk = FHEW.keygen(c.n)
    skN = FHEW.keygen(c.N)
    skNfft = FHEW.negacyclic_fft(skN, c.N, c.root_powers)

    ksk = FHEW.kskGenLWE(sk, skN, c)
    for i in 1:10
        for msg in 0:3
            ctLWE = encryptLWE(msg, c.N, c.Qks, skN, c.stddev)
            switched = FHEW.keyswitchLWE(ctLWE, ksk, c)
            mdec = decryptLWE(switched, sk, c.Qks)

            @test msg == mdec
        end
    end
end