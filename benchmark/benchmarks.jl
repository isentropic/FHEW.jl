using FHEW
using BenchmarkTools

const SUITE = BenchmarkGroup()

c = generate_context()
sk, evk = generate_keys(c)

m0 = 0
m1 = 0

ct0 = encryptLWE(m0, c.n, c.q, sk, c.stddev)
ct1 = encryptLWE(m1, c.n, c.q, sk, c.stddev)

res = nand_bootstrapping(ct0, ct1, evk, c)

decryptLWE(res, sk, c)
nand(false, true)


@profview nand_bootstrapping(ct0, ct1, evk, c)
@benchmark nand_bootstrapping($ct0, $ct1, $evk, $c)

SUITE["nand"] = @benchmarkable nand_bootstrapping($ct0, $ct1, $evk, $c)

