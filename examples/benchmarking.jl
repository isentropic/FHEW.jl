using FHEW
using BenchmarkTools
c = generate_context()
sk, evk = generate_keys(c)


m0 = 0
m1 = 1

ct0 = encryptLWE(m0, c.n, c.q, sk, c.stddev)
ct1 = encryptLWE(m1, c.n, c.q, sk, c.stddev)

@benchmark nand_bootstrapping(ct0, ct1, evk, c)
ctNAND = nand_bootstrapping(ct0, ct1, evk, c)

decryptLWE(ctNAND, sk, c)


