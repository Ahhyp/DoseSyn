# M6-A 结构差异

| name | params | dose | fusion |
| --- | --- | --- | --- |
| T0 b3_exact | 284545 | standardized scalar + zero flag | ordered concat(cell, ha, hb, da, db) |
| T1 c0_control | 559299 | Linear(2,64) dose embedding | ordered concat(cell, ha, hb, ea64, eb64) |
| T2 b3sym_scalar | 429954 | raw scalar + zero flag | sum/product/|diff| + raw dose |
| T3 b3sym_fourier | 492290 | Fourier embedding 64d | sum/product/|diff| + raw dose |
| T4 b3sym_film | 535682 | Fourier + FiLM(gamma,beta) | FiLM then sum/product/|diff| |
