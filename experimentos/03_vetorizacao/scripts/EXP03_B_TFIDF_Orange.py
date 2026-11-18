# EBRALC 2026 - Ramo B de controle: Python Script A -> Python Script B
# Entrada: tabela BoW 6x18 (contagens). Saida: tabela TF-IDF 6x18.
# Protocolo historico: TF=Count, IDF=ln(N/DF), Normalization=None.
import numpy as np
from Orange.data import Table

if in_data is None:
    raise ValueError('Conecte a saida Data do EXP03_A_BoW_Orange.py.')
X = np.asarray(in_data.X, dtype=float)
if X.shape != (6, 18):
    raise ValueError(f'Esperado corpus 6x18. Recebido: {X.shape}')
DF = (X > 0).sum(axis=0)
assert np.all(DF > 0)
TFIDF = X * np.log(len(X) / DF)
out_data = Table.from_numpy(in_data.domain, TFIDF,
                            Y=in_data.Y, metas=in_data.metas)
d12 = float(np.linalg.norm(TFIDF[0] - TFIDF[1]))
d34 = float(np.linalg.norm(TFIDF[2] - TFIDF[3]))
assert abs(d12) < 1e-12
assert abs(d34 - 2.5339305418927336) < 1e-6
print('TF-IDF Count x ln(N/DF), sem normalizacao; D1-D2:', d12)
print('D3-D4:', d34)
