# EBRALC 2026 - EXPERIMENTO III - roteiro de controle reexecutável no Orange.
# Widget Python Script SEM entrada obrigatória -> Data Table / Distances.
# Critério: 6 documentos, 34 ocorrências, 18 types; d(D1,D2)=0 no BoW.
# O código é uma transcrição autônoma do protocolo dos manuais, NÃO
# o script histórico originalmente usado pelo ministrante.
import re
import numpy as np
from collections import Counter
from Orange.data import Domain, ContinuousVariable, StringVariable, Table

TEXTOS = [
    'O cão mordeu o homem.',
    'O homem mordeu o cão.',
    'A manga da camisa ficou curta.',
    'A manga da camisa ficou comprida.',
    'Na feira, comprei uma manga madura.',
    'Na feira, comprei uma banana madura.',
]
TOKENS = [re.findall(r'\w+', texto, flags=re.UNICODE) for texto in TEXTOS]
VOCAB = sorted(set(token for seq in TOKENS for token in seq))
X = np.array([[Counter(seq)[token] for token in VOCAB] for seq in TOKENS],
             dtype=float)
DOM = Domain([ContinuousVariable(token) for token in VOCAB],
             metas=[StringVariable('Documento'), StringVariable('Texto')])
META = np.array([[f'D{i+1}', texto] for i, texto in enumerate(TEXTOS)],
                dtype=object)
out_data = Table.from_numpy(DOM, X, metas=META)
d12 = float(np.linalg.norm(X[0] - X[1]))
assert len(VOCAB) == 18 and sum(map(len, TOKENS)) == 34
assert X.shape == (6, 18) and abs(d12) < 1e-12
print('BoW | 6 linhas x 18 atributos; Regexp=\\w+; case preservado')
print('Ocorrencias:', sum(map(len, TOKENS)), 'Types:', len(VOCAB))
print('Distancia euclidiana D1-D2:', d12)
