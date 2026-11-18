"""Subexperimento de controle do Exp. III (sem Orange e sem checkpoint).
Dados literais dos seis enunciados e protocolo Regexp \\w+ / TF Count / IDF ln(N/DF).
O resultado numérico aqui é um NOVO teste local determinístico, não uma nova execução no Orange.
"""
import re
import math
from collections import Counter

TEXTOS = [
    'O cão mordeu o homem.',
    'O homem mordeu o cão.',
    'A manga da camisa ficou curta.',
    'A manga da camisa ficou comprida.',
    'Na feira, comprei uma manga madura.',
    'Na feira, comprei uma banana madura.',
]


def calcular(textos=TEXTOS):
    tokens = [re.findall(r'\w+', texto, flags=re.UNICODE) for texto in textos]
    vocabulario = sorted({t for seq in tokens for t in seq})
    contagens = [Counter(seq) for seq in tokens]
    bow = [[contagem[t] for t in vocabulario] for contagem in contagens]
    n = len(tokens)
    df = [sum(t in contagem for contagem in contagens) for t in vocabulario]
    idf = [math.log(n / freq) for freq in df]
    tfidf = [[valor * peso for valor, peso in zip(linha, idf)] for linha in bow]

    def d(a, b):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    return {
        'tokens': tokens, 'vocabulario': vocabulario, 'bow': bow,
        'tfidf': tfidf, 'n_tokens': sum(map(len, tokens)),
        'n_types': len(vocabulario), 'bow_d1_d2': d(bow[0], bow[1]),
        'bow_d3_d4': d(bow[2], bow[3]),
        'tfidf_d1_d2': d(tfidf[0], tfidf[1]),
        'tfidf_d3_d4': d(tfidf[2], tfidf[3]),
    }


if __name__ == '__main__':
    res = calcular()
    assert res['n_tokens'] == 34 and res['n_types'] == 18
    assert res['bow_d1_d2'] == res['tfidf_d1_d2'] == 0
    assert abs(res['bow_d3_d4'] - math.sqrt(2)) < 1e-12
    assert abs(res['tfidf_d3_d4'] - 2.5339305418927336) < 1e-10
    for key in ['n_tokens','n_types','bow_d1_d2','bow_d3_d4','tfidf_d1_d2','tfidf_d3_d4']:
        print(f'{key}: {res[key]}')
