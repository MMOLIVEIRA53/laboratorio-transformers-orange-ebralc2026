"""EBRALC 2026 - EXPERIMENTO IV, RAMO B HISTORICO.
Entrada: Create Corpus -> Python Script B.
Saidas: Data Table B e Select Rows -> Bar Plot.
Reproduz a organizacao sem variavel Grupo dos prints originais.
"""
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from Orange.data import Domain, ContinuousVariable, StringVariable, Table

MODELO = 'pierreguillou/gpt2-small-portuguese'
out_data = None
if in_data is None or len(in_data) != 2:
    raise ValueError('Conecte Create Corpus com exatamente 2 documentos ao Python Script B.')

def coluna(nome):
    variaveis = (list(in_data.domain.attributes) +
                 list(in_data.domain.class_vars) + list(in_data.domain.metas))
    return next((v for v in variaveis if v.name.casefold() == nome.casefold()), None)

col_texto = coluna('Document')
if col_texto is None:
    raise ValueError('Coluna Document nao encontrada. Nao use Preprocess Text.')
textos = [str(doc[col_texto]) for doc in in_data]
cache = globals().setdefault('_exp04_b_cache', {})
if MODELO not in cache:
    tokenizer = AutoTokenizer.from_pretrained(MODELO, local_files_only=True)
    modelo = AutoModelForCausalLM.from_pretrained(
        MODELO, local_files_only=True, use_safetensors=False)
    modelo.to('cpu').eval()
    cache[MODELO] = tokenizer, modelo
else:
    tokenizer, modelo = cache[MODELO]
WTE, WPE = modelo.transformer.wte, modelo.transformer.wpe
ids1, ids2 = [tokenizer.encode(t, add_special_tokens=False) for t in textos]
if not ids1 or len(ids1) != len(ids2):
    raise ValueError('As duas frases devem ter o mesmo numero de tokens, nao vazio.')
if len(ids1) > WPE.num_embeddings:
    raise ValueError('Comprimento maior que o limite de posicoes do modelo.')
with torch.inference_mode():
    L1 = WTE.weight[ids1].detach().cpu().numpy().copy()
    L2 = WTE.weight[ids2].detach().cpu().numpy().copy()
    P = WPE.weight[list(range(len(ids1)))].detach().cpu().numpy().copy()
C1, C2 = L1 + P, L2 + P
print('Tokens por documento:',len(ids1),len(ids2))
print('Mesmos IDs / frequencias:', sorted(ids1) == sorted(ids2))
registros = []
for palavra in ('cão','homem'):
    tid_lista = tokenizer.encode(' ' + palavra, add_special_tokens=False)
    if len(tid_lista) != 1:
        raise ValueError(f'{palavra!r} nao constitui token unico neste modelo.')
    tid = tid_lista[0]
    if ids1.count(tid) != 1 or ids2.count(tid) != 1:
        raise ValueError(f'{palavra!r} deve ocorrer uma vez por documento.')
    i, j = ids1.index(tid), ids2.index(tid)
    medidas = [('Lexical', np.linalg.norm(L1[i]-L2[j])),
               ('Posicional',np.linalg.norm(P[i]-P[j])),
               ('Combinado', np.linalg.norm(C1[i]-C2[j]))]
    for rep, distancia in medidas:
        registros.append([i,j,float(distancia),palavra,rep])
    assert np.isclose(medidas[1][1],medidas[2][1],atol=1e-5)
    print(palavra, 'ID=',tid,'posicoes=',i,j,'distancias=',[float(v) for _,v in medidas])
matrizes = float(np.linalg.norm(C1-C2))
medias = float(np.linalg.norm(C1.mean(axis=0)-C2.mean(axis=0)))
registros.extend([[-1,-1,matrizes,'Frases completas','Matrizes ordenadas'],
                  [-1,-1,medias,'Frases completas','Vetores medios']])
dominio = Domain([ContinuousVariable('Posicao_D1'),
                  ContinuousVariable('Posicao_D2'),
                  ContinuousVariable('Distancia')],
                 metas=[StringVariable('Comparacao'), StringVariable('Representacao')])
X = np.asarray([row[:3] for row in registros], dtype=float)
M = np.asarray([row[3:] for row in registros], dtype=object)
out_data = Table.from_numpy(dominio, X=X, metas=M)
assert len(out_data) == 8 and np.isfinite(X).all()
print('8 resultados | matrizes ordenadas:', f'{matrizes:.6f}',
      '| medias:', f'{medias:.6f}')
print('RAMO B CONCLUIDO')
