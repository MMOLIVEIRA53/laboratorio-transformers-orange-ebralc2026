"""EBRALC 2026 / EXP04 — RAMO A.
Entrada: Create Corpus -> Python Script. Saída: Python Script -> Data Table.
Mostra 4 coordenadas de cada vetor; normas usam TODAS as 768 coordenadas.
"""
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from Orange.data import Domain, ContinuousVariable, StringVariable, Table

MODELO = 'pierreguillou/gpt2-small-portuguese'
out_data = None
if in_data is None or len(in_data) != 2:
    raise ValueError('Conecte Create Corpus com exatamente 2 documentos ao Python Script A.')

def coluna(nome):
    variaveis = (list(in_data.domain.attributes) +
                 list(in_data.domain.class_vars) + list(in_data.domain.metas))
    return next((v for v in variaveis if v.name.casefold() == nome.casefold()), None)

col_texto = coluna('Document')
col_titulo = coluna('Title')
if col_texto is None:
    raise ValueError('Coluna Document não encontrada. Insira textos no Create Corpus.')

# Cache específico para reutilizar parâmetros enquanto o widget permanecer aberto.
cache = globals().setdefault('_exp04_a_cache', {})
if MODELO not in cache:
    tokenizer = AutoTokenizer.from_pretrained(MODELO, local_files_only=True)
    modelo = AutoModelForCausalLM.from_pretrained(
        MODELO, local_files_only=True, use_safetensors=False)
    modelo.to('cpu').eval()
    cache[MODELO] = tokenizer, modelo
else:
    tokenizer, modelo = cache[MODELO]
wte, wpe = modelo.transformer.wte, modelo.transformer.wpe
print('WTE:', tuple(wte.weight.shape), '| WPE:', tuple(wpe.weight.shape))

numeros, metas, todos_ids, matrizes = [], [], [], []
for numero, doc in enumerate(in_data, 1):
    texto = str(doc[col_texto])
    titulo = str(doc[col_titulo]) if col_titulo is not None else f'D{numero}'
    ids = tokenizer.encode(texto, add_special_tokens=False)
    if not ids or len(ids) > wpe.num_embeddings:
        raise ValueError('Sequência vazia ou excedeu o limite posicional do modelo.')
    todos_ids.append(ids)
    lexicais, posicionais, combinados = [], [], []
    print(f'\nD{numero}: {titulo} | tokens: {len(ids)}')
    with torch.inference_mode():
        for pos, ident in enumerate(ids):
            e = wte.weight[ident].detach().cpu().numpy().copy()
            p = wpe.weight[pos].detach().cpu().numpy().copy()
            c = e + p
            lexicais.append(e); posicionais.append(p); combinados.append(c)
            numeros.append([numero, pos, ident, *e[:4], *p[:4], *c[:4],
                           np.linalg.norm(e), np.linalg.norm(p), np.linalg.norm(c)])
            bruto = tokenizer.convert_ids_to_tokens(ident)
            metas.append([titulo, texto, str(bruto), tokenizer.decode([ident])])
            print(f'  posição {pos}: ID {ident} | token {bruto!r}')
    matrizes.append((np.stack(lexicais), np.stack(posicionais), np.stack(combinados)))

atributos = (['documento_num', 'posicao', 'token_id'] +
             [f'lex_{i}' for i in range(1, 5)] +
             [f'pos_{i}' for i in range(1, 5)] +
             [f'soma_{i}' for i in range(1, 5)] +
             ['norma_lex', 'norma_pos', 'norma_soma'])
dominio = Domain([ContinuousVariable(n) for n in atributos],
                 metas=[StringVariable(n) for n in ['Documento', 'Texto', 'Token', 'Token_decodificado']])
X = np.asarray(numeros, dtype=float)
out_data = Table.from_numpy(dominio, X=X, metas=np.asarray(metas, dtype=object))
assert np.isfinite(X).all(), 'Há números não finitos na tabela A.'
print('\nTabela A criada:', len(out_data), 'linhas')
if len(todos_ids[0]) == len(todos_ids[1]):
    l1, p1, c1 = matrizes[0]
    l2, p2, c2 = matrizes[1]
    print('Mesmos IDs com mesmas frequências:', sorted(todos_ids[0]) == sorted(todos_ids[1]))
    print('WPE por posição idênticos:', np.allclose(p1, p2))
    print('Distância matrizes ordenadas:', f'{np.linalg.norm(c1-c2):.6f}')
    print('Distância das médias:', f'{np.linalg.norm(c1.mean(0)-c2.mean(0)):.6f}')
print('RAMO A CONCLUÍDO')
