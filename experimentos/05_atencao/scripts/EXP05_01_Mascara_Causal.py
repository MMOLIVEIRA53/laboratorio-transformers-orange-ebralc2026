# EBRALC 2026 | Experimento V | Script 1
# Entrada: Create Corpus -> Python Script
# Saida: 98 pares Query-Key em out_data
import numpy as np
from transformers import AutoTokenizer
from Orange.data import Domain, ContinuousVariable, StringVariable, Table


MODELO = 'pierreguillou/gpt2-small-portuguese'
out_data = None
if in_data is None:
   raise ValueError('Conecte Create Corpus diretamente a este Python Script.')


def campo(nome):
  for v in list(in_data.domain.attributes) + list(in_data.domain.class_vars) + list(in_data.domain.metas):
     if v.name.lower() == nome.lower():
        return v
  return None


v_texto, v_titulo = campo('Document'), campo('Title')
if v_texto is None:
   raise ValueError('A coluna Document nao foi encontrada.')


cache = globals().setdefault('_exp05_tokenizer', {})
if MODELO not in cache:
   cache[MODELO] = AutoTokenizer.from_pretrained(MODELO, local_files_only=True)
tokenizer = cache[MODELO]


linhas, metas = [], []
print('=== EXPERIMENTO V | MASCARA CAUSAL ===')
for numero, doc in enumerate(in_data, start=1):
   texto = str(doc[v_texto])
   titulo = str(doc[v_titulo]) if v_titulo is not None else f'Documento {numero}'
   ids = tokenizer.encode(texto, add_special_tokens=False)
   if not ids:
      raise ValueError('Documento sem tokens: ' + titulo)
   tokens = [tokenizer.decode([x]) for x in ids]
   n = len(ids)
   mascara = np.tril(np.ones((n, n), dtype=int))
   print('\n', titulo, '| IDs:', ids, '| tokens:', tokens)
   print(mascara)
   for i in range(n):
      print(f'Query {i} consulta posicoes {list(range(i + 1))}')
      for j in range(n):
        linhas.append([numero, i, j, int(mascara[i, j])])
        metas.append([titulo, texto, tokens[i], tokens[j]])


dominio = Domain([ContinuousVariable(x) for x in
  ['documento_num', 'posicao_query', 'posicao_key', 'permitida']],
  metas=[StringVariable(x) for x in
  ['Documento', 'Texto', 'Token_query', 'Token_key']])
out_data = Table.from_numpy(dominio, X=np.asarray(linhas, dtype=float),
                metas=np.asarray(metas, dtype=object))
print('Total de pares Query-Key:', len(out_data))
