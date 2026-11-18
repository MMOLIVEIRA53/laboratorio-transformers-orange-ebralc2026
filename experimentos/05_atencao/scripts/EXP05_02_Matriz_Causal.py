# EBRALC 2026 | Experimento V | Script 2
# Entrada: Script 1 -> Python Script
# Saida: duas matrizes de 7 x 7 em formato tabular (14 linhas)
import numpy as np
from Orange.data import Domain, ContinuousVariable, StringVariable, Table
out_data = None
if in_data is None:
   raise ValueError('Conecte a saida do Script 1.')


def var(nome):
  return in_data.domain[nome]


vd, vq, vk = [var(x) for x in ['documento_num', 'posicao_query', 'posicao_key']]
vp, vt = var('permitida'), var('Documento')
numeros = sorted({int(row[vd]) for row in in_data})
matrizes, registros, metas = [], [], []
print('=== EXPERIMENTO V | MATRIZES CAUSAIS ===')
for numero in numeros:
   rows = [row for row in in_data if int(row[vd]) == numero]
   titulo = str(rows[0][vt])
   n = 1 + max(max(int(row[vq]), int(row[vk])) for row in rows)
   matriz = np.full((n, n), np.nan)
   for row in rows:
      matriz[int(row[vq]), int(row[vk])] = float(row[vp])
   if not np.isfinite(matriz).all():
      raise ValueError('Matriz incompleta: ' + titulo)
   matrizes.append(matriz)
   print('\n', titulo, '\n', matriz.astype(int))
   for i in range(n):
      registros.append([numero, i, *matriz[i]])
      metas.append([titulo])
if len({m.shape[1] for m in matrizes}) != 1:
   raise ValueError('Documentos de diferentes comprimentos: processe separadamente.')
n = matrizes[0].shape[1]
dominio = Domain([ContinuousVariable(x) for x in
   ['Documento_num', 'Query_pos'] + [f'Key_{i}' for i in range(n)]],
   metas=[StringVariable('Documento')])
out_data = Table.from_numpy(dominio, X=np.asarray(registros, dtype=float),
                   metas=np.asarray(metas, dtype=object))
print('Linhas enviadas:', len(out_data))
