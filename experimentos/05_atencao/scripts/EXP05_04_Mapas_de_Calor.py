# EBRALC 2026 | Experimento V | Script 4
# Entrada: saida do Script 3 -> Python Script
# Saida: figuras Matplotlib e matriz tabular de cada documento
import numpy as np
import matplotlib.pyplot as plt
from Orange.data import Domain, ContinuousVariable, StringVariable, Table
out_data = None
if in_data is None:
   raise ValueError('Conecte a saida do Script 3 (pesos reais).')
vd, vq, vk = [in_data.domain[x] for x in ['Documento_num', 'Query_pos', 'Key_pos']]
vp, vt = in_data.domain['Peso_atencao'], in_data.domain['Documento']
vtq, vtk = in_data.domain['Token_query'], in_data.domain['Token_key']
docs = sorted({int(row[vd]) for row in in_data})
linhas, metas, comprimentos = [], [], []
print('=== EXPERIMENTO V | MAPAS DE CALOR ===')
for numero in docs:
   rows = [row for row in in_data if int(row[vd]) == numero]
   titulo = str(rows[0][vt])
   n = 1 + max(max(int(row[vq]), int(row[vk])) for row in rows)
   A = np.full((n, n), np.nan)
   tq, tk = [''] * n, [''] * n
   for row in rows:
      i, j = int(row[vq]), int(row[vk])
      A[i, j] = float(row[vp])
      tq[i], tk[j] = str(row[vtq]), str(row[vtk])
   if not np.isfinite(A).all():
      raise ValueError('Pesos incompletos: ' + titulo)
   print('\n', titulo, '\n', np.round(A, 4))
   print('Somatorio por linha:', np.round(A.sum(axis=1), 6))
   fig, ax = plt.subplots(figsize=(9, 7))
   im = ax.imshow(A, cmap='YlGnBu', vmin=0, vmax=1, interpolation='nearest')
   ax.set_title('Self-attention — ' + titulo + '\nCamada 1 / Cabeca 1')
   ax.set_xlabel('Key — posicao consultada')
   ax.set_ylabel('Query — posicao que consulta')
   ax.set_xticks(range(n)); ax.set_yticks(range(n))
   ax.set_xticklabels(tk, rotation=45, ha='right')
   ax.set_yticklabels(tq)
   for i in range(n):
      for j in range(n):
          ax.text(j, i, f'{A[i,j]:.2f}', ha='center', va='center',
               fontsize=9, color='white' if A[i,j] > .55 else 'black')
   fig.colorbar(im, ax=ax, label='Peso de atencao', fraction=.046, pad=.04)
   fig.tight_layout()
   plt.show()
   comprimentos.append(n)
   for i in range(n):
      linhas.append([numero, i, *A[i]])
      metas.append([titulo, tq[i]])
if len(set(comprimentos)) != 1:
   raise ValueError('Comprimentos diferentes: processe os documentos separadamente.')
n = comprimentos[0]
dominio = Domain([ContinuousVariable(x) for x in
   ['Documento_num', 'Query_pos'] + [f'Key_{j}' for j in range(n)]],
   metas=[StringVariable('Documento'), StringVariable('Token_query')])
out_data = Table.from_numpy(dominio, X=np.asarray(linhas, dtype=float),



               metas=np.asarray(metas, dtype=object))
print('Linhas de saida:', len(out_data))
