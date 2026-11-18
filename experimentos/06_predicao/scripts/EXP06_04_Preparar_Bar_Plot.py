# EXP06_04 — PREPARAR BAR PLOT POR FRASE
# Entrada: Select Rows -> Python Script; Saida -> Bar Plot
import numpy as np
from Orange.data import Domain, DiscreteVariable, Table
out_data = None
if in_data is None: raise ValueError("Conecte Matching Data do Select Rows.")
v = in_data.domain["Documento"]
rotulos = [str(r[v]) for r in in_data]
nomes = list(dict.fromkeys(rotulos))
frase = DiscreteVariable("Frase", values=nomes)
base = in_data.domain
novo = Domain(list(base.attributes)+[frase], list(base.class_vars), list(base.metas))
X = np.column_stack([in_data.X, np.array([nomes.index(s) for s in rotulos],dtype=float)])
out_data = Table.from_numpy(novo, X=X, Y=in_data.Y, metas=in_data.metas)
print("Frases disponiveis para Group by/Color:", nomes)
