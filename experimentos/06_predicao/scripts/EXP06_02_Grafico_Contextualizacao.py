# ================================================================
# EBRALC 2026 — EXPERIMENTO VI — ETAPA 2
# VISUALIZACAO DAS DISTANCIAS ENTRE ESTADOS POR CAMADA
# Entrada: Python Script (etapa 1) -> Python Script (1)
# Saida:  Python Script (1) -> Data Table (1), mais grafico Matplotlib
# Nao carrega novamente o modelo: usa os dados da etapa 1.
# ================================================================
 
import numpy as np
import matplotlib.pyplot as plt
 
out_data = None
if in_data is None:
    raise ValueError("Conecte a saida do script de estados por camada.")
 
v_camada = in_data.domain["Camada"]
v_pos = in_data.domain["Posicao"]
v_distancia = in_data.domain["Distancia"]
v_t1 = in_data.domain["Token_D1"]
v_t2 = in_data.domain["Token_D2"]
 
registros = []
for linha in in_data:
    registros.append({
        "camada": int(linha[v_camada]),
        "posicao": int(linha[v_pos]),
        "distancia": float(linha[v_distancia]),
        "token_1": str(linha[v_t1]).strip(),
        "token_2": str(linha[v_t2]).strip()
    })
 
camadas = sorted(set(r["camada"] for r in registros))
posicoes = sorted(set(r["posicao"] for r in registros))
ultima = max(posicoes)
foco = [p for p in [0, 2, 3, ultima] if p in posicoes]
foco = list(dict.fromkeys(foco))
 
fig, ax = plt.subplots(figsize=(9, 5.2))
for pos in foco:
    selecionados = sorted(
        [r for r in registros if r["posicao"] == pos],
        key=lambda r: r["camada"]
    )
    valores = [r["distancia"] for r in selecionados]
    r0 = selecionados[0]
    rotulo = ("Pos %d: %s / %s" %
              (pos, r0["token_1"] or "[espaco]",
               r0["token_2"] or "[espaco]"))
    ax.plot(camadas, valores, marker="o", markersize=4, label=rotulo)
 
ax.set_title("Mudanca das diferencas entre D1 e D2 ao longo do Transformer")
ax.set_xlabel("Estado 0 = entrada; estados 1–12 = processamento")
ax.set_ylabel("Distancia euclidiana entre estados da mesma posicao")
ax.set_xticks(camadas)
ax.grid(alpha=0.2)
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()
 
out_data = in_data
print("Grafico gerado para posicoes:", foco)
print("Lembrete: as escalas podem mudar com a normalizacao entre camadas.")
print("Uma distancia maior nao significa maior compreensao linguistica.")
