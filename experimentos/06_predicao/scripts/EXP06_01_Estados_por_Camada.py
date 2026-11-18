# ================================================================
# EBRALC 2026 — AULA 2 — EXPERIMENTO VI
# ETAPA 1: ESTADOS CONTEXTUALIZADOS AO LONGO DAS 12 CAMADAS
# Entrada: Create Corpus -> Python Script
# Saida:  Python Script -> Data Table / Python Script (grafico)
# Modelo: pierreguillou/gpt2-small-portuguese
# ================================================================
 
import io
import numpy as np
import torch
from contextlib import redirect_stdout, redirect_stderr
from transformers import AutoTokenizer, AutoModelForCausalLM
from Orange.data import Domain, ContinuousVariable, StringVariable, Table
 
MODELO = "pierreguillou/gpt2-small-portuguese"
out_data = None
 
print("=" * 64)
print("EXPERIMENTO VI | ESTADOS DO TRANSFORMER POR CAMADA")
print("=" * 64)
 
if in_data is None or len(in_data) != 2:
    raise ValueError("Conecte Create Corpus com exatamente dois documentos.")
 
variaveis = (list(in_data.domain.attributes)
             + list(in_data.domain.class_vars)
             + list(in_data.domain.metas))
 
def localizar(nome):
    return next((v for v in variaveis if v.name.lower() == nome.lower()), None)
 
v_texto = localizar("Document")
v_titulo = localizar("Title")
if v_texto is None:
    raise ValueError("Nao foi encontrada a coluna Document.")
 
textos = [str(doc[v_texto]) for doc in in_data]
titulos = [str(doc[v_titulo]) if v_titulo is not None
           else "Documento " + str(i + 1)
           for i, doc in enumerate(in_data)]
 
# Reutiliza o modelo na memoria do proprio widget entre execucoes.
cache = globals().setdefault("_exp06_cache_estados", {})
if MODELO not in cache:
    print("Carregando modelo do cache local...")
    tokenizer = AutoTokenizer.from_pretrained(MODELO, local_files_only=True)
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        modelo = AutoModelForCausalLM.from_pretrained(
            MODELO, local_files_only=True, use_safetensors=False
        )
    modelo.to("cpu")
    modelo.eval()
    cache[MODELO] = (tokenizer, modelo)
else:
    tokenizer, modelo = cache[MODELO]
    modelo.eval()
    print("Reutilizando modelo em memoria.")
 
print("Camadas:", modelo.config.n_layer)
print("Dimensoes do estado:", modelo.config.n_embd)
 
ids_documentos = []
estados_documentos = []
tokens_documentos = []
 
for numero, (titulo, texto) in enumerate(zip(titulos, textos), start=1):
    ids = tokenizer.encode(texto, add_special_tokens=False)
    if not ids:
        raise ValueError("Documento vazio apos tokenizacao: " + titulo)
    if len(ids) > modelo.transformer.wpe.num_embeddings:
        raise ValueError("Limite de posicoes do modelo excedido.")
 
    entrada = torch.tensor([ids], dtype=torch.long)
    with torch.inference_mode():
        saida = modelo(
            input_ids=entrada,
            output_hidden_states=True,
            return_dict=True
        )
 
    # 13 estados: entrada (indice 0) + 12 checkpoints de blocos.
    # A ultima saida ja inclui a normalizacao final ln_f do GPT-2.
    estados = [h[0].detach().cpu().numpy().copy()
               for h in saida.hidden_states]
    if len(estados) != modelo.config.n_layer + 1:
        raise ValueError("Quantidade inesperada de estados retornados.")
 
    ids_documentos.append(ids)
    estados_documentos.append(estados)
    tokens = [tokenizer.decode([token_id]) for token_id in ids]
    tokens_documentos.append(tokens)
 
    print("\nDocumento", numero, "|", titulo)
    print("Texto:", texto)
    print("Tokens:", len(ids), "| Estados:", len(estados))
    for pos, (token_id, token) in enumerate(zip(ids, tokens)):
        print("  pos=%02d | id=%05d | %r" % (pos, token_id, token))
 
if len(ids_documentos[0]) != len(ids_documentos[1]):
    raise ValueError("Compare duas frases com a mesma quantidade de tokens.")
 
n = len(ids_documentos[0])
print("\nMesmos IDs (com as mesmas frequencias):",
      sorted(ids_documentos[0]) == sorted(ids_documentos[1]))
 
linhas = []
metas = []
 
for camada in range(len(estados_documentos[0])):
    H1 = estados_documentos[0][camada]
    H2 = estados_documentos[1][camada]
    for pos in range(n):
        a, b = H1[pos], H2[pos]
        d = float(np.linalg.norm(a - b))
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        denom = norm_a * norm_b
        cosseno = float(np.dot(a, b) / denom) if denom > 0 else np.nan
        linhas.append([camada, pos, d, cosseno, norm_a, norm_b])
        metas.append([
            "Entrada (WTE+WPE)" if camada == 0
            else ("Camada %d (saida final com ln_f)" % camada
                  if camada == modelo.config.n_layer
                  else "Apos camada %d" % camada),
            tokens_documentos[0][pos],
            tokens_documentos[1][pos],
            titulos[0], titulos[1]
        ])
 
atributos = [ContinuousVariable(nome) for nome in [
    "Camada", "Posicao", "Distancia", "Cosseno", "Norma_D1", "Norma_D2"
]]
dominio = Domain(atributos, metas=[
    StringVariable("Etapa"),
    StringVariable("Token_D1"),
    StringVariable("Token_D2"),
    StringVariable("Documento_1"),
    StringVariable("Documento_2")
])
X = np.array(linhas, dtype=float)
M = np.array(metas, dtype=object)
out_data = Table.from_numpy(dominio, X=X, metas=M)
 
print("\n" + "=" * 64)
print("CONTROLES DE INTERPRETACAO")
print("=" * 64)
print("A posicao 0 tem o mesmo token:",
      ids_documentos[0][0] == ids_documentos[1][0])
print("Distancia da posicao 0 em todas as camadas:",
      [round(float(np.linalg.norm(e1[0] - e2[0])), 6)
       for e1, e2 in zip(*estados_documentos)])
 
for pos in [0, 2, 3, n - 1]:
    if pos >= n:
        continue
    print("\nPosicao", pos,
          "| D1:", repr(tokens_documentos[0][pos]),
          "| D2:", repr(tokens_documentos[1][pos]))
    for camada in range(len(estados_documentos[0])):
        a = estados_documentos[0][camada][pos]
        b = estados_documentos[1][camada][pos]
        print("  estado=%02d | distancia=%.6f" %
              (camada, float(np.linalg.norm(a - b))))
 
print("\nLinhas enviadas ao Orange:", len(out_data))
print("Valores finitos nas distancias:", bool(np.isfinite(X[:, 2]).all()))
print("Experimento VI — etapa 1 concluida.")
