# ================================================================
# EBRALC 2026 — AULA 2 — EXPERIMENTO VI — ETAPA 3
# ESTADO FINAL -> LOGITS -> SOFTMAX -> PROXIMO TOKEN
# Entrada: Create Corpus -> Python Script (2) [ramo independente]
# Saida:  Python Script (2) -> Data Table (2) -> Bar Plot
# Analisa a previsao do token seguinte ao texto completo.
# ================================================================
 
import io
import numpy as np
import torch
from contextlib import redirect_stdout, redirect_stderr
from transformers import AutoTokenizer, AutoModelForCausalLM
from Orange.data import Domain, ContinuousVariable, StringVariable, Table
 
MODELO = "pierreguillou/gpt2-small-portuguese"
TOP_K = 10
out_data = None
 
if in_data is None or len(in_data) != 2:
    raise ValueError("Conecte Create Corpus com exatamente dois documentos.")
 
vars_corpus = (list(in_data.domain.attributes)
               + list(in_data.domain.class_vars)
               + list(in_data.domain.metas))
 
def localizar(nome):
    return next((v for v in vars_corpus if v.name.lower() == nome.lower()), None)
 
v_texto = localizar("Document")
v_titulo = localizar("Title")
if v_texto is None:
    raise ValueError("Nao foi encontrada a coluna Document.")
 
cache = globals().setdefault("_exp06_cache_logits", {})
if MODELO not in cache:
    print("Carregando GPorTuguese-2 do cache local...")
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
 
print("=" * 64)
print("EXPERIMENTO VI | DO ESTADO FINAL AS PROBABILIDADES")
print("=" * 64)
 
linhas = []
metas = []
distribuicoes = []
 
for numero, documento in enumerate(in_data, start=1):
    texto = str(documento[v_texto])
    titulo = (str(documento[v_titulo]) if v_titulo is not None
              else "Documento " + str(numero))
 
    ids = tokenizer.encode(texto, add_special_tokens=False)
    if not ids:
        raise ValueError("Documento sem tokens: " + titulo)
    if len(ids) > modelo.transformer.wpe.num_embeddings:
        raise ValueError("Limite posicional excedido.")
 
    entrada = torch.tensor([ids], dtype=torch.long)
    with torch.inference_mode():
        resultado = modelo(
            input_ids=entrada,
            output_hidden_states=True,
            return_dict=True
        )
        estado_final = resultado.hidden_states[-1][0, -1]
        logits = resultado.logits[0, -1].float()
        # Checagem: a mesma camada de saida e aplicada ao estado final.
        logits_recalculados = modelo.lm_head(estado_final).float()
        conferiu = torch.allclose(
            logits, logits_recalculados, atol=1e-4, rtol=1e-4
        )
        probabilidades = torch.softmax(logits, dim=-1)
        k = min(TOP_K, probabilidades.numel())
        maiores_probs, maiores_ids = torch.topk(probabilidades, k=k)
        escolhido = int(torch.argmax(probabilidades).item())
 
    distribuicoes.append(probabilidades.detach().cpu().numpy().copy())
    proximo_token = tokenizer.decode([escolhido])
    texto_extendido = tokenizer.decode(ids + [escolhido])
 
    print("\nDocumento", numero, "|", titulo)
    print("Entrada:", repr(texto))
    print("Tokens de entrada:", len(ids))
    print("Dimensoes do estado final:", estado_final.numel())
    print("Quantidade de logits:", logits.numel())
    print("Logits conferidos com lm_head:", bool(conferiu))
    print("Soma das probabilidades:", float(probabilidades.sum().item()))
    print("\nTop", k, "candidatos (probabilidades sobre TODO o vocabulario):")
 
    for posto, (prob, token_id_tensor) in enumerate(
        zip(maiores_probs, maiores_ids), start=1
    ):
        token_id = int(token_id_tensor.item())
        prob_float = float(prob.item())
        token_texto = tokenizer.decode([token_id])
        token_interno = tokenizer.convert_ids_to_tokens(token_id)
        logit = float(logits[token_id].item())
        print(" %2d | id=%5d | token=%-20r | logit=%9.4f | p=%10.6f%%"
              % (posto, token_id, token_texto, logit,
                 100.0 * prob_float))
        linhas.append([
            numero, len(ids), posto, token_id,
            logit, prob_float, 100.0 * prob_float
        ])
        metas.append([titulo, texto, str(token_interno),
                      repr(token_texto), repr(proximo_token)])
 
    print("Massa de probabilidade do Top %d: %.4f%%" %
          (k, float(maiores_probs.sum().item() * 100)))
    print("Selecao GREEDY (maior probabilidade):", repr(proximo_token))
    print("Texto apos um token:", repr(texto_extendido))
 
if len(distribuicoes) == 2:
    # Distancia de variacao total: distribuicoes completas sobre vocabulario.
    tv = 0.5 * float(np.abs(
        distribuicoes[0] - distribuicoes[1]
    ).sum())
    print("\nDistancia de variacao total das distribuicoes:",
          "%.6f" % tv)
    print("0 = distribuicoes iguais; 1 = disjuntas.")
 
atributos = [ContinuousVariable(nome) for nome in [
    "Documento_num", "Proxima_posicao", "Rank", "Token_id",
    "Logit", "Probabilidade", "Probabilidade_pct"
]]
dominio = Domain(atributos, metas=[
    StringVariable("Documento"),
    StringVariable("Input"),
    StringVariable("Token_interno"),
    StringVariable("Token_visivel"),
    StringVariable("Token_greedy")
])
out_data = Table.from_numpy(
    dominio,
    X=np.array(linhas, dtype=float),
    metas=np.array(metas, dtype=object)
)
 
print("\nTabela do Orange:", len(out_data), "candidatos.")
print("Cada probabilidade foi calculada sobre o vocabulario inteiro.")
print("A soma apenas dos candidatos Top K normalmente nao e 100%.")
print("Experimento VI — etapa 3 concluida.")
