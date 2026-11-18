# EBRALC 2026 | Experimento V | Script 3
# Entrada: Create Corpus -> Python Script (ramo independente)
# Saida: 98 pares Query-Key com PESOS REAIS da camada 1/cabeca 1
# Reproducao dos pesos de atencao na primeira camada do GPT-2.
import io
import numpy as np
import torch
from contextlib import redirect_stdout, redirect_stderr
from transformers import AutoTokenizer, AutoModelForCausalLM
from Orange.data import Domain, ContinuousVariable, StringVariable, Table


MODELO = 'pierreguillou/gpt2-small-portuguese'
CAMADA = 0
CABECA = 0
out_data = None
if in_data is None or len(in_data) != 2:
   raise ValueError('Conecte Create Corpus com exatamente dois documentos.')


def campo(nome):
  for v in list(in_data.domain.attributes) + list(in_data.domain.class_vars) + list(in_data.domain.metas):
     if v.name.lower() == nome.lower():
        return v
  return None


v_texto, v_titulo = campo('Document'), campo('Title')
if v_texto is None:
   raise ValueError('Coluna Document nao encontrada.')
cache = globals().setdefault('_exp05_modelo_real', {})
if MODELO not in cache:
   tokenizer = AutoTokenizer.from_pretrained(MODELO, local_files_only=True)
   with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
     modelo = AutoModelForCausalLM.from_pretrained(
        MODELO, local_files_only=True, use_safetensors=False)
   modelo.to('cpu').eval()
   cache[MODELO] = (tokenizer, modelo)
else:
   tokenizer, modelo = cache[MODELO]
   modelo.eval()


cfg = modelo.config
n_heads, dim = cfg.n_head, cfg.n_embd
head_dim = dim // n_heads
bloco = modelo.transformer.h[CAMADA]
attn = bloco.attn
if not (0 <= CABECA < n_heads):
   raise ValueError('Cabeca fora dos limites.')
print('=== EXPERIMENTO V | SELF-ATTENTION REAL ===')
print('Camadas:', cfg.n_layer, '| Cabecas:', n_heads,
    '| Dimensoes:', dim, '| Dimensoes/cabeca:', head_dim)


linhas, metas, matrizes = [], [], []
for numero, doc in enumerate(in_data, start=1):
   texto = str(doc[v_texto])
   titulo = str(doc[v_titulo]) if v_titulo is not None else f'Documento {numero}'
   ids = tokenizer.encode(texto, add_special_tokens=False)
   n = len(ids)
   if not n or n > modelo.transformer.wpe.num_embeddings:
      raise ValueError('Comprimento de sequencia invalido.')
   inp = torch.tensor([ids], dtype=torch.long)
   pos = torch.arange(n, dtype=torch.long).unsqueeze(0)
   with torch.no_grad():
      e = modelo.transformer.wte(inp) + modelo.transformer.wpe(pos)
      h = bloco.ln_1(modelo.transformer.drop(e))
      q, k, v = torch.split(attn.c_attn(h), dim, dim=-1)
      def separar(t):
          return t.reshape(1, n, n_heads, head_dim).transpose(1, 2)
      q, k, v = separar(q), separar(k), separar(v)
      scores = torch.matmul(q, k.transpose(-1, -2)).float()
      if getattr(attn, 'scale_attn_weights', True):
          scores = scores / np.sqrt(head_dim)
      if getattr(attn, 'scale_attn_by_inverse_layer_idx', False):
          scores = scores / (CAMADA + 1)
      mascara = torch.tril(torch.ones((n, n), dtype=torch.bool))
      scores = scores.masked_fill(~mascara[None, None], torch.finfo(scores.dtype).min)
      pesos = torch.softmax(scores, dim=-1)
      A = pesos[0, CABECA].detach().cpu().numpy().copy()
      contexto = torch.matmul(pesos[:, CABECA], v[:, CABECA].float())
   tokens = [tokenizer.decode([x]) for x in ids]
   matrizes.append(A)
   print('\nDOCUMENTO:', titulo, '| Tokens:', tokens)
   print('MATRIZ:', np.round(A, 4))
   print('Somas por linha:', np.round(A.sum(axis=1), 6))
   print('Pesos futuros zerados:', np.all(A[np.triu_indices(n, 1)] == 0))
   print('Linhas somam 1:', np.allclose(A.sum(axis=1), 1, atol=1e-5))
   if n > 3:
      print('Atencao de posicao 3:', np.round(A[3], 6))
   print('Atencao da ultima posicao:', np.round(A[-1], 6))
   print('Norma de contexto da cabeca na ultima posicao:',
        float(torch.linalg.vector_norm(contexto[0, -1]).item()))
   for i in range(n):
      for j in range(n):
          linhas.append([numero, i, j, float(mascara[i, j]), float(A[i, j])])
          metas.append([titulo, tokens[i], tokens[j]])


if len(matrizes) == 2 and matrizes[0].shape == matrizes[1].shape:
   print('\nDistancia de Frobenius entre matrizes:',
      float(np.linalg.norm(matrizes[0] - matrizes[1])))
dominio = Domain([ContinuousVariable(x) for x in
   ['Documento_num', 'Query_pos', 'Key_pos', 'Permitida', 'Peso_atencao']],
   metas=[StringVariable(x) for x in ['Documento', 'Token_query', 'Token_key']])
out_data = Table.from_numpy(dominio, X=np.asarray(linhas, dtype=float),
                 metas=np.asarray(metas, dtype=object))
print('Pares enviados ao Orange:', len(out_data))
