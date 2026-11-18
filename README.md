# Seis experimentos de Transformers com Orange Data Mining — EBRALC 2026

**Material didático e experimental do minicurso** *Como os Grandes Modelos de Linguagem representam a linguagem? Uma introdução aos Transformers para linguistas com Orange Data Mining*.

**Autor e ministrante:** Marcos Moisés Crisóstomo de Oliveira — Instituto Federal da Bahia (IFBA), PPGEL/UNEB.  
**Contato:** marcosoliveira@ifba.edu.br  
**ORCID:** https://orcid.org/0000-0001-5291-6000

Este repositório disponibiliza **somente os seis experimentos computacionais**. Não contém o artigo-tutorial destinado aos anais do ELC/EBRALC nem o Volume VII de leitura científica.

## Conteúdo

| Experimento | Tema | Materiais disponíveis |
|---|---|---|
| [I — Completação](experimentos/01_completacao/) | Completação textual e predição | Manual PDF e figura de apoio |
| [II — Tokenização](experimentos/02_tokenizacao/) | Tokens, IDs e vocabulário | Manual PDF e captura do Orange |
| [III — Vetorização](experimentos/03_vetorizacao/) | BoW, TF-IDF e embeddings | Manual, três scripts, microcorpus e figura |
| [IV — Posicionamento](experimentos/04_posicionamento/) | WTE, WPE e ordem | Manual, dois scripts, slides e figura |
| [V — Atenção](experimentos/05_atencao/) | Q/K/V, máscara causal, pesos | Manual, quatro scripts, slides e figura |
| [VI — Predição](experimentos/06_predicao/) | Estados por camada, logits e softmax | Manual, quatro scripts, slides e figuras |

Os códigos dos Experimentos I e II estão documentados nos seus manuais. Não foram localizados arquivos `.py` independentes desses dois volumes no pacote preparado.

## Como começar

1. Instale o **Orange Data Mining** e, quando for necessário, o complemento **Orange Text**; verifique a disponibilidade do widget **Python Script**.
2. Abra `experimentos/01_completacao/manual.pdf` e siga a coleção na ordem I → VI.
3. Para os Experimentos III a VI, copie os scripts da respectiva pasta `scripts/` para os widgets indicados no manual.
4. Ao trabalhar com o GPT-2 em português, use o checkpoint indicado nos manuais: `pierreguillou/gpt2-small-portuguese`. A instalação e o cache local do modelo dependem da configuração efetiva do computador.
5. Confira as saídas com os registros apresentados nos manuais, registrando diferenças de versões e parâmetros.

Para um controle independente do Orange, o arquivo `experimentos/03_vetorizacao/scripts/EXP03_controle_reproduzivel.py` utiliza Python padrão. Ele verifica operações matemáticas específicas, **não** a execução completa de um workflow no Orange.

## Reprodutibilidade e limites

- As capturas e os valores numéricos pertencem a execuções históricas documentadas no minicurso.
- A verificação de sintaxe de um arquivo Python não equivale à execução do checkpoint no Orange.
- **Não há workflows `.ows` validados** neste conjunto. Leia `workflows/README.md`.
- Não há ambiente histórico congelado ou hash de revisão do checkpoint comprovado. Para registrar novas instalações, utilize `ferramentas/diagnostico_ambiente.py`.
- As figuras de apoio podem incluir recortes e ampliações; o manual contém a documentação interpretativa.

## Direitos autorais

Os materiais didáticos autorais pertencem ao autor. Nenhuma licença de reutilização foi escolhida neste pacote. Antes de tornar o repositório público, revise os direitos de quaisquer imagens ou trechos de terceiros e decida se deseja aplicar uma licença. O modelo do Hugging Face, o Orange e suas bibliotecas têm licenças próprias.

## Referência provisória do material

Oliveira, Marcos Moisés Crisóstomo de (2026). *Seis experimentos de Transformers com Orange Data Mining: materiais do minicurso ELC/EBRALC 2026*. Repositório de materiais didáticos (endereço a definir após publicação).
