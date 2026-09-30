# 17 · Laboratório de geração: schemas de chat, Verbalized Sampling, "nunca…" como identidade e cabeçalho de roleplay

> Prefixo `c1`. Scripts em `scripts/analysis/c1_*.py`; saídas pequenas em `analysis/data/c1_*`; gerações e avaliações
> brutas em `data/processed/c1_*` (fora do git).
> **Status: RASCUNHO em andamento.** Este arquivo é reescrito à medida que os experimentos terminam.

## Desenho (resumo)

- **Pontos:** os 179 pontos de decisão reais do maichat do a9/b4 (dev = 60 pontos em 6 conversas, teste = 119 pontos em
  35 conversas; a divisão é por conversa). Cada bolha do histórico entra com o seu **timestamp real**.
- **Atores:** `inception/mercury-2.5`, `~deepseek/deepseek-flash-latest`, `openai/gpt-6-luna`, `google/gemini-3.5-flash-lite`.
- **Métricas:** D do relatório 13 (erro de tamanho + Σ|taxa − taxa humana| em 10 vícios); bolhas e intervalos propostos ×
  reais; invenção da fala do usuário; formato quebrado; coincidência de movimento (Choice do Jev sobre a resposta × sobre a
  resposta humana, mesmo formato de chamada); coerência (Noul do Jev); escore "código + banco atômico do Jev"
  (regressão logística do relatório 10, com cross-fitting por conversa; P(LLM), ↓ = mais humano).
- **Experimentos:** (1) 7 schemas × com/sem nota do diretor; (2) Verbalized Sampling N = 3 e 5 contra 4 seeds, com 5
  estratégias de escolha; (3) restrições como identidade × imperativo × nada, em 20 fichas e 75 contextos que tentam o
  personagem; (4) cabeçalho de roleplay × persona mínima; (5) pipeline combinado, validado no teste.

(resultados a seguir, preenchidos conforme cada etapa termina)
