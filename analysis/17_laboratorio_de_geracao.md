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

## Achados parciais (DEV, 60 pontos; serão substituídos pelos números do teste)

- **Schemas sem nota do diretor (dev):** trocar "persona + mensagens" por um log de exportação corta o D da lite de 3,90
  para 1,30 (Messenger JSON) e o do luna de 3,68 para 1,48, sem nenhuma instrução de estilo. No deepseek e no mercury o
  ganho é menor (1,64 → 1,37–1,45; 1,93 → 1,50–1,55). O melhor formato no dev foi o **Messenger JSON**, não o WhatsApp.
- **Risco medido:** sem nota, o deepseek inventa a fala do usuário em 42–65% dos logs de linha (WhatsApp, IRC) e o mercury
  em 18–38%; a lite e o luna quase nunca. O Snapchat JSON faz o deepseek recopiar o histórico (63% quebrado).
- **Com a nota do diretor, o schema quase não importa** (D 1,15–1,42 em todos) e a nota **faz 3 dos 4 atores abandonarem o
  formato** (resposta solta em 38–100% dos casos; só a lite mantém o log). Pôr a nota no system em vez de antes do log
  não resolve (deepseek/mercury: 93–100% soltas).
- **Tempos propostos pelo log:** medianas plausíveis (9–22 s × 15 s reais), mas sem relação com o tempo real de cada ponto
  (ρ 0–0,3; erro |log2| ≈ 1,1, igual ao de prever sempre a mediana, 1,03).
- **Cabeçalho de roleplay com ficha de comportamentos situacionais (sem nota)** chega a D 1,15–1,37, tão bom quanto a
  nota; a mesma ficha com **adjetivos** fica em 2,9–3,3 (lite, luna), igual à LLM pura. No banco do Jev, o cabeçalho sem
  nota é o melhor para o deepseek (P(LLM) 0,48 × humano 0,45).
- **VS:** com a instrução de VS só no system (como no paper), o luna e o deepseek **ignoram o pedido** (JSON em 63% e 3%);
  com a instrução como última mensagem, 93–100% de JSON válido. Sem nota, VS-5 + filtro de código leva a lite de D 3,90 para
  1,15 e o luna de 3,68 para 1,44 (4 seeds + o mesmo filtro: 2,67 e 2,96).

## (checkpoint) Restrições como identidade × imperativo — TESTE concluído

53 contextos de teste (14 personagens) × 4 atores. Sem restrição, o personagem viola o que a ficha proibiria em 64–83%
dos casos; o imperativo derruba para 15–34% e a identidade em 3ª pessoa para 26–32%. **Identidade não é melhor que
imperativo** (ICs sobrepostos em todos os atores; no deepseek o imperativo é melhor: 15% × 28%). Os dois formatos têm o
mesmo custo colateral: menção indireta ao tema 30–55% (× 5–10% sem restrição), esquiva 19–36%, recusa explícita 13–30%,
resposta "fria" 28–51% (× 2–11%) e coerência 0,72–0,85 (× 0,87–0,91). Checagem manual: 60 casos (não cega) + 20 cegos,
concordância 100% com o Noul do Jev nos dois.

## (checkpoint) Pipeline no DEV e regra de escolha fixada antes do teste

No dev (60 pontos) todos os pipelines com cabeçalho/nota ficam em D 1,13–1,35 (ICs sobrepostos; A = 1,64–3,90). Escolher
pelo banco do Jev e avaliar com o mesmo banco é circular; por isso o banco foi partido em meia-bateria A (código + 43
perguntas, para ESCOLHER; AUC-CV 0,885 no b1) e meia-bateria B (42 outras perguntas, sem código, para AVALIAR; AUC-CV
0,821). No dev, escolher com A ainda baixa o escore B (ex.: lite 0,61 → 0,55; deepseek 0,67 → 0,63), mas custa 0,03–0,10
de coerência. Levados ao teste: A, NCT+N (rel. 13), P1 (cabeçalho + nota no fim, 1 chamada), P4-bankA (cabeçalho + nota
no fim, 4 seeds, filtros de código + escolha pela meia-bateria A, normalizador) e P5-viol (cabeçalho + nota + VS-5 numa
chamada, escolha por código).
