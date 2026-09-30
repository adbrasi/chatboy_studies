# 06 — O Jev como infraestrutura: confiabilidade, desenho de perguntas, custo e latência

Dimensão a6. Pergunta central: dá para apoiar um chatbot de companhia em **dezenas a centenas de perguntas ao Jev por
mensagem do usuário**? Com que formato, quanto contexto, em que idioma, com que limiares, a que custo e com que latência?

**Como medi.** Foram 2.717 chamadas novas ao Jev (`typesafe/jev-1.13-20260917`), 4,18 M tokens de entrada, **US$ 0,175 no
total**, com no máximo 4 chamadas simultâneas enquanto outros agentes usavam a mesma chave. Houve zero erros e zero
*retries*. Os dados vêm de turnos reais do maichat (EN) e do whatsapp_nl (NL), além do EmpatheticDialogues quando havia
rótulo-ouro, e reaproveitei `jev_base.jsonl` sempre que deu (passes D e P). Os scripts estão em `scripts/analysis/a6_*.py`
e as saídas em `analysis/data/a6_*.json`. O cache bruto das chamadas deste estudo ficou fora do repositório, em
`<scratchpad>/a6_cache.jsonl` (6 MB). Sem ele, rodar os scripts de novo refaz as chamadas.

| Exp. | Script | O que mede | n | Chamadas |
|---|---|---|---|---|
| E1 | `a6_e1_consistency.py` | repetição idêntica, perturbação irrelevante, paráfrases, Noul × Choice × Score | 60 turnos (40 mai + 20 wa) | 300 |
| E3 | `a6_e3_context.py` | 0 / 2 / 8 / 20 turnos anteriores no state | 100 turnos (60 + 40) | 300 (+ base p/ n=8) |
| E4 | `a6_e4_fanout.py` + `a6_pool.py` | 1 / 10 / 40 / 100 / 132 perguntas por chamada; 1×132 vs 2×66 vs 4×33 em paralelo; 132 chamadas avulsas | 12 turnos | 540 |
| E5 | `a6_e5_language.py` + `a6_translations.py` | EN × PT-BR (traduzido à mão) e NL × EN | 60 turnos PT + 40 NL | 360 |
| E6 | `a6_e6_format.py` | JSON × texto; A/B × user/assistant; foco explícito × implícito | 50 turnos | 400 |
| E7 | `a6_e7_empathetic.py` | Choice de 32 classes × cascata 8→fina; contexto; calibração da confiança | 192 diálogos (6 por emoção) | 768 |
| E8 | `a6_e8_statesize.py` | latência e custo com 8 / 40 / 150 / 400 turnos no state | 12 turnos | 48 |
| E9/E10 | `a6_e9_offline.py`, `a6_e10_stats.py` | estatísticas de latência; calibração do passe P contra o que a pessoa fez de fato; estabilidade de `relationship`; ICs e McNemar | 5.491 turnos da base | 0 |

---

## 1. Resumo dos achados

- **O Jev não é determinístico, mas o ruído é pequeno.** A mesma chamada, repetida, varia em média 0,006–0,013 num Noul
  (p95 ≤ 0,04) e 0,02–0,03 num Score. Numa Choice o rótulo vencedor muda em 0–1,7 % dos casos. Chamadas feitas alguns
  minutos depois, em outro processo (base × agora), mostram o mesmo ruído, ou seja, não houve deriva nesse intervalo. Um campo irrelevante no state (`run_id`)
  aumenta um pouco a variação: Noul 0,008–0,022, e trocas de rótulo em até 8 % (phase) e 6,7 % (emotion). Um Noul cruza
  0,5 em 0,9 % dos casos com a chamada idêntica e em 2,4 % com a perturbação.
- **A confiança prevê a instabilidade quase perfeitamente.** Nas Choices com confiança ≥ 0,7, o rótulo não mudou em
  nenhuma de 153 respostas, somando repetições e perturbações. Abaixo de 0,5, mudou em 27 % delas (n = 41). Nos Scores,
  foram 0 % instáveis com confiança ≥ 0,7 e 33 % abaixo de 0,5.
- **As perguntas são mesmo isoladas.** Colocar 1, 10, 40, 100 ou 132 perguntas na mesma chamada não muda a resposta de
  cada uma além do ruído: 0,0065–0,0075 de diferença num Noul contra 0,006–0,009 de ruído entre repetições, e 0–2,8 % de
  trocas numa Choice.
- **A latência praticamente não depende do número de perguntas nem do tamanho do state.** Com 1 pergunta, a mediana foi
  0,52 s; com 132 perguntas, 0,55 s; com 400 turnos de state (~20 mil tokens), 0,61 s. Em todas as 2.717 chamadas:
  p50 0,49 s, p90 0,62 s, p99 1,36 s, e só 2,6 % passaram de 1 s.
- **Custo.** Um briefing de 132 perguntas numa só chamada custa **US$ 0,00023** (~5,5 mil tokens). Fazer as mesmas 132
  perguntas em chamadas separadas custa 11× mais (US$ 0,0026) e levou 17,8 s com 4 workers. Dividir em 2 ou 4 chamadas
  paralelas **não reduz a mediana** (0,55 → 0,53 / 0,60 s) e paga o state de novo em cada chamada. Cada pergunta custa
  ~34 tokens (≈ US$ 0,0000014).
- **Contexto: 0 turnos é ruim; 2 turnos recuperam a maior parte; 8 turnos são um bom padrão.** Sem contexto, a emoção
  discorda da versão com 8 turnos em 37 % dos casos; com 2 turnos, em 15 %; com 20, em 9 %. Isolada, uma mensagem parece
  **mais séria** (seriedade média 0,99 contra 0,64 com 8 turnos, numa escala de 0 a 3) e **menos brincalhona** (playful
  0,43 contra 0,55). No EmpatheticDialogues, com rótulo-ouro, o diálogo inteiro acerta mais que só a primeira fala:
  50,5 % contra 43,2 % (McNemar p = 0,04).
- **`relationship` por turno é instável.** Na base, só 65–70 % dos turnos de uma conversa concordam com o rótulo mais
  frequente dela, e cada conversa recebe em média 3,5 rótulos distintos. Com 20 turnos de contexto, o rótulo muda em 31 %
  dos casos em relação a 8 turnos. Essa variável deve ser agregada ou vir da configuração do app, e não ser perguntada a
  cada mensagem.
- **Português funciona, mas perde ironia e zoeira.** Com o state traduzido à mão para PT-BR informal e as perguntas em
  inglês, os Nouls correlacionam 0,87–0,94 com o inglês (o ruído correlaciona 0,99) e a emoção concorda em 78 % (contra
  98 % no ruído). O PT sai **mais sério** (+0,08; +0,13 com as perguntas também em PT) e o `playful` concorda no binário em
  só 83 %. Perguntas em inglês com o state em PT saem mais próximas do inglês do que perguntas em PT. O holandês com
  perguntas em inglês fica um pouco mais perto da tradução para o inglês (emoção 82,5 %, Nouls com correlação 0,87–0,98).
- **O formato do state importa pouco.** Todas as variantes ficaram parecidas. JSON com `current_turn` explícito foi a mais
  estável (confiança média das Choices 0,76). Texto corrido com o foco implícito foi a que mais se afastou (14 % de trocas
  numa Choice, 0,17 de diferença num Score). Os rótulos "user"/"assistant" funcionam tão bem quanto "A"/"B".
- **Com muitas classes, a Choice plana ganha da cascata.** Na emoção de 32 classes do ED, a Choice plana acertou 50,5 %
  (top-3 72 %; o acaso é 3 %). A cascata grupo → fina acertou 44,3 % (p = 0,004 contra a plana) e com *beam* de 2, 49,0 %.
  Mapear a resposta plana para um dos 8 grupos acerta **75 %**, mais do que perguntar o grupo diretamente (70 %,
  p = 0,035).
- **A confiança da Choice é superconfiante, mas ordena bem.** No ED, ECE = 0,31. Com confiança ≥ 0,85 (51 % dos itens), o
  acerto é de 62 % na classe fina e 88 % no grupo. Abaixo de 0,7, de 26–31 % na fina.
- **As probabilidades preditivas (passe P) não vêm calibradas para eventos raros.** `p_laugh` tem média 0,34 contra uma
  taxa real de 4,7 % (maichat) e `p_emoji`, 0,29 contra 4,4 %. As AUCs ficam entre 0,61 e 0,79: servem para ordenar, não
  como frequência. `p_n_msgs` respondeu "1" em 99,9 % dos turnos, com confiança ≥ 0,9 em 94 %, e acertou exatamente a taxa
  da classe majoritária (68 % e 53 %).
- **Paráfrases e tipos de pergunta concordam na ordem, não no nível.** Nouls parafraseados correlacionam 0,89–0,97. A
  negação ("está calmo e sem preocupação?", usando 1 − p) só concorda em 85 % no binário. A Choice sim/não é mais extrema
  que o Noul: para ansiedade, P(sim) média 0,10 contra 0,21, com correlação 0,93. "É um momento sério?" como Noul tem
  correlação de apenas 0,63 com o Score de seriedade. Limiares **não se transferem** entre formulações.

---

## 2. Achados detalhados

### A1. Consistência: repetição e perturbação irrelevante (E1, 60 turnos × 17 perguntas × 4 chamadas)

| Tipo | Idêntica agora (média / p95) | Idêntica, minutos depois (base, outro processo) | Com `run_id` × idêntica | Com `run_id` × `run_id` |
|---|---|---|---|---|
| Noul (10 perguntas) | 0,006–0,013 / ≤ 0,041 | 0,006–0,015 | 0,008–0,022 / ≤ 0,061 | 0,005–0,016 |
| Score (valence, arousal, seriousness, engagement) | 0,018–0,034 | 0,022–0,038 | 0,028–0,058 | 0,022–0,041 |
| Choice: troca do rótulo (emotion / intent / phase / relationship) | 1,7 / 0 / 0 / 0 % | 1,7 / 0 / 1,7 / 6,7 % | 6,7 / 0 / 8,3 / 3,3 % | 3,3 / 1,7 / 3,3 / 0 % |

- Um Noul que cruza 0,5 aparece em 0,9 % dos pares idênticos e em 2,4 % dos pares com `run_id`.
- **Confiança × instabilidade** (E10; 4 chamadas por turno: 2 idênticas e 2 com `run_id`; instável significa que o rótulo
  mudou em alguma delas ou que o Score variou mais de 0,15):

| Confiança | Choice: n / % instável | Score: n / % instável |
|---|---|---|
| < 0,5 | 41 / **27 %** | 21 / **33 %** |
| 0,5–0,7 | 46 / 2 % | 96 / 12,5 % |
| 0,7–0,9 | 63 / 0 % | 69 / 0 % |
| ≥ 0,9 | 90 / 0 % | 54 / 0 % |

**Exemplo.** "🥺🥺🥺 yes please / I miss you" recebeu ansiedade 0,37 (formulação original), 0,56 ("sounds nervous or
insecure") e 0,11 (afirmação "is feeling worried"): é um caso ambíguo por natureza.

**Confiança: alta** para o ruído (n grande, 4 condições). **Média** para a tabela de confiança, porque há poucos casos
abaixo de 0,5.

### A2. Paráfrases e tipos de pergunta para o mesmo conceito (E1 parte B, 60 turnos)

| Conceito | Variante | Média | r com a original | Concordância binária |
|---|---|---|---|---|
| anxious | Noul original | 0,21 | 1 | — |
| | "sound nervous or insecure?" | 0,25 | 0,93 | 93 % |
| | afirmação "is feeling worried." | 0,16 | 0,89 | 97 % |
| | com critérios true/false | 0,17 | 0,97 | 97 % |
| | Choice {yes, no}: P(yes) | **0,10** | 0,93 | 97 % |
| | Score de 4 níveis (0–3) | 0,27 | 0,95 | 95 % (limiar 1,5) |
| | negação "calm and free of worry?" (1 − p) | **0,32** | 0,88 | **85 %** |
| playful | 2 paráfrases de Noul | 0,52–0,56 | 0,96–0,97 | 92 % |
| | Choice {yes, no} | 0,54 | 0,98 | 93 % |
| | Score 0–3 | 1,12 | 0,93 | 73 % (limiar 1,5) |
| seriousness | Score original (0–3) | 0,56 | 1 | — |
| | Choice de 4 opções (valor esperado) | 0,55 | **0,98** | — |
| | escala invertida (3 − x) | 0,43 | 0,93 | — |
| | paráfrase "heavy or emotional" | 0,47 | 0,79 | — |
| | Noul "Is this a serious moment?" | 0,15 | **0,63** | — |
| emotion (12) | mesmas opções em outra ordem | — | — | 88 % |
| | sem as descrições das opções | — | — | **78 %** |
| valence | 5 níveis × 10 níveis (normalizados) | 0,60 / 0,62 | 0,96 | — |

Leitura:

- A **ordem** é robusta: correlações de 0,9 ou mais entre formulações do mesmo conceito.
- O **nível absoluto** depende da forma. A Choice sim/não empurra a probabilidade para os extremos. A negação não é o
  complemento. O Noul de "sério" responde a outro construto: tem um limiar implícito de "muito sério".
- A **descrição de cada opção** pesa mais que a ordem delas: tirá-la muda 22 % dos rótulos; embaralhar muda 12 %.

**Confiança: alta.**

### A3. As perguntas são isoladas, a latência é plana e o custo é linear (E4, 12 turnos × 5 tamanhos × 2 repetições)

| Perguntas/chamada | Latência p50 / p90 / máx (s) | Tokens | Custo por chamada | Custo por pergunta |
|---|---|---|---|---|
| 1 | 0,52 / 0,60 / 0,75 | 575 | US$ 0,000024 | 0,000024 |
| 10 | 0,47 / 0,69 / 0,79 | 1.465 | US$ 0,000062 | 0,0000062 |
| 40 | 0,54 / 0,60 / 0,73 | 2.466 | US$ 0,000104 | 0,0000026 |
| 100 | 0,52 / 0,65 / 1,48 | 3.761 | US$ 0,000158 | 0,0000016 |
| 132 | 0,55 / 0,71 / 2,07 | 5.543 | US$ 0,000233 | 0,0000018 |

- **Isolamento.** A diferença nas 10 perguntas comuns entre as chamadas de 10 e de 40/100/132 perguntas foi de 0,0065–0,0075
  (Noul), abaixo ou igual ao ruído entre repetições (0,006–0,009). Uma pergunta sozinha contra ela dentro das 132: 0,008
  (máx. 0,05), sem troca de Choice. Confirmado.
- **Uma chamada por pergunta**, 132 chamadas com 4 workers: 17,8 s de relógio e US$ 0,0026, **11×** o custo da chamada única.
- **Dividir o briefing de 132 perguntas em chamadas paralelas** (24 medições por configuração):

| Configuração | Relógio p50 | p90 | máx. |
|---|---|---|---|
| 1 × 132 | 0,55 s | 0,78 s | 1,53 s |
| 2 × 66 simultâneas | 0,53 s | 0,61 s | 2,16 s |
| 4 × 33 simultâneas | 0,60 s | 0,73 s | 0,81 s |

  Na mediana não há ganho. A cauda varia ao acaso, porque cada chamada tem a sua (com 24 medições, não dá para afirmar
  que "4 × 33 tira a cauda"). Dividir paga o state N vezes.
- **Tamanho do state** (E8, whatsapp, 17 perguntas): com 8 turnos, 2,0 mil tokens e 0,50 s; com 40, 3,5 mil e 0,54 s;
  com 150, 8,4 mil e 0,55 s; com 400, 19,9 mil tokens, **0,61 s** e US$ 0,00084. O custo cresce linearmente com o state,
  a latência quase não cresce.
- **Todas as 2.717 chamadas:** p50 0,49 s, p90 0,62 s, p99 1,36 s, máx. 2,16 s; 2,6 % acima de 1 s e 0,6 % acima de
  1,5 s. Preço efetivo: **US$ 0,042 por milhão de tokens**; os tokens de saída são grátis.

**Confiança: alta** para custo e isolamento. **Média** para a cauda de latência: rede via OpenRouter e outros agentes
usando a mesma chave ao mesmo tempo.

### A4. Quanto contexto pôr no state (E3, 100 turnos com ≥ 20 turnos anteriores; referência: 8 turnos)

Não há rótulo-ouro no maichat/whatsapp, então a tabela mede a distância para a versão com 8 turnos. O ruído de
referência (A1) é de 1,7 % de trocas na emoção e 0,02–0,03 num Score.

| Pergunta | n=0 × 8 | n=2 × 8 | n=20 × 8 | n=20 × 2 | Médias com n = 0 / 2 / 8 / 20 |
|---|---|---|---|---|---|
| emotion (troca) | 37 % | 15 % | 9 % | 21 % | — |
| seriousness (Δ na escala 0–3) | 0,52 | 0,24 | 0,14 | 0,30 | **0,99 / 0,77 / 0,64 / 0,62** |
| valence (Δ 0–4) | 0,44 | 0,24 | 0,15 | 0,31 | 2,09 / 2,24 / 2,33 / 2,39 |
| playful (Δp) | 0,18 | 0,09 | 0,05 | 0,12 | **0,43 / 0,50 / 0,55 / 0,58** |
| anxious (Δp) | 0,11 | 0,07 | 0,05 | 0,09 | 0,27 / 0,27 / 0,25 / 0,24 |
| engagement (Δ 0–4) | 0,68 | 0,29 | 0,18 | 0,39 | 1,45 / 1,89 / 2,06 / 2,16 |
| phase (troca) | 57 % | 25 % | 11 % | 31 % | — |
| relationship (troca) | 63 % | 26 % | **31 %** | 47 % | — |
| topic_shift / mirrors | sem sentido com n=0 (0,76 / 0,25) | | | | 0,23 e 0,67 com n=8 |

- **Sem contexto, a mesma mensagem parece mais séria e mais zangada.** Três exemplos do maichat:
  - "stop judging me im fragile" sai `frustration_anger` com 0 e com 2 turnos, e `playful_teasing` com 8 e com 20.
  - "i live with you. i have to." vai de `frustration_anger` (n=0) a `playful_teasing` (n ≥ 2).
  - "good" vai de `neutral_informational` (n=0) a `affection` (n ≥ 2).
- **Retorno decrescente.** 2 turnos cobrem cerca de metade a dois terços da distância entre 0 e 8. De 8 para 20 ainda há
  mudança (cerca de 5× o ruído), mas sem ouro não dá para dizer se é melhora.
- **Com ouro (ED):** diálogo inteiro (4–8 falas) 50,5 % [IC95 44–57]; só a 1ª fala 43,2 % [36–51]; as duas primeiras
  falas 42,2 %. Isto é, +7 pontos com o diálogo inteiro (McNemar 27 × 13, p = 0,04). Curiosamente, acrescentar só a
  resposta do ouvinte não ajudou.
- Com 0, 2 ou 20 turnos, a latência é a mesma (p50 ≈ 0,48 s). Os 20 turnos custam +40 % de tokens (1,6 → 2,3 mil com
  17 perguntas).
- **`relationship` é um traço da conversa, não do turno.** Na base (n = 8), só 70 % dos turnos (maichat) e 65 %
  (whatsapp) concordam com o rótulo mais frequente da conversa, com 3,5–3,7 rótulos distintos por conversa. Somar as
  probabilidades dos 10 primeiros turnos dá o mesmo rótulo que somar a conversa inteira em apenas 69 % das conversas. Um
  exemplo: "Yes - possibly February. / If you are free it would be great." recebeu colleagues (n=0) → close_friends
  (n=2) → family (n=8) → romantic_partners (n=20).

**Confiança: alta** para "0 turnos é ruim" e para o viés de seriedade. **Média** para escolher entre 8 e 20, porque não
há ouro.

### A5. Idioma: português e holandês (E5)

O state tinha 2 turnos anteriores + o turno atual. Foram 12 janelas do maichat, 84 turnos traduzidos à mão para PT-BR
informal (com "vc", "tbm", "kkkk", "mt"), com 60 turnos avaliados, e 8 janelas do whatsapp, 56 turnos traduzidos de NL
para EN, com 40 avaliados.

| Comparação | emotion (troca) | Nouls: r (faixa) | Nouls: Δ médio | seriousness Δ / médias |
|---|---|---|---|---|
| EN × EN com `run_id` (ruído) | 1,7 % | 0,99–1,00 | 0,010–0,015 | 0,03 |
| EN × PT (perguntas em EN) | **21,7 %** | 0,87–0,94 | 0,04–0,085 | 0,16; 0,53 → **0,61** |
| EN × PT (perguntas em PT) | 25 % | 0,84–0,92 | 0,075–0,10 | 0,20; 0,53 → **0,65** |
| NL × NL com `run_id` (ruído) | 0 % | 0,99–1,00 | 0,008–0,018 | 0,035 |
| EN traduzido × NL original | 17,5 % | 0,87–0,98 | 0,033–0,056 | 0,10 |

- **Em PT, a zoeira vira coisa séria.** Exemplos:
  - "para de me julgar eu sou frágil": playful 0,75 → 0,33 e seriedade 0,98 → 2,35 (EN → PT).
  - "o prof olhou pra mim uma vez / eu entrei em pânico": playful 0,85 → 0,37 e seriedade 0,20 → 1,41.
  - "ai não" (de "oh no"): ansiedade 0,75 → 0,27 e emoção `surprise` → `frustration_anger`.

  A concordância binária do `playful` foi de 83 %; a da `tension`, 87 %.
- **Perguntas em PT deslocam os níveis para cima:** flirting +0,06, vulnerable +0,07, seeks_support +0,05,
  seriousness +0,12 em relação ao EN. Perguntas em EN sobre o state em PT ficam mais próximas do EN em todas as dimensões.
- **O holandês com perguntas em EN funciona razoavelmente.** As correlações são de 0,93–0,98 em valence, anxious,
  playful, vulnerable e seriousness, mas só 0,87 em `tension`. Exemplo: "Jij bent mijn inspiratie" sai `gratitude` em NL
  e `playful_teasing` em EN ("You are my inspiration").
- **Confundidores.** A diferença EN × PT mistura o idioma com a minha tradução: "cap" → "mentira" e o trocadilho
  chilli/chilly adaptado não são equivalentes perfeitos. As janelas de PT e NL são diferentes, então as linhas PT e NL não
  se comparam diretamente. Com n = 60 e n = 40, o IC da troca de emoção é de ±10 pontos.

**Confiança: média.** O Jev funciona em PT, mas com mais ruído e com viés de seriedade. É preciso calibrar com dados
reais em PT.

### A6. Formato do state (E6, 50 turnos; perturbação `run_id` em 30)

| Formato | Instabilidade Noul / Score / Choice | Distância ao JSON A/B explícito: Noul / Score / Choice | Confiança média Choice | Sonda de foco "o turno atual tem pergunta?" (AUC / acurácia) | Tokens |
|---|---|---|---|---|---|
| JSON, A/B, `current_turn` explícito (base) | 0,013 / 0,037 / 6,7 % | — | **0,760** | **0,93** / 90 % | 1.953 |
| JSON, user/assistant, explícito | 0,014 / 0,037 / **2,5 %** | 0,032 / 0,081 / 6,5 % | 0,754 | 0,91 / 90 % | 1.953 |
| JSON, foco implícito (último item da lista) | 0,017 / 0,048 / 8,3 % | 0,037 / 0,116 / 9,5 % | 0,734 | 0,91 / 88 % | 1.996 |
| texto corrido, "CURRENT TURN:" explícito | 0,015 / 0,043 / 4,2 % | 0,027 / 0,104 / 6,5 % | 0,759 | 0,92 / 90 % | 1.791 |
| texto corrido, foco implícito | 0,017 / **0,054** / 5,8 % | **0,055 / 0,169 / 14 %** | 0,730 | 0,92 / 88 % | 1.832 |

- O foco explícito é um pouco mais estável e mais confiante que o implícito, tanto em JSON quanto em texto. No texto
  implícito, a seriedade média cai de 0,64 para 0,53 e o `topic_shift` sobe de 0,23 para 0,28: o Jev mistura mais o turno
  atual com os anteriores.
- Os rótulos "user"/"assistant" não atrapalham, e a pergunta `relationship` não "vira" relação humano-máquina: continuou
  dando close_friends em 34 de 50.
- O texto corrido usa ~8 % menos tokens.
- A sonda "O turno atual contém risada?" acertou 98 % em todos os formatos.

**Confiança: baixa a média.** As diferenças são pequenas (n = 50/30); a direção é coerente com a documentação.

### A7. Muitas classes × cascata e calibração da confiança (E7, EmpatheticDialogues, 192 diálogos, 32 emoções-ouro)

| Método (diálogo inteiro) | Acerto [IC95] |
|---|---|
| Choice plana de 32 classes | **50,5 %** [44–57] (top-3 71,9 %) |
| cascata gulosa: grupo (8) → fina | 44,3 % [38–51]; perde para a plana: 14 × 2, p = 0,004 |
| cascata com beam 2 (fina entre as classes dos 2 melhores grupos) | 49,0 % [42–56]; empate com a plana (p = 0,51) |
| produto P(grupo) × P(fina \| grupo) | 47,9 % |
| **grupo**: perguntado diretamente (8 opções) | 70,3 % [64–77] |
| **grupo**: derivado da resposta plana | **75,0 %** [69–81]; ganha: 12 × 3, p = 0,035 |

Calibração da Choice plana (diálogo inteiro):

| Confiança | n | Acerto fino | Acerto no grupo |
|---|---|---|---|
| 0,30–0,50 | 19 | 26 % | 47 % |
| 0,50–0,70 | 42 | 31 % | 57 % |
| 0,70–0,85 | 34 | 56 % | 76 % |
| ≥ 0,85 | 97 | 62 % | **88 %** |

- ECE (por p_max) = 0,31: a Choice é superconfiante. Mesmo assim, a confiança **ordena** bem: com o limiar em 0,7 ficam
  68 % dos itens, com 60 % de acerto fino e 85 % no grupo.
- Com só a primeira fala a confiança perde o sentido na faixa média (0,70–0,85: 29 % de acerto). Ela só volta a informar
  acima de 0,85 (64 %).
- Os erros são, em grande parte, **quase-sinônimos**: devastated → sad (5), furious → angry (5), ashamed → embarrassed (4),
  terrified → afraid (3), sentimental → nostalgic (3). Os rótulos do ED também são ruidosos; humanos também confundem
  esses pares.
- Os grupos mais difíceis foram proud_confident (46 %), warm_caring (50 %) e surprised (33 %). Os mais fáceis:
  fear_worry (88 %), self_conscious (89 %) e happy_excited (87 %).
- Exemplo de erro com confiança baixa: "I was in a hospital and a guy came in with some really bad burns." (ouro:
  disgusted; saída: sad, confiança 0,37).

**Confiança: média-alta** (n = 192, com ICs). O ED é induzido e em inglês.

### A8. As probabilidades preditivas (passe P) ordenam bem, mas não vêm calibradas (E9, base: 2.826 turnos do maichat e 2.665 do whatsapp)

| Pergunta P | Taxa real (mai / wa) | p médio (mai / wa) | AUC (mai / wa) | Brier do Jev × Brier da taxa-base (mai) |
|---|---|---|---|---|
| p_laugh | 4,7 % / 19 % | **0,34** / 0,29 | 0,63 / 0,70 | 0,146 × 0,045 |
| p_question | 9,4 % / 22,5 % | 0,34 / 0,37 | 0,61 / 0,61 | 0,153 × 0,085 |
| p_emoji | 4,4 % / 15 % | 0,29 / 0,36 | 0,70 / 0,67 | 0,107 × 0,042 |
| p_end | 2,4 % / 4,6 % | 0,11 / 0,16 | **0,79** / 0,63 | 0,030 × 0,023 |

- A relação entre p e a frequência real é monotônica, mas fica deslocada. Por exemplo, `p_laugh` no maichat: com p entre
  0,10 e 0,20, a frequência real é 2,4 %; com p entre 0,65 e 0,80, é 12,7 %. Ou seja, o Jev ordena e o código calibra.
  O Brier bruto do Jev é **pior que o de chutar a taxa-base**.
- `p_n_msgs` (Choice 1/2/3/4+) respondeu "1" em 2.826 de 2.826 turnos (maichat) e em 2.663 de 2.665 (whatsapp), com
  confiança ≥ 0,9 em ~93 %. O acerto (68 % e 53 %) é exatamente o da classe majoritária. Aqui **a confiança alta não
  significa nada**: é o prior. P("1") como *score* tem AUC de 0,61 (maichat) e 0,54 (whatsapp) para separar uma bolha de
  várias.

**Confiança: alta** (n grande). O maichat é ao vivo e em inglês; o whatsapp é holandês.

---

## 3. Padrões "invisíveis"

Aqui, "invisíveis" são coisas que as pessoas fazem sem notar e que também explicam como o Jev reage:

1. **A zoeira depende do histórico.** Uma pessoa lê "stop judging me im fragile" ou "i live with you. i have to." como
   brincadeira porque viu os turnos anteriores. Isoladas, essas frases parecem raiva: o Jev mudou de `frustration_anger`
   para `playful_teasing` ao receber contexto. Um bot que reage só à última mensagem vai levar a sério quem está brincando,
   e essa é uma das falhas que mais quebram a ilusão. A média de seriedade cai de 0,99 para 0,64 quando entra contexto.
2. **A informalidade em PT (e a tradução) apaga o tom de brincadeira.** A mesma zoeira em PT informal saiu mais séria.
   Humanos brasileiros ironizam o tempo todo ("sou mt burra kkk", "para de me julgar") e esperam que o outro entre no
   jogo. O bot precisa de uma proteção extra contra ler o sarcasmo ao pé da letra em PT.
3. **As pessoas não recalculam a relação a cada mensagem.** Humanos mantêm um modelo estável de quem é o outro. O Jev, se
   perguntado a cada turno, oscila entre amigos, colegas, família e casal; em conversa real isso seria bizarro. A relação
   é uma variável lenta.
4. **Emoções vizinhas são intercambiáveis na prática.** Furious/angry, devastated/sad e ashamed/embarrassed são
   confundidas pelo Jev e pelos próprios anotadores do ED. Uma pessoa responde igual às duas de cada par. O bot só precisa
   da família (8 grupos), que tem 75–88 % de acerto.
5. **O comportamento mais comum é o default.** Uma bolha só, sem risada, sem emoji: o Jev "acerta" dizendo sempre o
   óbvio. As variações (várias bolhas, "kkkk", figurinha) são raras e é justamente nelas que está o realismo. Elas precisam
   ser sorteadas de forma calibrada pelo código, não decididas pelo argmax.

---

## 4. Tradução para o sistema

### 4.1 Regras de engenharia

1. **Uma chamada "briefing" por mensagem do usuário, com todas as perguntas.** Até ~150 perguntas numa chamada: a
   latência é igual à de 1 pergunta e o custo é ~1/11 do de chamadas avulsas. Divida em 2–4 chamadas só se for preciso
   reduzir a cauda, e com *hedging* (veja 4.3), não por padrão.
2. **O state é um JSON com o foco explícito:**
   `{"persona": {...fixos...}, "relationship_context": "<do app>", "memory": "<resumo curto>", "previous_turns": [8 turnos], "current_turn": {speaker, text}}`.
   Nas perguntas, cite `current_turn` e `previous_turns` entre crases. Os rótulos podem ser "user"/"assistant" ou os
   nomes das pessoas. Junte as bolhas de um burst num texto só com " / ".
3. **Contexto:** 8 turnos por padrão, nunca 0, e no mínimo 2 para qualquer pergunta sobre tom, emoção, ironia, seriedade,
   `topic_shift` ou `mirrors`. Mais do que ~20 turnos só entra como **resumo** no campo `memory`: 400 turnos custam 10× e
   não mostraram ganho.
4. **Formulação:**
   - Uma pergunta afirmativa e literal por conceito, **sem negação**.
   - Critérios em **todas** as opções: descrever as opções muda 22 % dos rótulos, então vale a pena. Pode acrescentar
     `other`/`none`.
   - Para um conceito, escolha **um** tipo (Noul, Choice ou Score) e **uma** formulação, e congele. Limiares não se
     transferem: Choice sim/não ≠ Noul ≠ Score.
   - Instruções em **inglês**, mesmo com o state em PT.
5. **Taxonomias grandes:** use uma Choice plana com todas as classes (até 255) e **derive o grupo em código** somando as
   probabilidades. Não faça cascata gulosa. Só use uma segunda chamada quando ela trouxer evidência nova (outro state),
   não para refinar a mesma pergunta.
6. **Confiança:** trate a confiança < 0,5 como "não sei" (27–33 % de instabilidade) e aja com ≥ 0,7 (0 % de
   instabilidade). A confiança **não** garante acerto em perguntas preditivas cuja resposta é o prior (`p_n_msgs`).
7. **Nouls são rankers.** Probabilidades preditivas de eventos raros devem passar por uma **calibração isotônica ou Platt
   no código**, ajustada com logs do próprio produto, por idioma. Não use `p > 0,5` direto.
8. **Histerese e suavização.** O ruído de ±0,01–0,04 e 1–8 % de trocas de rótulo pedem faixas: ligar acima de 0,6 e
   desligar abaixo de 0,4. Estados de humor devem passar por uma EMA ao longo dos turnos.
9. **O que calcular em código:**
   - contagens: bolhas, caracteres, emojis, "kkk";
   - tempos: latência, horário, tempo desde a última mensagem;
   - comprimento e presença de pergunta ou risada (regex);
   - número de bolhas e atraso da resposta, sorteados a partir de distribuições condicionais, com o Jev fornecendo só o
     estado emocional ou a seriedade como condição.
10. **O que cachear:**
    - por **sessão ou conversa**: relationship/intimidade (agregada com EMA, ou vinda da configuração da persona), idioma,
      estilo do usuário (fingerprint calculado em código);
    - por **turno**: o briefing inteiro, com chave = hash(state, perguntas), porque a mensagem só é avaliada uma vez;
    - as perguntas congeladas (strings exatas), versionadas junto com o modelo (`jev-1.13-20260917`). Fixe a versão, não
      use `jev-latest`, e recalibre os limiares quando mudar de versão.

### 4.2 Cada achado vira um detector ou uma regra

| Achado | Detector Jev (tipo, instrução, critérios) | State | Onde roda no fluxo | Regra de código |
|---|---|---|---|---|
| A1 ruído / A1 confiança | qualquer Choice/Score do briefing | — | pós-processamento do briefing | Choice com confiança < 0,5 → rótulo = `unknown` e usa o default da persona. Entre 0,5 e 0,7 → só ações reversíveis (tom, emoji). ≥ 0,7 → ações fortes (mudar de modo, consolar, encerrar). Noul com histerese 0,6/0,4 |
| A2 formulação | ex.: `anxious` = Noul "Is the speaker of `current_turn` anxious, worried or insecure?", com true/false descritos | 8 turnos + current_turn | briefing (rodada 1) | limiar calibrado **nessa** formulação; mudou o texto, recalibra |
| A3 fan-out | briefing de 40–150 perguntas numa chamada | o mesmo JSON para todas | rodada 1, logo que a mensagem do usuário chega | uma chamada. Timeout de 0,9 s → *hedge* (reenvia uma cópia e fica com a primeira resposta). Ao passar de 1,8 s → segue com o que tiver: defaults + estado do turno anterior |
| A4 contexto / seriedade | `seriousness` Score de 4 níveis ["playful banter", "casual", "somewhat serious", "very serious / emotional"] | **≥ 2**, idealmente 8 turnos | rodada 1 | EMA(0,5) da seriedade: ≥ 1,8 → modo sério (bolhas longas, sem "kkk", sem figurinha). ≤ 0,7 → modo leve |
| A4 relationship | Score lento "How intimate/close is the relationship between the two people, as shown in `previous_turns`?" ["strangers", "acquaintances", "friends", "close friends", "romantic/very intimate"] | 20 turnos + memória | **a cada ~10 mensagens ou no início da sessão**, não por turno | EMA; muda de nível só com 3 leituras consecutivas iguais. Se a persona define a relação (namorada virtual), o dado vai no state, sem perguntar |
| A5 PT / sarcasmo | Noul "Is `current_turn` said jokingly or ironically, not meant literally?", com critério true "teasing, irony, self-deprecating humor (e.g. 'sou mt burra kkk')" e false "sincere, literal" | 8 turnos, texto original em PT | rodada 1, junto com `playful` | se `ironic > 0,5` e `playful > 0,4` → não entra em modo de consolo, mesmo com `anxious`/`seeks_support` altos. Faixas calibradas com dados PT |
| A6 formato | — | JSON + `current_turn` explícito | construção do state | template fixo. Os rótulos "user"/"assistant" podem ficar |
| A7 emoção fina | `emotion` Choice plana de ~32 classes com descrição curta; o grupo é derivado em código (8 famílias) | diálogo da sessão (até 8 turnos) | rodada 1 | usa a **família** quando a confiança está entre 0,5 e 0,85 e a emoção fina quando é ≥ 0,85. Abaixo de 0,5 → "neutra/indefinida" |
| A8 previsão de eventos raros | `p_laugh`, `p_emoji`, `p_question`, `p_end` como Nouls sobre "o próximo turno" (state sem o turno a gerar) | até o último turno do usuário | rodada 1 (chamada paralela, com outro state) | p_cal = isotônica(p) por idioma. Sorteio de Bernoulli(p_cal) para usar risada, emoji ou pergunta; **não** usar o argmax |
| A8 número de bolhas | **não perguntar** `p_n_msgs` | — | — | nº de bolhas ~ distribuição empírica condicionada a (seriedade, arousal, anxious, tamanho do texto gerado), calculada em código |

### 4.3 Fluxo recomendado e orçamento de latência (por mensagem do usuário)

```
t=0     mensagem chega → código: features (regex/contagens/tempos), montagem do state (8 turnos + current_turn)
t≈0     RODADA 1 (em paralelo): [A] briefing D (40–150 perguntas, state com o turno do usuário)
                                [B] briefing P (10–20 perguntas "próximo turno", state até o turno do usuário)
        timeout 0,9 s → hedge (duplica a chamada atrasada); corte duro em 1,8 s → usa defaults/EMA anterior
t≈0,55  código: calibração, EMA, histerese → modo (leve/sério/consolo/flerte), plano de entrega (nº de bolhas, atraso, "digitando…")
t≈0,6   RODADA 2 (opcional, só quando depende da rodada 1 E precisa de outro state):
         ex.: escolher entre 3–5 respostas candidatas geradas pela LLM (Choice) ou checar a candidata ("Does `reply` make fun of
         something the user is sensitive about?")
t≈1,2   LLM gera o texto a partir do briefing (o custo dominante é a LLM, não o Jev)
```

- **Com uma rodada**, dá para ficar abaixo de 1 s em ~97 % das mensagens (p90 0,62 s, p99 1,36 s).
- **Com duas rodadas sequenciais**, a mediana fica em ~1,0–1,1 s e o p99 em ~2,5 s.
- O "digitando…" e o atraso humano de resposta (segundos) escondem folgadamente essas latências. O Jev não é o gargalo.

### 4.4 Custo por mensagem do usuário (preço medido: US$ 0,042 por milhão de tokens; ~34 tokens por pergunta; ~35–45 tokens por turno de chat no state)

| Perfil | Chamadas | Perguntas | Tokens (≈) | Custo por mensagem | Por 1.000 mensagens | Latência Jev (p50 / p99) |
|---|---|---|---|---|---|---|
| **Leve** | 1 (briefing D) | ~20 | ~2.000 | **US$ 0,00008** | US$ 0,08 | 0,5 s / 1,4 s |
| **Médio** | 2 em paralelo (D 60 + P 12) + 1 condicional (rodada 2, ~5 perguntas) | ~77 | ~5.500 | **US$ 0,00023** | US$ 0,23 | 0,55 s (1 rodada) a 1,1 s (2 rodadas) / ~2,5 s |
| **Pesado** | 2 em paralelo (D 150 com 20 turnos + memória; P 20) + 2 condicionais (candidatas, checagens) | ~180 | ~12.000 | **US$ 0,0005** | US$ 0,50 | 1,1 s / ~2,5 s |

- Se o pesado for dividido em 4 × 38 perguntas, o state é pago 4 vezes: ~+4,5 mil tokens, cerca de +40 %, sem ganho na
  mediana.
- Um usuário muito ativo, com 200 mensagens por dia, custa em Jev: leve US$ 0,016/dia (~US$ 0,50/mês); médio
  US$ 0,05/dia (~US$ 1,40/mês); pesado US$ 0,10/dia (~US$ 3/mês).
- As estimativas foram calibradas nas medições: 17 perguntas + 8 turnos = 1,95 mil tokens; 132 perguntas + 8 turnos =
  5,5 mil tokens; 400 turnos = 19,9 mil tokens.

---

## 5. Limitações

- **Sem rótulo-ouro para tom, seriedade ou relationship no chat real.** Em E3 e E6, a estabilidade e a concordância
  medem consistência, não acerto. O único ouro é o do ED (emoção de uma situação induzida por crowdworkers, em inglês, com
  rótulos ruidosos) e as features de código (risada, pergunta, emoji, bolhas) usadas como alvo do passe P.
- **As amostras são pequenas em vários experimentos:** 60 turnos (E1), 50 (E6), 60 PT + 40 NL (E5), 12 (E4, E8). As
  diferenças de 3–5 pontos percentuais em E6 estão dentro do erro. O maichat tem 42 conversas, e parte delas é encenada
  ou entre amigos com estilo de meme.
- **A tradução para PT-BR foi feita por mim, à mão.** Ela mistura o efeito do idioma com o efeito da tradução (gírias
  adaptadas, trocadilhos perdidos), e o PT informal real de usuários de app de companhia é diferente de uma tradução do
  maichat. Os números servem de alerta, não de estimativa final: repita com conversas nativas em PT.
- **O whatsapp_nl é holandês de 2012–2014.** A anonimização (`[REMOVED]`, `[numerical_value]`) atrapalha algumas
  leituras.
- **A latência foi medida via OpenRouter**, a partir deste container, com até 4 chamadas simultâneas e outros agentes
  usando a mesma chave. A API direta (api.typesafe.ai) e outra região podem dar números diferentes. Os limites de taxa
  (40 req/s, 100 mil tokens/s) não foram testados.
- **O ruído entre repetições mostra que o endpoint não é determinístico.** Todas as comparações com uma única chamada por
  condição carregam ±0,01–0,03 de ruído. As diferenças que reportei estão comparadas com esse ruído.
- **As calibrações (ED, passe P) valem para essas perguntas e esses dados.** Os limiares devem ser reajustados com os
  logs do produto, por idioma e por versão do modelo.
