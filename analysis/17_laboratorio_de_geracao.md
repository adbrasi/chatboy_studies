# 17 · Laboratório de geração: schemas de chat, Verbalized Sampling, "nunca…" como identidade e cabeçalho de roleplay

> Prefixo `c1`. Scripts em `scripts/analysis/c1_*.py`, saídas pequenas em `analysis/data/c1_*` e gerações e avaliações
> brutas em `data/processed/c1_*` (fora do git).
> **Status: FINAL, mas incompleto.** Os créditos do OpenRouter acabaram no meio da validação no teste (toda chamada passou
> a devolver HTTP 402). Cada número abaixo vem marcado com **[TESTE]** ou **[DEV]**. O que faltou rodar está listado na §9.

## 1. Resumo (números)

- **O pipeline combinado resolve a forma, e a maior parte do ganho sai de uma só chamada.** [TESTE, 119 pontos, 35
  conversas] O D (distância às taxas humanas, ↓ = melhor) cai de **3,79 → 1,26** na lite, **3,16 → 1,19** no luna,
  **1,55 → 1,17** no deepseek e **2,15 → 1,19** no mercury. A configuração é cabeçalho de roleplay + ficha de
  comportamentos situacionais + nota do diretor do Jev no FIM, uma chamada, + normalizador (P1). Todas as reduções têm IC
  pareado longe de 0: lite −2,53 [−2,96; −2,06], luna −1,97, deepseek −0,38 [−0,81; −0,08], mercury −0,96. Gerar 4 ou
  pedir 5 variações (VS) e escolher em código melhora pouco ou nada: só a lite ganha 0,06–0,19 (IC exclui 0); nos outros
  três atores o IC inclui 0.
- **Ator mais perto do humano no teste (D):** lite NCT+N 1,07 [0,87–1,30]; luna NCT+N 1,10 [0,96–1,32]; deepseek
  VS-5 sem nota + código 1,15 [0,93–1,37] ≈ P1 1,15; mercury VS-5 top 1,14 ≈ NCT+N 1,17. **Todos ficam entre 1,07 e 1,20
  e os ICs se sobrepõem.** Neste teste, portanto, não há vencedor entre as 7 combinações por ator; a diferença real é
  "LLM pura × qualquer combinação com briefing".
- **Schemas de chat (C)** [TESTE]:
  - **Sem instrução nenhuma de estilo,** trocar "persona + mensagens" por um log de exportação já corta muito o D. Com o
    Messenger JSON: lite 3,79 → 1,14 e deepseek 1,55 → 1,25. Com o WhatsApp: luna 3,16 → 1,79 e mercury 2,15 → 1,54.
  - **O preço é a invenção da fala do usuário.** O deepseek escreve a próxima fala do Alex em 38–48% dos logs de linha
    (WhatsApp, IRC), o mercury em 9–34% e a lite em até 17% (SMS). O luna fica em ≤ 3%.
  - **Os JSONs quebram o formato:** Snapchat 23–52% e Messenger 10–37% no luna, no deepseek e no mercury.
  - **Com a nota do diretor, o schema deixa de importar:** D 1,03–1,34 em todos os formatos. Luna, deepseek e mercury
    **abandonam o log** e respondem solto em 21–100% dos casos; só a lite mantém o formato (exceto no SMS).
- **Tempos propostos pelo log** [TESTE]:
  - As medianas são plausíveis: 11–24 s propostos contra 16–20 s reais.
  - O ρ contra o tempo real fica em 0,32–0,57, mas **copiar a latência anterior do Sam no histórico já dá ρ 0,59 e erro
    |log2| 0,83**, contra 0,89–1,28 da LLM. A LLM não acrescenta nada ao ritmo que já está no histórico. No dev,
    ρ = 0–0,3.
  - Nos JSONs, a lite escreve sempre **1 bolha**, contra 1,46 do humano.
- **Verbalized Sampling (A)** [DEV]:
  - O formato exato do paper, com a instrução só no system, é **ignorado**: JSON válido em 63% no luna e 3% no deepseek.
    Com a instrução como última mensagem, 90–100%.
  - VS-5 dá mais movimentos distintos por ponto que 4 seeds: 1,9–2,6 contra 1,6–2,1.
  - A estratégia "a de maior probabilidade" nunca é a melhor. A escolha por violação em código dá D 1,13–1,24.
  - **Sem nota,** VS-5 + filtro de código leva a lite a 1,15 e o luna a 1,44; 4 seeds + o mesmo filtro dão 2,67 e 2,96.
    Esse é o único cenário em que o VS vence com folga.
  - **VS dentro do Messenger JSON** (combinação com o melhor schema): D 1,06–1,14 com escolha por código na lite, no
    deepseek e no mercury. O luna quebra (JSON 58%).
- **Restrições "nunca…" (B)** [TESTE, 53 contextos × 4 atores]:
  - Sem restrição, o personagem viola o que a ficha proibiria em 64–83% dos casos. O imperativo derruba para 15–34% e a
    identidade em 3ª pessoa para 26–32%. **Identidade não é melhor que imperativo:** em todos os atores os ICs se
    sobrepõem, e no deepseek o imperativo é melhor (15% × 28%).
  - Os dois formatos têm o mesmo efeito colateral: menção indireta 30–55% (× 5–10% sem restrição), resposta "fria"
    28–51% (× 2–11%), recusa explícita 13–30%, e coerência de 0,87–0,91 para 0,72–0,85.
  - O Noul do Jev bateu com a checagem manual em 60/60 casos (não cega) e em 20/20 casos cegos.
- **Cabeçalho de roleplay (D):**
  - [TESTE] Sozinho (H0, sem nota), dá D 1,10–1,33, contra 1,55–3,79 da persona mínima. Com a nota no fim (H1e), dá
    1,15–1,25, **igual à nota sozinha** (M1: 1,13–1,28).
  - [DEV] A mesma ficha escrita com **adjetivos** fica em 2,9–3,3 na lite e no luna, igual à LLM pura. O que funciona é
    descrever comportamentos situacionais, não o tom de "você é um personagem".
- **Banco atômico do Jev** (P(LLM), ↓ = mais humano; humano = 0,53) [TESTE]:
  - LLM pura: 0,71–0,91. Cabeçalho: 0,60 na lite e 0,75 no luna. Messenger sem nota: 0,68–0,81.
  - [DEV] Escolher pelo banco o baixa ainda mais, até abaixo do humano, mas é parcialmente circular. Com meia-bateria A
    para escolher e meia-bateria B para avaliar, o ganho persiste (por exemplo, lite 0,61 → 0,55) ao custo de 0,03–0,10
    de coerência. **Esse braço não chegou a ser medido no teste.**
- **Custo/latência por resposta** [TESTE, LLM]:
  - P1: US$ 0,02–0,15/mil respostas, p50 0,8–1,2 s.
  - NCT+N e 4 seeds: 2,5–4× o custo, p50 1,1–1,7 s.
  - VS-5: 1 chamada mais longa, p50 0,9–1,9 s.
  - Somar a leitura do Jev (~0,5 s, 1 chamada) e, se a escolha for pelo banco, 1 chamada por candidata em paralelo
    (+~0,5 s).

## 2. O que é teste e o que é dev

| Experimento | Gerações | Métricas de código (D, bolhas, invenção, formato, tempos) | Métricas do Jev (movimento, coerência, banco) |
|---|---|---|---|
| (1) Schemas × nota | dev + teste, 4 atores, 7 schemas × {sem, com nota} | **TESTE** completo | **TESTE** em livre\|0, livre\|1 e messenger\|0 (4 atores); o resto só no DEV |
| (2) Verbalized Sampling | só dev (60 pts), mais VS-5 em pipeline no teste | VS isolado **DEV**; VS no pipeline **TESTE** | **DEV** |
| (3) Restrições | 75 contextos (22 dev / 53 teste) × 3 formatos × 4 atores | **TESTE** | **TESTE** (Nouls de violação e efeito colateral) |
| (4) Cabeçalho | H0/H1e dev + teste; H1s/HA0 só dev | **TESTE** | **TESTE** só lite e luna (o 402 parou o deepseek e o mercury) |
| (5) Pipeline combinado | dev + teste, 4 atores, todas as configurações | **TESTE** (configurações sem banco) | **DEV**; no teste só o humano e A |

## 3. Desenho

- **Pontos:** os 179 pontos de decisão reais do maichat do a9/b4, com dev = 60 pontos em 6 conversas e teste = 119
  pontos em 35 conversas; a divisão é por conversa. Cada bolha do histórico entra com o seu **timestamp real** do
  maichat.
- **Atores:** `google/gemini-3.5-flash-lite` (lite), `openai/gpt-6-luna` (luna), `~deepseek/deepseek-flash-latest`
  (deepseek) e `inception/mercury-2.5` (mercury). Parâmetros aceitos sondados: o luna não aceita temperature nem stop, e o
  mercury não aceita seed. O raciocínio ficou desligado em todos.
- **Métricas:**
  - D do relatório 13: erro de tamanho + Σ|taxa − taxa humana| em 10 vícios.
  - Bolhas e intervalos propostos contra os reais.
  - Invenção da fala do usuário: a 1ª linha de outro falante é cortada e contada.
  - Formato quebrado (nada utilizável) e "solto" (a LLM largou o schema e respondeu em texto livre).
  - Coincidência de movimento: Choice do Jev de 16 movimentos sobre a resposta × sobre a resposta humana.
  - Coerência: Noul do Jev "faz sentido como resposta?".
  - Escore "código + banco atômico do Jev" (LR do relatório 10, 87 perguntas), com cross-fitting por conversa.
  - IC 95% por bootstrap por conversa (1000 reamostragens); diferenças pareadas por ponto.
- **Seleção honesta:**
  - As escolhas foram feitas no dev e as configurações levadas ao teste foram fixadas antes de vê-lo (checkpoint do
    rascunho).
  - Para não escolher e avaliar com o mesmo pontuador, o banco foi partido em duas metades. A meia-bateria A (código +
    43 perguntas, AUC-CV 0,885 no b1) serve para ESCOLHER. A meia-bateria B (42 perguntas, sem código, AUC-CV 0,821)
    serve para AVALIAR.
- **Nota do diretor:** o briefing de alvos T do relatório 13, calculado em código a partir da leitura do Jev do a9.
- **Normalizador:** o `normalize` do b4.

## 4. Resultados

### 4.1 Schemas de chat [TESTE; humano: 6 palavras, 12% "?", 1,46 bolhas, banco 0,53]

D [IC95] · inv = inventa a fala do usuário (%) · qbr = quebrado (%) · solto (%) · mov = movimento igual ao humano (%) ·
banco = P(LLM).

**Sem nota do diretor**

| schema | lite | luna | deepseek | mercury |
|---|---|---|---|---|
| (a) livre (A) | 3,79 [3,39–4,17] · mov 36 · banco 0,91 | 3,16 [2,80–3,56] · mov 37 · banco 0,89 | 1,55 [1,34–1,90] · mov 36 · banco 0,71 | 2,15 [1,86–2,50] · mov 34 · banco 0,76 |
| (b) WhatsApp export | 1,30 [1,13–1,52] · inv 2 | 1,79 [1,59–2,05] · inv 1 | 1,47 [1,29–1,75] · **inv 38** | 1,65 [1,42–1,90] · inv 9 |
| (c) WhatsApp só hora | 1,36 [1,21–1,57] · inv 1 | 1,98 [1,80–2,24] · inv 3 | 1,46 [1,26–1,74] · **inv 39** | 1,54 [1,29–1,84] · inv 9 |
| (d) Messenger JSON | **1,14 [1,02–1,35]** · inv 0 · mov 40 · banco 0,75 | 1,77 [1,53–2,04] · qbr 10 · banco 0,81 | 1,25 [1,08–1,53] · inv 9 · mov 41 · banco 0,68 | 1,83 [1,56–2,17] · **qbr 37** · banco 0,75 |
| (e) Snapchat JSON | 1,28 [1,11–1,49] | 1,82 [1,66–2,04] · qbr 23 | 1,16 [1,04–1,44] · **qbr 52** | 1,39 [1,17–1,67] · qbr 27 |
| (f) IRC/Discord | 1,50 [1,25–1,81] · inv 8 | 2,19 [1,96–2,44] | 1,32 [1,14–1,60] · **inv 48** | 1,86 [1,63–2,21] · inv 22 |
| (g) SMS | 1,54 [1,29–1,83] · inv 17 | 2,21 [1,96–2,50] | 1,42 [1,25–1,66] · inv 17 · solto 60 | 2,01 [1,74–2,32] · **inv 34** |

**Com a nota do diretor antes do log**

| schema | lite | luna | deepseek | mercury |
|---|---|---|---|---|
| (a) livre | 1,10 [0,92–1,34] · banco 0,69 | 1,19 [1,03–1,40] · banco 0,83 | 1,23 [1,01–1,51] · banco 0,64 | 1,28 [1,12–1,54] · banco 0,66 |
| (b) WhatsApp export | 1,16 · solto 0 | 1,14 · solto 39 | 1,20 · solto 82 | 1,19 · solto 99 |
| (c) WhatsApp só hora | 1,26 · solto 0 | 1,20 · solto 67 | 1,18 · solto 94 | 1,25 · solto 100 |
| (d) Messenger JSON | 1,17 · solto 0 | 1,21 · solto 28 | 1,09 · qbr 15 · solto 15 | 1,34 · solto 99 |
| (e) Snapchat JSON | 1,11 · solto 0 | 1,17 · solto 21 | 1,03 · qbr 44 · solto 14 | 1,33 · solto 93 |
| (f) IRC | 1,20 · solto 1 | 1,19 · solto 68 | 1,13 · solto 100 | 1,25 · solto 99 |
| (g) SMS | 1,18 · solto 94 | 1,14 · solto 98 | 1,20 · solto 100 | 1,09 · solto 100 |

Com a nota, a invenção da fala do usuário cai a 0% em todos os casos.

- [DEV] Pôr a nota dentro do system em vez de antes do log (|2) não segura o formato: deepseek e mercury respondem
  soltos em 93–100% dos casos.
- **Bolhas:** com o log, o deepseek e o mercury propõem 1,3–1,7 bolhas, inflado porque continuam escrevendo. A lite
  propõe 1,00–1,03 nos JSONs, e o humano manda 1,46.
- **Tempos propostos** (sem nota; latência mediana proposta/real · ρ · erro |log2|):

| schema | lite | luna | deepseek | mercury |
|---|---|---|---|---|
| WhatsApp export | 23/18 s · 0,41 · 0,97 | 21/19 s · 0,41 · 1,08 | 19/18 s · 0,49 · 0,94 | 13/19 s · 0,43 · 1,07 |
| WhatsApp só hora | 23/18 s · 0,45 · 0,95 | 16/18 s · 0,32 · 1,06 | 21/18 s · 0,53 · 0,89 | 14/19 s · 0,44 · 1,02 |
| Messenger JSON | 24/18 s · 0,52 · 0,96 | 18/20 s · 0,43 · 1,09 | 16/19 s · 0,57 · 0,89 | 11/16 s · 0,41 · 1,08 |
| **referência: latência anterior do Sam** | ρ 0,59 · erro 0,83 | | | |
| **referência: mediana constante** | erro 1,12 | | | |

No dev, ρ ficou em 0–0,3 e a referência "latência anterior" em 0,16. Os intervalos entre bolhas propostos (0–21 s) não
têm relação com os reais.

### 4.2 Verbalized Sampling [DEV, 60 pontos]

- **Formato:** o prompt do paper (§G.3, Dialogue Simulation: "Generate N plausible responses… JSON 'responses' com text
  e probability") foi usado de duas formas.
  - Só no system, como no paper: JSON válido 98% na lite, **63% no luna, 3% no deepseek** (ele ignora e responde normal)
    e 87–93% no mercury.
  - Como última mensagem ("t"): 90–100% em todos.
- **Diversidade:**
  - Textos distintos: 100% das candidatas.
  - Movimentos distintos por ponto: VS-3 1,65–1,82 e **VS-5 1,88–2,55**, contra 1,60–2,13 com 4 seeds (a 5ª candidata
    ajuda pouco).
  - A probabilidade da candidata "top" é 0,31–0,36, exceto no mercury, que dá 0,65–0,69 e cuja soma passa de 1.
- **D por estratégia** (VS-5t com nota; top · lowpass · viol · bank_pass):
  - lite 1,15 · 1,23 · 1,16 · 1,19
  - luna 1,25 · **1,08** · 1,13 · 1,14
  - deepseek 1,26 · 1,37 · 1,17 · 1,30
  - mercury 1,32 · 1,32 · 1,23 · 1,35
  - Referência com 4 seeds + viol: 1,26 / 1,26 / 1,13 / 1,14.
  - **Com nota, VS não vence 4 seeds** (diferenças de ≤ 0,1 com ICs de ±0,5). "Maior probabilidade" é a pior
    estratégia em 3 de 4 atores.
- **Sem nota** (o VS "puro"): VS-5t + viol dá 1,15 / 1,44 / 1,17 / 1,16, contra 4 seeds + viol 2,67 / 2,96 / 1,13 /
  1,45. Aqui o VS vence com folga na lite e no luna: pedir 5 variações faz a LLM oferecer opções curtas e secas, e o
  código escolhe uma.
- **Escolha pelo banco:** baixa o P(LLM) (0,68 → 0,48–0,58) e baixa a coerência (0,84–0,89 → 0,76–0,84). Ver o
  controle de circularidade na §4.5.
- **VS dentro do Messenger JSON** (combinação com o melhor schema, sem Jev):
  - Sem nota, com viol: 1,06 / 1,87 / 1,14 / 1,06. O luna quebra: JSON válido 58%, 42% de queda para a LLM pura.
  - Com nota: 1,27 / 1,30 / 1,24 / 1,14.
  - Não supera o cabeçalho + nota em modo livre.

### 4.3 Restrições como identidade × imperativo [TESTE: 14 personagens, 53 contextos que tentam o personagem, 4 atores]

| ator | formato | violação % [IC95] | menção indireta % | esquiva % | recusa % | "fria" % | coerência | palavras |
|---|---|---|---|---|---|---|---|---|
| lite | nenhuma | 75 [65–85] | 5 | 6 | 8 | 11 | 0,89 | 27 |
| lite | imperativo | 28 [17–40] | 40 | 19 | 17 | 30 | 0,80 | 14 |
| lite | identidade | 26 [12–45] | 35 | 34 | 13 | 43 | 0,72 | 12 |
| luna | nenhuma | 83 [74–91] | 5 | 2 | 2 | 2 | 0,91 | 28 |
| luna | imperativo | 26 [14–39] | 55 | 21 | 26 | 28 | 0,85 | 22 |
| luna | identidade | 28 [17–40] | 55 | 25 | 30 | 30 | 0,84 | 20 |
| deepseek | nenhuma | 64 [53–74] | 10 | 6 | 2 | 11 | 0,90 | 20 |
| deepseek | imperativo | **15 [6–28]** | 55 | 36 | 30 | 51 | 0,79 | 15 |
| deepseek | identidade | 28 [17–42] | 45 | 36 | 19 | 40 | 0,77 | 12 |
| mercury | nenhuma | 70 [59–81] | 10 | 6 | 0 | 8 | 0,87 | 17 |
| mercury | imperativo | 34 [20–49] | 50 | 28 | 23 | 34 | 0,81 | 11 |
| mercury | identidade | 32 [20–44] | 30 | 32 | 25 | 38 | 0,76 | 9 |

- **Juntando os 4 atores:** nas restrições de TEMA, a violação vai de 90% (nenhuma) para 30% (imperativo) e 33%
  (identidade). Nas de COMPORTAMENTO, de 63% para 23% e 27%. A restrição de tema é a mais tentada e a mais violada sem
  instrução.
- **No dev** (22 contextos) a identidade pareceu um pouco melhor na lite e no deepseek (14% e 9% × 18% e 14%). **Isso
  não se replicou no teste.**
- **Validação:** 60 casos checados manualmente com concordância de 100% (checagem não cega). Uma 2ª amostra **cega** de
  20 casos também deu 20/20 (`c1_constraint_manual_blind.json`).
- **Efeito colateral:** as duas formas trocam "falar do pai" por "desviar do pai". A menção indireta sobe de 5–10% para
  30–55%, a esquiva robótica de 2–6% para 19–36%, e as respostas encurtam (27 → 9–22 palavras). O D_ref cai por isso, não
  porque a resposta fique mais humana.

### 4.4 Cabeçalho de roleplay × persona mínima

**[TESTE]** (M0 = persona mínima = A; M1 = persona + nota; H0 = cabeçalho + ficha situacional; H1e = H0 + nota no FIM)

| ator | M0 | M1 | H0 | H1e | banco M0 → H0 / H1e |
|---|---|---|---|---|---|
| lite | 3,79 [3,40–4,18] | 1,13 [0,95–1,37] | 1,33 [1,13–1,60] | 1,25 [1,05–1,49] | 0,91 → 0,60 / 0,61 |
| luna | 3,16 [2,81–3,55] | 1,19 [1,04–1,42] | **1,15** [0,98–1,44] | 1,18 [1,00–1,41] | 0,89 → 0,75 / 0,82 |
| deepseek | 1,55 [1,34–1,93] | 1,23 [1,00–1,53] | **1,10** [1,01–1,39] | 1,15 [0,95–1,44] | 0,71 → (não medido) |
| mercury | 2,15 [1,87–2,48] | 1,28 [1,13–1,54] | 1,29 [1,13–1,61] | 1,19 [0,97–1,44] | 0,76 → (não medido) |

**[DEV]** Duas variantes testadas só no dev:
- Ficha de **adjetivos** (HA0): 3,30 / 2,94 / 1,52 / 1,51, contra H0 1,37 / 1,29 / 1,15 / 1,34. Os adjetivos não servem.
- Nota no começo (H1s) contra nota no fim (H1e): sem diferença (1,27 × 1,25; 1,19 × 1,17).

No dev, a coerência do cabeçalho sem nota caiu um pouco (0,79–0,85 × 0,84–0,88), com respostas às vezes secas demais
("don't tell me what to do", ver exemplos).

### 4.5 Pipeline combinado [TESTE para D/custo/latência; DEV para o Jev]

Configurações:
- **A:** LLM pura.
- **NCT+N:** o melhor do relatório 13 (4× briefing T, escolhe a de menor violação e normaliza), agora medido no teste.
- **P1:** cabeçalho + ficha situacional + nota no fim, 1 chamada, + normalizador.
- **P4:** P1 × 4 seeds, filtros de código e escolha por violação.
- **P5:** P1 + VS-5 numa chamada, escolha por violação ou top.
- **P6:** cabeçalho + VS-5 **sem** nota, escolha por violação.

Nenhum pipeline usa schema de log, porque com a nota o log é abandonado (§4.1).

**D [IC95] no TESTE (119 pontos); humano = 0**

| config | lite | luna | deepseek | mercury | custo US$/1000 resp (lite/luna/ds/merc) | LLM p50 s |
|---|---|---|---|---|---|---|
| A (LLM pura) | 3,79 [3,39–4,17] | 3,16 [2,78–3,56] | 1,55 [1,34–1,91] | 2,15 [1,84–2,45] | 0,09 / 0,02 / 0,03 / 0,01 | 0,8–1,3 |
| NCT+N (rel. 13) | **1,07 [0,87–1,30]** | **1,10 [0,96–1,32]** | 1,22 [0,98–1,46] | 1,17 [0,95–1,39] | 0,36 / 0,14 / 0,15 / 0,06 | 1,1–1,7 |
| P1 cabeçalho + nota fim | 1,26 [1,04–1,54] | 1,19 [0,99–1,42] | 1,17 [0,95–1,43] | 1,19 [0,98–1,44] | 0,15 / 0,05 / 0,06 / 0,02 | 0,8–1,2 |
| P4 ×4 seeds + viol | 1,20 [0,98–1,45] | 1,17 [0,99–1,39] | 1,24 [1,01–1,47] | 1,20 [0,98–1,45] | 0,58 / 0,22 / 0,14 / 0,09 | 1,1–1,6 |
| P5 VS-5 + viol | 1,18 [0,95–1,42] | 1,21 [0,99–1,44] | 1,24 [1,00–1,49] | 1,19 [0,93–1,46] | 0,53 / 0,11 / 0,12 / 0,05 | 0,9–1,9 |
| P5 VS-5 + top | 1,22 | 1,21 | 1,27 | **1,16** [0,95–1,47] | idem | idem |
| P5 VS-5 + lowpass | 1,28 | 1,14 | 1,40 | 1,31 | idem | idem |
| P6 VS-5 sem nota + viol | 1,36 [1,11–1,62] | 1,20 [0,94–1,43] | **1,15 [0,95–1,36]** | 1,17 [0,95–1,46] | 0,44 / 0,09 / 0,12 / 0,04 | 0,9–1,8 |

Diferenças pareadas no teste (IC95; negativo = o 1º está mais perto do humano):
- P1 − A: lite −2,53 [−2,96; −2,06] · luna −1,97 [−2,38; −1,52] · deepseek −0,38 [−0,81; −0,08] · mercury −0,96
  [−1,35; −0,56].
- NCT+N − P1: lite **−0,19 [−0,29; −0,10]** · luna −0,09 [−0,20; +0,03] · deepseek +0,05 · mercury −0,02 (os dois
  últimos com IC incluindo 0).
- P4 ou P5 − P1: lite −0,06 e −0,08 (IC exclui 0 por pouco) · nos outros três atores −0,02 a +0,07, IC incluindo 0.
- P1 − M1 (nota sozinha, sem cabeçalho): lite **+0,14 [+0,02; +0,24]** (o cabeçalho piora a lite: respostas curtas
  demais, 3–4 palavras contra 6 do humano) · luna −0,01 · deepseek −0,06 · mercury −0,10, com IC incluindo 0.
- P1 − Messenger sem nota: lite +0,13 (IC incluindo 0) · luna −0,59 · deepseek −0,08 · mercury −0,64.

Forma no teste (humano: 12% "?", 7% "!", 5% clichê, 36% multi-bolha):
- Todos os pipelines têm 0–11% de "?", 0–1% de "!" e 0–6% de clichê.
- Multi-bolha fica em 11–29%. Os pipelines passam a **sub-usar** "!", "?" e riso (0–2% × 7%). É uma correção
  excessiva, já vista no relatório 13.

**Jev [DEV, 60 pontos]** (humano: banco 0,45, meia-B 0,60, coerência 0,78, entropia de movimento 3,11):

| config | movimento = humano (lite/luna/ds/merc) | coerência | banco completo | meia-B (avaliação independente) |
|---|---|---|---|---|
| A | 40 / 22 / 28 / 33 | 0,84–0,88 | 0,65–0,91 | 0,73–0,92 |
| NCT+N | 49 / 43 / 50 / 40 | 0,83–0,89 | 0,57–0,81 | 0,64–0,75 |
| P1 | 42 / 37 / 45 / 32 | 0,83–0,88 | 0,57–0,78 | 0,61–0,75 |
| P4 + viol | 43 / 35 / 49 / 35 | 0,82–0,89 | 0,59–0,78 | 0,61–0,75 |
| P4 + escolha pela meia-A | 42 / 37 / 43 / 40 | 0,78–0,85 | 0,46–0,67 | **0,55–0,72** |
| P5 + escolha pela meia-A | 41 / 37 / 40 / 43 | 0,72–0,81 | 0,46–0,64 | 0,45–0,63 |
| P6 + escolha pelo banco | 27 / 25 / 31 / 39 | 0,65–0,77 | 0,29–0,42 | 0,36–0,49 (abaixo do humano) |

- **A escolha pelo banco não é só circular:** escolhendo com a meia-A, a meia-B também cai (lite 0,61 → 0,55; deepseek
  0,67 → 0,63; mercury 0,65 → 0,60).
- Mas ela troca coerência e coincidência de movimento por "parecer humano" no banco. Com VS sem nota, passa do humano
  (0,36–0,49 × 0,60) e a coerência cai para 0,65–0,77, abaixo da do próprio humano no dev (0,78).
- A coincidência de movimento com o humano fica em 32–50% em todos os pipelines, contra 22–40% da LLM pura. É o mesmo
  teto dos relatórios 12/13: o briefing acerta a forma, não a escolha do movimento.

### 4.6 Custo e latência por resposta (sistema completo)

- **LLM** (teste, medido): a tabela da §4.5 dá de US$ 0,00001 a US$ 0,0006 por resposta.
- **Jev:** a leitura do a9 (1 chamada, ~0,5 s) vale para todas as configurações. A escolha pelo banco acrescenta 1
  chamada por candidata (4–5, em paralelo, +~0,5 s, US$ 0,04/Mtok ≈ US$ 0,00016 por candidata com ~4 k tokens de
  estado + 87 perguntas).
- **Código:** < 5 ms.
- **Total por resposta:**
  - P1: ~1,3–1,7 s, 1 chamada de LLM + 1 de Jev.
  - NCT+N e P4: ~1,6–2,2 s, 4 de LLM + 1 de Jev.
  - P4 com escolha pelo banco: ~2,1–2,7 s, 4 de LLM + 5 de Jev.

## 5. Exemplos reais lado a lado [TESTE]

Contexto `conv049_11` (…"Btw its chilli on monday for tea is that okah"). **Humano:** "Yeah i like your chilli loads /
Have you got rice".

| ator | A (LLM pura) | Messenger sem nota | livre + nota | cabeçalho (H0) | NCT+N | P4 ×4 + viol |
|---|---|---|---|---|---|---|
| lite | "Chilli sounds great! Perfect for a Monday. / And yeah I'll bring the pecans over then too 👍" | "Yeah chilli sounds great" | "yeah chilli sounds good I will bring the nuts then" | "Chilli is fine" | "yeah chilli works for me i can do that night" | "chilli is fine with me" |
| luna | "Yeah, chilli sounds great! And I'll save you some pecans for Monday 😊" | "Yeah chilli sounds lovely, see you Monday 😊" | "Yeah chilli sounds good, and I'll bring you some pecans on Monday" | "Aye, chilli sounds good for Monday" | "yeah, chilli sounds good, and i'll bring the pecans over on monday for tea" | "Yeah, chilli's good for Monday, I'll bring the pecans then too" |
| deepseek | "Yeah chilli's sound" | "Yeah chilli sounds good / Got loads of pecans so no rush / Just finished the cottage pie / On the jaffas now / How's the mousse" | "Think you said tea Monday chilli is fine" | "Chilli's fine / I'll bring your nuts" | "Chilli is fine and i will get the pecans ready" | "Chilli's fine but go easy on the beans" |

Contexto `conv063_58` ("sleep well" / "u too"). **Humano:** "night cousin". Luna A: "thank u 🥺 miss u guys too, sleep
well ❤️" · luna P4: "night, talk soon" · deepseek H0: **"don't tell me what to do"** (secura exagerada do cabeçalho) ·
mercury H0: "got it".

**Invenção da fala do usuário** (deepseek, WhatsApp só hora, sem nota, `conv086_1`, último: "hello? u there son"):
```
[21:15:02] Sam: yeah im here dad, whats up
[21:15:30] Alex: just checking in. hows the job hunt going      ← inventado (cortado pelo código)
[21:16:11] Sam: slow tbh. sent out like 15 applications this week...
```

**VS-5** (luna, cabeçalho + nota, `conv089_1`, "hellooo are u ignoring me again"; humano: "nooo i was in the shower 😩"):
0,38 "oh, no, i got pulled into work and forgot to reply" · 0,27 "i wasn't ignoring you, my phone was on silent" ·
0,18 "yeah, sorry, i got caught up with stuff earlier" · 0,10 "oh, no, i've been dealing with a bunch of errands" ·
0,07 "i was in the middle of cooking and missed your texts". As 5 opções são variações do mesmo movimento (desculpa +
motivo): o VS diversifica o texto, não a decisão.

**Restrições** [TESTE]:

| contexto (ator) | nenhuma | imperativo | identidade |
|---|---|---|---|
| "how's your dad doing? you never really mention him" (mercury) | "he's good, thanks. it's just never come up. what made you ask" | "let's not go there. what about that gym you mentioned" | "changing the subject. what's on your mind" |
| "u forgot again didnt u" — nunca pedir desculpa primeiro (deepseek) | "im sorry. got called back in. shouldve texted" | "shit. we got a call right as i was leaving. big one. i shouldve texted you" | "call came in. big one. couldnt leave. / i know. im sorry." (viola) |
| "should i text her back or just leave it" — não dar conselho (deepseek) | "why are you asking me if you should text her back? she's your coworker, not mine" | "why is she texting you" | "leave it. dinner at 8 then" (viola) |

Mais exemplos em `analysis/data/c1_examples_test.json`.

## 6. Recomendação para o sistema

1. **Padrão para os 4 atores:** cabeçalho de roleplay com a ficha escrita como **comportamentos situacionais** + nota
   do diretor do Jev **no fim** (uma mensagem de system depois da última fala do usuário) + normalizador, em **uma
   chamada** (P1).
   - Leva todos os atores a D 1,15–1,26 [TESTE], contra 1,55–3,79 da LLM pura, com latência de LLM p50 0,8–1,2 s.
   - Na **lite**, trocar P1 por NCT+N ou por "nota sozinha" (M1): o cabeçalho a deixa curta demais (P1 − M1 = +0,14) e
     NCT+N é 0,19 melhor que P1 [TESTE].
2. **Gerar várias e escolher (4 seeds ou VS-5) não compensa** como padrão: +0 a 0,08 de D, 2–4× o custo.
   - Se for usar, **VS-5 numa chamada com a instrução no fim**, escolhendo por violação em código. Nunca "maior
     probabilidade". Custa 1 chamada, contra 4.
   - O VS vale de verdade **quando não há nota**, por exemplo como modo degradado se o Jev falhar: lite 3,79 → 1,15 e
     luna 3,16 → 1,44 [DEV].
3. **Não use schemas de log como formato de produção.** Eles funcionam sem instrução (lite 1,14 no Messenger), mas:
   - o deepseek e o mercury inventam a fala do usuário em até 48%;
   - os JSONs quebram em até 52%;
   - com a nota, o schema é abandonado.
   - Se usar, **só com a lite**, no Messenger JSON, e com o corte na 1ª linha de outro falante sempre ligado.
4. **Tempos:** não peça à LLM para propor horários. A latência anterior do personagem no histórico prevê melhor (ρ 0,59
   × ≤ 0,57). O tempo deve continuar a sair de código ou do Jev.
5. **Restrições da ficha:** escreva no formato que for mais natural. Identidade ("{{char}} nunca…") e imperativo dão a
   mesma violação (26–34% × 15–34%). Três cuidados:
   - A violação não vai a zero. **Verifique com um Noul do Jev por restrição** (concordância 80/80 com a checagem
     manual) e regenere quando violar.
   - Prefira dar à restrição um **comportamento alternativo** ("quando perguntam do pai, fala de outra coisa da família
     / responde curto e muda de assunto com naturalidade"). As duas formas testadas geram esquiva robótica e menção
     indireta em 30–55%.
   - Restrições de tema são as mais violadas sem instrução (90%).
6. **Escolha pelo banco atômico do Jev:** use só com **meia-bateria de escolha separada da de avaliação** e com um piso de
   coerência (Noul "faz sentido?" ≥ 0,8). Sem esse piso, o banco empurra para respostas secas e incoerentes, mais
   "humanas" que o humano pelo escore.

## 7. Limitações

- **Validação interrompida (HTTP 402):** o banco, a coerência e o movimento dos pipelines no teste não foram medidos. O
  braço "escolha pela meia-bateria A" só existe no dev. O cabeçalho no teste tem Jev só na lite e no luna.
- **Tamanho:** 119 pontos de teste (35 conversas) dão IC de D com ±0,2–0,3. As diferenças entre pipelines (≤ 0,2)
  ficam, em geral, dentro do ruído. A escolha por ator no dev (60 pontos, 6 conversas) é quase ruído: a melhor
  configuração no dev não foi a melhor no teste em 2 de 4 atores.
- **D mede forma, não conteúdo.** Os pipelines reduzem a forma para perto do humano e às vezes passam do ponto: "!",
  "?" e riso ficam abaixo do humano, e a lite escreve 3–4 palavras contra 6.
- Parte das gerações do dev e as condições NCT/T saíram do cache do b4, com corpo de chamada idêntico. Os custos desses
  registros são os custos originais.
- **Contextos das restrições:** foram gerados por LLM (deepseek), a partir de especificações escritas à mão. São mais
  "tentadores" que conversas reais.
- A checagem manual de 60 casos não foi cega. A amostra cega é pequena (20).
- **Pontuador:** o banco e as meias-baterias foram treinados nos dados do b1 (incluindo conversas do maichat), com
  cross-fitting por conversa. Ainda assim, o humano recebe P(LLM) 0,45–0,53 (e 0,60–0,62 na meia-B), e não 0.
- Um único corpus (maichat, inglês) e um personagem-alvo por ponto (Sam).

## 8. Variantes tentadas (todas)

- **Schemas:** 7 schemas × {sem nota, nota antes do log} (dev + teste). Nota no system (|2) em 4 schemas (dev).
- **VS:**
  - VS-3 e VS-5 com a instrução no system (formato do paper) e como última mensagem ("t"), com e sem nota.
  - 4 seeds com e sem nota.
  - VS-5 dentro do Messenger JSON, com e sem nota.
  - 8 estratégias de escolha: top, probw (sorteio ∝ p), rand, filt_top, lowpass, viol, bank, bank_pass. Mais bankA_pass.
- **Restrições:** nenhuma × imperativo × identidade, 20 fichas, 15 tipos de restrição, 75 contextos.
- **Cabeçalho:** H0, H1e, H1s e HA0 (adjetivos).
- **Pipelines:** P1, P4 (viol / bank / bankA), P5 (top / low / viol / bank / bankA), P6 (low / viol / bank / bankA),
  cada um com e sem normalizador. O normalizador muda D em ≤ 0,03.

## 9. O que falta rodar (quando houver crédito)

Todas as gerações de LLM do teste já estão em `data/processed/c1_gen.jsonl`. Falta só o Jev, na ordem abaixo, de
`scripts/analysis/`:

```bash
python3 c1_header.py jev test H0,H1e                                   # ~500 chamadas restantes (deepseek, mercury)
python3 c1_pipeline.py jev test P1_hdr,P4_hdr_seed4_bankA "A,NCT+N,P1_hdr+N,P4_hdr_seed4_bankA+N,P4_hdr_seed4_bankA-N,P5_hdr_vs5_viol+N"   # ~3.000 chamadas
python3 c1_header.py analyze test H0,H1e
python3 c1_pipeline.py analyze test P1_hdr,P4_hdr_seed4_viol,P4_hdr_seed4_bankA,P5_hdr_vs5_top,P5_hdr_vs5_low,P5_hdr_vs5_viol,P6_hdr0_vs5_viol
python3 c1_tables.py pipe test          # tabelas para atualizar as §4.4–4.5
# opcional: Jev nos demais schemas do teste (≈ 2.400 chamadas)
python3 c1_schemas.py jev test "wa_full|0,wa_time|0,snapchat|0,irc|0,sms|0" && python3 c1_schemas.py analyze test
```

## 10. Arquivos

- **Scripts** (`scripts/analysis/`):
  - `c1_common.py`: pontos com a linha do tempo real, 7 schemas e seus parsers, cliente LLM com cache, avaliação do Jev,
    pontuador do banco com cross-fitting, meias-baterias A/B e `jev_recover_cached` (só local).
  - `c1_metrics.py`, `c1_schemas.py`, `c1_vs.py`, `c1_constraints.py`, `c1_header.py`, `c1_pipeline.py`,
    `c1_examples.py` e `c1_tables.py`.
- **Resultados** (`analysis/data/`):
  - `c1_schema_{dev,test}.json`, `c1_vs_dev.json`, `c1_constraints.json`, `c1_constraint_contexts.json`,
    `c1_constraint_manual.json` e `c1_constraint_manual_blind.json`.
  - `c1_header_{dev,test}.json`, `c1_pipeline_{dev,test}.json`, `c1_pipeline_test_diffs.json`,
    `c1_latency_baselines.json` e `c1_examples_test.json`.
- **Brutos:** `data/processed/c1_gen.jsonl` (gerações), `c1_jev_eval.jsonl` (≈ 11,5 mil avaliações),
  `c1_constr_jev.json` e `c1_llm_cache.jsonl`.
- **Gasto:** LLM ≈ US$ 2,17 (21,9 mil chamadas no cache c1; parte veio de graça do cache do b4). Jev ≈ 13 mil chamadas.
