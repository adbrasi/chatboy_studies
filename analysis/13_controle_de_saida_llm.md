# 13 · Controle de saída: como impedir os excessos da LLM (tamanho, pergunta, molde, entusiasmo)

> Prefixo `b4`. Scripts em `scripts/analysis/b4_*.py`; saídas em `analysis/data/b4_*`.
> **Status: PARCIAL.** A rodada foi interrompida por falta de créditos. O que está aqui foi medido **só no dev**
> (60 pontos reais do maichat, 6 conversas, os mesmos do a9), com os **4 atores permitidos** (`google/gemini-3.5-flash-lite`,
> `openai/gpt-6-luna`, `~deepseek/deepseek-flash-latest`, `inception/mercury-2.5`). **Não rodei o teste (119 pontos)**, a
> avaliação de coerência e perda de conteúdo pelo Jev, os mecanismos de várias etapas (reescrita guiada, rascunho → chat,
> plano → Jev → texto, intensidade) nem a simulação do controlador de distribuição. Os scripts estão prontos (§6).
> Gasto: ~4.500 gerações de LLM (≈ US$ 0,20) e ≈ 100 chamadas de Jev (< US$ 0,01).

---

## 1. Resumo com números (dev, n = 60; sem IC confiável, pois são só 6 conversas)

Métrica principal: **D = erro de tamanho** (média de |log2((palavras+1)/(humano+1))|) **+ Σ|taxa − taxa humana|** em 10 vícios
(pergunta, "!", emoji, riso, lista LLM-ish, 3+ frases, molde de 3 tempos, eco/paráfrase, abertura performática, pergunta
recíproca genérica). Menor é melhor; o humano tem D = 0.

- **Os atores partem de lugares muito diferentes.** Com a persona pura (A), o D foi de **3,90 (flash-lite)**, **3,68 (luna)**,
  1,93 (mercury) e 1,64 (deepseek). Mediana de palavras: 15 / 13 / 7 / 8, contra **5 do humano**. O luna põe emoji em **75%**
  das respostas (humano: 5%); o flash-lite pergunta em 50% e ri em 42% (humano: 15% e 3%).
- **O normalizador de código sozinho (sem LLM e sem Jev) corta de 25% a 47% do D.** Ele tira "!", emoji fora do orçamento,
  riso não autorizado, abertura performática ("omg", "wait", "aww"), travessão, vocativo e HTML. Na lite, "!" cai de 28% para
  0%, emoji de 37% para 7% e riso de 42% para 3%. **Não resolve tamanho nem pergunta.** Custo zero e latência zero.
- **A alavanca maior é o briefing de alvos (T)**: "about N words, one idea", pergunta e "!" sorteados com a taxa humana e uma
  linha positiva de como começar. O D cai para **1,32 / 1,40 / 1,25 / 1,34** (lite / luna / mercury / deepseek), e a mediana de
  palavras fica em 5–8. O molde de 3 tempos vai a 0% em todos. O briefing do a9 (B9) fica quase igual (1,38 / 1,31 / 1,39 / 1,33).
- **O melhor no dev foi "gerar 4 com o briefing T e escolher em código" (NCT)**, pela menor soma de violações do orçamento:
  D de **1,19 / 1,21 / 1,15 / 1,16**, com 4× o custo e +0,2–0,4 s de latência (as chamadas rodam em paralelo). Na lite, o
  **plano em JSON numa chamada só (PL1) + normalizador** foi ainda melhor (**1,08**). Mas o luna **ignorou o formato JSON em 90%**
  dos casos, então o PL1 não serve para ele.
- **A poda de frases guiada pelo Jev funciona, mas ganha pouco da regra** (só na lite): Jev "qual frase manter" (J1) + normalizador
  deu D **1,24**; a regra "1ª frase" (R1) + normalizador deu 1,38; o normalizador sem poda, 2,60. O Choice do Jev não mostrou viés
  de posição (escolheu a 1ª unidade em 22 de 52 casos, a 2ª em 25 e a 3ª em 5).
- **Os controles "de API" quase não servem com estes 4 atores:**
  - **`max_tokens` justo quebra a frase**: resposta cortada no meio em **90% (lite), 60% (luna), 40% (deepseek) e 33% (mercury)**.
    Com o conserto em código (voltar até a última frase completa), a quebra cai para 3–5%, mas o D não melhora e "!"/emoji ficam;
  - **`logit_bias`** não existe na lite, no luna nem no mercury; o deepseek **aceita e ignora** (o "!" continuou em 4 de 4 com todos
    os tokens de "!" em −100);
  - **prefill**: nenhum dos 4 aceita (a lite recusa com 400; os outros ignoram). "Comece com a palavra X" por instrução ajuda pouco
    (lite 2,50; +normalizador 1,79);
  - **temperatura e top_p**: a lite **ignora** (60 de 60 textos idênticos ao A); no mercury, não reduzem vício nenhum (D 2,31 × 1,93);
  - **`stop`**: parar em "?" zera a pergunta, mas deixa o resto (D quase igual); parar em "\n" ajuda pouco. O `stop` **equivale a
    um corte em código** (as métricas ficaram idênticas na lite), e isso é necessário para o luna, que não aceita `stop`.
- **Vícios que continuaram sem solução no dev:**
  - a **pergunta** fica ou alta demais (A: 25–50%) ou baixa demais (T na lite/mercury/deepseek: 2–5%, contra 15% humano). Mesmo
    quando o briefing manda "termine com uma pergunta", a lite obedece raramente. Só o luna seguiu (22%);
  - o **eco/paráfrase** do luna (17–20% com T, contra 5% humano);
  - a **lista LLM-ish** residual na lite com T (12%).

---

## 2. Arquiteturas testadas

Legenda: `LLM(x)` = chamada à boca; `J[...]` = chamada ao Jev; `C:` = código. O **orçamento** do ponto (alvo de palavras,
pergunta sim/não, riso, emoji, "!", minúsculas, sem ponto final) é calculado **em código** a partir da leitura do Jev do a9
(~30 perguntas, feita sem ver a resposta) e da impressão digital de estilo da persona, com limiares congelados no dev do a9.
A pergunta usa o limiar do a9 mais um sorteio de 7%, o que dá ~12–17% de pontos com pergunta.

```
A    persona+histórico ─► LLM ─► texto                                  (baseline)
S    A + prompt de estilo estático
B9   S + briefing do a9 ("Max N words", "No question"…)                 (1ª rodada)
T    S + briefing de ALVOS ─► LLM          briefing = C:orçamento(J[leitura a9], persona)
       "about N words, one idea" · pergunta/!/riso sorteados · "start plainly (oh/ok/yeah/i/so)"
F1   persona + "ONE short sentence, at most N words"
X+N  qualquer X ─► C:normalizador (tira !, emoji, riso, abertura performática, —, nome, HTML; minúscula/sem ponto da persona)

── mecanismo 1: na geração ──────────────────────────────────────────────────────────────
MT    A com max_tokens = 1,5·N+4            MTfix  MT ─► C:volta até a última frase completa
MT12  A com max_tokens = 12 (corte cego)
STnl  A com stop ["\n"]      STq  A com stop ["?"]      (…E = o mesmo corte feito em código)
TLO/THI  A com temperatura 0,3 / 1,2 + top_p 0,9
LB    A com logit_bias −100 em "!", totally, amazing, definitely, absolutely, honestly, 😂 (só deepseek)
PFX   persona + "Begin your reply with the word 'X'" (X sorteado das aberturas humanas: i/oh/so/yes/ok…)
NCA   4× A (seeds 0–3) ─► C:escolhe a de menor violação do orçamento     SHORT: escolhe a mais curta
NCT   4× T (seeds 0–3) ─► C:escolhe a de menor violação do orçamento
PL1   LLM("planeje em JSON {move, idea, length_words, question} e depois text") ─► C:pega o text

── mecanismo 2: poda por frase (só lite) ───────────────────────────────────────────────
A ─► C:quebra em unidades u1..un ─► J[state: 5 turnos + last_message + draft_reply{u1..un};
        por unidade: Noul essencial? · Noul pergunta genérica? · Noul valida/parafraseia? · Noul entusiasmo vazio?
        + Choice "se só pudesse mandar UMA, qual?" (opções embaralhadas)] ─► C:monta
  J1 = só a unidade do Choice · J2 = unidades essenciais sem vício (limiar 0,5), cortadas pelo orçamento
  J3 = J2 + política de pergunta do orçamento
  regras: R1 1ª unidade · R2 última · R3 tira as perguntas · R4 1ªs sem pergunta até o orçamento · R5 R4 + 1 pergunta se o orçamento pede
```

**Escritos, mas não rodados** (a rodada foi interrompida): reescrita cirúrgica guiada pelo Jev em até 3 voltas (RW), rascunho →
versão de chat com e sem alvos (D5/D5x), plano → código fixa os números → Jev valida a ideia → texto (PL2), controle de intensidade
"resposta ≤ usuário" por escolha entre 4 candidatas ou reescrita (INT/INTr), e o controlador de distribuição em 30 turnos seguidos
(`b4_sim.py`).

---

## 3. Resultados (dev, n = 60; humano: palavras 5, pergunta 15%, "!" 0%, emoji 5%, riso 3%)

| condição | lite D | luna D | deepseek D | mercury D | o que muda (lite, salvo nota) |
|---|---|---|---|---|---|
| A (persona pura) | 3,90 | 3,68 | 1,64 | 1,93 | 15 palavras; ? 50%; ! 28%; emoji 37%; riso 42%; molde 15%; abertura performática 35% |
| **A+N** (só normalizador) | 2,60 | 1,97 | 1,39 | 1,39 | ! 0%; emoji 7%; riso 3%; performática 12%; tamanho e ? iguais |
| S (estilo estático) | 2,43 | 2,80 | 1,48 | 1,94 | 9 palavras; mantém riso (48%) e emoji |
| S+N | 1,64 | 1,51 | 1,24 | 1,35 | |
| B9 (briefing a9) | 1,38 | 1,31 | 1,33 | 1,39 | 4 palavras; ? 3% (corrige demais) |
| **T** (briefing de alvos) | 1,32 | 1,40 | 1,34 | 1,25 | 5 palavras; ? 2% (luna 22%); molde 0%; LLM-ish 12% |
| T+N | 1,30 | 1,36 | 1,34 | 1,24 | |
| F1 ("1 frase, ≤ N palavras") | 1,78 | 2,03 | 1,40 | 1,65 | ! 33% e LLM-ish 25% (instrução de formato não tira o entusiasmo) |
| F1+N | 1,41 | 1,24 | 1,24 | 1,45 | |
| MT (max_tokens justo) | 2,07 | 2,90 | 1,13 | 1,70 | **frase quebrada em 90% / 60% / 40% / 33%** |
| MTfix (conserta o corte) | 2,07 | 2,46 | 1,17 | 1,74 | quebra cai para 3–5% |
| MT12 (corte cego) | 1,82 | 2,76 | 1,19 | 1,61 | quebrada em 90% / 67% / 48% / 45% |
| STq (stop "?") | 3,12 | 3,44ᵉ | 1,70 | 2,17 | ? 0%; o resto intacto |
| STnl (stop "\n") | 2,61 | 3,64ᵉ | 1,57 | 2,22 | |
| TLO / THI (temperatura) | 3,90 / 3,90 | n/a | não rodado | 2,31 / 2,31 | lite: textos idênticos ao A (ignora temperatura) |
| LB (logit_bias) | n/a | n/a | não rodado* | n/a | *sonda: aceito e ignorado |
| PFX ("comece com X") | 2,50 | 3,35 | não rodado | 1,56 | +N: 1,79 / 1,87 / – / 1,37 |
| NCA (4× A, escolhe em código) | 2,67 | 2,96 | 1,13 | 1,45 | +N: 1,99 / 1,58 / 1,12 / 1,18 |
| SHORT (4× A, a mais curta) | 3,21 | 3,27 | 1,28 | 1,46 | escolher pela violação > escolher pela brevidade |
| **NCT** (4× T, escolhe em código) | **1,19** | **1,21**⁺ | **1,16** | **1,15** | 6 palavras; ? 2% (luna 18%); ! 0%; LLM-ish 5% |
| **PL1** (plano+texto em JSON) | 1,27 (+N **1,08**) | 3,55 (JSON ignorado 90%) | não rodado | 1,46 (+N 1,28) | lite: ? 15% (= humano), tamanho 0,71 |
| R1+N (1ª frase, regra) | 1,38 | – | – | – | 3 palavras; performática 17% |
| R4+N / R5+N (regra + orçamento) | 1,55 / 1,47 | – | – | – | |
| **J1+N** (Jev escolhe a frase) | **1,24** | – | – | – | ? 18%; performática 7%; tamanho 0,96 |
| J2+N / J3+N (Nouls por frase) | 1,29 / 1,31 | – | – | – | Nouls quase sempre caem no fallback do Choice |

ᵉ = emulado em código (o luna não aceita `stop`). ⁺ = com normalizador. Custo por resposta (dev): A ≈ US$ 0,00009 (lite),
0,00003 (luna, deepseek), 0,00001 (mercury); NCT = 4×. Latência p50: 0,8–1,5 s por chamada; NCT +0,2–0,4 s.
Tabela completa, com todas as taxas: `analysis/data/b4_results_dev.json`.

**Leituras:**
1. **Entusiasmo ("!", emoji, riso, "omg/aww/wait") se resolve em código**, com precisão e sem custo. Pedir por instrução deixa
   resíduo (F1 com 33% de "!"), e os parâmetros de API não resolvem nada disso com estes 4 atores.
2. **Tamanho e molde se resolvem com alvos no briefing**, não com corte. O corte por `max_tokens` quebra frase em 33–90% dos casos,
   e o conserto não recupera o que a resposta queria dizer (a frase que sobra é o começo, em geral a reação).
3. **Pergunta é o vício mais difícil de calibrar.** Sem controle ela sobra (25–50%); com proibição ou alvo, ela some (2–5%). Os
   modelos baratos obedecem "não pergunte" e desobedecem "pergunte". Os caminhos que chegaram perto da taxa humana foram o PL1 na lite
   (15%), a poda do Jev J1 (18%) e o NCT no luna (18%). A proposta que falta testar é **gerar a pergunta separado** (2ª chamada curta
   só quando o orçamento sortear "sim").
4. **Selecionar entre candidatas é melhor que cortar uma.** O NCT foi o melhor em 3 dos 4 atores, porque a violação medida em código
   escolhe a candidata que já saiu certa. O custo é 4×, mas isso significa US$ 0,0001–0,0004 por resposta.
5. **O Jev na poda:** o Choice "qual frase um amigo mandaria" rende um pouco melhor que "a 1ª frase" (D 1,24 × 1,38), mas a
   diferença é pequena, é só na lite e está sem IC. Os 4 Nouls por frase (J2/J3) não acrescentaram nada ao Choice.

---

## 4. Tabela vício → mecanismo recomendado → eficácia medida → custo

| vício | mecanismo recomendado | eficácia medida (dev, 4 atores) | custo / latência |
|---|---|---|---|
| "!" | normalizador de código (orçamento sorteia "!" com a taxa humana; o resto sai; "!" no meio vira quebra de bolha) | 8–28% → 0–2% | 0 / 0 |
| emoji | normalizador (≤ 1 e só se o orçamento libera) | 10–75% → 2–8% | 0 / 0 |
| riso | normalizador + orçamento (espelha o riso do usuário; nunca em momento sério) | 8–45% → 0–7% | 0 / 0 |
| abertura performática ("omg", "wait", "aww", "wow") | normalizador + linha positiva no briefing ("start plainly…") | lite 35% → 2–12% | 0 / 0 |
| tamanho / nº de frases | briefing de ALVOS ("about N words, one idea", N do Jev + persona) + escolha entre 4 candidatas | erro de tamanho 0,98–1,47 → 0,77–0,93; mediana 13–15 → 5–6 palavras (humano 5) | T ≈ 1× (+0,5 s de Jev já pago); NCT 4× |
| molde "reação → comentário → pergunta" | briefing T (1 ideia) ou poda por frase | 15% → 0% | ≈ 1× |
| pergunta no fim | **não resolvido.** Hoje: orçamento sorteado + NCT/PL1; testar a "pergunta como 2ª chamada" | sobra 25–50% → falta 2–5%; mais perto: PL1-lite 15%, J1 18%, NCT-luna 18% (humano 15%) | — |
| vocabulário LLM-ish | briefing T + seleção em código (lista negra conta como violação) | lite 22% → 5% (NCT); no T sozinho ainda 12% | NCT 4× |
| eco/paráfrase | **não resolvido** no dev (o luna com T: 17%) | — | — |
| frase quebrada | **não usar `max_tokens` para controlar tamanho**; usar só como teto de segurança largo | MT quebra 33–90% | — |
| `logit_bias`, prefill, temperatura/top_p | **não usar** com estes atores (sem suporte, ignorados ou sem efeito) | nenhum efeito mensurável | — |

---

## 5. O pipeline recomendado (provisório: escolhido no dev, **não confirmado no teste**)

```
mensagem do usuário
 ─► J[leitura do momento, ~30 perguntas, ≈0,5 s]  ─► C:orçamento (N palavras, pergunta/riso/emoji/"!" sorteados)
 ─► C:briefing T curto e imperativo
 ─► LLM × 4 em paralelo (seeds diferentes)                 ← 4 candidatas; +0,2–0,4 s de latência
 ─► C:violações por candidata (tamanho vs N, "?" não autorizado, "!", emoji, riso, LLM-ish, abertura performática,
       pergunta recíproca, molde, HTML) ─► escolhe a de menor violação
 ─► C:normalizador (garante "!"/emoji/riso/abertura/minúsculas/sem ponto final)
 ─► [a testar] se o orçamento pede pergunta e a escolhida não tem: 2ª chamada curta só para a pergunta
 ─► [a testar] controlador por conversa (janela de 20 msgs: taxas de ?/riso/emoji; muletas ≤ 1 a cada 15)
 ─► entrega (bolhas/tempo: relatórios 01/10)
```

**Tradução para o sistema:**
- Trate "!", emoji, riso, abertura performática, travessão, nome do usuário e HTML como **responsabilidade do código**. É
  determinístico, grátis e funciona igual em qualquer modelo. A LLM não precisa "acertar" isso.
- Trate tamanho, "1 ideia" e o molde de 3 tempos como **responsabilidade do briefing**, com números de alvo, e use **seleção entre
  candidatas** como rede de segurança. Não use corte por token.
- **Não dependa de parâmetros do provedor.** Entre estes 4 atores, só `stop` e `max_tokens` funcionam em todos (e o luna nem aceita
  `stop`). `logit_bias`, prefill e temperatura não têm efeito ou não existem. Um pipeline portátil entre modelos (o usuário vai
  trocar de ator) precisa fazer o controle em código.
- **O que depende do ator:** o deepseek e o mercury já saem bem menos excessivos (D 1,6–1,9 × 3,7–3,9 da lite e do luna), mas todos
  convergem para o mesmo patamar com controle (D ≈ 1,15–1,35). O ator mais "comportado" precisa de menos controle, mas não fica
  melhor que os outros depois do controle. O luna segue ordens de pergunta melhor e ignora formatos JSON; a lite ignora temperatura
  e falha (vazio) em ~2–5% dos briefings (a mesma falha determinística vista no a9).

---

## 6. Limitações e o que ficou incompleto

- **Só dev (60 pontos, 6 conversas).** Não há teste nem IC utilizável. A escolha "NCT + normalizador" está **sujeita a
  overfitting** e precisa ser confirmada em `b4_* test` (119 pontos, 35 conversas). Rodar:
  `b4_gen.py test …` → `b4_prune.py` → `b4_multi.py` → `b4_post.py` → `b4_eval.py` → `b4_analyze.py test`.
- **Sem coerência nem perda de conteúdo.** O `b4_eval.py` (Nouls de coerência, "responde à pergunta?", "mantém a ideia do
  rascunho?" e Score de intensidade, com lote de 8 por chamada validado contra chamadas unitárias) não chegou a rodar. Por isso,
  as podas (R1, J1) e o NCT podem estar ganhando em "forma" à custa de responder menos ao que foi dito. **Sem checagem manual
  sistemática.** Numa leitura informal de ~10 exemplos da lite, a poda às vezes fica com a frase de reação ("no way, seriously?")
  e perde o conteúdo, e o J1 às vezes escolhe a pergunta.
- **Não medidos:** a reescrita guiada (quantas voltas, custo), rascunho → chat, plano → Jev → texto, o controle de intensidade
  ("resposta ≤ usuário") e o controlador de distribuição em 20–40 turnos (vícios substitutos). O código está em `b4_multi.py` e
  `b4_sim.py`.
- deepseek: TLO/THI/PFX/PL1/LB ficaram sem geração no dev (o processo foi interrompido). O `logit_bias` foi avaliado só na sonda
  (4 gerações).
- A distância D pesa todos os vícios igualmente e mede **forma**, não "soar humano". Os alvos vêm do maichat (inglês, pessoas
  que se conhecem), e a tradução para PT-BR continua sendo inferência.

### Arquivos
- Scripts: `scripts/analysis/b4_llm.py` (cliente com stop/top_p/logit_bias/prefill/reasoning e custo), `b4_probe.py`/`b4_probe2.py`
  (o que cada provedor aceita), `b4_common.py` (orçamento, métricas, normalizador, segmentação, bootstrap), `b4_gen.py`
  (mecanismo 1 e baselines), `b4_prune.py` (poda Jev × regra), `b4_post.py` (derivadas em código: +N, stop emulado, NCA/NCT),
  `b4_multi.py` (RW, D5, PL2, INT: não rodado), `b4_eval.py` (Jev: não rodado), `b4_sim.py` (controlador: não rodado),
  `b4_analyze.py`.
- Saídas: `analysis/data/b4_probe.json`, `b4_probe2.json`, `b4_probe3.json` (suporte a parâmetros), `b4_results_dev.json`
  (todas as métricas por ator × condição). As gerações brutas estão em `scratchpad/b4/gen.jsonl` e o cache em
  `data/processed/b4_llm_cache.jsonl`.
