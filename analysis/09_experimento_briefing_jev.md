# 09 · Experimento A/B: LLM pura × LLM + briefing do Jev × humano real

> Prefixo `a9`. Scripts em `scripts/analysis/a9_*.py`; saídas em `analysis/data/a9_*`.
> Corpus: **maichat** (inglês, amigos, casais e família, com chat ao vivo). 119 pontos de decisão de teste em 35 conversas,
> mais 60 de dev (calibração) em 6 conversas disjuntas. "Boca": `google/gemini-3.5-flash-lite`.

---

## 1. Resumo dos achados

- **O briefing do Jev deixa a LLM barata muito mais parecida com o humano *na superfície*.** Humano / A (LLM pura) / B (LLM + briefing):
  mediana de palavras **6 / 15 / 4**; pergunta em **12% / 57% / 1%** dos turnos; riso **7% / 37% / 8%**; emoji **5% / 39% / 4%**;
  "!" **7% / 29% / 0%**; vocabulário "LLM-ish" **5% / 21% / 3%**; gíria "de internet" **5% / 21% / 5%**.
  O erro de tamanho |log2(razão)| cai de 1,27 para 0,78 (diferença de −0,50, IC95% −0,72 a −0,28).
- **"O que escrever" melhora um pouco.** O movimento da resposta (responder, zoar de volta, reagir, perguntar…) coincide com o do
  humano em **23,5% (A) → 34,5% (B)**, ou +11 pp (IC95% +2 a +20). O tom coincide em 23,5% → 36%. Com prompt de estilo estático
  (S) o valor é 28,6%; com a LLM cara (D, claude-haiku-4.5) é 26%.
- **A LLM "cara" sem briefing não resolve.** A claude-haiku-4.5 (D) é tão "LLM" quanto a barata: 13 palavras de mediana, pergunta
  em 54%, riso em 47%, "!" em 32%, "honestly" em 13% das respostas. **Barato + briefing (B) supera caro sem briefing (D)** em
  todas as métricas de código e em coincidência de movimento (+8 pp, IC 0 a +16). B custa US$0,00017 por resposta e D, US$0,00026.
- **Os dois juízes automáticos estão invertidos, e isso é um achado em si.** Tanto o Jev quanto o `gpt-4o-mini`, perguntados
  "qual das duas o humano real escreveu?", apontam a resposta da LLM pura como humana em **71% e 85%** dos pares, porque escolhem
  a **mais longa** como humana em 63% e 70% dos casos. A "taxa de engano" ingênua premia o vício. Leio então a
  **distinguibilidade** |2p−1|: A = 0,43 (Jev) e 0,70 (LLM); B = **0,06 e 0,08** (acaso). O Jev também avalia o humano real como
  *menos* adequado ao clima (0,83) do que a LLM (0,89).
- **A maior parte do ganho vem da calibração às taxas humanas feita em código, não da leitura contextual do Jev.** O controle
  "Bnojev" usa o mesmo formato de briefing sem nenhum conteúdo do Jev (proibições constantes, tamanho pela mediana humana e o
  estilo da persona). No subconjunto de 36 pontos, ele chega quase ao mesmo lugar: distinguibilidade de 0,06 no juiz Jev, pergunta
  em 3% e mediana de 4 palavras. O Jev acrescenta pouco e com IC largo: coincidência de movimento +5,6 pp (IC 0 a +13,5) e
  acompanhamento do tamanho humano (Spearman **0,47 × 0,28**). Em tom, o Bnojev foi melhor (44% × 28%, n=36).
- **B corrige demais.** Sai curto demais (mediana de 4 palavras contra 6), quase nunca pergunta (1% contra 12%), nunca usa "!"
  (0% contra 7%) e fica genérico ou seco: "hey there" 3× nas aberturas; "delete your account" em resposta a "love u dad". O limite
  "Max N words" ficou abaixo do que o humano escreveu em **43%** dos pontos.
- **Os briefings longos pioram e os concretos funcionam.** Despejar as probabilidades do Jev em prosa (Blong, ~145 palavras)
  devolve a LLM ao baseline: riso 22%, emoji 28%, distinguibilidade 0,44 (Jev) e 0,39 (LLM). O curto e imperativo (~68 palavras)
  fica em 0,06. Número em prosa não é ordem.
- **"Não use X" não fez a LLM usar X.** Os itens proibidos aparecem em **1,7%** das respostas de B, contra **39,5%** de A (a maior
  parte é "!"). Sem a lista (Bnoban), o uso foi de 0% no subconjunto, então a lista é redundante quando o resto do briefing é
  concreto. O que **gerou gíria forçada** foi outra coisa: listar as gírias da própria persona ("words you use: idk, omg") fez a
  LLM enfiá-las em tudo no dev ("idk / omg", "idk tbf"). Removi essa linha.
- **Padrões invisíveis.** Humanos abrem o turno com **marcador simples** ("ok", "so", "oh", "yeah", "same", "i…") em 32% dos
  turnos do maichat. Abrem com **interjeição performática** ("omg", "wait", "ooh", "lol", "honestly") em só 5%; a LLM pura faz
  isso em 37%. A **pergunta recíproca genérica** ("what about you?", "how's your day?", "what's up?") aparece em 1,2% dos turnos
  humanos e em 17,6% (A) e 19,3% (D). Na **abertura**, 13 de 17 respostas humanas já trazem conteúdo ("guess what",
  "just got home / im exhausted"); a LLM pergunta em 17 de 17.
- **O gate (C: 3 candidatos + escolha do Jev) não compensou.** Não houve ganho em movimento (34,5% = B), a distinguibilidade no
  juiz LLM piorou (0,18 × 0,08), o custo foi 2,2× maior e a latência subiu 0,7 s. O `Choice` entre candidatos mostrou **viés de
  posição** forte: escolheu o 1º, 2º e 3º em 53%, 32% e 15% dos casos, com candidatos intercambiáveis. `Noul` por candidato
  reduz o viés (0,73 / 0,69 / 0,67).
- **Custo e latência são viáveis.** O briefing do Jev (~30 perguntas numa chamada) leva **p50 de 0,49 s** (p90 0,61 s) e custa
  US$0,00008. O pipeline B tem p50 de 1,75 s, contra 1,29 s de A (+0,46 s, que cabem no "digitando…").
- **O poder preditivo do Jev na posição do bot é desigual.** No teste, antes de ver a resposta: tamanho Spearman **0,49**
  (útil); emoji AUC 0,71; pergunta AUC 0,57 (fraco); **riso AUC 0,41 (pior que o acaso)**; nº de bolhas Spearman 0,04 (inútil);
  movimento top-1 32% (contra 16% da classe majoritária).

---

## 2. Desenho (e por que mudei o sugerido)

**Pontos de decisão.** Cada ponto é o turno T do falante real ("Sam") logo após um turno do parceiro ("Alex") na mesma sessão.
O histórico tem até 12 turnos, e o alvo é o que Sam respondeu (bolhas unidas por quebra de linha). A estratificação usa o
**D do turno do parceiro** (T−1), porque é isso que o bot conhece na hora de responder. São 7 estratos com 17 pontos cada:
abertura, fechamento, sério/vulnerável, flerte/afeto, logística, brincadeira e casual. Limitei a 4 pontos por conversa. O rótulo
de estrato vem do Jev e é ruidoso: "im on a budget ok" caiu em "sério".

**Condições** (a mesma boca, temperatura 0,8, persona "You are Sam, chatting with Alex on a messaging app. Reply as Sam."):

| Cond. | O que é | n |
|---|---|---|
| **A** | persona + histórico, nenhuma dica | 119 |
| **S** | A + prompt de estilo **estático** genérico ("casual e curto, minúsculas ok, sem cara de assistente, espelhe a energia") | 119 |
| **B** | S + **briefing curto do Jev** (gerado em código a partir de ~30 perguntas) | 119 |
| **C** | 3 candidatos de B (seeds 0, 1 e 2) + gate do Jev (1 `Noul` por candidato; o `Choice` foi guardado como "Cchoice") | 119 |
| **D** | `anthropic/claude-haiku-4.5`, persona + histórico, sem briefing (referência "cara") | 119 |
| Bpure | persona + briefing, **sem** o prompt estático (o "A + briefing" literal do enunciado) | 36 |
| Bnoban | B sem a lista "Never use" | 36 |
| Blong | o Jev despejado em prosa longa (probabilidades), sem ordens concretas | 36 |
| **Bnojev** | **controle-chave**: o mesmo formato de B, mas sem nada do Jev (taxas-base humanas no lugar) | 36 |

**Por que B = S + briefing, e não A + briefing.** Qualquer produto já teria um prompt de estilo fixo. A pergunta útil para a
tese é qual o valor **marginal** do Jev sobre isso. O "A + briefing" literal (Bpure) foi rodado como ablação e deu praticamente
o mesmo que B: com o briefing, o prompt estático não acrescenta nada.

**Dev e teste.** Os limiares que convertem as respostas do Jev em ordens foram calibrados no dev e congelados antes do teste
(`a9_calibrate.py` → `a9_thresholds.json`). Cada `Noul` "p_*" tem um corte que faz a fração de pontos marcados igualar a taxa
humana (o Jev superestima: média de 0,40 para "vai perguntar?" contra 9,5% reais). O tamanho usa casamento de quantis entre o
Score do Jev e o nº de palavras humano. Iterei o briefing em 12 pontos de dev: três versões e ~40 chamadas de LLM.

**Avaliação** (tudo cego, humano × condição):
1. juiz Jev em pares, `Choice` "qual o real Sam mandou?", nas **duas ordens**, com a média das duas;
2. métricas de código (tamanho, pergunta, riso, emoji, "!", lista LLM-ish, gíria, bolhas, minúsculas);
3. qualidade isolada pelo Jev (`Noul`s: cabe no clima, soa IA, formal demais, gíria forçada, coerente, entusiasmo demais);
4. juiz LLM independente (`gpt-4o-mini`, temperatura 0, ordem balanceada por hash);
5. **extra, para o "o que escrever"**: o Jev rotula o **movimento** e o **tom** de cada resposta (a do humano inclusive) com o
   mesmo `Choice` do briefing, em chaves embaralhadas. A métrica é a coincidência com o humano.

Os ICs são bootstrap **por conversa** (cluster), com 2.000 reamostras.

---

## 3. Achados detalhados

### 3.1 A LLM pura tem uma "assinatura" muito distante do humano, e o modelo caro tem a mesma (confiança **alta**)

| Métrica (n=119) | Humano | A (flash-lite) | S (estilo estático) | **B (+Jev)** | C (gate) | D (haiku-4.5) |
|---|---|---|---|---|---|---|
| palavras (mediana) | **6** [5–7] | 15 [13–17] | 9 [8–10] | 4 [3–5] | 4 [3–4] | 13 [11–16] |
| erro de tamanho \|log2\| | – | 1,27 | 0,91 | **0,78** | 0,80 | 1,15 |
| tem pergunta | **12%** [6–18] | 57% | 27% | 1% | 1% | 54% |
| riso (haha/lol/lmao/😂/💀) | **7%** | 37% | 36% | 8% | 8% | 47% |
| emoji | **5%** | 39% | 16% | 4% | 4% | 20% |
| "!" | **7%** | 29% | 16% | 0% | 0% | 32% |
| lista LLM-ish | **5%** | 21% | 13% | 3% | 3% | 25% |
| gíria "de internet" | **5%** | 21% | 13% | 5% | 7% | 18% |
| >1 bolha | **36%** | 48% | 28% | 23% | 23% | 34% |
| abre em minúscula | **80%** | 77% | 97% | 100% | 100% | 84% |
| abertura performática¹ | **5%** | 37% | 29% | 8% | 7% | 32% |
| pergunta recíproca genérica² | **5%** (1,2% no maichat todo) | 18% | 13% | 1% | 0% | 19% |
| artefatos (HTML `</p>`, `*ação*`, turnos inventados) | 0 | 5 | 1 | 0 | 0 | 3 |

¹ Começa com omg/wait/ooh/lol/haha/lmao/hey!/aww/ugh/honestly/stop/literally. ² "what about you", "wbu", "how's your day",
"what's up", "how are you"…

- O prompt estático (S) encurta e corta pergunta, mas **mantém o riso (36%)** e a interjeição performática (29%).
- **D ≈ A.** Pagar ~3× mais pela boca não tira os vícios. O que tira é a instrução concreta.
- Exemplos do baseline: "Hey! Perfect timing. How was your day?" (para "hey love / im here"; o humano mandou "guess what");
  "aww hang in there! what language are you trying to learn? coding can definitely be super frustrating at first 😭".
- A Gemini às vezes **vaza formatação e inventa turnos**, por exemplo
  `💀 pls dont remind me</p><p id=727 pid=727 uid=>*reads your message*…`, ou um link falso do Unsplash para "send pic".
  Isso ocorreu em 5 de 119 respostas de A e em nenhuma de B.

### 3.2 O briefing do Jev corrige a superfície, mas corrige demais (confiança **alta**)

- B chega perto do humano em riso (8% contra 7%), emoji (4% contra 5%), gíria (5% contra 5%) e LLM-ish (3% contra 5%).
- A correção passa do ponto em **tamanho** (razão mediana 0,75), **pergunta** (1% contra 12%), **"!"** (0 contra 7%) e **bolhas**
  (23% contra 36%). O limite "Max N words" foi respeitado em 97% dos casos, mas ficou **abaixo do tamanho humano em 43%** dos
  pontos.
- A aderência ao briefing é alta: "no question" 100%, "no laughing" 100%, "no emoji" 97%, minúsculas 100% e o movimento pedido
  foi realizado em 73% (rótulo do Jev).
- O defeito típico de B é ficar **seco, genérico ou "engraçadinho"**: "hey there" (3 aberturas), "stop it", "fair",
  "im a chaotic potato" (para "what horoscope are you"; o humano disse "aries - I'm so firey / wbu").

### 3.3 "O que escrever": movimento e tom (confiança **média**)

| Coincide com o humano (n=119) | A | S | **B** | C | D |
|---|---|---|---|---|---|
| movimento | 23,5% | 28,6% | **34,5%** | 34,5% | 26,1% |
| tom | 23,5% | 29,4% | **36,1%** | 33,6% | 23,5% |

- B − A = **+10,9 pp** (IC +1,8 a +19,8); B − S = +5,9 pp (IC −1,6 a +13,4, não significativo); B − D = +8,4 pp (IC 0 a +15,8).
- A distribuição de movimentos explica parte do efeito. O humano faz "ask_follow_up" em 13% dos turnos; A faz em 27% e D em 26%
  (as LLMs "entrevistam"); B faz em 10%.
- Antes de ver a resposta, o Jev prevê o movimento humano com top-1 de **32%**, contra 16% da classe majoritária. É informativo,
  mas erra 2 de cada 3.
- **Os erros de leitura do Jev viram erros de conteúdo.** Para "love u dad", o Jev leu "being sweet" + "tease back, dry" e B
  respondeu "delete your account"; o humano: "love u too / now go eat real food". Para "hi beautiful / what's up", o tom
  "flirty" produziu "not much, just thinking of you"; o humano: "just got home / im exhausted". Com o briefing, a LLM obedece,
  então um rótulo errado sai caro.
- No subconjunto de 36 pontos, B ficou em 47% de movimento e Bnojev em 42% (Δ +5,6 pp, IC 0 a +13,5). Em tom, B ficou em 28% e
  Bnojev em **44%**. **Não dá para afirmar que o conteúdo contextual do Jev melhora o "o que dizer"** além de "seja curto e não
  pergunte".

### 3.4 O valor específico do Jev (contra o controle sem Jev) é pequeno (confiança **média**)

No subconjunto de 36 pontos, com as mesmas conversas:

| | A | S | B | **Bnojev** | Bpure | Bnoban | Blong |
|---|---|---|---|---|---|---|---|
| distinguibilidade, juiz Jev (\|2p−1\|) | 0,18 | 0,24 | **0,01** | 0,05 | 0,02 | 0,02 | 0,19 |
| palavras (mediana) | 7,5 | 6 | 3 | 4 | 3 | 3 | 4 |
| erro de tamanho \|log2\| | 0,97 | 0,94 | 0,78 | **0,65** | 0,75 | 0,80 | 0,95 |
| acompanha o tamanho humano (Spearman) | 0,20 | 0,09 | **0,47** | 0,28 | 0,43 | 0,20 | 0,20 |
| pergunta | 64% | 36% | 3% | 3% | 3% | 3% | 11% |
| coerente (Jev) | 0,91 | 0,91 | 0,89 | 0,88 | 0,87 | 0,86 | 0,90 |
| cabe no clima (Jev) | 0,91 | 0,90 | 0,86 | 0,85 | 0,86 | 0,86 | 0,89 |

(A distinguibilidade nesta tabela usa a probabilidade média; na seção 1 uso o voto duro, n=119.)

- O que o Jev comprovadamente acrescenta: **o tamanho acompanha o momento** (Spearman 0,47 contra 0,28), porque o Score de
  tamanho do Jev tem Spearman 0,49 com o tamanho humano no teste. Há ainda um leve ganho em movimento.
- O que **não** acrescenta de forma mensurável: a distinguibilidade e as métricas de superfície (o controle sem Jev chega lá
  sozinho).
- Leitura honesta: com o Jev lendo o momento, ~80% do caminho foi feito pelas **regras de código calibradas às taxas humanas**. O
  Jev entra como ajuste fino e, quando lê errado, piora.

### 3.5 Os juízes automáticos não medem "humanidade" aqui (confiança **alta**)

- O juiz Jev aponta a condição como humana em 71% dos pares para A, 76% para S, 53% para B, 59% para C e 68% para D (voto duro).
  O `gpt-4o-mini` faz o mesmo em 85%, 82%, 54%, 59% e 82%.
- O motivo: nos pares com tamanhos diferentes, os juízes escolhem a **mais longa** como humana em **63% (Jev) e 70% (gpt-4o-mini)**
  das vezes. O juiz LLM ainda tem viés de posição (escolhe o candidato 1 em 62%). O Jev, com a média das duas ordens, fica em 52%.
  A concordância entre as ordens do Jev é de 78%, e a concordância Jev × LLM, de 69%.
- A **qualidade isolada pelo Jev também favorece a LLM**. "Cabe no clima": humano 0,83, A 0,89, D 0,89, B 0,86. "Coerente":
  humano 0,81, A 0,90. "Soa IA": humano 0,18, A 0,19, ou seja, não separa. Humanos reais desviam, zoam e mudam de assunto, e a
  rubrica "adequado ao momento" pune exatamente isso.
- Consequência: **não use o Jev nem uma LLM pequena como juiz de "parece humano"**. Eles têm o estereótipo "o bem articulado é o
  humano". Para avaliar, use as distâncias de código e juízes humanos. Para o Jev, use perguntas **literais** sobre o momento
  ("ela está zoando?") e não perguntas de gosto ("isso é natural?").
- Mesmo assim, o dado diz algo: com B os dois juízes caem no **acaso** (0,06 e 0,08), ou seja, o sinal que usavam para separar
  humano e LLM some. Isso não prova que B soe humano a um leitor humano.

### 3.6 O gate (C) não se paga (confiança **média**)

- C − B: juiz Jev +0,03 (IC +0,01 a +0,04, a favor de C, mas o mesmo modelo escolheu e julgou, o que é circular); "cabe no clima"
  +0,009; movimento 0; distinguibilidade no juiz LLM 0,18 × 0,08 (pior); tamanho igual.
- O `Choice` "qual candidato é o mais natural" escolhe **por posição** (53/32/15% para candidatos intercambiáveis; probabilidade
  média de 0,48/0,31/0,21). `Noul` absoluto por candidato é bem menos enviesado (0,73/0,69/0,67), e o gate final usou `Noul`.
- O preço: C custa 2,2× B e soma 0,73 s no p50. Vale só se a geração tiver alta variância, por exemplo com temperatura mais alta.

### 3.7 O que ajudou e o que não ajudou no briefing (confiança **média**, n=36 nas ablações)

| Variante | Efeito observado |
|---|---|
| Ordens **concretas e curtas** (~68 palavras) | ✔ melhor combinação: aderência de 97–100%, distinguibilidade 0,06 |
| **Prosa com probabilidades** (Blong, ~145 palavras) | ✘ volta ao baseline: riso 22%, emoji 28%, "!" 14%, pergunta 11%, distinguibilidade 0,44 (Jev) e 0,39 (LLM). A LLM lê "laugh 43%" como licença |
| Lista **"Never use: …"** | neutro: B 2,8% × Bnoban 0% de itens proibidos (n=36). **Não houve efeito rebote.** Ela é redundante quando o resto é concreto; já a proibição de "!" corrigiu demais (0% contra 7% humano) |
| **"words you use: idk, omg"** (gírias da persona) | ✘ no dev, gíria enfiada à força ("idk / omg", "idk tbf"). Removido |
| "Style: all lowercase; no final period" (da persona, em código) | ✔ 100% de aderência; os humanos do maichat usam minúscula em 77–80% dos turnos e ponto final em 2% |
| Rótulo de momento pelo **maior `Noul`** + guardas | ✔ necessário: os `Noul`s "vulnerable" e "annoyed" do Jev disparam em provocação de brincadeira ("if i fail im blaming you" → vulnerable 0,62). Descontar pelo `teasing` resolveu no dev |
| Score de **nº de bolhas** do Jev | ✘ Spearman −0,16 (dev) e 0,04 (teste). Troquei pelo hábito da persona (Spearman 0,22 no dev) |
| `Noul` "vai rir?" do Jev | ✘ AUC 0,41 no teste, pior que o acaso. Precisa de outra formulação ou de regra de código |
| Prompt estático junto do briefing | neutro (Bpure ≈ B) |
| "It must make sense as a direct reply…" | entrou depois do dev ("gluttony probably", "i have existential dread…" eram desconexos). Não testei isoladamente |

---

## 4. Padrões "invisíveis" que um bot deveria imitar

1. **Começar sem cerimônia.** 32% dos turnos humanos do maichat começam com marcador discursivo simples (ok, so, oh, yeah, same,
   i, but, why…), e só 5% com interjeição performática. A LLM inverte isso (37% performático, 13% simples). B elimina o
   performático (8%), mas não adota o simples (17%): "não faça X" não ensina "faça Y".
   *PT-BR (inferência):* "ah", "então", "sim"/"ss", "mesmo", "pois é", "tb", "ok"/"blz", "é que…", contra os performáticos
   "nossa!", "meu deus", "pera", "sério?!", "que incrível!".
2. **Não devolver a pergunta.** A pergunta recíproca genérica ("e você?", "como foi seu dia?", "e aí, tudo bem?") aparece em 1,2%
   dos turnos humanos e em 18–19% das respostas da LLM. Quando o humano pergunta (12%), a pergunta é específica e curta ("which
   gym though", "how fat is he").
3. **Abrir com conteúdo, não com "what's up".** Em 13 de 17 aberturas (contagem manual), o humano responde ao "oi" já com um estado ou gancho: "guess
   what", "just got home / im exhausted", "nooo i was in the shower 😩", "why does 2pm feel illegal". A LLM pergunta em 17 de 17
   ("Hey! How's it going?"), e B responde oi vazio ("hey there").
4. **Afeto se devolve com cuidado prático ou com deboche, não com derretimento.** "love u dad" → "love u too / now go eat real
   food"; "im proud" → "dont sound like my mom"; "u deserve a raise tbh" → "say that louder". A LLM responde "aw thank u 🥺 i
   could really use those rn".
5. **Responder à palavra, não ao tema.** Muita graça humana pega uma palavra específica e a vira: "my desk" → "thats not temporary
   thats kidnapping"; "so how has your day been" → "why are you talking like a customer survey". A LLM responde ao tema de
   forma genérica ("productive. how was yours, fellow human?").
6. **Fechar com âncora concreta.** "OK - see you Friday." contra "You're welcome, Alex. Good luck with the meeting tomorrow! See
   you for Christmas." (A) e "goodnight / sleep well" (B, genérico). Atenção: muitas vezes o "tchau" do outro **não** encerra a
   conversa, e o humano segue ("ok i should actually start my hw" → "same / this chat looks long enough"). B se despediu cedo
   demais nesses casos.
7. **Vulnerabilidade leve pede ação concreta, não validação.** "kinda / still adjusting" (cabelo novo) → "send pic"; "its so hard
   :(" (programação) → "did u try watching videos on coding…". A LLM valida ("that's so valid tbh") e B dá chavão ("coding is no
   joke, you got this").
8. **Sem pontuação de assistente.** "!" em 7% (LLM 29–32%), ponto final em 2%. Minúscula em ~80%.
   *PT-BR:* "kkkk"/"kkkkk" no lugar de "haha"/"lol", "vc", "tb", "pq", "q"; o "😭"/"💀" como riso também é comum aqui (inferência).

---

## 5. Tradução para o sistema

**Fluxo proposto.** A mensagem do usuário chega. Em paralelo, o código calcula a impressão digital de estilo da persona e as
features da mensagem. Uma única chamada ao Jev, com ~30 perguntas (~0,5 s) e o "digitando…" já na tela, lê o momento. O código
monta o briefing, a LLM gera, um filtro de código revisa e a entrega sai (bolhas e atraso: ver relatórios de entrega).

**State do Jev:** `{"conversation": [até 7 turnos anteriores {from, text}], "last_message": {from: USER, text}}`. As bolhas de um
turno vão juntas com "\n". É enxuto: 8 turnos no total.

| # | Achado | Detector Jev (tipo · instrução · critérios) | Quando | Regra de código que consome |
|---|---|---|---|---|
| 1 | Rótulo do momento | **`Noul`s** atômicos sobre `last_message`: teasing, flirting, affection, vulnerable, good_news, complaint, closing, greeting, logistics, minimal, annoyed | toda mensagem | momento = argmax, com `vulnerable` e `annoyed` multiplicados por (1 − teasing); se max < 0,55 → "casual". Seriousness ≥ 1,8 força "sério". **Adicionar** `Noul` "Is the affection in `last_message` joking/ironic?" para não responder a carinho sincero com zoeira |
| 2 | Movimento | **`Choice`** (16 movimentos: react_only, answer, tease_back, joke_riff, empathize, reassure, share_own, ask_follow_up, flirt_back, compliment, agree, disagree, plan, goodbye, greet_back, new_topic) | toda mensagem | ordem de 1 linha por movimento. **Sugestão:** se confidence < 0,4, não ditar o movimento (a LLM decide) e ditar só forma. Top-1 = 32% contra 16% majoritário |
| 3 | Tamanho | **`Score`** 5 níveis ("1-3 words" … "over 30 words") | toda mensagem | palavras = quantil humano do score (mapa em `a9_thresholds.json`), misturado 60/40 com a mediana da persona. **Corrigir:** usar como *alvo* com folga (+50%), não como teto; o teto atual cortou 43% dos humanos |
| 4 | Pergunta | `Noul` "Will Sam's next message ask Alex a question?" (AUC 0,57, fraco) | toda mensagem | permitir pergunta se `move ∈ {ask_follow_up}` ou p ≥ 0,61. **Proibir sempre, em código,** a pergunta recíproca genérica (regex), salvo na abertura sem conteúdo. Meta: ~12% dos turnos com pergunta |
| 5 | Riso | ✘ o `Noul` "would Sam laugh?" falhou (AUC 0,41) | – | usar regra de código (espelhar o riso do parceiro e o token da persona: "haha", "kkkk", "💀"). Reformular e validar antes de usar o Jev |
| 6 | Emoji | `Noul` "Will Sam's next message contain an emoji?" (AUC 0,71) | toda mensagem | liberar só se p ≥ 0,51 **e** a persona usa emoji em ≥ 5% das mensagens |
| 7 | Nº de bolhas | ✘ `Score` do Jev inútil (ρ = 0,04) | – | hábito da persona: 2 bolhas se bolhas/turno ≥ 1,5 e alvo ≥ 7 palavras |
| 8 | Abertura | `Noul` greeting | quando greeting > 0,6 | ordem: "cumprimente + 1 coisa concreta do seu momento (estado, novidade); sem 'e aí, tudo bem?'" |
| 9 | Fechamento | `Noul` closing | quando closing > 0,6 | despedida curta com âncora concreta ("até sexta"). Se closing estiver entre 0,4 e 0,6, **não** se despedir (os falsos positivos geraram despedidas prematuras) |
| 10 | Seriedade / exagero | `Score` seriousness (0–3); `Noul`s p_overkill e p_excitement | toda mensagem | seriedade ≥ 1,8 → "no jokes, simple and sincere"; p_overkill > 0,5 e p_excitement < 0,5 → "low-key, not gushing" |
| 11 | Estilo da persona | – (código) | cadastro da persona e a cada N turnos | minúsculas se lower_frac > 0,7; sem ponto final se period_frac < 0,2. **Não** listar as gírias da persona (causa gíria forçada) |
| 12 | Filtro anti-LLM | – (código, regex) | após a geração | regerar ou aparar se houver abertura performática, pergunta recíproca genérica, termos LLM-ish, travessão, `</p>`, `*ação*`, "Alex:"/"Sam:", links inventados |
| 13 | Gate | se usar: **um `Noul` por candidato**, com a ordem embaralhada; nunca `Choice` entre candidatos (viés de posição) | opcional | como não mostrou ganho, desligado por padrão |
| 14 | Avaliação offline | ✘ não usar o Jev nem o gpt-4o-mini como juiz "qual é humano?" | – | usar distâncias de código às taxas humanas + painel humano |

**Briefing final (o que foi usado em B)**, com um exemplo real:

```
Moment: light banter, Alex is teasing/joking.
Answer Alex's question directly and plainly. Tone: playful.
Max 6 words, one message.
Laugh the way you usually do ('😭').
No question.
Keep it low-key, not gushing.
Style: all lowercase; no final period.
It must make sense as a direct reply to Alex's last message.
Never use: aww, totally, absolutely, amazing, sounds like, that sounds, I'd love, honestly, vibe(s), super, definitely, journey, em dashes (—), exclamation marks.
```

As ~30 perguntas estão em `scripts/analysis/a9_brief.py` (`Q`), e a conversão está em `build_brief()`. **Próxima versão
recomendada** (hipóteses, não testadas):
- pergunta e "!" **sorteados** com a probabilidade humana, em vez de proibidos;
- tamanho como alvo com folga;
- uma linha positiva "start plainly (ok/so/oh/yeah) or straight with content";
- na abertura, "greet + one concrete thing";
- o `Noul` de afeto irônico;
- o riso por regra de código.

---

## 6. Exemplos lado a lado

(contexto → **humano** | A = LLM pura | B = + briefing Jev | C = + gate | D = haiku-4.5; "/" separa bolhas)

1. **Abertura.** Alex: "hey love / im here" → **"guess what"** | A: "Hey! Perfect timing. How was your day?" | B/C: "hey there" |
   D: "hey! 💙 good to hear from you, how's your day going?". *O humano abre com gancho; a LLM entrevista; B fica vazio.*
2. **Abertura / afeto.** Alex: "hi beautiful / what's up" → **"just got home / im exhausted"** | A: "Hey! Not much, just relaxing.
   How are you doing?" | B: "not much, just thinking of you" | C: "hey, just thinking about you". *O tom "flirty" do Jev gerou
   flerte forçado.*
3. **Afeto (pai e filho).** Alex: "love u dad" → **"love u too / now go eat real food"** | A: "go to your room" | B: "delete your
   account" | C: "keep dreaming kid" | D: "lol stop / you're gonna make me sentimental". *O Jev leu "tease back, dry": erro de
   leitura virou grosseria.*
4. **Elogio.** Alex: "im proud" → **"dont sound like my mom"** | A: "thx someone has to be lol / what about u? got plans?" | B/C:
   "stop it" | D: "thanks lol, means a lot coming from you".
5. **Brincadeira.** Sam: "STOP 😭" / Alex: "so how has your day been" → **"why are you talking like a customer survey"** | A:
   "productive. how was yours, fellow human?" | B/C: "productive mostly 😭" | D: "lol ok fine, it's been pretty chill honestly…
   you?". *O humano reage à forma da pergunta; ninguém mais faz isso.*
6. **Brincadeira.** Alex: "did you see the cat one i sent after" → **"THE ORANGE ONE???"** | A: "omg yes 💀 i needed that eye bleach
   after watching you faceplant / how's your head tho fr?" (inventa fatos) | B/C: "i don't watch your spam" | D: "nah what was it".
7. **Sério (leve).** Alex: "i have been trying to learn some coding since morning / its so hard :(" → **"did u try watching videos
   on coding tht could possibly help u"** | A: "aww hang in there! what language are you trying to learn? coding can definitely be
   super frustrating at first 😭" | B/C: "coding is no joke, you got this" | D: "Ah yeah coding can be tough at first! What
   language are you learning? I found it gets easier…" (38 palavras).
8. **Sério (leve).** Alex: "kinda / still adjusting" (cabelo novo) → **"send pic"** | A: "that's so valid tbh, every time i change my
   hair i stare in the mirror like 'who is this stranger'…" | B/C: "takes a minute" | D: "that's fair, takes a bit to get used to
   a new cut. does it feel different?".
9. **Ansiedade.** Alex: "no im just scared / should i follow him first" → **"how many followers does he have"** | A: "wait no don't
   double the anxiety dose. let him find u" | B: "shoot your shot" | C: "do it coward" | D: "okay but what's the vibe? like is he
   someone you actually know or random follow".
10. **Fechamento (família).** Alex: "Anyway dad I need to go now / thanks for chatting" → **"OK - see you Friday."** | A: "You're
    welcome, Alex. Good luck with the meeting tomorrow! See you for Christmas." | B/C: "goodnight / sleep well" | D: "No problem!
    Good luck with your meeting tomorrow - you'll be fine. Enjoy the rest of your day and I'll chat to you soon. Take care!"
11. **Fechamento.** Alex: "ok im actually gonna try studying" → **"proud of u"** | A: "oh wow look at u being all productive / fine
    go do work, don't let me distract u 🫡" | B/C: "got u covered" | D: "good luck bestie 💪 you got this" | Bnoban: "we die like
    men".
12. **Logística.** Alex: "…if it is like this on monday do you wanna go for a walk up the braids" → **"Aye that'd be lovely and we
    could take the chilli and sit up on those old logs if you like / But onky if its not too CHILLY LOLLLL"** | A: "Ooh I would
    love that! Count me in. Fingers crossed the weather holds up 🤞 x" | B: "sounds good / see you then". *O humano às vezes é mais
    longo e mais bobo do que qualquer condição.*
13. **Pedido de prova.** Alex: "send pic / proof of real food" → **"u dont trust me at all 😭"** | A:
    `![Ramen pic](https://images.unsplash.com/…)` (link falso) | B/C: "its just instant noodles / nothing special" | D: "*sends
    blurry pic of pot with water boiling* / there. happy now?".
14. **Pergunta simples.** Alex: "what horoscope are you / (you can make one up if you want)" → **"aries - I'm so firey / wbu"** | A:
    "im a gemini / what about u" | B/C: "im a chaotic potato" | D: "i'm a scorpio lol / what about you? and yeah we can totally read
    them together". *Aqui o A acertou a forma; B tentou "ser engraçado" à força.*
15. **Artefato do baseline.** Alex: "mood during exams" → **"exactly / ok i should actually start hw"** | A: "💀 pls dont remind
    me</p><p id=727 pid=727 uid=>*reads your message* / my bad 😭 but fr tho, u ready for bio?</p>…" (inventa turnos dos dois
    lados) | B/C: "real 😭".

---

## 7. Custo e latência medidos (por resposta, no teste)

| Cond. | Custo (US$) | Latência p50 / p90 (s) | Composição |
|---|---|---|---|
| A | 0,000089 | 1,29 / 1,59 | 1 geração |
| S | 0,000071 | 1,32 / 1,69 | 1 geração |
| **B** | **0,000166** | **1,75 / 2,17** | Jev (0,49 / 0,61 s; US$0,000077, ~1.800 tokens, 30 perguntas) + geração |
| C | 0,00037 | 2,48 / 2,98 | Jev + 3 gerações em paralelo (máx.) + gate Jev (0,52 s) |
| D | 0,00026 | 1,43 / 2,17 | 1 geração (haiku-4.5) |

- A latência de geração foi medida só em chamadas fora do cache (n = 116–119 por condição). O pipeline B/C soma as etapas
  sequenciais por ponto.
- Falhas do provedor: a Gemini devolveu `finish_reason=error` com conteúdo vazio de forma **determinística** em 1 dos 119
  pontos, com o mesmo prompt de briefing, em qualquer seed. Nesse ponto usei a resposta de A como fallback (marcado). No dev,
  1–2 de 12 pontos tiveram o mesmo problema.
- Uso total: **~1.620 chamadas de LLM (~US$0,13)**, ~8% acima do teto de ~1.500 chamadas por causa das iterações de dev, das
  retentativas e das ablações julgadas; o custo ficou muito abaixo de US$1,50. **~2.810 chamadas de Jev (~US$0,10)**, dentro do
  teto de 3.000.

---

## 8. Limitações

- **Nenhum juiz humano.** Os dois juízes automáticos mostraram viés sistemático (preferem o texto mais longo e articulado). Por
  isso tratei "engano" como distinguibilidade e dei mais peso às métricas de código e à coincidência de movimento, que também é
  rotulada pelo Jev e é ruidosa. A pergunta "soa humano para uma pessoa?" continua **aberta** e merece um teste com avaliadores
  humanos.
- **O maichat é pequeno e peculiar**: 42 conversas, pessoas que já se conhecem, jovens britânicos, e uma tarefa de estudo ("we
  need like 60 msgs", "this chat looks long enough"). Parte das respostas humanas é meta-conversa impossível de prever. Os
  estratos têm n = 17 (ICs largos) e as ablações, n = 36.
- **O alvo é uma única resposta humana**, mas há muitas respostas humanas boas para cada ponto. Coincidência de movimento
  de 34% não significa que os outros 66% estejam errados.
- **Circularidade:** o Jev escreveu o briefing, escolheu no gate, julgou os pares e rotulou os movimentos. Mitiguei com o juiz
  LLM, as métricas de código e o controle sem Jev, mas a vantagem de C no juiz Jev deve ser descontada.
- **Uma única boca** (gemini-3.5-flash-lite) e um único modelo caro de referência (haiku-4.5). Um modelo que obedeça menos ao
  briefing pode se comportar de outro jeito.
- **Idioma:** o experimento é todo em inglês. As traduções para PT-BR ("kkkk", "vc", "nossa!", "e você?") são **inferência
  minha**, não dado. É preciso repetir com conversas brasileiras e calibrar as taxas-base locais (riso e emoji devem ser bem mais
  frequentes em PT-BR).
- **O briefing foi iterado em 12 pontos de dev** (três versões), o que é pouco. Os limiares foram calibrados em 60. Houve cuidado
  para não olhar o teste antes de congelar tudo, mas o desenho do briefing ainda é uma primeira versão, e a correção excessiva
  (curto demais, sem pergunta) é consertável.

---

### Arquivos

- `scripts/analysis/a9_common.py`: carga, features de código, lista LLM-ish, gírias, estilo da persona, chamadas cronometradas
- `a9_sample.py` (amostra estratificada), `a9_brief.py` (perguntas do Jev + `build_brief`), `a9_calibrate.py` (limiares no dev),
  `a9_generate.py` (condições), `a9_gate.py` (C), `a9_judge.py` (juízes), `a9_moves.py` (movimento e tom), `a9_extra.py`,
  `a9_analyze.py`
- `analysis/data/`: `a9_points.jsonl` (pontos, briefings e respostas), `a9_judge.jsonl`, `a9_moves.jsonl`, `a9_results.json`
  (todas as métricas com IC), `a9_extra.json`, `a9_vocab.json`, `a9_examples.json`, `a9_thresholds.json`,
  `a9_calibration_diag.json`, `a9_latency.json`, `a9_costs.json`
