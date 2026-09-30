# 04 — Estilo, registro e espelhamento (acomodação)

Dimensão a4. Scripts em `scripts/analysis/a4_*.py`, saídas em `analysis/data/a4_*`.
Dados: `messages.jsonl`/`turns.jsonl`/`jev_base.jsonl` (maichat EN, whatsapp_nl NL, nps_chatroom EN, nus_sms EN,
empathetic EN). Jev: 1.740 chamadas novas (≈US$0,04); LLM (gemini-3.5-flash-lite via `scripts/llm.py`): 120 chamadas (≈US$0,01).

> **Aviso sobre os dados (vale para outros agentes):** em `scripts/features.py` a classe `EMOJI` termina com um `-`
> literal, então `f.n_emoji` (e o `n_emoji` dos turnos) **conta hífens**. Além disso, o whatsapp_nl de 2012–2014
> usa emoji do iPhone antigo (SoftBank, na área de uso privado U+E001–U+E53E, ex.: `` = 😂), que nenhum regex
> pega. Refiz as features de estilo em `a4_style_feats.py` com o regex corrigido: a taxa de mensagens com emoji no
> whatsapp_nl vai de 5,3% para 11,0% (emoji ou emoticon: 17%).

---

## 1. Resumo dos achados

- **Cada pessoa tem uma impressão digital estável, mas por taxa, não por mensagem.** Numa mensagem isolada o estilo
  varia muito: a parte da variância explicada pelo falante (η²) fica em 0,03–0,12 para riso, emoji, alongamento e
  abreviação. Já a taxa de cada pessoa se repete entre as metades pares e ímpares das mensagens dela: r = 0,85–0,98
  no whatsapp_nl (≥100 msgs/pessoa) e 0,45–0,97 no maichat (~50 msgs/pessoa). Com um vetor de 13 features, a
  1ª metade das mensagens de alguém acha a 2ª metade da mesma pessoa em 59% dos casos no whatsapp (66 falantes,
  acaso = 1,5%), 62% no SMS (87 falantes), 33% no NPS e 18% no maichat. Contra o próprio parceiro de conversa, o
  vetor acerta quem é quem em 89% (whatsapp) e 69% (maichat).
- **Os traços mais "de assinatura" são a caixa inicial e o ponto final.** Começar em minúscula tem η² = 0,59 no
  maichat, com a mediana por pessoa em 93% e o p10 em 4%, e confiabilidade de 0,97. No ponto final, η² vai de 0,33
  (maichat) a 0,52 (NPS). **Cinco pessoas produzem 62% (maichat) e 72% (whatsapp) de todos os pontos finais do
  corpus.** O ponto final é um hábito da pessoa, não um termômetro do humor.
- **Cada pessoa ri sempre do mesmo jeito.** A forma dominante de riso (haha, hahaha, lol, hehe, emoji) responde por
  71% dos risos de cada pessoa no whatsapp (mediana, n = 59), 81% no SMS e 83% no NPS. Um usuário de SMS escreveu
  "Haha" em 572 dos seus 573 risos.
- **O espelhamento local existe e é forte para riso, emoji, alongamento e exclamação, mas quase nulo para caixa e
  abreviação.** O lift foi ajustado pela taxa de base do próprio B. Riso puxa riso com lift de 2,8× no maichat
  (IC95 2,2–4,3) e 1,5× no whatsapp (1,4–1,6). Emoji puxa emoji: 1,9× e 1,2×. Alongamento: 1,25× (n.s.) e 1,5×.
  Exclamação: 8,8× (n = 38) e 1,4×. Minúscula e abreviação ficam em 1,0× no maichat e 1,2× no whatsapp. Pergunta
  **não** puxa pergunta (0,81×: quem recebe pergunta responde).
- **O contágio vale só para a resposta seguinte.** No whatsapp, o lift do riso é 1,51 na resposta imediata e cai para
  1,02 duas respostas depois. **O riso é mais reação ao outro do que tique próprio:** o riso de A prevê o riso de B
  na resposta seguinte (2,8×/1,5×) melhor do que prevê o próximo riso do próprio A (0,86×/1,2×).
- **O tamanho acompanha pouco.** A correlação do comprimento dentro do falante é r = 0,12 (maichat) e 0,14
  (whatsapp). Quando A escreve no quartil mais longo (relativo a ele mesmo), B escreve 1,51–1,72× a sua mediana,
  contra 1,24–1,27× quando A é curto. Rajadas puxam rajadas: se A mandou ≥3 bolhas, B manda ≥2 em 50%, contra 26%
  se A mandou uma (maichat).
- **Não há convergência dentro da conversa, mas os pares já começam parecidos.** A diferença de estilo entre os
  parceiros não diminui do 1º para o último terço: 0 de 10 features no whatsapp; só o riso no maichat, com p = 0,006
  e 45% dos pares convergindo. Mesmo assim, os pares são muito mais parecidos que pares sorteados: comprimento com
  Spearman 0,91 (maichat) e 0,62 (whatsapp), ausência de pontuação 0,58/0,62, riso 0,33/0,48 (p_perm < 0,02).
- **Tensão e raiva matam o riso e o emoji.** Nos turnos com tensão: riso 0,9% contra 5,5% (maichat) e 6,4% contra 19%
  (whatsapp); emoji 2,6% contra 6,0% (maichat); alongamento 1,8% contra 6,8% (whatsapp, p < 0,01 dentro do falante).
  Em raiva, o riso vai a 0% no maichat (n = 203).
- **A hipótese "em momento sério usa pontuação completa" não se confirma; acontece o contrário.** Em turnos
  vulneráveis a fração de bolhas com pontuação final **cai**: 19% contra 30% no whatsapp e 4% contra 11% no
  maichat (p < 0,01). A pessoa escreve **mais e picotado**: 2,8× mais caracteres e 2,5 contra 1,65 bolhas no
  whatsapp, 1,7× e 1,8 contra 1,4 no maichat, com mais reticências (8,5% contra 2,8%). A parte do "para de alongar"
  se confirma só em tensão.
- **O ponto final soa um pouco frio, mas o efeito é pequeno.** Na leitura do Jev com pares mínimos (60 respostas
  curtas reais), acrescentar "." muda p(frio) em +0,03 (sobe em 80% dos pares). Acrescentar "haha" muda p(frio) em
  −0,24 e p(caloroso) em +0,35; ":)" dá −0,20 e +0,36; "!" dá +0,13 em caloroso. A resposta curta e seca já lê
  como fria: p(frio) = 0,42. Nos dados reais, o turno com ponto tem valência um pouco menor (−0,24 dentro do falante
  no whatsapp, p = 0,01, 17 falantes), mas a reação do parceiro não muda.
  **O que esfria é a falta de amaciador, não o ponto.**
- **Registro real.** Ficam sem pontuação final 44% das mensagens no SMS, 80% no NPS, 89% no maichat e 76% no
  whatsapp. No crowdworker "caprichado" (empathetic) são 25%. Começam em minúscula 75% no maichat e no NPS, mas só
  12% no SMS e 2% no whatsapp, porque o celular capitaliza sozinho. Têm abreviação da lista 51% no SMS, 24% no NPS e
  19% no maichat. O riso ou emoji ocupa o lugar da pontuação final em 20% das mensagens do whatsapp.
- **O Jev não detecta LLM casual quando a pergunta é holística; com pergunta atômica, detecta.** Com 60 casos do
  maichat, o Jev escolhe a resposta real contra a reescrita formal em 95%. Contra o gemini em tom casual, escolhe a
  real em só 38%, e contra o gemini instruído a imitar o estilo, em 23%: ele **prefere** a LLM. Os Nouls "é chatbot?"
  e "mesma pessoa?" dão AUC de 0,43–0,51. A pergunta atômica "usa mais gíria/emoji/riso do que X costuma usar?" dá
  AUC 0,81 (LLM simples) e 0,69 (LLM com estilo), no mesmo nível de uma regra de código de excesso de marcadores
  (0,79/0,69). A LLM erra **por excesso**: riso em 40% das bolhas contra 4% do humano, emoji 25% contra 0%, "fr fr",
  "💀", "bestie 🫡".

---

## 2. Achados detalhados

### 2.1 Impressão digital de estilo (a4_fingerprint.py → `a4_fingerprint.json`)

Inclui falantes com ≥30 mensagens (maichat, 79 falantes), ≥100 (whatsapp, 66), ≥40 (NPS, 52) e ≥60 (SMS, 87).
A tabela mostra a taxa média, a faixa entre pessoas (p10–p90), η² (parte da variância explicada pelo falante) e
r_I (correlação entre metades pares e ímpares das mensagens de cada pessoa).

| feature | maichat média (p10–p90) | η² | r_I | whatsapp média (p10–p90) | η² | r_I |
|---|---|---|---|---|---|---|
| começa em minúscula | 0,76 (0,04–1,00) | **0,59** | **0,97** | 0,02 (0–0,05) | 0,22 | 0,99 |
| termina em ponto final | 0,02 (0–0,06) | **0,33** | 0,92 | 0,02 (0–0,05) | 0,05 | 0,92 |
| sem pontuação final | 0,83 (0,70–0,95) | 0,12 | 0,77 | 0,52 (0,28–0,75) | 0,12 | 0,96 |
| riso | 0,03 (0–0,06) | 0,05 | 0,51 | 0,10 (0,02–0,21) | 0,05 | 0,85 |
| emoji/emoticon | 0,04 (0–0,07) | 0,05 | 0,45 | 0,17 (0,03–0,38) | 0,12 | 0,98 |
| alongamento | 0,04 (0–0,08) | 0,06 | 0,52 | 0,04 (0,01–0,09) | 0,03 | 0,84 |
| abreviação | 0,19 (0,08–0,33) | 0,06 | 0,52 | 0,09 (0,04–0,17) | 0,04 | 0,91 |
| reticências | 0,01 | 0,03 | 0,29 | 0,02 (0–0,05) | 0,29 | 0,97 |
| log(caracteres) | mediana de 19 caracteres | 0,15 | 0,74 | mediana de 22 caracteres | 0,07 | 0,95 |

- **Leitura.** Uma mensagem sozinha diz pouco: η² baixo significa que a pessoa "às vezes ri, às vezes não". A taxa
  dela, porém, é muito estável. A confiabilidade cai no maichat só porque cada pessoa tem ~50 mensagens.
  Metades temporais (início × fim) dão r um pouco menor que metades intercaladas: 0,47 contra 0,51 no riso do
  maichat e 0,85 contra 0,85 no whatsapp. A deriva ao longo da conversa é pequena.
- **Traços binários de "política pessoal".** Minúscula inicial e ponto final funcionam quase como configuração do
  teclado ou da pessoa. No maichat, p10 = 4% e p90 = 100% de minúscula: há quem nunca capitaliza e quem sempre
  capitaliza. Ex.: conv054 B escreve *"OK - see you Friday."* e *"OK so you won't be back till late then."*
  (sempre com ponto; é a conversa de família). O parceiro dele escreve *"oh I just got the message | yes it is a
  nice locket isnt it."*.
- **Riso: forma fiel.** Distribuição das formas no whatsapp: haha 60%, hahaha+ 26%, emoji 9%. No NPS, lol domina
  (74%). No SMS: haha 51%, lol 26%. A forma dominante de cada pessoa cobre 71%/81%/83% dos seus risos (mediana;
  maichat sem n suficiente). Há idioletos: um falante do whatsapp usa "hah" (74×) e outro alterna
  "Hahah"/"Hahaha"/"Hahahah" sem nunca usar "Haha" curto.
- **Identificação.** Vetor z-normalizado de 13 features, 1ª metade × 2ª metade, vizinho mais próximo. Top-1 fica em
  59% (whatsapp), 62% (SMS), 33% (NPS) e 18% (maichat), contra acaso de 1–2%. Top-5 fica em 85%, 78%, 73% e 48%.
- **Confiança: alta** para "estilo = taxas estáveis por pessoa" (consistente nos 4 corpora). É média para os valores
  do maichat, com poucos dados por pessoa.

### 2.2 Espelhamento local turno a turno (a4_mirroring.py → `a4_mirroring.json`)

Pares consecutivos A(t) → B(t+1), com falantes diferentes e na mesma sessão, em chats diádicos: maichat
n = 2.826 pares em 42 díades, whatsapp n = 29.472 em 57, empathetic n = 82.370. O `lift_spk` é a razão entre o
observado e o esperado pela taxa do próprio B naquela conversa (deixando o turno de fora). Isso remove o efeito "B
já ri muito" e o clima geral da conversa. IC95 por bootstrap de díades.

| feature em A → mesma feature em B | maichat lift_spk [IC95] | whatsapp lift_spk [IC95] | empathetic | autopersistência de A (maichat/whatsapp) |
|---|---|---|---|---|
| riso | **2,78** [2,15–4,32] | **1,51** [1,41–1,64] | 2,26 | 0,86 / 1,20 |
| emoji/emoticon | 1,89 [0,82–2,53] | 1,20 [1,13–1,29] | 3,04 | 1,15 / 1,10 |
| alongamento | 1,25 [0,87–2,05] | 1,51 [1,41–1,81] | 3,6 (n pequeno) | 0,78 / 1,30 |
| "!" no fim | 8,79 [1,44–13,3] (n = 38) | 1,43 [1,24–1,69] | 1,10 | 1,36 / 1,22 |
| palavra em CAPS | 2,25 [0,94–3,82] | 3,99 [2,50–8,05] | 0,90 | — |
| "!!"/"??" | 1,55 [1,05–2,85] | 1,54 [1,36–2,07] | 1,24 | — |
| ponto final | 2,30 [0–5,1] | 2,31 [1,00–3,18] | 1,00 | 0,98 / 1,86 |
| abreviação | 0,98 [0,89–1,08] | 1,19 [1,13–1,28] | 1,78 | — |
| começa em minúscula | 1,01 [1,00–1,01] | 1,96 [1,59–3,48] | 1,15 | — |
| pergunta | 1,07 [0,87–1,31] | **0,81** [0,73–0,90] | 0,87 | — |

- **O lift bruto engana.** Sem ajustar pela taxa de B, o riso do maichat daria 3,5×, a abreviação do empathetic
  1,4×, o CAPS 4–7×. Parte disso é "gente que ri conversa com gente que ri" (ver 2.4).
- **Decaimento (lift_spk no k-ésimo turno seguinte de B).** O riso no whatsapp vai 1,51 → 1,02 (k = 2) → 1,03
  (k = 5); o alongamento vai 1,51 → 1,04; o emoji 1,20 → 1,04. **O efeito é local.** O ponto final vai
  2,31 → 2,26 → 1,52 e a minúscula 1,96 → 1,39 → 1,68: esses marcam um **modo** que dura (trechos mais "formais" da
  conversa), não um reflexo.
- **O riso é resposta, não tique.** No maichat, o riso de A no turno t prevê o riso de B em t+1 (2,78×) e não o
  próximo riso do próprio A (0,86×). Exemplos:
  - conv007 · A: *"hahaha yesssss"* → B: *"hahahhaa | oops"*
  - conv022 · A: *"you know that panda caretakers dress up in panda suits...? | lol"* → B: *"wait no way | really?"*
    (a reação veio sem riso: o lift não é determinismo; o P(B ri | A riu) é de 17%)
- **Tamanho.** A correlação de log(caracteres) A → B dentro do falante é 0,12 (maichat), 0,14 (whatsapp) e 0,03
  (empathetic). Comprimento relativo de B (em relação à média dele) por quartil do comprimento relativo de A:
  maichat 1,24 / 1,19 / 1,31 / 1,51; whatsapp 1,27 / 1,32 / 1,38 / 1,72. A elasticidade fica em ~0,12–0,15: B segue
  o comprimento de A muito parcialmente.
- **Número de bolhas.** P(B manda ≥2 bolhas) é 0,50 se A mandou ≥3 e 0,26 se A mandou 1 (maichat). No whatsapp,
  0,58 contra 0,44. A correlação de n_msgs dentro do falante é 0,09 e 0,11.
- **Confiança: alta** para riso, emoji e alongamento no whatsapp (n grande, IC estreito). Média no maichat, com IC
  largo e n_A = 135 risos. A exclamação e o CAPS do maichat têm n pequeno. No empathetic a conversa é induzida e
  curta (4 falas): serve só como réplica de direção.

### 2.3 Convergência ao longo da conversa × semelhança entre parceiros

- **Convergência.** A diferença |taxa_A − taxa_B| foi comparada entre o 1º e o último terço da conversa (Wilcoxon
  sobre díades). No maichat, só o riso converge: 0,077 → 0,038, p = 0,006, mas só 45% das díades convergem. Emoji,
  alongamento, abreviação, minúscula, pontuação, comprimento e bolhas não convergem (p de 0,12 a 0,95). No whatsapp,
  nenhuma das 10 features converge. **A hipótese de convergência progressiva não se confirma.**
- **Semelhança entre parceiros** (díade real × pareamento aleatório, 2.000 permutações). Spearman entre as taxas de A
  e B: comprimento 0,91 (maichat) e 0,62 (whatsapp); sem pontuação 0,58/0,62; minúscula 0,51/0,45; abreviação
  0,45/0,37; riso 0,33/0,48; emoji 0,32/0,34. Todos com p_perm < 0,03. O ponto final não é compartilhado
  (0,36/0,16; p = 0,03/0,35).
- **Leitura.** O alinhamento aparece "pronto" desde o início: amigos já têm normas comuns, e a mesma plataforma e o
  mesmo tema também pesam, o que é um confundidor. O ajuste que se vê dentro da conversa é **local**, turno a turno
  (2.2), e não uma deriva lenta. Confiança média (só 42 e 57 díades).

### 2.4 Estilo × momento da conversa (a4_moments.py → `a4_moments.json`)

Os momentos vêm do passe D do `jev_base`. Há dois enquadramentos:

- **(same)**: o momento do próprio turno. É parcialmente circular, porque o Jev viu o texto, inclusive o "haha".
- **(prev)**: o momento do turno **anterior do parceiro**, que é a posição do bot. Não é circular.

As comparações são dentro do falante. Uma estrela marca p < 0,05 e duas, p < 0,01 (teste t das diferenças por
falante). A tabela usa o enquadramento same.

| momento (same) | riso | emoji | alongamento | fração com pontuação final | caracteres (log) | bolhas |
|---|---|---|---|---|---|---|
| tensão, maichat (n = 462) | **0,009** vs 0,055** | 0,026 vs 0,060** | 0,019 vs 0,066 | 0,08 vs 0,11 | 2,94 vs 3,27 | **1,19** vs 1,51** |
| tensão, whatsapp (n = 110) | 0,064 vs 0,192 | 0,145 vs 0,265 | **0,018** vs 0,068** | 0,28 vs 0,29 | 3,55 vs 3,69 | 1,75 vs 1,76 |
| raiva, maichat (n = 203) | **0,000** vs 0,051** | 0,030 vs 0,057 | 0,030 vs 0,061 | 0,10 vs 0,10 | 3,06 vs 3,23 | 1,27 vs 1,47 |
| vulnerável, maichat (n = 249) | 0,060 vs 0,046 | 0,076 vs 0,053 | 0,048 vs 0,060* | **0,04** vs 0,11** | **3,72** vs 3,17** | **1,79** vs 1,42** |
| vulnerável, whatsapp (n = 375) | 0,20 vs 0,19 | **0,38** vs 0,24** | 0,08 vs 0,07 | **0,19** vs 0,30** | **4,59** vs 3,56** | **2,52** vs 1,65** |
| flerte, maichat (n = 106) | 0,057 vs 0,047 | **0,207** vs 0,049** | 0,028 vs 0,060 | 0,05 vs 0,11 | 2,98 vs 3,23 | 1,33 vs 1,46 |
| brincalhão, whatsapp (n = 1.430) | **0,39** vs 0,01** | 0,42 vs 0,12** | 0,11 vs 0,03** | 0,22 vs 0,36** | 3,80 vs 3,59** | 1,95 vs 1,59** |
| valência negativa, whatsapp (n = 305) | 0,08 vs 0,20** | 0,24 vs 0,26 | 0,05 vs 0,07 | 0,21 vs 0,30** | 3,92 vs 3,66** | 2,11 vs 1,72** |

- **Momento sério ou vulnerável.** A pessoa escreve **mais**, em **mais bolhas**, com **menos** pontuação final e
  mais reticências (whatsapp 8,5% contra 2,8%). A hipótese "pontuação completa" cai, porque o desabafo sai picotado.
  Exemplo, conv031 B: *"truly | I keep thinking too meta | I wonder if this captures things I almost type | I've
  backspaced over severeal sentences"*.
- **Tensão ou conflito.** Some o riso, cai o emoji, cai o alongamento (whatsapp) e os turnos ficam **mais curtos** e
  em uma bolha (maichat). Exemplos:
  - conv091: *"did. you. move. it."* → *"maybe temporarily relocated"* → *"TO WHERE"*. Aqui o ponto e o CAPS servem
    de ênfase na briga de brincadeira.
  - conv088: *"shut up"* / *"never"*.
- **Ponto final no conflito.** Tendência fraca. Fase de conflito no whatsapp: 10,4% contra 4,2% (n = 48, n.s.
  dentro do falante). Raiva no maichat: 5,9% contra 2,3%. Valência negativa no maichat: 3,7% contra 2,3% (p < 0,05).
  Confiança baixa.
- **Enquadramento prev (como a pessoa responde ao momento do outro).**
  - Depois de um turno vulnerável do parceiro (whatsapp, n = 347), a resposta é **1,36× mais longa** (log 3,89
    contra 3,58**), traz mais perguntas (27% contra 22%*) e **não** perde o riso (28% contra 19%*): no holandês,
    "haha" funciona como amaciador. No maichat não há diferença.
  - Depois de um turno brincalhão, a resposta traz mais riso (whatsapp 31% contra 11%**, maichat 5,5% contra 3,6%**)
    e mais emoji.
  - Depois de raiva, a resposta traz menos emoji (whatsapp 17% contra 26%**).
  - Depois de flerte, menos pontuação final e menos pergunta (maichat, *).
- **Confiança.** Alta para riso × tensão/brincadeira e para comprimento/bolhas × vulnerabilidade (os dois corpora
  concordam). Média para o resto. O enquadramento same infla os efeitos de riso e emoji por circularidade.

### 2.5 O ponto final soa frio? (a4_period.py → `a4_period.json`)

- **Dados reais.** Turnos terminados em "." (maichat n = 53, whatsapp n = 93) têm valência (D) menor dentro do
  falante: −0,24 no whatsapp (p = 0,012, 17 falantes) e −0,18 no maichat (n.s., 9 falantes). A tensão não muda. O
  parceiro **não** reage diferente: a tensão e a valência do turno seguinte ficam iguais (p > 0,2).
- **Concentração.** O ponto é hábito de poucos: 5 pessoas fazem 62%/72% dos pontos. Nas 2 conversas de família do
  maichat, 24% dos turnos têm ponto, contra 1,6% entre amigos (n minúsculo).
- **Pares mínimos no Jev** (60 respostas curtas reais do maichat sem pontuação, riso ou emoji, no contexto; ex.:
  *"sounds good xx"*, *"pls dont"*, *"then stop supervising my life"*). p(frio/seco) e p(caloroso), com o Wilcoxon
  pareado contra a versão sem nada:

  | variante | p(frio) | Δ | p(caloroso) | Δ |
  |---|---|---|---|---|
  | sem nada | 0,42 | — | 0,40 | — |
  | + "." | 0,46 | **+0,03** (80% sobem) | 0,38 | −0,02 |
  | + "!" | 0,36 | −0,06 | 0,53 | **+0,13** |
  | + " haha" | 0,18 | **−0,24** | 0,75 | **+0,35** |
  | + " :)" | 0,22 | −0,20 | 0,76 | **+0,36** |

  Nenhuma variante muda o "parece bot" (Δ ≤ 0,04).
- **Conclusão.** O ponto final esfria pouco. O que torna uma resposta curta fria é a **falta de amaciador**
  (riso, emoji, "!"). Confiança média: o Jev é um leitor-modelo, não um humano, mas o resultado bate com a
  literatura (Gunraj et al., 2016).

### 2.6 Registro: SMS, chat room e o "quão humano" (a4_register.py → `a4_register.json`)

| | SMS (nus) | chat room (NPS) | maichat | whatsapp (NL) | empathetic (crowd) |
|---|---|---|---|---|---|
| sem pontuação final | 44% | 80% | 89% | 76% | 25% |
| termina em ponto final | 19% | 4% | 2% | 2% | 43% |
| começa em minúscula | 12% | 75% | 75% | 2% | 11% |
| abreviação/gíria (lista) | **51%** | 24% | 19% | 10% | 9% |
| "im/dont/thats" sem apóstrofo | 4% | 5% | 8% | — | 4% |
| "i" minúsculo | 7% | 7% | 8% | — | 6% |
| reticências | 12% | 7% | 0,7% | 2% | 1,6% |
| riso | 17% | 13% | 3% | 10% | 2,5% |
| termina em riso/emoji | 6% | 11% | 5% | **20%** | 2% |
| tem token fora do léxico* | 69% | 38% | 36% | — | — |
| mediana de caracteres/palavras | 36/7 | 20/3 | 19/4 | 22/4 | 59/12 |

\* O léxico são as palavras com frequência ≥5 no empathetic. É um proxy grosseiro de typo/abreviação e é inflado
por nomes e pelo singlish (lor, lah, leh).

Inventário das formas mais marcadas contra o inglês "caprichado" (log-odds com prior):

- **SMS (Singapura):** u (11k), haha, lol, ok/okay, ur, k, 2, 4, n, r, ya, dun, wat, tmr, pls, abt, coz/cos, ppl,
  jus/juz, nvm, okie, gd/gud, dat, nite, hav, thks, shld, frm, tink. Locais: lor, lah, leh, liao, meh, sia.
- **NPS:** lol (10% dos posts), lmao, u, wanna, ya, im, wb, hiya, ty, brb, yw, tc, sup, afk, prolly, ppl, nite,
  asl, heyy/heyyy.
- **maichat (britânicos jovens, 2023):** u, im, dont, thats, idk, btw, tho, omg, ur, bc, kinda, gonna, tbh, tbf, fr,
  rn, ig, lowkey, defo, yh, wbu, rly; formas alongadas (yesss, stoppp, hellooo, byeee); autocorreção com "*": na
  conv003, *"shape themb"* → *"*them"*.
- **Leitura.** O texto humano é curto (4 palavras de mediana no chat), quase nunca fecha com ponto e mistura poucas
  abreviações por mensagem: 81% das mensagens do maichat **não** têm nenhuma. O "sotaque" vem de 1 ou 2 marcadores
  frequentes da pessoa ou do grupo (u, im, lol), não de uma chuva de gírias. A caixa depende do aparelho: no celular
  o corretor capitaliza (whatsapp 2%, SMS 12%), no desktop não (maichat 75%).

### 2.7 O Jev como filtro de naturalidade/estilo (a4_jev_style.py, a4_jev_style2.py)

**Montagem.** 60 turnos reais de 60 falantes/momentos do maichat, no máximo 2 por conversa. O state tem ~10
mensagens do falante X (mais de 3 turnos longe do alvo) e os 8 turnos anteriores. Candidatas:

- `real`: o que X mandou.
- `formal`: o real limpo por código (caixa, apóstrofos, gírias expandidas, sem riso/emoji/alongamento, com ponto, em
  1 bolha).
- `llm_plain`: gemini-3.5-flash-lite respondendo como X, só com o contexto.
- `llm_styled`: a mesma LLM, com as amostras de X e a instrução de imitar o estilo.
- `partner`: a resposta real do parceiro ao mesmo contexto.

O Choice foi rodado nas duas ordens e a acurácia usa a média das duas.

| real contra… | Jev Choice "qual X digitou?" | Noul "é chatbot" (AUC) | Noul "mesma pessoa" (AUC) | regra de código (perfil de features) |
|---|---|---|---|---|
| formal | **95%** | 0,72 | 0,77 | 97,5% |
| llm_plain | **38%** | 0,51 | 0,44 | 82% |
| llm_styled | **23%** | 0,43 | 0,32 | 52% |
| partner (outro humano) | 68% | 0,64 | 0,68 | 50% |

Com o enquadramento explícito ("uma delas é uma IA imitando X; qual é a genuína?"), o Choice acerta 30% contra as
duas LLMs. Perguntas atômicas sobre a candidata (AUC para detectar a LLM):

| Noul/Score | llm_plain | llm_styled |
|---|---|---|
| "usa **mais** gíria/abreviação/emoji/riso do que X costuma" | **0,81** | **0,69** |
| "soa como IA imitando jovem (gíria forçada, piada performática)" | 0,69 | 0,62 |
| "tenta ser mais espirituoso/engraçado que o X de sempre" | 0,66 | 0,54 |
| Score "quão típico de X" (0–4) | 0,46 | 0,41 |
| regra de código: excesso de marcadores (candidata − amostras de X) | 0,79 | 0,69 |

Estatísticas de superfície das candidatas (por bolha):

| | bolhas/resposta | caracteres | começa minúscula | pontuação final | riso | emoji | abreviação |
|---|---|---|---|---|---|---|---|
| real | 1,55 | 42 | 76% | 11% | **4%** | **0%** | 28% |
| llm_plain | 1,00 | 60 | 88% | 23% | **40%** | **25%** | 50% |
| llm_styled | 1,92 | 29 | 89% | 3% | 9% | 7% | 24% |
| partner | 1,45 | 32 | 63% | 15% | 3% | 5% | 8% |

Exemplos (real → LLM):

- *"they have such good food"* → plain: *"hate costello coffee lol"* / styled: *"wait | did u just get snipped |
  middle of a sentence frfr"*
- *"Yeah same I didn't understand that well"* → plain: *"linear algebra is wild lol. good luck figuring out null
  space, that one broke my brain for a minute 😂"* / styled: *"oh god linear algebra 💀"*
- *"72 is close enough"* → plain: um parágrafo de 60 palavras terminando em *"wish me luck bestie 🫡"*
- *"ahh cool"* → styled: *"ooh cool | makes sense tho her shows are basically storytelling anyway"*

**Leitura.**

1. A LLM de hoje, com prompt de "amigo no chat", já **não soa formal**: ela exagera o casual, com mais riso, emoji,
   gíria da moda e piada de efeito. O humano real é **mais banal**, com respostas como "ahh cool", "pls dont" e
   "we are intellectuals".
2. O Jev pega o formal (95%) mas não pega o casual exagerado quando a pergunta é holística ("é chatbot?", "mesma
   pessoa?", "qual é o real?"). Nessas perguntas ele até prefere a LLM, que é mais **prototípica** do estilo: uma
   caricatura.
3. Com a pergunta **atômica e comparativa** ("mais marcadores do que X costuma?"), o Jev chega a AUC 0,81, empatado
   com a regra de código. Isso segue a boa prática oficial: perguntas atômicas e literais.
4. Entre dois humanos, o Jev distingue X do parceiro em 68%, e a regra de código em 50%. Ele capta algo do jeito de X
   além da contagem de marcadores.

**Confiança:** média (60 casos, uma LLM, um corpus). Custo: 1.440 chamadas de Jev (≈US$0,04) e 120 de LLM.

---

## 3. Padrões "invisíveis" (o que as pessoas fazem sem perceber e o bot deveria imitar)

1. **Fidelidade de forma.** Quem ri com "Haha" ri com "Haha" 70–80% das vezes e quase nunca troca para "lol". Numa
   persona PT-BR isso seria algo como "kkkk", "hahaha" ou "rsrs". Com base no efeito medido, a persona deve escolher
   **uma forma dominante e mais uma variante rara**, e não copiar a forma do usuário.
2. **O riso responde ao riso do outro, não ao próprio.** O contágio vale só para a resposta imediata e some duas
   trocas depois. Rir porque o usuário riu é natural; rir de novo só porque o bot riu antes, não.
3. **Riso e emoji fazem o papel do ponto.** No whatsapp, 20% das mensagens **terminam** em riso ou emoji, e quase
   nenhuma em ".". O fim da mensagem é onde mora o tom.
4. **Desabafo sai picotado e sem pontuação.** No momento vulnerável as pessoas escrevem 1,7–2,8× mais, em mais
   bolhas, e pontuam **menos**. Uma LLM "séria" faz o oposto: um parágrafo único e bem pontuado.
5. **Resposta curta sem amaciador soa seca.** "sounds good" lê como frio (0,42). "sounds good haha" cai para 0,18.
   O ponto final pesa pouco (+0,03).
6. **Pergunta não puxa pergunta.** Depois de uma pergunta, a chance de devolver outra cai 19% (lift 0,81): primeiro
   se responde. O vício de LLM de terminar tudo com "e você?" vai contra isso.
7. **Na tensão, a brincadeira some antes de tudo.** O riso vai a ~0–1% no maichat, o alongamento cai 4× no whatsapp
   e o turno encurta (1,2 bolha). O ponto e o CAPS aparecem como ênfase ("did. you. move. it.").
8. **Não há "convergência lenta".** Os parceiros já chegam parecidos e o ajuste é local, turno a turno. Um bot que
   "vai virando o usuário" ao longo da conversa não reproduz o que se observa.
9. **Os humanos são banais.** Parte grande das respostas reais é curta, sem piada e sem gíria de efeito
   ("ahh cool", "72 is close enough"). A LLM erra por excesso de personalidade, não por falta.
10. **Autocorreção com "*" e typos.** São 0,4% das mensagens no maichat e 0,3% no whatsapp e no NPS. É raro, mas é
    um sinal fortíssimo de digitação humana. Isso se usado com parcimônia (≤1 a cada ~200 mensagens).

---

## 4. Tradução para o sistema

Princípio geral: **o estilo é contado em código; o Jev julga momento e excesso.** O Jev não conta bem. As taxas de
estilo do usuário e da persona saem de regex determinístico, e o Jev entra onde é preciso julgar o momento, a
adequação e o excesso relativo.

### 4.1 Style profile da persona (configuração fixa, código)

Uma persona é um vetor de **taxas-alvo e políticas binárias** que não mudam ao longo das conversas. Os valores
abaixo são o padrão "amigo(a) jovem no chat", calibrado no maichat, no whatsapp e no NPS. Para PT-BR, usar os
equivalentes, mas **a calibração PT-BR é hipótese: não há dado em português aqui**.

| traço | tipo | padrão sugerido (faixa humana p10–p90) |
|---|---|---|
| caixa inicial | política binária (95/5) | escolher "sempre minúscula" (estilo desktop/jovem) **ou** "capitaliza como o corretor do celular" |
| ponto final | política binária | "quase nunca" (≤2%; é o padrão de 80–90% das pessoas); só em persona "mãe/pai/formal" usar ~20% |
| mensagem sem pontuação final | taxa | 75–90% |
| forma de riso | 1 dominante (≥70%) + 1 variante | ex.: "kkkk" 75% / "kkkkkkk" 25%; EN: "haha" / "hahaha" |
| taxa de riso por mensagem | taxa-base | 5–15% (maichat 3%, whatsapp 10%, SMS 17%) |
| emoji por mensagem | taxa-base | 4–15% (whatsapp p10–p90: 3–38%); repertório fixo de 3–5 emojis da persona |
| alongamento ("simmm", "nossaaa") | taxa | 3–5% |
| abreviações | inventário fixo de 3–6 formas | ex. PT: vc, tb/tbm, pq, q, msm, blz; cerca de 1 a cada 5 mensagens tem alguma (maichat 19%) |
| reticências | taxa | ≤2% (a não ser persona "reticente": até 10%) |
| mediana de tamanho | caracteres | 20–40 caracteres/bolha, 4–7 palavras |
| typos/autocorreção "*" | taxa | ≤0,5% das mensagens |

O código também guarda um **banco de 10–15 mensagens-exemplo da persona**. Ele é o `style_samples_of_X` dos
detectores do Jev.

### 4.2 Perfil do usuário (código, contínuo)

Janela deslizante das últimas 30 mensagens do usuário: taxa de riso e forma de riso, emoji, alongamento, CAPS,
política de caixa e de ponto, abreviações, mediana de caracteres e de bolhas. Sem Jev. Cada detector abaixo consome
esse perfil.

### 4.3 Detectores do Jev (por turno do usuário, na chamada de "leitura do momento")

Todos na mesma chamada, avaliados em paralelo. O state tem os últimos 6–8 turnos, com o último turno do usuário
juntado em um texto e em inglês se possível.

| id | tipo | instrução (literal, atômica) | uso |
|---|---|---|---|
| `tension` | Noul | "The user's last message shows tension, irritation or conflict with the other person." | ≥0,5 → modo sem brincadeira |
| `anger` | Noul | "The user is annoyed or angry in their last message." | ≥0,5 → corta emoji e riso |
| `vulnerable` | Noul | "The user is sharing something personal, sad or worrying in their last message." | ≥0,5 → modo acolhimento |
| `playful` | Noul | "The user's last message is joking, teasing or being silly." | ≥0,5 → libera riso e emoji |
| `seriousness` | Score 0–3 | playful banter / casual / somewhat serious / very serious | ≥2 → modo sério |
| `joke_welcome` | Noul | "A light joke in the next reply would be welcome." | combina com a regra do riso |

(Esses já existem no passe D e P do `jev_base`; reaproveitar as mesmas instruções.)

### 4.4 Regras de acomodação (código: quanto espelhar e quando **não**)

A probabilidade de cada marcador na resposta é a taxa-base da persona × multiplicador de espelhamento × multiplicador
de momento, com teto.

| marcador | espelhar? | multiplicador se o usuário usou no último turno | trava por momento |
|---|---|---|---|
| **riso** (o ato) | **SIM, só na resposta imediata** | ×2,5 (faixa observada 1,5–2,8), teto 0,6; na troca seguinte volta à base | tension ≥0,5 ou anger ≥0,5 → 0; seriousness ≥2 → ×0,3; vulnerable → não iniciar riso, mas "kkk" amaciador depois de desabafo leve é aceitável se joke_welcome ≥0,5 |
| forma do riso | **NÃO** | usar sempre a forma da persona | — |
| **emoji** | SIM | ×1,5 (1,2–1,9), repertório da persona | anger → ×0,6; flerte/afeto → ×2 (flerte tem 4× mais emoji no maichat) |
| **alongamento** | SIM, leve | ×1,5 | tension → 0 |
| **"!" / "!!"** | SIM | ×1,5 | conflito → 0 |
| CAPS de ênfase | só em brincadeira ou empolgação | ×2 se playful ≥0,5 | nunca em conflito real |
| **tamanho** | parcial | alvo = mediana da persona × (tamanho relativo do usuário)^0,15; teto 1,7× | vulnerable (prev) → ×1,35 e incluir 1 pergunta de acompanhamento |
| **nº de bolhas** | parcial | usuário ≥3 bolhas → P(bot ≥2 bolhas) = 0,5; usuário 1 bolha → 0,25 | vulnerable → permitir 2–3 bolhas curtas em vez de um parágrafo |
| caixa (minúscula/maiúscula) | **NÃO** | traço fixo da persona (lift 1,0 no maichat) | — |
| ponto final | **NÃO** | traço fixo; não "esfriar" por imitação | — |
| abreviações/gírias | **NÃO** a forma; no máximo a densidade | se o usuário é muito abreviado (>40%), subir a densidade da persona até ×1,3 **usando o inventário dela**, nunca as gírias do usuário | nunca gíria regional/dialeto do usuário (o análogo de "lor/lah") |
| pergunta | **anti-espelhar** | se o usuário perguntou, responder primeiro; P(devolver pergunta) ×0,8 | — |
| typos do usuário | **NÃO** | — | — |

Regra de sessão (convergência): no início do relacionamento (onboarding ou 1ª sessão), escolher entre 2 e 3
variantes de persona a mais próxima do registro do usuário (distância entre o vetor do usuário e o da persona, em
código). Depois, **não derivar**. Os pares humanos já começam parecidos (Spearman 0,3–0,9) e não convergem dentro
da conversa.

### 4.5 Pós-processador e validador de estilo (código + Jev), depois da LLM

Fluxo: briefing (momento + alvos de 4.4) → a LLM gera 2–4 candidatas → etapa **A** em código → etapa **B** no Jev →
entrega.

**A. Normalização determinística em código**, aplicada a todas as candidatas:

1. Impor a política de caixa e de ponto da persona: tirar o "." final com probabilidade 1 − taxa_persona e ajustar a
   caixa.
2. Trocar formas de riso alheias ("lol", "😂", "hahaha") pela forma da persona.
3. Cortar emojis fora do repertório.
4. Cortar abreviações fora do inventário.
5. **Limitar os marcadores**: nº de risos + emojis + gírias por bolha ≤ alvo de 4.4 + 1 desvio. A LLM casual gera
   riso em 40% e emoji em 25% das bolhas, contra 4% e 0% do humano.
6. Quebrar em bolhas conforme o alvo de bolhas; tirar "e você?" final se o usuário acabou de perguntar.
7. Lista negra de expressões de LLM: ver o relatório a8.

**B. Jev (uma chamada por candidata, em paralelo).** State = `{style_samples_of_X: banco da persona,
conversation_so_far: últimos 8 turnos, X_reply: candidata}`. Nouls:

- `more_markers`: "X_reply uses more slang, abbreviations, emoji or laughter than X usually does in
  style_samples_of_X." É o melhor detector medido (AUC 0,81/0,69).
- `ai_imitating`: "X_reply sounds like an AI imitating a young person texting (forced or trendy slang, performative
  jokes) rather than X's own plain way of texting." (AUC 0,69/0,62)
- `tryhard`: "X_reply tries harder to be witty, funny or clever than X's usual messages." (AUC 0,66)
- `cold` (só em resposta ≤6 palavras e fora de tensão): "X's reply sounds cold, curt or annoyed."
- `joke_misfit` (se a candidata tem riso ou emoji): "X_reply jokes or laughs although the other person is upset or
  talking about something serious."

**Regra de escolha (código).** Custo = 1,0·more_markers + 0,6·ai_imitating + 0,4·tryhard + 2,0·joke_misfit, mais a
penalidade de código por excesso de marcadores e desvio de tamanho. Entregar a de menor custo.

- Se todas tiverem more_markers > 0,6 ou joke_misfit > 0,5, regenerar uma vez com o briefing "mais simples, menos
  gíria, sem piada".
- Se a escolhida for curta e tiver cold > 0,5 em momento neutro ou brincalhão, acrescentar o amaciador da persona
  (riso ou emoji do repertório). No teste, isso baixou p(frio) em 0,20–0,24.

**O que NÃO fazer.** Não usar o Jev com pergunta holística ("isto parece um chatbot?", "qual soa mais como a mesma
pessoa?", "quão típico de X, 0–4") como filtro principal. Contra LLM casual deu AUC 0,41–0,51 e ele **prefere a
caricatura**. O Choice "qual é da mesma pessoa" só serve para separar formal × informal (95%) ou candidatas humanas
entre si (68%).

**Custo e latência.** 2–4 candidatas = 2–4 chamadas de Jev em paralelo (~0,5 s p50), ≈US$0,0001 por resposta.

---

## 5. Limitações

- **Idioma.** Não há dado em português. As taxas e as formas (kkkk, rsrs, vc, pq) para PT-BR são **transposições**
  dos padrões medidos em EN e NL, não medições. O whatsapp_nl é holandês e o Jev é melhor em inglês: os rótulos D do
  whatsapp (momento) são mais ruidosos, e "haha" em holandês tem função de amaciador que talvez não seja igual.
- **maichat pequeno.** São 42 conversas e ~50 mensagens por pessoa: os ICs são largos, e riso e emoji são raros
  (3–4%). Várias conversas parecem roteirizadas (ex.: a conv019 tem um conflito "encenado"). O NPS é em grupo e de
  2006. O SMS são mensagens avulsas de Singapura, com singlish. O empathetic é induzido, tem 4 falas por conversa e
  crowdworkers "caprichando".
- **Dispositivo como confundidor.** A caixa inicial e parte da pontuação vêm do teclado (autocorreção do celular ×
  desktop), não da pessoa.
- **Circularidade.** No enquadramento same, o Jev viu o texto com os marcadores, então os efeitos de riso e emoji ×
  brincadeira/tensão estão inflados. Por isso reportei também o enquadramento prev.
- **Espelhamento × tópico.** O lift ajustado pela taxa do falante remove o "B é risonho", mas não remove um trecho
  engraçado que faz os dois rirem. O decaimento rápido (lag 2 ≈ 1,0) sugere efeito local, mas a causalidade
  "contágio" contra "momento compartilhado" não se separa aqui. Para o bot, a regra prática é a mesma.
- **Convergência.** Com 3 terços por conversa e 42/57 díades o poder estatístico é limitado. "Não convergem" quer
  dizer "sem evidência de convergência média", não "prova de ausência".
- **Experimento do filtro.** Foram 60 casos, uma única LLM (gemini-3.5-flash-lite) e dois prompts. As amostras de
  estilo de X foram sorteadas de novo no follow-up (a4_jev_style2): o mesmo conjunto de mensagens, mas outro sorteio.
  A 1ª rodada usou `hash()`, que é aleatório por processo, e já corrigi para `crc32`. O "real" às vezes é ambíguo:
  respostas humanas banais, sem marcas de estilo, podem ser de qualquer pessoa. O Jev julgou a partir de ~10
  mensagens-exemplo; com um banco maior, o resultado pode mudar.
- **Pares mínimos do ponto.** O leitor é o Jev, não humanos. O efeito de "." (+0,03) é consistente, mas pequeno.
- **Regex.** O léxico de abreviações e a detecção de riso são listas manuais (EN e NL). O "fora do léxico" é um proxy
  grosseiro de typo. Emoji antigo do iPhone foi mapeado pela faixa U+E001–U+E53E, e só 😂 (U+E412) é tratado como
  riso.

---

### Arquivos

- Scripts: `scripts/analysis/a4_style_feats.py` (features corrigidas), `a4_common.py`, `a4_fingerprint.py`,
  `a4_mirroring.py`, `a4_moments.py`, `a4_period.py`, `a4_register.py`, `a4_jev_style.py`, `a4_jev_style2.py`.
- Saídas: `analysis/data/a4_fingerprint.json`, `a4_mirroring.json`, `a4_moments.json`, `a4_period.json`,
  `a4_register.json`, `a4_jev_style.json`, `a4_jev_style2.json`, `a4_jev_style_cases.csv`,
  `a4_jev_style_answers.csv`.
- Tabelas intermediárias grandes (`a4_msgs.pkl` e `a4_turns.pkl`) ficam no scratchpad, fora do git. Para
  regenerar: `A4_OUT=<dir> python3 scripts/analysis/a4_style_feats.py` e depois
  `A4_OUT=<dir> python3 scripts/analysis/a4_common.py`.
