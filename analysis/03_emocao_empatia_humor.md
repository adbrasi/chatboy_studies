# 03: Emoção, empatia, seriedade e humor

**Dimensão:** como as pessoas sentem, acolhem, ficam sérias e riem na conversa, e como o Jev pode detectar cada momento.
**Dados:** EmpatheticDialogues (24.850 conversas, rótulo-ouro de 32 emoções), `jev_base` (5.968 turnos: 2.871 do maichat e 3.097 do whatsapp_nl).
**Jev novo:** 2.977 chamadas (≈US$0,18): 1.472 na validação de emoção, 608 nas estratégias do ouvinte e 895 na função do riso.
**Scripts:** `scripts/analysis/a3_*.py`. **Saídas:** `analysis/data/a3_emotion_eval.json`, `a3_listener.json`, `a3_repertoire.json`, `a3_seq.json`, `a3_laugh.json`.

Convenções: "sério" = `D.seriousness ≥ 1,25` (escala 0–3). "Brincalhão" = `D.playful ≥ 0,5`. "Riso" = regex de riso das features, mais os emojis SoftBank antigos do WhatsApp (😂/😁/😄 como ``, `` e ``), que o regex original não pegava (+80 turnos). IC95% por bootstrap de conversas quando indicado.

---

## 1. Resumo dos achados

- **O detector de emoção fina do Jev acerta 45% no top-1 e 69–70% no top-3 entre 32 classes** (acaso = 3,1%; n = 608, estratificado). Metade dos erros cai em quase-sinônimos da mesma família (terrified→afraid, furious→angry). **Na família (7 grupos), acerta 70–74%; na polaridade, 85–90%.**
- **A cascata não ajudou.** Família → emoção fina deu 43–44% contra 45% do Choice plano (McNemar p = 0,28–0,69). Mesmo acertando a família sempre (oráculo), a etapa fina só chega a 61–62%. A cascata vale para **decidir o comportamento** (família/polaridade), não para ganhar precisão no rótulo fino.
- **O contexto ajuda só no nível grosso.** Ver A1+B1+A2 em vez de só A1 não muda o top-1 (44,7%→45,1%). Melhora a família (70,2%→73,8%, p = 0,025) e a polaridade direta (75,6%→84,4%), porque tira o "neutro/misto" de falas ambíguas. A conversa inteira não acrescenta nada.
- **A confiança do Jev ordena bem os acertos, mas tem valor absoluto inflado no rótulo fino.** Com confiança ≥0,8 no Choice de 32 classes, o acerto é 54% (confiança média 0,95; ECE ≈ 0,34). Na família, confiança ≥0,8 dá 80–81% de acerto (ECE ≈ 0,16), e o limiar ≥0,9 cobre 67% dos casos com 83–84% de acerto.
- **O ouvinte humano responde com uma pergunta exploratória, qualquer que seja a emoção.** É a estratégia principal em 41% das respostas a emoções negativas e em 46% das positivas (Jev, n = 608). **O que muda com a emoção é a abertura.** Diante de emoção negativa, a resposta abre com validação 30% das vezes (contra 11% nas positivas). Diante de positiva, abre com parabéns 18% das vezes (contra 3%). No corpus inteiro, a pergunta aparece em **58,0% das respostas a emoções negativas e em 52,6% das positivas** (z = 8,3): o efeito é real, mas pequeno. **O tamanho é o mesmo** (mediana de 9 palavras, 46 caracteres).
- **Pergunta-se na 1ª resposta e quase não se pergunta na 2ª.** 55% das primeiras respostas de B têm "?", contra 19% das segundas. Quem conta a história quase nunca devolve a pergunta (3%).
- **O repertório real é curto e formulaico.** Na tristeza, "oh"/"oh no" abre 20–30% das respostas e "I'm sorry / sorry to hear" aparece em 19%. No medo, só 2,7% dizem "sorry": aí se pergunta mais (62%). Na notícia boa, "that's great/awesome" aparece em 12%, e "congrats" só em 4,7%. Frases de assistente como "I understand" (1%) e "don't worry" (1%) são raras.
- **O contágio emocional existe, mas é moderado.** A valência de um turno correlaciona r = 0,44 com a do turno anterior do parceiro. Dentro da mesma conversa (tirando o "clima" geral), fica em r = 0,29. No riso, que é medido em código, a chance de rir é **32% se o parceiro acabou de rir, contra 10%** quando ele não riu. Pareando dentro da conversa, o efeito encolhe (whatsapp: 29% contra 22%). O "contágio" de emoji é quase todo estilo da conversa, não espelhamento turno a turno.
- **Depois de um turno sério, o parceiro brinca menos e usa menos emoji, mas não demora mais para responder.** Dentro da conversa: turnos brincalhões caem 7–10 pp, emoji cai 6,8 pp no whatsapp (IC −11,8 a −1,7), e o texto cresce cerca de 6 caracteres (+15–20%). O riso **não** cai de forma confiável no pareado, e a latência não aumenta (maichat: 6 s mais rápido; IC toca o zero).
- **O humor volta rápido e é puxado por quem ouviu.** Um episódio sério dura em média 2,2 turnos. No turno seguinte ao fim dele, há humor em 59% dos casos, contra 36% na linha de base. Em 84% das voltas, quem traz o humor é o outro (não quem fechou o momento sério).
- **Rir é mais social do que reação a piada.** O Jev classificou 745 risadas: 33% são reação a humor alheio, 24% são o próprio humor ("marcar a piada" ou rir da própria história) e **43% são sociais** (23% suavizam algo, 19% são simpatia ou backchannel). **O "haha" suavizador tem forma própria:** é curto (só 9% têm 6 caracteres ou mais, contra 38% nas reações), fica no fim ou no meio do turno (70%) e nunca vem sozinho. A risada de reação abre o turno (59%) e vem sozinha em 19% dos casos.
- **O `p_joke_welcome` funciona como detector de "hora de brincar".** Ele prevê se a pessoa vai brincar com AUC 0,78. Quando a pessoa brinca, a piada é recebida (o parceiro ri ou continua brincando) em **88% dos casos se p ≥ 0,6, contra 52% se p < 0,3**. Depois de um turno sério, o valor cai de 0,58 para 0,39 (maichat) e de 0,47 para 0,33 (whatsapp).

---

## 2. Achados detalhados

### 2.1 Validação do detector de emoção (EmpatheticDialogues, 32 classes)

**Desenho.** Amostra estratificada de 19 conversas por emoção (608 no total). A mesma chamada leva 10 perguntas isoladas:

- um Choice plano com as 32 classes;
- um Choice de polaridade;
- um Choice de família (7 grupos);
- 7 Choices finos, um por família.

A cascata "dura" usa o fino da família escolhida; a "suave" usa P(família)·P(emoção | família). As condições de contexto são: **A1** (primeira fala de A), **A1B1A2** (as duas primeiras falas de A com a resposta de B no meio) e **FULL** (conversa inteira, subamostra de 256).

| métrica (n) | A1 (608) | A1B1A2 (608) | FULL (256) |
|---|---|---|---|
| Plano top-1 | **44,7%** [40,8–48,7] | 45,1% [41,1–49,0] | 44,9% |
| Plano top-3 | 68,8% | **70,1%** | 68,0% |
| Cascata dura top-1 | 43,3% | 44,4% | 41,8% |
| Cascata suave top-1 / top-3 | 43,4% / 68,1% | 44,6% / 71,9% | 42,6% / 67,6% |
| Média plano+suave top-1 | 45,6% | 44,7% | 44,9% |
| Família direta (7) | 70,2% | **73,8%** | 72,7% |
| Família derivada do plano | 71,5% | 73,0% | 75,4% |
| Oráculo da 2ª etapa (fino com a família-ouro) | 61,3% | 62,5% | 60,2% |
| Polaridade, pergunta direta (sem "surprised") | 75,6% | 84,4% | 82,7% |
| Polaridade derivada do plano | **85,2%** | **89,5%** | 89,9% |

- **Plano contra cascata:** nenhuma diferença significativa. A1: 32 casos só o plano acerta, 23 só a cascata (p = 0,28). A1B1A2: 30 contra 26 (p = 0,69). A cascata perde onde a 1ª etapa erra, e a 2ª etapa tem teto de cerca de 62%.
- **Contexto:** no top-1 fino, A1 contra A1B1A2 dá 56 contra 58 casos discordantes (p = 0,93). Na família, 33 contra 55 (p = 0,025). Na polaridade direta, o ganho de 9 pp vem de o Jev parar de responder "neutral_or_mixed" (93 casos em A1 contra 38 com contexto). Exemplo de fala ambígua sozinha: *"I remember going to see the fireworks with my best friend…"* (ouro: sentimental).
- **Pergunta direta de polaridade é pior do que somar as probabilidades do Choice plano por polaridade** (−5 a −10 pp): a opção "neutro/misto" funciona como ralo.
- **Onde erra:** anticipating 11%, apprehensive 16%, faithful 16%, ashamed, terrified e furious 21% cada. **Onde acerta:** nostalgic 84%; content, proud e disgusted 68%. A família "esperança/confiança" só é reconhecida 45% das vezes: 34 de 114 casos viram "alegria". **Confusões principais:** terrified→afraid (13 de 19), furious→angry (12 de 19), anticipating→excited (9), apprehensive→anxious (8), grateful→joyful (8), sentimental→nostalgic (8). **O Jev prefere o termo básico ao intenso.** Nota prática: um bot que não distingue "terrified" de "afraid" perde pouco; um que confunde "esperança" com "alegria" responde "que ótimo!" a quem ainda está torcendo.
- **Calibração.** Choice plano, confiança 0,2–0,4 → 13–18% de acerto; 0,4–0,6 → 28–34%; 0,6–0,8 → 36–41%; ≥0,8 → 53–55%. É monotônico, mas inflado. Família: ≥0,8 → 80–81%; 0,6–0,8 → 44–57%. Uso seletivo: família com confiança ≥0,9 cobre 67–68% dos casos com 83–84% de acerto; o plano com ≥0,9 cobre 42–47% com 57–58%.
- **Confiança do achado:** alta para os números. O ouro do ED é ruidoso (um rótulo por conversa, escolhido pelo próprio autor entre quase-sinônimos), então 45% é um piso da utilidade real: o top-3 de 70% e a família de 74% mostram melhor o que o bot consegue usar.

### 2.2 Como o ouvinte (B) responde a emoções negativas e positivas

**Jev** (1ª resposta de B, n = 608: 304 negativas, 285 positivas e 19 "surprised"). Estratégia principal e primeira estratégia:

| estratégia | principal (neg) | principal (pos) | 1ª (neg) | 1ª (pos) |
|---|---|---|---|---|
| pergunta exploratória | **41,1%** | **46,0%** | 24,3% | 22,5% |
| validação/empatia | 23,4% | 7,7% | **29,6%** | 10,9% |
| parabéns/compartilhar alegria | 0,7% | 11,6% | 3,0% | **17,5%** |
| autorrevelação ("comigo também") | 11,5% | 17,5% | 6,2% | 11,6% |
| reação genérica ("wow", "oh") | 1,6% | 3,5% | 20,4% | 23,5% |
| opinião/julgamento | 5,6% | 7,0% | 5,9% | 10,2% |
| conselho | 7,2% | 3,5% | 4,9% | 1,4% |
| tranquilizar/minimizar | 6,2% | 1,1% | 3,6% | 1,8% |
| humor | 2,6% | 2,1% | 2,0% | 0,7% |

Por família (estratégia principal):

- tristeza: validação 38%, pergunta 34%;
- medo: pergunta 45%, validação 24%, tranquilizar 9%;
- raiva: pergunta 42%, validação 18%, autorrevelação 14%, opinião 9% ("that's so rude of her");
- vergonha/culpa: pergunta 44%, validação 12%, tranquilizar 11%, humor 5%;
- alegria: pergunta 45%, parabéns 20%;
- nostalgia/ternura: pergunta 37%, **autorrevelação 26%**.

**Regex no corpus inteiro** (n = 24.847):

- a 1ª resposta de B tem "?" em 58,0% nas emoções negativas e em 52,6% nas positivas (z = 8,3). No medo são 62,1%; na nostalgia, 50,7%;
- o tamanho mediano é 46 caracteres nas duas polaridades (9 palavras; A1 tem mediana de 15). A correlação entre o tamanho de A1 e o de B1 é ρ = 0,26;
- "sorry/oh no/that's too bad" aparece em 30,5% das respostas à tristeza, 17,3% à vergonha e 3% às positivas. "congrat/that's awesome" aparece em 14% das respostas à alegria. Conselho ("you should…") aparece em 6% das negativas contra 2% das positivas;
- a **2ª resposta de B** tem "?" em só 19,7% (neg) e 18,4% (pos), e quem conta a história pergunta de volta em só 3%.

**Respondendo às hipóteses:** a pergunta aparece mais em emoções negativas, mas o efeito é pequeno (+5 pp). A estratégia que vem primeiro é uma **reação curta ou validação** ("Oh no, that's terrible!") e, logo depois, a pergunta. O tamanho não muda com a emoção. **Confiança:** alta no corpus inteiro; média no Jev (n ≈ 300 por polaridade). O ED é induzido: o ouvinte foi instruído a ser empático, o que infla a validação em relação ao chat real (ver 2.3 e 2.6).

### 2.3 Repertório real por momento (o QUE as pessoas escrevem)

Fonte: 1ª resposta de B no EmpatheticDialogues (inglês, conversas induzidas, 24,8 mil) e respostas reais no maichat/whatsapp. "% abre" = proporção de respostas que **começam** com a expressão; "% contém" = aparece em qualquer lugar.

**Consolar tristeza** (sad, lonely, devastated, disappointed; n = 3.152):

- tamanho: mediana de 10 palavras (IQR 7–14), 2 frases;
- 56% têm pergunta e 21% são só uma pergunta;
- 21% seguem o molde **"reação curta. + pergunta?"**;
- 45% começam com interjeição.

Aberturas e expressões mais comuns:

- "oh…": 19,5% abre, das quais "oh no" 9,7%;
- "I'm sorry / sorry to hear": 18,8% contém;
- "that is…": 4,2% abre;
- "that sucks / that's terrible": 6,5% contém;
- "I can imagine / must be…": 3% contém.

Exemplos típicos:

- *"oh, dear. That sounds lonely. How are you holding up?"*
- *"Oh my god, Are you doing okay? What happened?"*
- *"Oh man! What did you want to do with it?"*

Raros: "I understand" (1%), "me too" (1%), "you should" (2,3%).

**Consolar medo/ansiedade** (n = 2.960):

- é o momento com **mais perguntas**: 62% têm pergunta e 25% são só uma pergunta;
- "sorry" aparece em só 2,7%;
- aberturas: "oh" 19%, "oh no" 6,7%, "why…" 5,3% contém, "did you…" 2,2%, "that sounds…" 2,3%;
- exemplos: *"Scary!!! What was it?"*, *"That must have frightened you. Was it…?"*, *"Oh i can relate i am also afraid of that"*.

**Raiva/indignação** (n = 4.039):

- "oh no" 7,6%, "that is (terrible)" 4,7%, "did you…" 2,5%, "I hate when…" 1,3%;
- a pessoa **toma partido**: *"What a jerk! Why is she always in your biz?"*, *"That's a rip off at that price."*;
- pergunta prática: *"Did you complain to the landlord?"*.

**Vergonha/culpa** (n = 2.126):

- "oh no" abre **12,8%**, a maior taxa entre os momentos; "why did you…", "what happened";
- é onde mais aparece riso na resposta (4,9% contra ~1% nos outros) e tranquilização (11%): *"It's okay to have cheat days sometimes!"*, *"Ooooof lmao do you work in an office?"*.

**Notícia boa** (alegria, orgulho, gratidão, surpresa; n = 6.210):

- tamanho: mediana de 9 palavras;
- 53% têm pergunta;
- aberturas: "that/that's…" 25% (**"that's great / that's awesome / that is great"** 12,3% contém), "wow" 6,4%, "oh wow" 1,7%;
- "congrat*" em só **4,7%**, "good for you" em 0,7%, "I bet…/you must be…" em ~1%;
- molde típico: *"That's awesome. Are you using any special types of lotion?"* (elogio curto + pergunta concreta sobre o detalhe);
- **o que evitar:** o parabéns sozinho e protocolar é raro.

**Expectativa/esperança** (hopeful, anticipating, confident, prepared…):

- "I hope…/good luck" 5,6%;
- "what kind of…" 1%;
- exemplo: *"Good luck with that. Are you well prepared?"*

**Nostalgia/ternura:**

- "I…" abre 13,2%, porque a pessoa conta a própria memória: *"I love them too. What kind of dog do you have?"*;
- "aww" 1,9%;
- é o momento com **menos pergunta** (51%).

**Engajamento de quem conta:** medido pelo tamanho da fala seguinte de A (A2), as diferenças entre expressões são pequenas, de ±10–20%:

- "what happened?" no medo → A2 18% mais longo;
- "why…?" → A2 11–16% mais curto (no medo, na vergonha e na tristeza);
- "I can imagine / must be" → +9 a 22%;
- conselho → +11 a 16%;
- terminar ou não com pergunta → ≈ 0.

Isso é fraco e o ED tem duração fixa, então não há "frase mágica": o que importa é a estrutura.

**Reagir a uma piada** (chat real; turnos logo depois de uma piada do parceiro):

- **maichat** (n = 1.429): **94% não riem**, mas **85% continuam a brincadeira**. Mediana de 5 palavras. A resposta à piada é **outra piada ou continuar a zoeira**, não "haha". Exemplos:
  - "your journey lasted 4 minutes" → *"ok but mentally i worked 8 hours"*;
  - "ur evil" → *"slightly"*;
  - "productive queen" → *"thank u"*.
- **whatsapp** (n = 897): 29% riem e comentam, 3% só riem, 1,4% só emoji, 67% não riem; 73% continuam brincando. Mediana de 9 palavras. As formas de riso são "haha" (195 casos) ≫ "hahaha" (63) > "hahah", "hahahaha", "hihi". "haha" abre 12,3% das respostas. Exemplo: *"Ahh oke, haha vet leuk"*.

**Momento sério** (turno logo depois de um turno sério do parceiro):

- mediana de 6 palavras no maichat (contra 5 na base) e 11 no whatsapp (contra 9);
- aberturas: "yeah/ja" (7% no maichat, 14,6% no whatsapp), "I…", "oh", "same", "nee";
- intenção (Jev): responder 18–24%, perguntar 16–23%, **consolar explicitamente só 4–7%**, logística 12% (whatsapp).

Mesmo quando o parceiro **pede apoio** (`seeks_support ≥ 0,5`, n = 244), só 8,5% (maichat) e 14% (whatsapp) das respostas são "consolo" explícito. O resto é pergunta (15–17%), ajuda prática (14% no whatsapp) ou zoeira de amigo (31% no maichat):

- "dont mock my journey" → *"your journey lasted 4 minutes"*;
- "same / we're such failures" → *"speak for urself"*.

No whatsapp, o acolhimento real é curto e prático:

- *"Aaahzo / Sterkte ermee dan!"* ("Ah, entendi / Força com isso!");
- *"Goed bezig, nog even volhouden!"* ("Mandou bem, aguenta mais um pouco!");
- *"Succes joh! … Hahaha"* ("Boa sorte, hein!…").

**O que evitar (pelos dados):**

1. Parágrafo de validação: a mediana é de 9–10 palavras.
2. Fazer mais de uma pergunta ou voltar a perguntar na 2ª resposta: 19%.
3. "Sorry" diante de medo ou raiva: 3%.
4. Conselho logo de cara: 5% das aberturas nas negativas.
5. Minimizar ("don't worry"): 1–4%, e só na vergonha passa de 10%.
6. Responder a uma piada só com "haha" quando a conversa é de zoeira (no maichat, 94% não fazem isso).
7. Consolo solene entre amigos íntimos: no chat real, o acolhimento mistura praticidade e humor leve.

### 2.4 Contágio emocional e seriedade (jev_base)

| correlação com… | valência | seriedade | playful | arousal |
|---|---|---|---|---|
| turno anterior do **parceiro** | 0,44 | 0,80 | 0,55 | 0,28 |
| turno anterior **próprio** (t−2) | 0,47 | 0,71 | 0,52 | 0,37 |
| turno aleatório do parceiro na mesma conversa | 0,22 | 0,41 | 0,31 | 0,14 |
| dentro da conversa (sem a média da conversa) | **0,29** | 0,67 | 0,34 | 0,17 |

(n = 5.455 pares; os dois corpora dão números quase iguais.)

- **A valência acompanha a do parceiro além do "clima" da conversa** (0,44 contra 0,22 embaralhado), mas tanto quanto acompanha a própria valência anterior. É o *momento* que é compartilhado. Quando o parceiro está negativo, a valência de quem responde fica em média 1,9–2,0; quando está positivo, 2,65–2,76 (escala 0–4). Há **atração, não espelhamento 1:1**.
- A seriedade tem r = 0,80, mas é quase definicional: a pergunta do passe D é "quão sério é este momento da conversa".
- **Checagem sem Jev:** a chance de rir é 32,4% se o parceiro acabou de rir, contra 10,1% se não riu. Pareando por conversa: whatsapp 28,7% contra 21,6% (só 55% das conversas mostram o efeito); maichat 12,0% contra 6,1%. Com emoji, o dado bruto é 32% contra 7%, mas no pareado dá 21,6% contra 17,0% (whatsapp) e 6,1% contra 6,3% (maichat). **O emoji é estilo da dupla ou da conversa, não espelhamento turno a turno.**
- **Confiança:** média. O passe D vê os 8 turnos anteriores, então o rótulo do turno t pode "herdar" o do parceiro e inflar as correlações; por isso as checagens determinísticas são o número mais confiável.

### 2.5 Quando alguém traz algo sério: o que acontece no turno seguinte do outro

Comparação bruta, turno seguinte ao turno sério contra os outros turnos:

| | maichat (n = 128 contra 2.698) | whatsapp (n = 316 contra 2.313) |
|---|---|---|
| ri | 3,1% contra 4,8% | 13,0% contra 22,8% |
| emoji | 3,9% contra 4,4% | 11,1% contra 15,5% |
| brincalhão | 44% contra 68% | 30% contra 50% |
| caracteres (mediana) | 27,5 contra 24,5 | 51,5 contra 39 |
| nº de bolhas (média) | 1,36 contra 1,46 | 1,89 contra 1,75 |
| tem pergunta | 17% contra 9% | 27% contra 22% |
| latência | mediana 12,0 s contra 13,9 s | <2 min: 60% contra 55% |
| intenção "consolar" | 3,9% contra 0,7% | 7,0% contra 2,4% |

**Pareado dentro da conversa** (diferença média, IC95% por bootstrap):

- brincalhão: −9,9 pp no maichat [−21, 0]; −7,0 pp no whatsapp [−17, +2,5];
- emoji: −6,8 pp no whatsapp [−11,8; −1,7];
- tamanho: +6,4 caracteres no maichat [0,5; 14,1]; +6,0 no whatsapp [−5,5; 18];
- riso: +4,2 pp no maichat [−3,5; 14]; −0,4 pp no whatsapp [−8,7; 7,6], ou seja, **nenhum efeito**;
- bolhas: 0;
- latência: −6,3 s no maichat [−15,9; 0,5].

Hipóteses do usuário:

- "sério → escreve mais": confirmado de leve, cerca de +15–20% de caracteres, **sem mais bolhas**;
- "brinca menos": confirmado no playful e no emoji;
- "ri menos": só no dado bruto, porque conversas sérias são menos risonhas no todo;
- "demora mais": **não se confirma**. Se muda algo, a resposta vem mais rápido.

Turno **vulnerável** (autorrevelação ou pedido de apoio) é diferente de turno sério. No whatsapp, o parceiro **ri mais** depois dele (+9,1 pp [2,9; 16]) e escreve mais (+13,7 caracteres [2,6; 25]). Muitas confissões chegam em tom leve (*"ik ben gesloopt van die verhuizing haha"*, "estou moído da mudança haha"), e o outro acompanha o tom. **Confiança:** média-baixa. São poucos turnos sérios (2,4% do maichat e 7,4% do whatsapp com ≥1,5), e a "seriedade" do maichat é branda (amigos fazendo uma tarefa).

### 2.6 A volta do humor depois de um momento sério

Foram 224 episódios sérios (66 no maichat, 158 no whatsapp), com duração média de 2,2 turnos.

- Contando do início do episódio, o humor (riso ou playful) volta na mesma janela em 70% dos casos, com **mediana de 1 turno** (IQR 1–3). Em 69% das voltas, isso acontece em até 2 turnos.
- Contando **do fim** do episódio, o humor aparece no turno seguinte em **59%** dos casos, contra **36%** a partir de um turno qualquer não sério e sem humor. Em até 10 turnos: 82% contra 74%.
- **Quem puxa a volta:** o outro, e não quem falou o último turno sério, em **84%** das voltas (linha de base: 68%). 75% das voltas são puxadas por quem *ouviu* o momento sério.
- **Como volta:** em só 25% das vezes com "haha"; no resto, com uma frase brincalhona. Exemplos (maichat):
  - "im concerned" → *"its fineee"*;
  - "if i survive this week" → *"same tbh"*;
  - "I miss you 🥺" → *"I will make sure you enjoy dancing by the end of this conversation😂"*.
- **Confiança:** média. O fim do episódio é definido pela queda da seriedade, o que puxa o turno seguinte para "leve" (há viés de seleção). O que é robusto: episódios sérios são curtos (cerca de 2 turnos) e quem volta ao leve é quem ouviu.

### 2.7 Humor: o que precede o riso, quem ri, riso de reação e riso social

**Antecedentes:**

- no whatsapp, a chance de rir é 36% depois de uma piada do parceiro, contra 14% depois de outra coisa. No maichat, 5,5% contra 3,9%: lá, a piada é respondida com piada (ver 2.3);
- depois de o parceiro rir, a chance de rir é 32%, contra 13% na base: **o riso chama riso mais do que a piada**;
- 58% das risadas vêm depois de uma piada do parceiro, sendo que 43% de todos os turnos vêm depois de uma piada;
- a intenção do próprio turno com riso é mais "reagir" (26%, contra 13% nos turnos sem riso) e "piada" (15% contra 10%).

**Função das risadas** (Jev Choice, 745 turnos com riso: 612 do whatsapp e 133 do maichat):

| função | todos | maichat | whatsapp |
|---|---|---|---|
| reação a humor do outro | **33,2%** | 43,6% | 30,9% |
| marcar a própria piada | 16,1% | 19,5% | 15,4% |
| rir da própria história | 7,4% | 10,5% | 6,7% |
| **suavizador** (reclamação, recusa, correção, confissão) | **23,4%** | 14,3% | 25,3% |
| simpatia ou backchannel (nada engraçado) | 19,1% | 10,5% | 20,9% |
| nervoso ou constrangido | 0,9% | 1,5% | 0,8% |

- **Riso "social" (suavizar, simpatia, nervoso) = 43%.** Riso de reação = 33%. Próprio humor = 24%. Só com as risadas de confiança ≥0,6: reação 41%, suavizador 22%, backchannel 17%. Em 37% das risadas, o Jev não vê nada engraçado nem antes nem no próprio turno.
- **Quem ri mais, quem fez a piada ou quem ouviu?** Quem ouviu, mas não por muito. Das risadas, 33% são reação e 24% são o próprio humor: **uma em cada quatro risadas é da pessoa rindo da própria piada ou história.** No rótulo do passe D, quem fez a piada põe riso no próprio turno em 25% das piadas, e o ouvinte ri no turno seguinte em 17%. Esse número é circular, porque o "haha" ajuda o Jev a rotular "piada".
- **O "haha" suavizador existe, mas não é a maioria dos risos em reclamações.** Turnos com reclamação ou crítica têm *menos* riso do que a média (maichat 1,2% contra 4,6%; whatsapp 14,7% contra 19,8%), e "face threat" aparece em 42% dos turnos com riso contra 38% no controle (n = 150). O suavizador é um uso frequente do riso (23% das risadas), não uma regra de toda reclamação. Exemplos:
  - *"i like the one with the cat lol / hated the others"*;
  - *"haha i didn't think it sounded very me"*;
  - *"Al ben ik er nu wel klaar mee haha"* ("Mas agora já cansei disso haha");
  - *"Hahaha you are taking so long to type"*: o exemplo do próprio usuário, que o Jev leu como riso sobre a própria história. Confusões assim são esperadas.
- **A forma denuncia a função:**

| função | riso longo (≥6 caracteres) | riso sozinho | posição: início | posição: fim ou meio |
|---|---|---|---|---|
| reação | **38%** | **19%** | **59%** | 22% |
| suavizador | 9% | 0% | 31% | **70%** |
| backchannel | 11% | 8% | 44% | 48% (28% são emoji 😁😄) |
| própria piada | 23% | 1% | 29% | 71% |

  CAPS ("HAHAHA", "LMAOOO") só aparece em reações.
- **Depois de cada tipo de riso, o parceiro ri de volta:** 42% depois de "marcar a própria piada", 37% depois de suavizador, 28% depois de reação, 23% depois de backchannel.
- **Confiança:** média. O Jev não foi validado contra rótulos humanos de função do riso, e 82% das risadas são do whatsapp em holandês. Os padrões de forma são medidos em código e consistentes nos dois corpora (reação longa: maichat 43%, whatsapp 37%; suavizador longo: 21% e 7%).

### 2.8 Piada bem-vinda ou mal-vinda (`P.p_joke_welcome`)

- **Prevê se a pessoa vai brincar** no próximo turno: AUC 0,78 (maichat 0,77; whatsapp 0,76). Prevê playful com AUC 0,80. Prevê riso com AUC só 0,56.
- **Recepção**, quando a pessoa de fato brincou (n = 2.238); "recebida" = o parceiro ri ou continua brincando:

| p_joke_welcome | recebida | n |
|---|---|---|
| < 0,3 | **52%** | 60 |
| 0,3–0,6 | 75% | 940 |
| ≥ 0,6 | **88%** | 1.238 |

  AUC 0,66 (maichat 0,61; whatsapp 0,68). No whatsapp, com p < 0,3, só 45% das piadas foram recebidas.
- **O contexto sério derruba a estimativa:** 0,58→0,39 (maichat) e 0,47→0,33 (whatsapp).
- **Detectores de "hora de…" que já existem no passe P** (AUC contra o que a pessoa fez):
  - tom "supportive_serious" → o turno foi sério: 0,86 (acolher com intenção de consolo: 0,77);
  - tom "playful" → o turno foi brincalhão: 0,78;
  - `p_laugh` → riu: 0,64;
  - `p_question` → perguntou: **0,62, fraco**. Para "hora de perguntar", regras de código funcionam melhor (ver 4).
- **Confiança:** alta para "hora de brincar" e "hora de ser sério"; média para a recepção, porque a faixa baixa tem n = 60.

---

## 3. Padrões "invisíveis" (o que as pessoas fazem sem perceber)

1. **O tamanho do "haha" é um sinal.** A risada de verdade é longa, vem no início do turno e às vezes sozinha ou em CAPS ("HAHAHAHA", "LMAOOO"). A risada social é curta ("haha", "lol"), fica no fim e nunca vem sozinha. Um bot que usa sempre o mesmo "haha" soa falso nos dois casos.
2. **Uma em cada quatro risadas é da pessoa rindo da própria piada ou história.** O riso marca "isto é brincadeira" e convida o outro: depois dele, o parceiro ri de volta 42% das vezes, a maior taxa.
3. **Em conversa de zoeira, não se responde piada com "haha": responde-se com outra piada.** No maichat, 94% das respostas a piada não têm riso e 85% continuam a brincadeira. O "kkk" é mais comum em grupos e apps como o whatsapp (29% "riso + comentário"), e quase nunca vem sozinho (3%).
4. **O momento sério é curto: cerca de 2 turnos.** Quem sai dele é quem ouviu, no turno seguinte, com uma frase leve ("its fineee", "same tbh"), não com "haha".
5. **Depois de algo sério, a resposta fica um pouco maior (+15–20%), perde o emoji e a zoeira, mas não fica mais lenta nem mais picotada.** A demora dramática não é humana.
6. **Confissões vulneráveis costumam vir com riso** ("estou moído haha"), e o parceiro ri junto (+9 pp). Tratar toda autorrevelação como tragédia é erro de tom.
7. **Pergunta-se uma vez só.** A 1ª resposta a uma história tem pergunta em 55% dos casos; a 2ª, em 19%. Quem conta quase nunca devolve a pergunta (3%).
8. **A abertura é quase fixa e curta:** "oh no", "oh", "that's…", "wow", "I'm sorry". Frases de terapeuta ("I understand", "that must be…", "don't worry") somam menos de 5%. Para o medo, pergunta-se o que aconteceu em vez de dizer "sorry".
9. **Entre amigos, "acolher" raramente é consolo explícito.** Mesmo quando alguém pede apoio, só 8–14% das respostas são consolo; o resto é pergunta, ajuda prática ("breng ze maar mee", "traz aqui que eu vejo") ou provocação carinhosa.
10. **O estilo de emoji é da dupla, não do turno.** Espelhar o emoji do usuário turno a turno não é o que humanos fazem. O que convergem é o nível médio da conversa.

---

## 4. Tradução para o sistema

Princípio geral: o state é **curto** (os últimos 1–3 turnos do usuário, mais o último do bot), porque contexto a mais não melhorou a emoção fina. Perguntas vão **em paralelo numa única chamada** por turno do usuário. Contas, contagens e formas de riso ficam no código.

### 4.1 Detector de emoção do usuário (família → comportamento; o fino só dá "sabor")

- **Perguntas (1 chamada):**
  - `fam = Choice(7 famílias com descrição)`;
  - `emo = Choice(32 emoções com descrição curta)` (ou o conjunto de emoções do produto);
  - (opcional) `valence = Score 0–4`.
- **State:** `{"user_recent": [os últimos 1–3 turnos do usuário], "bot_last": "…"}`.
- **Quando:** a cada turno do usuário, antes do roteamento.
- **Regras de código:**
  - a polaridade vem da soma das probabilidades de `emo` por polaridade (85–90%), **não** de um Choice de polaridade (75–84%);
  - só muda o comportamento com `fam.confidence ≥ 0,8` (≈80% de acerto) e agir forte com ≥0,9 (≈84%, 67% dos casos). Abaixo disso, usa o modo neutro e curioso (pergunta exploratória);
  - `emo` fino entra no briefing como "sabor" (por exemplo, "terrified" deixa o texto mais enfático), nunca como gatilho. Se `emo.confidence < 0,6`, passa as top-3 para a LLM ("parece medo ou ansiedade");
  - "esperança/expectativa" é confundida com "alegria" (45%). Se `fam ∈ {joy, hope}` e a mensagem fala de algo **futuro** (Noul "Is the user talking about something that has not happened yet?"), use "torcer" ("good luck / I hope") em vez de "comemorar".

### 4.2 "Hora de acolher" (e como)

- **Cascata:**
  1. `Noul`: "Is the user telling about something bad/difficult that happened to them or that they feel?" (state: os últimos 2 turnos do usuário).
  2. Se p ≥ 0,6: `Choice` da família (4.1) e `Noul` "Is the user asking for help or advice explicitly?".
  3. `Noul` "Is the user joking or laughing about it?", ou código: riso no fim do turno.
- **Regras de código para montar o briefing:**
  - **estrutura:** reação curta e depois uma pergunta exploratória (molde de 21%). Mediana de 9–10 palavras, no máximo 2 frases, 1 bolha;
  - **abertura por família** (sorteio com os pesos observados):
    - tristeza: "oh no" ou "I'm so sorry" (≈30%);
    - medo: "oh no/scary" e a pergunta "what happened?" (sem "sorry");
    - raiva: tomar partido ("what a jerk") e uma pergunta prática;
    - vergonha: "oh no" e tranquilizar ou humor leve se o usuário riu;
  - conselho só se o Noul "pede ajuda" der ≥0,6;
  - minimizar só na vergonha e com o usuário rindo;
  - se o usuário **riu no fim** (suavizador) ou o tom é leve, acolher com leveza: resposta prática, "força!", e humor permitido (a vulnerabilidade com riso é correspondida com riso em +9 pp).

### 4.3 "Hora de perguntar mais"

- **Regra de código primeiro, porque `p_question` é fraco (AUC 0,62).** Contador de perguntas do bot sobre o tópico atual: pergunta na 1ª resposta a uma história do usuário (alvo: 55–60%), e na 2ª a taxa cai para no máximo 20%.
- **Jev:** `Noul` "Did the user share a new experience, news or feeling that has not been asked about yet?" (state: os últimos 3 turnos, com o bot incluído). Se p ≥ 0,6 e o contador for 0 → pergunta exploratória ("what happened?/how did it go?"; evitar "why…?", que teve A2 11–16% mais curto). Se o contador for ≥1 → reação, opinião ou autorrevelação, sem pergunta.
- **No medo:** aumentar a chance de pergunta (62%). **Na nostalgia:** preferir autorrevelação ("I…", 26%).

### 4.4 "Hora de ser sério"

- **Pergunta:** `Score` seriousness 0–3 do **momento** (a mesma do passe D), mais `Choice` do tom esperado (p_tone com "supportive_serious"; AUC 0,86 para prever um turno sério).
- **State:** os últimos 4–6 turnos.
- **Regras (quando seriousness ≥ 1,25 ou P(supportive_serious) ≥ 0,4):**
  - sem emoji (−7 pp) e sem riso próprio;
  - texto 15–20% mais longo do que o normal da persona, **mesmo número de bolhas**;
  - **sem aumentar o atraso** (os humanos não demoram mais; até respondem um pouco antes);
  - no máximo uma pergunta;
  - p_joke_welcome é forçado para 0 nesses turnos.

### 4.5 "Hora de brincar" e a volta do humor

- **Pergunta:** `Noul` "Would a joke be a good idea for the bot's next turn?" (igual ao `p_joke_welcome`). **State:** a conversa até o último turno do usuário, sem a resposta do bot.
- **Regras:**
  - p ≥ 0,6 → pode brincar (88% de recepção);
  - 0,3–0,6 → só com humor leve ou continuando uma brincadeira que o **usuário** começou;
  - p < 0,3 → nada de piada (52% de recepção, 45% no whatsapp).
- **Continuar a zoeira:** se `playful` do último turno do usuário for ≥0,5, a resposta padrão é **continuar a brincadeira** (85% no maichat), não rir. O riso entra conforme 4.6.
- **Volta do humor:** o código mantém o estado `serious_episode` (turnos seguidos com seriousness ≥1,25). Quando a seriedade do último turno do usuário cai abaixo de 1 e o episódio durou pelo menos 1–2 turnos, o bot **pode** puxar a volta com uma frase leve (humanos: 59% no turno seguinte; 84% das voltas são de quem ouviu). A condição: p_joke_welcome ≥ 0,5 e nenhum Noul de "ainda chateado" (`seeks_support`) ≥ 0,5.

### 4.6 Função do riso do usuário e forma do riso do bot

- **Leitura do riso do usuário** (a cada turno com riso, detectado por regex: "haha", "kkk", "rs", 😂…):
  - `Choice` função {reação, marcar própria piada, rir da própria história, suavizador, simpatia};
  - features de código: tamanho do riso, posição e se vem sozinho;
  - state: o turno anterior do bot e o turno atual.
- **Regra de pré-classificação barata:** riso longo (≥6 caracteres), no início ou sozinho → reação (é o padrão de 38–59% das reações). Riso curto no fim, junto com conteúdo → suavizador ou simpatia. Só chama o Jev nos ambíguos.
- **Consumo:**
  - **suavizador** → responder ao **conteúdo** (a reclamação ou recusa é real), sem rir de volta, com leveza. Exemplo: "hahaha you are taking so long to type" pede uma desculpa rápida, não um "haha";
  - **marcar própria piada / rir da própria história** → rir de volta e continuar (o parceiro ri em 42% dos casos);
  - **reação ao bot** → o bit funcionou; continuar ou escalar a brincadeira uma vez;
  - **simpatia** → nada especial.
- **Forma do riso do bot** (código, não LLM):
  - reação genuína: riso longo (no PT-BR, "kkkkkk"/"KKKKK"/"hahahaha"), no **início** e às vezes sozinho em bolha própria;
  - suavizar: curto ("kkk", "haha"), no **fim** da frase;
  - nunca riso sozinho para suavizar.
  - A taxa de riso-alvo depende do canal: ≈5% dos turnos num estilo maichat, ≈20% num estilo whatsapp. Espelhar o **nível** do usuário na conversa, não turno a turno. Depois de o usuário rir, o bot pode rir com mais probabilidade (32% contra 10%).
  - Limitação: o mapeamento "haha" → "kkk" é suposição, porque os corpora não têm PT-BR.

### 4.7 Contágio de valência (tom do bot)

- Nenhuma pergunta nova: usa `valence` (4.1) do último turno do usuário.
- **Regra:** a valência-alvo do bot = persona + 0,3 × (valência do usuário − 2), com atração parcial (r dentro da conversa ≈ 0,3; os humanos ficam em ~1,9 quando o parceiro está negativo, não em 0).
- **Emoji:** segue a taxa média da conversa (janela de 20 turnos), não o último turno.

---

## 5. Limitações

- **EmpatheticDialogues** é induzido (crowdworkers instruídos a ser empáticos), em inglês e de conversa curta e fixa. Superestima validação e pergunta em relação ao chat real, e o tamanho de A2 é uma medida fraca de engajamento. O ouro é um rótulo por conversa, escolhido pelo narrador entre quase-sinônimos; o teto humano de concordância no top-1 é baixo, então o 45% do Jev deve ser lido junto com o top-3 (70%) e a família (74%).
- **Circularidade:** o passe D vê os 8 turnos anteriores, o que infla as correlações de contágio (os números de riso, feitos em código, são a referência). O rótulo "piada" do passe D usa o texto com o "haha", então "quem fez a piada ri mais" pelo passe D está contaminado; a classificação de função do riso é a medida mais limpa.
- **Função do riso:** foi classificada pelo Jev, sem validação humana. 82% das risadas são do whatsapp **em holandês** (o Jev é melhor em inglês), e o próprio exemplo do usuário foi classificado de forma discutível.
- **Seriedade** é rara (maichat: 4,5% com ≥1,25; whatsapp: 11,7%). O maichat é de estudantes amigos fazendo uma tarefa, e a "seriedade" deles é branda ("remember this chat is for that assignment"). As estimativas pareadas têm IC largo (26–58 conversas).
- **Latência:** o whatsapp tem resolução de minuto (a mediana fica em 60 s nas duas condições). O maichat tem só 42 conversas.
- **Volta do humor:** o fim do episódio é definido pela queda da seriedade (viés de seleção a favor do "leve" no turno seguinte).
- **Riso:** o regex não pega todas as variantes (por exemplo "xD", ":D", "ahaha" colado a letras). Os emojis SoftBank foram incluídos à mão. O repertório de PT-BR ("kkk", "rs") não foi medido.
- **Orçamento:** foram usadas 2.977 chamadas novas do Jev (US$0,18). A validação de emoção usou uma única formulação de pergunta e descrições curtas; outras formulações podem render mais.
