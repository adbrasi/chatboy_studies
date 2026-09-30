# 02: Aberturas ("oi"), reaberturas e fechamentos

**Dimensão:** como as conversas começam, recomeçam depois de um silêncio e terminam. **Prefixo:** `a2`.
**Dados:** `whatsapp_nl` (57 chats diádicos, 3.711 sessões com nova sessão após mais de 3 h de silêncio), `maichat` (42 conversas
síncronas em laboratório), `nps_chatroom` (15 salas públicas, 7.935 posts, 698 "primeiros posts" de usuários, 1.021 entradas
(JOIN)), `jev_base` (5.968 turnos com os passes D e P).
**Experimentos novos de Jev:** cerca de 2.620 chamadas, por uns US$ 0,10 no total: (1) tipologia de 1.109 aberturas de sessão do
WhatsApp, com até 30 por chat e o fim da sessão anterior no state; (2) relação por chat (57 chamadas); (3) tipologia e flerte dos
698 primeiros posts do NPS; (4) quatro detectores novos de fim em 757 turnos do WhatsApp.
**Scripts:** `scripts/analysis/a2_*.py`. **Saídas:** `analysis/data/a2_*`.
Para reproduzir (cache em pickle em `$A2_SCRATCH`), rode nesta ordem: `a2_sessions` → `a2_jev_openings` → `a2_nps` →
`a2_jev_nps` → `a2_closings` → `a2_pend` → `a2_jev_end` → `a2_close_rules` → `a2_end_signals` → `a2_oi` →
`a2_nps_model` → `a2_nps_join` → `a2_reply_success` → `a2_summary`. Os rótulos do Jev linha a linha estão em
`analysis/data/a2_jev_openings.csv.gz`, `a2_jev_nps_firstposts.csv.gz` e `a2_jev_end.csv.gz`.

---

## 1. Resumo dos achados

- **"Oi" puro é raro onde a relação é íntima e contínua.** No WhatsApp, só **21%** das aberturas de sessão têm saudação
  (classificação do Jev; pelas regras, 14%), e **"oi" sozinho é só 2%**. O mais comum é entrar direto: pergunta ou pedido novo (20%),
  saudação + assunto (19%), **resposta atrasada à última mensagem** (15%), logística (13%), notícia (11%) e mídia (8%). No MaiChat
  (síncrono, "vamos conversar agora"), **84%** abrem com saudação e **31%** abrem checando presença ("u free?", "yo u alive").
  Entre desconhecidos (NPS), **41%** dos primeiros posts são saudação.
- **A saudação é um "pedágio" que cresce com o tempo de silêncio.** A taxa de saudação sobe de **4%** (silêncio de 3 a 6 h) para
  **13%** (6 a 12 h), 18% (12 a 24 h), 19% (1 a 3 dias) e **34% e 32%** (3 a 7 dias e mais de 7 dias). As retomadas que se referem à
  conversa anterior caem de **65% para 19%**, e as respostas atrasadas caem de **42% para 1%**.
- **Quase ninguém comenta o tempo que passou.** Só **3,7%** das reaberturas reconhecem o silêncio (41/1.109), e depois de mais de
  7 dias a taxa é de **0,6%**. Quando aparece, é quase sempre **quem devia uma resposta pedindo desculpa** (em 68% dos casos quem
  abre não é quem mandou a última mensagem; mediana de 23 h): "Sorry voor mijn late reactie", "Telefoon was leeg sorry".
  "Quanto tempo!" praticamente não existe nesses dados.
- **Quem retoma não é necessariamente "a vez do outro".** Em **45%** das reaberturas quem escreve é a mesma pessoa que mandou a
  última mensagem antes do silêncio. Isso sobe para **60%** depois de 3 a 7 dias. Mas, se a sessão acabou com uma **pergunta
  pendente**, o outro reabre em **66%** dos casos, e **50%** dessas reaberturas são a resposta atrasada.
- **Resposta ao "oi": devolve a saudação, com outra palavra e com a mesma energia.** Quando a abertura tem saudação, a resposta
  cumprimenta de volta em **25%** dos casos, contra **3%** quando não tem (WhatsApp, n=876, ~8×). No MaiChat essa taxa é de 71%, e
  no NPS **64%** das respostas a uma saudação nominal são saudações. Para o "oi" puro (n=23), **83%** devolvem a saudação, **43%**
  já põem um assunto no mesmo turno (muitas vezes numa 2ª bolha) e 26% perguntam "e aí?". A palavra é **a mesma só em 26%**
  dos casos no 1:1 ("Hi!" → "Helluuuu"; "Hoi" → "Hallo / Hoe staat het leven?"). A energia é espelhada: abertura alongada
  ("Heeeey") → resposta alongada em **32%** dos casos, contra 8% (n=2.936).
- **O que acompanha as sessões longas:** uma abertura com gancho (pergunta, pedido ou convite) é respondida em **85%** dos casos,
  contra **70%** sem gancho (+15 pp, IC ±5 pp; dentro do mesmo chat, +13 pp), e rende sessões ~1,4× maiores. Na **primeira
  resposta**, uma pergunta vem junto de mais turnos depois (mediana de 8 contra 5; +0,35 em log dentro do chat). Também contam
  2+ bolhas (+0,30) e responder em até 1 min (+0,26). Mídia sem texto fica **sem resposta na sessão em 51%** dos casos (n=111).
- **Entre desconhecidos, citar o nome dobra a resposta, e a "chamada genérica" de flerte é ignorada.** No NPS, uma saudação
  endereçada ("hi User5") recebe resposta nominal em **63%** dos casos, contra 31% do "hi" solto e 38% do "hi all". Num logit
  com efeito fixo de sala, a saudação endereçada tem OR ≈ 2,0. Os "any girls wanna chat" / "18/m pm me" têm **12%** de resposta
  pública (OR 0,42, p<0,001), e repetir não ajuda (8% a 17%). Quem **é cumprimentado pelo nome ao entrar** posta em **66%** dos
  casos, contra 26% de quem não é.
- **Conversas íntimas assíncronas não "terminam": elas pausam.** Só **7,5%** das sessões do WhatsApp acabam com despedida;
  **61%** acabam numa frase comum, 11% numa risada, 7% em "ok/top", 7% em emoji ou mídia e 7% numa pergunta sem resposta. No MaiChat
  (síncrono), **55%** terminam com "bye"; em **88%** dos casos o "bye" é devolvido, e em **21%** o ritual se estende por 3 ou mais
  turnos ("bye / bye / bye / byyyyyeeeeeee…").
- **Sinais de fim são visíveis 1 ou 2 turnos antes.** No WhatsApp (1.245 sessões com ≥8 turnos), a última mensagem tem **18
  caracteres (mediana), contra 37 a 40**. As perguntas caem de 25–31% para 19% → 14% → **7%**, as respostas lentas (≥5 min) sobem
  de ~19–23% para 27% → **31%**, e o emoji sobe para 23%. No Jev D, o gancho cai de 0,60 para 0,34 e `phase=winding_down` sobe de
  4–6% para 34%.
- **O `P.p_end` prevê bem o fim na conversa síncrona e mal no WhatsApp.** No MaiChat a AUC é **0,91** (0,86–0,96) para "este é o
  último turno", **0,87** dois turnos antes e **0,75** quatro turnos antes. No WhatsApp a AUC é **0,66** (0,62–0,69) e já cai para
  0,60 um turno antes. O `p_end` é mal calibrado: quase nunca passa de 0,5, e o limiar útil fica perto de **0,3** (precisão de
  ~45% a 55%). Os quatro detectores novos testados ficaram com AUC entre 0,59 e 0,69; o melhor (`closure`, score 0–3) chega a
  **0,69**, que não é significativamente melhor.

---

## 2. Achados detalhados

### 2.1 O que as pessoas mandam como primeira mensagem

**Tipologia do Jev nas aberturas do WhatsApp** (n=1.109; até 30 sessões por chat, 57 chats; ponderada por chat entre parênteses):

| tipo (Jev `open_type`) | % | exemplo real (holandês) |
|---|---|---|
| pergunta ou pedido novo, sem saudação | 19,7 (18,6) | "Ga je vanavond trainen?" |
| saudação + assunto | 18,9 (22,5) | "Hee [X]! Veilig thuis gekomen? En de koffers al uitgepakt?" |
| **resposta atrasada** ao que o outro disse antes do silêncio | 15,2 (15,1) | prev: "Hoe gaat het eigenlijk met bas?" → (7 h) "Raar!" |
| logística | 12,9 (11,2) | "We zitten nu in de bus dus we zijn er over een half uur" |
| notícia, história ou comentário | 10,6 (10,6) | "Ik heb dikke griep:((" |
| mídia ou link | 7,8 (8,5) | "<afbeelding weggelaten> / Derde dag kapsalon" |
| retomada de assunto anterior | 5,8 (4,8) | "Heb je m ingeleverd?😍😍" |
| felicitações e votos | 3,5 (3,9) | "Gefeliciteerd! / [X] kampioen" |
| checagem de presença | 3,0 (2,2) | "Ben je nog wakker?? Kan ik je bellen?" |
| **só saudação** | 2,0 (2,0) | "Hoi", "Heeeeeeeeeeeeey", "Goedemorgen!" |
| saudação + "tudo bem?" | 0,6 (0,6) | "Hoihoi hoe gaat ie mt jullie?" |

- A concordância com as regras de código (`a2_sessions.py`) é boa nas classes óbvias: 144/175 das "greet+content" pelas regras
  viram `greeting_plus_topic`, e 36/36 das "media" viram `media_or_link`. O Jev achou saudações que o regex não pegou ("Heuj",
  "Joo") e separou as "respostas atrasadas", que o regex não conseguia ver.
- **MaiChat** (45 sessões): só saudação 38%, saudação + checagem de presença 27% ("hi u online", "yo bro u there"), saudação +
  assunto 18%, sem saudação 16% ("DID U SEE THAT ORANGE CAT VIDEO 😭😭", "knock knock"). O contexto é de laboratório: os dois
  foram chamados para conversar naquele momento, e isso infla a saudação.
- **NPS** (698 primeiros posts, Jev): saudação a uma pessoa 18%, saudação à sala 18%, saudação solta 10%, entra na conversa em
  andamento 18%, convite aberto ou "pm me" 11%, asl ou demografia 6%, pergunta à sala 4%, flerte ou elogio 1,4%, proposta sexual
  1,7%.
- **Por relação** (relação por chat classificada pelo Jev: 25 amigos próximos, 23 colegas/estudo, 4 irmãos ou família, 2 pais e
  filhos, 2 casais, 1 conhecido): a taxa de saudação é de 23% entre colegas, 21% entre amigos e irmãos, 13% entre pais e filhos e
  **8% entre casais** (40 aberturas, 2 chats). Os casais abrem mais com notícia ou história (30%). **Confiança baixa:** há poucos
  chats nas relações íntimas. O `D.relationship` do `jev_base` dá uma distribuição parecida (33 amigos, 20 colegas).
- **Conclusão (confiança alta):** a saudação depende de **sincronia × distância × tempo de silêncio**. Não é um ritual universal
  de abertura.

### 2.2 Reaberturas depois de silêncio longo (WhatsApp)

| silêncio | n | saudação | reconhece o silêncio | refere-se ao anterior | resposta atrasada | quem abre = quem falou por último |
|---|---|---|---|---|---|---|
| 3–6 h | 135 | 4% | 4% | 65% | 42% | 33% |
| 6–12 h | 158 | 13% | 4% | 60% | 37% | 29% |
| 12–24 h | 201 | 18% | 4% | 51% | 14% | 46% |
| 1–3 d | 260 | 19% | 5% | 34% | 8% | 50% |
| 3–7 d | 144 | 34% | 5% | 26% | 2% | 60% |
| > 7 d | 168 | 32% | 0,6% | 19% | 1% | 48% |

- **A saudação cresce com o silêncio, mas nunca vem sozinha:** depois de mais de 3 dias, 85% das saudações trazem gancho, e só
  4% são "oi" puro. Um exemplo depois de 10 dias: "Heeeee willy! / Hoe gaat het met jou? / Nog leuke vierdaagsedagen gehad? :)" →
  "Heeeee [X]! / Goooed! En met jou? / Jaaaaaa zeker!". A saudação vem com **callback específico** e é espelhada na mesma forma
  alongada.
- **O reconhecimento do silêncio é um pedido de desculpas de quem estava devendo**, não um comentário sobre o tempo. Dos 41
  casos, 34% são respostas atrasadas, e em 68% dos casos quem abre não foi o último a falar. Esses casos são respondidos em 73%
  das vezes, contra 79% nos demais (diferença não significativa). Exemplos: "Spijt me, lees je bericht nu pas / Ben net
  wakker"; "Aaaaaaa ik hd helemal niet meer geantwoord zie ik nu! Wat errug!".
- **Pergunta pendente vira resposta atrasada:** se o último turno antes do silêncio tinha "?", 50% das reaberturas são a resposta
  a ele (contra 11% sem "?"), e o outro reabre em 66% dos casos.
- **Iniciativa assimétrica:** no chat mediano, 60% das sessões são abertas pela mesma pessoa, e em 32% dos chats uma pessoa abre
  mais de 65%.
- **Hora do dia:** a saudação aparece em 20% das aberturas de manhã, 21% à tarde e 25% à noite. "Goedemorgen" é raro (23
  aberturas, mediana às 9 h). Efeito fraco.
- **Confiança:** média-alta para as tendências com o silêncio (monotônicas, n≥135 por faixa). Os rótulos vêm do Jev em holandês,
  mas as tendências concordam com as regras (o "sorry" pelo regex dá 2%).

### 2.3 Quando alguém manda só "oi": o que o outro responde

| fonte | n | devolve saudação | adiciona assunto | pergunta ("e aí?") | alonga | latência mediana |
|---|---|---|---|---|---|---|
| MaiChat, abertura "oi" puro | 15 | 87% | 40% | 20% | 20% | 19 s |
| WhatsApp, abertura "oi" puro | 8 | 75% | 50% | 38% | 12% | 60 s |
| NPS, saudação nominal respondida | 765 | 64% (ato Greet) | — | 10% | 2% | 4 posts |

- **Não é cópia literal:** entre os pares saudação → saudação, a mesma palavra aparece em só **26%** no 1:1 (n=19) e em 53% no
  NPS (n=393; "hi" → "hi" é o par mais comum entre desconhecidos). Exemplos: "Hi!" → "Helluuuu"; "hiiiiiiii" → "Hello!";
  "hey" → "Hii"; "yo" → "hey"; "Heeeeeeeeeeeeey" → "Hoooooi"; "Hoii" → "Heuj / Was ist loss?".
- **Espelhamento generalizado** (todas as aberturas do WhatsApp, n=2.936): se a abertura tem saudação, a resposta cumprimenta em
  23% dos casos (regras) ou 25% (Jev), contra 3–4% sem saudação. Se a abertura tem letras alongadas, a resposta alonga em 32%,
  contra 8%.
- **Saudação + 2ª bolha com conteúdo** é o formato típico de quem responde: "Hi / Did you get my message", "hello / that baby
  with ice cream / teeth must have hurt", "Hallo / Hoe staat het leven?".
- **O "oi" puro é lido como preâmbulo:** a resposta muitas vezes cobra o motivo ("Was ist loss?", "whats up", "yeah whats up").
  Nos dados, um "oi" sem nada já é um pedido de atenção.
- **Resposta que leva a conversa longa** (primeira resposta de todas as sessões do WhatsApp, n=2.937; log(turnos depois da
  resposta) centrado no chat):

  | característica da 1ª resposta | com | sem | Δ log |
  |---|---|---|---|
  | contém pergunta | +0,26 (mediana de 8 turnos) | −0,10 (5) | **+0,35** |
  | 2+ bolhas | +0,15 | −0,15 | +0,30 |
  | chegou em ≤1 min | +0,18 | −0,08 | +0,26 |
  | ≥40 caracteres | +0,10 | −0,10 | +0,20 |
  | devolve a saudação | +0,17 | −0,01 | +0,19 (IC ±0,15) |

  Nas aberturas com saudação (n=432) o padrão se repete: resposta com pergunta +0,23; com 2+ bolhas +0,20; saudação de volta
  +0,14. **Confiança:** média. É correlacional, e em parte mecânico: uma pergunta pede resposta. Para o "oi" puro isoladamente o
  n é pequeno demais (23) para dizer o que "salva" a conversa.

### 2.4 Salas de chat: quais aberturas são respondidas, e como começa o flerte

"Resposta" aqui significa que, nos 15 posts seguintes, outro usuário menciona o nome de quem postou (o NPS anonimiza os nomes
também no texto).

| tipo de post (todos os posts) | n | resposta nominal |
|---|---|---|
| saudação endereçada ("hi User5", "wb User7") | 937 + 127 | **63% / 61%** |
| saudação à sala ("hi all", "hello ladies") | 124 | 38% |
| saudação solta ("hello", "wussups") | 175 | 31% |
| convite aberto ou "pm me" | 200 | **12%** |
| elogio ou apelido carinhoso | 128 | 38% |
| sexual | 78 | 44% (em geral provocação do grupo) |

- **Logit com efeito fixo de sala** (n=7.935): endereçado OR 1,30 (1,15–1,45); endereçado × saudação OR 1,58 (1,17–2,13), o que
  dá ≈2,0 para a saudação endereçada; convite aberto OR **0,42** (0,27–0,64); asl, elogio, sexual e pergunta sem efeito
  significativo.
- **Primeiros posts com rótulos do Jev** (n=698): saudação a uma pessoa tem 53% de resposta e 72% de permanência (≥3 posts
  depois); convite aberto 13% e 20%; asl 22% e 25%; posts com flerte 21%, contra 39% sem flerte; posts "pushy/creepy" 27%, contra
  38%. **Mas com efeito fixo de sala** só a energia (score 0–3) fica significativa (OR 1,33 por nível, p=0,03). Flerte, pushy e
  endereçamento perdem significância, porque as salas de adolescentes (12–18% de resposta) concentram os convites abertos, e
  nas salas de 40 anos a resposta fica perto de 60%. O noul `easy_reply` **não discriminou** (36% contra 35%).
- **Como o flerte começa aqui:** autodescrição demográfica + convite ("18/m pm me if u tryin to chat", "15 f perth bi any bi
  girls wanna chat"), perguntas coletivas ("any hot girls wanna chat w/a college guy"), apelidos e emotes de abraço
  ("heyyyyyyyyyy User65 honey *smewchies*", "((((User49)))) Hi honey.....how are you?"). O asl clássico é raro (20 posts),
  porque as salas já são separadas por idade.
- **Entrar e ser cumprimentado:** 25% das entradas (JOIN) recebem saudação nominal (wb 11% para quem volta). Quem é
  cumprimentado **antes** de postar posta em 66% dos casos nos 25 posts seguintes (média de 1,6 posts), contra 26% (média de 0,5).
  Isso tem confusão: os "regulares" são mais cumprimentados, e muitos JOINs são de bots ou de quem só espia.
- **Confiança:** alta para "endereçado > genérico" e "convite aberto é ignorado em público". **Ressalva:** a resposta pode ter
  acontecido por PM (invisível), justamente no caso do flerte. É 2006, em inglês e em grupo.

### 2.5 Fechamentos

**Como a última mensagem da sessão se parece:**

| | WhatsApp (2.937 sessões diádicas) | MaiChat (42) |
|---|---|---|
| despedida ou "xx" | 7,5% | **54,8%** |
| frase comum | **61,4%** | 35,7% |
| risada ("Haha") | 10,6% | — |
| ok, top, goed | 7,0% | 7,1% |
| só emoji ou mídia | 6,7% | — |
| pergunta sem resposta | 6,7% | 2,4% |
| quem fecha = quem abriu | 45,6% | 52,4% |

- **O WhatsApp íntimo é um fio contínuo:** as sessões acabam em "Dankjee!", "Nicee:))", "Ik ben er", "Hahaha" ou num emoji. A
  "despedida" é a próxima mensagem horas depois, e 15% das reaberturas são respostas atrasadas.
- **Ritual síncrono (MaiChat):** 79% têm um "bye" nos últimos 10 turnos. O primeiro "bye" aparece em geral **1 turno antes do
  fim** (16/33, bye → bye de volta). Só 4/33 terminam sem devolução, e 7/33 (21%) continuam 3 ou mais turnos depois do primeiro
  "bye", com brincadeira no meio ("bye im gonna rewatch it one more time" / "liar u mean 10 more times" / "shhh"). A despedida
  é **espelhada com variação** e muitas vezes com afeto ou apelido: "night cousin" → "goodnight"; "bye dramatic cousin" → "bye
  chaotic one 😌"; "Love you xxx" → "Love you too"; "okay bye" → "Oh okay bye 😒". O pré-fechamento explícito ("gotta go", "text
  u later", "ok im gonna sleep soon") aparece em só 24% das conversas, 1 a 6 turnos antes do fim.

**Sinais antecipados** (d = turnos até o fim; WhatsApp: todas as 1.245 sessões com ≥8 turnos; Jev: janelas do `jev_base`):

| d | caracteres (mediana) | tem "?" | resposta ≥5 min | emoji | D.hook | D.engagement | D.winding_down | P.p_end |
|---|---|---|---|---|---|---|---|---|
| 0 (último) | **18** | **7%** | **31%** | 23% | 0,34 | 1,62 | 34% | 0,22 |
| 1 | 33 | 14% | 27% | 21% | 0,46 | 1,98 | 19% | 0,19 |
| 2 | 39 | 19% | 22% | 19% | 0,54 | 2,06 | 13% | 0,18 |
| 3–9 | 36–40 | 21–31% | 19–23% | 15–19% | 0,55–0,63 | 2,0–2,2 | 4–11% | 0,14–0,18 |

No MaiChat os sinais descritivos do Jev são mais nítidos: o engagement cai de ~2,0 (d≥5) para 1,8 → 1,5 → 1,2, e o
winding_down sobe de 7% (d=9) para 21–33% (d=3–6) e depois 69–71% (d≤1). A queda de tamanho só aparece no último turno (10,5
caracteres contra ~23).

**O `P.p_end` prevê o fim?** (`a2_pend.py`; AUC contra o que de fato aconteceu)

| alvo | MaiChat | WhatsApp |
|---|---|---|
| T é o último turno da sessão | **0,914** (0,86–0,96) | **0,656** (0,62–0,69) |
| T é despedida (regex ou D.intent=closing) | 0,774 | 0,690 |
| fim em ≤3 turnos | 0,863 | 0,611 |
| melhores sinais de código (sozinhos) | −P.p_question 0,82; prev winding 0,82; turn_in_session 0,80 | −P.p_question 0,59; −prev engagement 0,57 |
| combinação (p_end + código, CV por conversa) | 0,919 | 0,690 |

- **Horizonte** (AUC de "faltam k turnos" contra "faltam ≥8"): no MaiChat k=0: 0,94; k=1: 0,83; k=2: 0,87; k=3: 0,81; k=4:
  0,75; k=5–6: ~0,70. No WhatsApp k=0: 0,70; k=1: 0,60; k=2: 0,59; k=3: 0,58; k≥4: ~0,55. **Na conversa síncrona dá para detectar
  com 2 a 4 turnos de antecedência; na assíncrona, só no último momento.**
- **Calibração:** `p_end ≥ 0,5` marca só 1,1–1,2% dos turnos. Com `≥ 0,3`, marca 3,7% (MaiChat; precisão de 43% para
  despedida e 49% para fim em ≤3 turnos) e 9% (WhatsApp; 24% e 55%). Use limiares relativos.
- **Experimento novo** (757 turnos do WhatsApp; 337 últimos + 420 aleatórios; state do passe P + relógio): `quiet_after` (noul)
  AUC 0,65; `open_loop` (noul, invertido) 0,59; `resolved` (noul) 0,67; **`closure` (score 0–3) 0,69 (0,65–0,72)**; p_end
  base 0,66 (0,62–0,70); combinação 0,685. Para detectar a despedida em si, `closure` chega a 0,77. **Não há ganho significativo:**
  no WhatsApp o fim depende de coisas fora da conversa (aula, trabalho, bateria).
- **Confiança:** alta para o MaiChat (mas o fim é artificial, porque há sessão de laboratório, "voucher" e "Time over") e média
  para o WhatsApp.

---

## 3. Padrões "invisíveis" (o que as pessoas fazem sem perceber)

1. **O "oi" de volta nunca é cópia:** muda-se a palavra (74% no 1:1) e **sobe-se um degrau de energia** ("Hi!" → "Helluuuu",
   "hey" → "Hii"). Um bot que responde "Oi!" a "Oi!" soa como máquina.
2. **Devolver a saudação e já pôr algo na mesa, em duas bolhas:** "hello / that baby with ice cream / teeth must have hurt".
   A saudação sozinha é só a metade do turno.
3. **"Oi" sozinho é um pedido de atenção** e é respondido com "e aí?" ou "fala". Não é um "começo de small talk".
4. **Entre íntimos não se cumprimenta nem se despede:** a conversa é um fio que pausa. 61% das sessões acabam numa frase comum,
   e 15% das "novas" conversas são só a resposta à última mensagem.
5. **O pedágio da saudação é proporcional ao silêncio** (4% → 34%) e vem sempre com um **callback concreto** ("Nog leuke
   vierdaagsedagen gehad?"), nunca com "quanto tempo!".
6. **Ninguém tematiza o tempo que passou**, a não ser quem deve desculpas (3,7%). Cobrar "sumiu, hein?" seria anormal.
7. **Mandar outra mensagem depois do silêncio sem ter recebido resposta é normal:** em 45% das reaberturas quem escreve é quem
   falou por último. No chat mediano, 60% das aberturas vêm da mesma pessoa, e em 32% dos chats essa pessoa abre mais de 65%.
8. **A pergunta pendente é uma âncora:** quando a sessão acaba numa pergunta, a próxima começa com a resposta (50%).
9. **A última mensagem é curta, sem pergunta e muitas vezes emoji ou risada** (18 caracteres; 7% com "?"; emoji 23%). Quem quer
   encerrar **para de fazer perguntas** um ou dois turnos antes: é o sinal mais forte e mais barato.
10. **A despedida é espelhada e às vezes vira uma "guerra de bye"** (21% com 3 ou mais turnos depois do primeiro bye), com
    afeto ou apelido ("bye dummy <3", "bye smaller idiot 😎").
11. **Entre desconhecidos, o que funciona é falar com alguém pelo nome.** A chamada genérica "anyone wanna chat?" morre (12%), e
    repetir não adianta.

---

## 4. Tradução para o sistema (detectores Jev + regras de código)

Convenções: **N** = Noul, **S** = Score, **C** = Choice. As contas (silêncio, hora, tamanhos, "?", alongamento, latência) ficam no
código. Todos os detectores de um mesmo momento vão numa só chamada.

### 4.1 Na chegada de uma mensagem do usuário depois de silêncio (gap ≥ 3 h) ou no primeiro contato

**State (enxuto):** `{bot_last_turns: últimos 3–4 turnos da sessão anterior, silence: faixa (código), local_time: "terça 21:40",
pending_items: [itens abertos da memória, ≤5], user_message}`. Se for o primeiro contato: `earlier_messages: "(none)"`.

| id | tipo | instrução | critérios |
|---|---|---|---|
| `open_type` | C | "What best describes `user_message` (first message after the silence)?" | greeting_only, greeting_how_are_you, greeting_plus_topic, availability_check, late_reply, follow_up_earlier, new_news, new_question_or_request, logistics, media_or_link, wishes |
| `apologizes_delay` | N | "Does `user_message` apologize for or explain their absence/late reply?" | — |
| `answers_pending` | N | "Does `user_message` answer the bot's last unanswered question?" | — |
| `callback` | C | "Which pending item is most worth asking about now?" | itens de `pending_items` + `none` |
| `user_energy` | S | "How much energy does `user_message` show?" | flat, mild, lively, very excited |

**Regras de código:**
- `greeting_only` (conf ≥ 0,6), que corresponde ao "oi" puro:
  - responder rápido (alvo de 10 a 60 s, com "digitando…");
  - bolha 1: **saudação de volta com outra palavra** (sortear, evitando a do usuário ~75% das vezes) e **energia igual ou +1**
    (se o usuário alongou ou usou "!", alongar);
  - bolha 2: gancho. Se `callback ≠ none` e p ≥ 0,5, perguntar sobre o item ("e a prova, foi?"); senão, "e aí, o que manda?"
    (o "oi" é um pedido de atenção);
  - nunca mandar só "Oi! Como posso ajudar?".
- `greeting_how_are_you`: responder de forma concreta, com um detalhe do "dia" do personagem, e devolver a pergunta ("e você?").
  Opcionalmente, callback.
- `greeting_plus_topic`, `new_question`, `new_news`: saudação curta **só se o usuário saudou** (espelho; 25% contra 3% nos dados)
  e ir direto ao assunto. Sem saudação do usuário, nada de saudação.
- `late_reply` ou `answers_pending` > 0,6: **nenhuma saudação e nenhum "bem-vindo de volta"**. Continuar o fio como se não
  tivesse havido pausa e reagir ao conteúdo.
- `apologizes_delay` > 0,6: uma oração curta de "relaxa" (ex.: "imagina 😅") e seguir. Nunca cobrar, nunca mencionar a duração.
- `availability_check`: confirmar presença em poucas palavras ("tô aqui, fala") e, se houver, puxar o gancho.
- `media_or_link`: reagir ao conteúdo da mídia primeiro (mídia sem reação morre em 51% dos casos).
- **Não tematizar o tempo** ("quanto tempo!") a menos que o silêncio seja maior que 7 dias **e** `open_type ∈ {greeting_only,
  greeting_how_are_you}`. Mesmo assim, fazer isso só em uma resposta a cada 3 (nos dados, reconhecer o silêncio é raro).

### 4.2 Reabertura proativa do bot (quando o bot escreve primeiro)

**State:** `{last_session_tail (4 turnos), pending_items, silence, local_time, last_speaker}`.

| id | tipo | instrução |
|---|---|---|
| `worth_reopening` | N | "Is there something natural and specific to come back to (an event that has now happened, an open question, a plan)?" |
| `callback` | C | igual a 4.1 |
| `left_on_question` | N | "Did the conversation stop on a question that is still unanswered?" |

**Regras:**
- Só reabrir se `worth_reopening ≥ 0,6` ou se o silêncio for de 24 h ou mais com `callback ≠ none`.
- Formato: **saudação apenas se o silêncio for ≥ 24 h** (a taxa de saudação passa de ~4% para ~20–34%) e **sempre com gancho
  específico** (85% nos dados). "Oi" sozinho, nunca (4%).
- Se `left_on_question` e quem perguntou foi o bot: não repetir a pergunta; mudar para outro callback ou esperar.
- Mandar outra mensagem sem resposta é natural (45% das reaberturas), mas no máximo uma reabertura sem resposta por ciclo.
- Hora: o código evita reabrir entre 0 h e 7 h. Pela manhã, a variante "bom dia" é escolhida por código.

### 4.3 Primeiro contato com usuário novo (estilo sala e desconhecido)

**State:** `{user_message, persona_brief}`. Perguntas: `open_type` (C, com as classes do NPS: greet_person, greet_bare,
demographics_asl, open_invitation, compliment_or_flirt, sexual_proposition, question, self_statement), `flirting` (N),
`pushy_creepy` (N), `user_energy` (S).
**Regras:** responder **endereçando o usuário pelo nome ou apelido**, se conhecido (é o fator que dobra a resposta no NPS).
Espelhar a energia. Para `open_invitation` ou `demographics_asl`, responder com calor e **uma pergunta concreta**, sem
questionário. Se `flirting ≥ 0,6` e `pushy_creepy < 0,4`, espelhar o flerte **um degrau abaixo** até a relação estar
estabelecida (quem controla é o módulo de relação). Se `pushy_creepy ≥ 0,6`, mandar resposta curta e redirecionar.

### 4.4 Detecção de fim (roda em todo turno do usuário, na mesma chamada dos outros detectores)

**State:** o do passe P (últimos 6–8 turnos, com as faixas de "replied") + `clock_time`.

| id | tipo | instrução | níveis |
|---|---|---|---|
| `closure` | S | "How close is this conversation to ending for now?" | in full swing, slowing down, wrapping up, saying goodbye |
| `p_end` | N | "Will the user start ending the conversation (goodbye) in this turn?" | — |
| `farewell_type` | C (só se closure ≥ 2) | "What kind of ending is this?" | goodnight_sleep, gotta_go_busy, see_you_later, affectionate, abrupt_annoyed, implicit (ok/haha/emoji) |

**Sinais de código** (relativos ao próprio usuário): `short = chars < 0,5 × média móvel(4)`; `no_q = sem "?" nos 2 últimos turnos`;
`slow = latência ≥ 2 × mediana do usuário`; `minimal = só emoji, risada ou "ok"`.
**Regra:** `end_score = 0,5·(closure/3) + 0,3·min(1, p_end/0,3) + 0,1·short + 0,1·(no_q & (slow | minimal))`.
- `end_score < 0,4`: normal.
- `0,4–0,6` (**desacelerando**): responder mais curto, **não abrir tema novo nem fazer pergunta nova** (quem sai para de perguntar
  1 a 2 turnos antes). No máximo um gancho leve.
- `≥ 0,6` ou `farewell_type` explícito (**fechando**): espelhar a despedida **com variação e afeto conforme o tipo**
  (goodnight_sleep → "boa noite, dorme bem 💤"; affectionate → devolver o afeto; gotta_go → "vai lá! depois me conta X",
  deixando um fio aberto para a próxima). Permitir **uma** rodada extra se o usuário continuar brincando (a guerra de bye
  acontece em 21% dos casos). Nunca perguntar "posso ajudar em mais alguma coisa?".
- `implicit` (o usuário manda "haha", "ok" ou emoji com end_score alto): num contexto assíncrono, **responder com algo mínimo, ou
  nem responder**, e deixar a conversa pausar. É assim que 61% das conversas íntimas "terminam".
- Salvar como `pending_item` o que ficou aberto (evento futuro, pergunta sem resposta) para o callback de 4.1 e 4.2.
- **Limiares:** no modo síncrono (app aberto, respostas em segundos) confiar mais no `closure` e no `p_end` (AUC ~0,9). No modo
  assíncrono, tratar como sinal fraco (AUC ~0,66–0,69) e agir só com o código concordando.

### 4.5 Entrega (sem Jev)

- Resposta ao "oi": 1–2 bolhas; latência curta. Nos dados, respostas em até 1 min acompanham sessões mais longas (Spearman
  −0,14, p<0,001).
- Alongamento e "!" espelhados por código a partir das features do usuário (`elongation`, `n_excl`).
- A despedida vai numa bolha curta, podendo terminar em emoji (23% das últimas mensagens têm emoji).

---

## 5. Limitações

- **WhatsApp em holandês (2012–2014)** e Jev melhor em inglês: a tipologia de aberturas depende do Jev em holandês. As tendências
  batem com as regras, mas as porcentagens exatas podem mudar ±5 pp. São 57 chats, e poucos são de casais ou família (2 a 4):
  **não dá para generalizar por relação**.
- **A "sessão" é artificial:** o fim de sessão é só um silêncio de mais de 3 h, então muitos "fins" não são fechamentos, e muitos
  "inícios" são respostas atrasadas. Os timestamps têm resolução de minuto (latências "1 min" empatam).
- **O MaiChat tem 42 conversas em laboratório:** as aberturas são induzidas ("vamos conversar agora"), e os fins são forçados
  por tempo ou voucher ("Time over", "wrap this up / And get that sweet amazon voucher"). A AUC de 0,91 do `p_end` provavelmente
  **superestima** o que se veria em uso real. O lote de 2025 (conv063–095) tem estilo de roleplay entre primos ou irmãos
  ("yo idiot u alive").
- **O "oi" puro tem n pequeno** (23 casos no 1:1). As conclusões sobre a resposta a ele se apoiam no espelhamento geral (n≈900 a
  2.900) e no NPS (n=765).
- **NPS:** é grupo, 2006, com respostas por PM invisíveis (subestima a resposta ao flerte e ao "pm me"). A medida de "resposta" é a
  menção nominal, e as salas diferem muito entre si (confusão; os efeitos do Jev somem com o efeito fixo de sala).
- **Sucesso = tamanho da sessão** é uma proxy fraca e em parte mecânica (perguntas geram respostas). Tudo aqui é correlacional,
  sem experimento.
- Os detectores novos de fim foram avaliados em uma amostra enriquecida (45% positivos). As AUCs são comparáveis entre si, mas
  a precisão absoluta, não.
