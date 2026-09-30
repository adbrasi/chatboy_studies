# 07 · Flerte, afeto e intimidade: o que dizer e como responder

Prefixo `a7`. Scripts em `scripts/analysis/a7_*.py`; saídas em `analysis/data/a7_*`.
Dados: `maichat` (inglês, ao vivo, 42 conversas), `whatsapp_nl` (holandês, WhatsApp real), `nps_chatroom` (salas públicas
de 2006, desconhecidos) e gerações novas de LLM (80 contextos reais do maichat).

**O que foi rodado**

| etapa | script | n | custo |
|---|---|---|---|
| Estilo, vocabulário (log-odds com prior de Dirichlet), digitação | `a7_style.py`, `a7_reply_typing.py`, `a7_episodes.py` | 5.968 turnos anotados no jev_base (464 de flerte/afeto, 4.290 neutros) | 0 |
| Movimento de flerte do turno T + movimento da resposta R (Jev) | `a7_moves.py` → `a7_moves_analysis.py` | 705 chamadas → **413 pares** T→R (162 maichat, 251 whatsapp; 67 vêm de uma expansão com 300 turnos extras de 3 chats de casal) | Jev US$0,05 |
| NPS: flerte entre desconhecidos (Jev) | `a7_nps.py` → `a7_nps_analysis.py` | 804 posts (504 candidatos por regex + 300 de controle), 328 flertes, 87 flertes dirigidos, 55 respostas do alvo | Jev US$0,04 |
| LLM × humano: 80 contextos de flerte/afeto, 3 condições | `a7_llm_gen.py`, `a7_llm_moves.py` | 240 gerações (gemini-3.5-flash-lite e gpt-4o-mini de baseline; gemini com briefing) + 880 chamadas Jev de juízo/classificação | LLM US$0,02; Jev US$0,03 |

No total foram cerca de 2.460 chamadas novas ao Jev (≈US$0,13) e 242 chamadas de LLM (≈US$0,02).
Definição operacional de "turno de flerte/afeto" (FA) no jev_base: `D.flirting ≥ 0,5` **ou** `D.emotion = affection` **ou** `D.intent = compliment_affection`.
"Neutro" é o turno com `D.flirting < 0,15` e sem afeto. As intensidades vêm de um Score do Jev em 5 níveis: 0 = nada, 1 = carinho leve,
2 = flerte ou afeto claro, 3 = afeto romântico forte, 4 = intenso ou sexual.

---

## 1. Resumo dos achados

- **O afeto vem em pitadas; não é sustentado.** Só 7–8% dos turnos são FA (maichat 208/2.871; whatsapp 256/3.097). 79% dos
  episódios de FA no maichat duram **um único turno** (114/144; média de 1,4 turno). O afeto é contagioso (P(resposta FA | turno FA) = 33%
  contra 7% de base, lift de 4,5×), mas **em 2/3 das vezes a resposta a um flerte não é flerte**.
- **A resposta humana mais comum ao afeto é *não* responder a ele.** Nos 413 pares: ignora e segue o assunto em 32%, dá um ack mínimo
  ("hehe", "haha") em 13%, devolve a provocação em 18%, devolve o afeto em 13%, engaja com entusiasmo em 8%, agradece em 4%,
  fica "encabulada" ("stop 🥺") em só 1,7%, recusa educadamente em 1,7% e escala em 1,7%.
- **As pessoas respondem abaixo da intensidade do parceiro.** A diferença média é de −0,51 nível (escala 0–4). Só 8% respondem acima e
  3,4% respondem mais de 1 nível acima. Quem recebe um turno de nível 3 responde em média no nível 1,7. No NPS a diferença é −0,88.
  Isso sustenta a regra **"nunca mais que o usuário + 1"**, com o padrão real ficando em "igual ou um pouco abaixo".
- **Mas o que mantém o flerte vivo é igualar ou subir meio passo.** Veja se o turno seguinte do flertador ainda é FA
  (medida independente, do jev_base): depois de "encabulada" 6/7, depois de escalar 4/4, depois de ack mínimo 52%, devolver
  provocação 40%, devolver afeto 40%, ignorar 19% (n = 104), só agradecer 17% e recusar educadamente 0/5. Devolver o afeto também
  **fecha** a conversa: 24,5% dos "love you too" são o último turno da sessão.
- **Cada movimento tem a sua resposta típica.** Provocação → provocação de volta (20/70). Pirraça ("rude???") → provocação (5/10).
  Elogio indireto → encabulada, provocação ou desinflar com humor (12/26), como em "ugh ur sweet", "dont expose me", "i live with you. i have to."
  "Já comeu?" → resposta literal ("i had noodles", "coffee counts right"). "I miss you" → metade segue o assunto ou a logística (11/22) e
  4/22 devolvem; também aparece "its the same face as yesterday". Ninguém responde com um parágrafo.
- **O flerte escrito é curto, sem pergunta e com marcadores de carinho, não de riso.** No maichat, turnos FA × neutros: 21 × 26 caracteres
  (p < 0,001), pergunta em 2,9% × 10,2%, emoji em 10,6% × 3,4%, `<3`/xx/emoticon em 7,2% × 0,6% e apelido em 12% × 1,2%. O riso
  **não** aumenta (3,8% × 4,7%). No whatsapp_nl: xx em 11% × 0,8%, alongamento ("liefjeeee") em 17% × 6,5%, "!" em 34% × 21%.
- **Um padrão invisível de timing: pausa antes, digitação rápida depois.** Quem responde a um flerte no maichat demora mais para
  *começar* a digitar (mediana de 11,2 s × 4,4 s; latência total de 18 × 13 s, p = 0,0002). Depois digita mais rápido, com
  menos apagamentos (29% × 39% dos turnos com deleção) e manda algo mais curto (21,5 × 26 caracteres). Devolver o afeto é a resposta
  mais rápida (13 s); a provocação espirituosa é a mais lenta (22–30 s).
- **Vocabulário do afeto real: abreviação, cuidado e emoji, não adjetivo.** As palavras mais distintivas no lado FA (EN): *love, u, proud,
  miss, sweet, aww, tho, eat, too, btw, cute, <3, kinda, 🥺, ur, xx, sleep, pls, baby, babe, bye, 😘, 😇, 😌*. No lado neutro:
  *the, yeah, haha, lol, idk, hmm, really*. Em holandês: *xx, ;), schat, lief, 😘, slaap lekker, hihi, :D, ;p, succes, sterkte*.
- **A LLM baseline erra no formato e no movimento.** O gemini-flash-lite escreve 2,2× mais que o humano no mesmo contexto (mediana de
  62 × 23 caracteres), põe pergunta em 42,5% × 9%, emoji em 60% × 14%, riso em 31% × 6% e 0,49 "!" por mensagem × 0,09. Em 45% das
  respostas usa a estrutura "reação + \n\n + pergunta ou novo gancho", que nenhum humano usou, e vazou `</p>` 3 vezes. O gpt-4o-mini
  tem o vício oposto (assistente): 81% das respostas terminam com pontuação, 1,45 "!" por mensagem e 44% são "engajamento entusiasmado"
  (humano: 7,5%). Ninguém nas LLMs faz o que o humano mais faz, que é ignorar ou dar um ack mínimo (LLMs 12–16%, humanos 40%).
- **As LLMs ficam perto demais da intensidade do usuário.** Em relação ao turno do parceiro, o humano fica em média 0,49 nível abaixo;
  o gemini fica 0,12 abaixo e o gpt 0,22 abaixo. Em termos absolutos, a intensidade da resposta é 0,91 no humano, 1,23 no gemini e 1,15 no gpt.
  O excesso é modesto: o problema maior é **engajamento e validação demais**, não melodrama.
- **O Jev não serve de juiz de "soa humano?".** No pareado, ele escolheu o **gemini como "a pessoa real" em 75%** dos contextos e como "mais
  adequado ao momento" em 89%. O gpt-4o-mini perdeu para o humano em 66%. O juiz recompensa respostas engajadas e coerentes,
  e o "deixa pra lá" humano parece pior do que é. Validação de realismo deve usar estilometria determinística + distribuição de movimentos.
- **O briefing corrige o formato, mas trouxe dois problemas novos.** O gemini com briefing (movimento + intensidade + tamanho + exemplos
  + palavras proibidas) acertou o balde de tamanho em 40% (baseline: 21%), a presença de pergunta em 90% (59%) e a de emoji em 85% (49%), e teve
  a menor sobrevalidação no juízo do Jev (0,10 contra 0,15 do humano e 0,20–0,24 das LLMs). Mas o movimento previsto antes da resposta
  bateu com o humano em só 27,5%. Além disso, a LLM **copiou os exemplos literalmente** ("miss u too 😚" apareceu 4 vezes, "ugh ur sweet" 3,
  "keep dreaming" 3), às vezes de forma incoerente.

---

## 2. Achados detalhados

### A1. Inventário de movimentos (o que as pessoas fazem quando flertam ou dão carinho)

Distribuição dos 413 turnos FA com resposta (Jev Choice; confiança média de 0,72):

| movimento | % | exemplo real |
|---|---|---|
| provocação / banter (`tease_banter`) | 16,9 | "brat" · "define finish" · "Haha / Sukkel" |
| cuidado / check-in (`caring_checkin`) | 14,5 | "did you eat today btw" · "u eating properly tho" · "proud of u" |
| saudação com apelido (`pet_name_greeting`) | 11,6 | "Hello my love" · "yo babe u there" · "Isabellie schatje" |
| convite / plano (`invitation_plan`) | 9,2 | "shower / then maybe call?" · "read our horoscopes together?" |
| elogio direto | 7,0 | "hello handsome" · "Leuke foto👍" · "talented" |
| reasseguramento | 6,8 | "dont say that / i dont hate you xxx" |
| elogio indireto | 6,3 | "still my fav" · "that's the softest thing u've said" · "i love how you understand me" |
| despedida afetuosa | 6,1 | "sleep well" · "bye dummy <3" · "Ok tot dalijk xxx" |
| saudade (`miss_longing`) | 5,3 | "yes pls / i miss ur face" · "Kon ik je maar knuffelen" |
| insinuação / sexual | 4,1 | quase tudo de um casal holandês (wa_M_003) |
| só emoji ou beijo | 3,9 | "🥰🥰" · "Okeeeeeee / ❤️" |
| "te amo" | 2,9 | "love u tho" · "Love you xxx" |
| pirraça / fingir ofensa | 2,4 | "rude???" · "pls dont 😭" · "Oh okay bye 😒" |
| vulnerabilidade | 2,2 | "i like falling asleep to ur voice" |
| autodepreciação charmosa | 0,7 | "u know im weak" |

- **Leitura:** entre pessoas que já se conhecem, o afeto é sobretudo **cuidado cotidiano** (comer, dormir, orgulho, boa sorte) e
  **provocação**. As declarações explícitas ("I love you", "I miss you") somam só 8%. O elogio direto à aparência é raro; o elogio
  vem mais indireto ("still my fav").
- **NPS (desconhecidos) é outro mundo.** Dos 328 flertes, 34% são convites ("any girls wanna chat? pm me"), 18% saudações com apelido
  ("hey hun", "*hugs*"), 14% insinuação ou sexo explícito e 11% só beijo ("mwahhss", "<333"). A intensidade média é 2,08, contra 1,5 entre conhecidos.
  33% são anúncios para a sala inteira e só 27% são dirigidos a alguém. **Serve como anti-modelo** para um bot de companhia.
- **Confiança: média.** O Choice do Jev tem classes vizinhas que se confundem (elogio indireto × reasseguramento). Em 56 dos 405
  turnos que o jev_base marcou como FA, o Jev devolveu "none" (14%), o que mostra a fronteira difusa entre carinho e simpatia.

### A2. Como as pessoas respondem (e qual resposta mantém o flerte vivo)

| resposta (R) | % | vivo (Jev) | próximo turno do flertador ainda FA | R é o último turno da sessão | mediana de caracteres | exemplo |
|---|---|---|---|---|---|---|
| ignora e segue o assunto | 32,0 | 0,36 | **0,19** (n = 104) | 3,8% | 41 | "quality over quantity babe" → "deadline today right" |
| devolve a provocação | 18,2 | 0,66 | 0,40 (57) | 1,3% | 25 | "im proud" → "dont sound like my mom" |
| ack mínimo | 13,3 | 0,44 | 0,52 (33) | 16% | 8 | "NO" → "hehe" |
| devolve o afeto | 12,8 | 0,69 | 0,40 (35) | **24,5%** | 28 | "Love you xxx" → "Love you too" |
| engaja com entusiasmo | 8,2 | 0,70 | 0,47 (32) | 2,9% | 36 | "i saw a couple that reminded me of us" → "good or bad 😭" |
| agradece | 4,4 | 0,54 | 0,17 (12) | 33% | 22 | "im proud actually" → "thank you" |
| desinfla com humor | 3,6 | 0,43 | 0,36 (14) | 0% | 27 | "yes pls / i miss ur face" → "its the same face as yesterday" |
| encabulada / tímida | 1,7 | 0,47 | **0,86** (7) | 0% | 13 | "still my fav" → "ugh ur sweet" · "that was vulnerable" → "HELP" |
| escala | 1,7 | 0,90 | **1,00** (4) | 0% | 73 | (casal NL) "Ik ook schat" → proposta sexual |
| recusa educada | 1,7 | 0,33 | **0,00** (5) | 0% | 52 | "Do you just like me stretchy 🤣" → "Now, now. That's for the other chat😆" |

- **O que mata:** ignorar (19% de continuidade), agradecer seco (17%) e recusar (0/5). **O que mantém:** encabular-se, escalar meio
  passo, provocar de volta. Pela diferença de intensidade R − T: com 1,5 nível ou mais abaixo, o turno fica "vivo" em 0,30; perto de igualar
  (±0,5), em 0,59; com +0,5 a +1,5, em 0,80 (n = 33), e o próximo turno ainda é FA em 67% (contra 28% quando iguala).
  **Mas:** quando a resposta desinfla muito (≤ −1,5), o flertador *insiste* (47% de FA no turno seguinte, n = 43). Muitas vezes o flerte
  sobrevive a um "deixa pra lá" porque quem começou tenta de novo.
- **Por movimento:** provocação → provocação de volta (29%), ignora (24%), ack (16%). Pirraça → provocação (5/10), "jk jk".
  Elogio (direto e indireto, n = 55): agradecer é raro (2/55); predominam ignorar (17), provocar (10), engajar (6), devolver (6),
  desinflar (4) e encabular (3). "Já comeu?" / "orgulho de você" (n = 60): ignora com resposta literal (21), provoca (11), ack (10),
  agradece (7), desinfla (4). Despedida afetuosa: devolve (9/25), às vezes só com "u too" ou "Tot zoooxxx". Saudação com apelido
  (n = 48): segue direto para o assunto (17), devolve o apelido (11). "hey love / im here" → "guess what".
- **"I miss you" (n = 22) e "I love you" (n = 12):** devolvem em só 4/22 e 4/12. A devolução costuma ser **simétrica e do mesmo
  tamanho** ("love you <3" → "love you too <3"; mediana R/T de 1,28 no "te amo") e rápida (latência mediana de 8 s no "te amo" contra 31 s na
  saudade). No conv001, a devolução vem **um turno depois**: primeiro a pessoa termina o fio da dança ("I will make sure you enjoy
  dancing…😂"), depois manda "I miss you too😚😚", e o parceiro então faz pirraça ("Oh okay bye 😒").
- **Esquiva do flerte não correspondido:** quase nunca é explícita (7 recusas educadas e 3 grosseiras, contra 132 "segue o assunto").
  Esquiva-se **mudando de assunto**, com humor ("Now, now. That's for the other chat😆") ou com um "Nee / Niet moe :(". No NPS,
  entre desconhecidos: "11-09-20sUser80 i like you" → "lol thanks i think". As rejeições grosseiras aparecem nos atos `Reject`
  ("get outta my PM box.. Im with my fiance!!!").
- **Tamanho, emoji e riso da resposta** (FA × resposta a turno neutro): no maichat, 21 × 26 caracteres (p = 0,003), pergunta em 5,6% × 9,8% e
  emoji em 9,3% × 3,7%. No whatsapp, o tamanho não muda (41 × 40) e emoji fica em 14% × 4%. A razão R/T de tamanho tem mediana de 1,0 no maichat
  (simétrico) e 0,73 no whatsapp. O emoji é espelhado: P(R tem emoji | T tem emoji) = 24% contra 10% quando T não tem.
- **Confiança:** média para a distribuição; baixa para as classes raras (encabulada, escala e recusa têm n = 4–7).

### A3. A intensidade é calibrada *para baixo*

- Spearman(t_int, r_int) = 0,36. Média da resposta por nível do gatilho: T = 1 → 0,82; T = 2 → 1,18; T = 3 → 1,73; T = 4 → 1,81 (n = 4).
  73% das respostas ficam a até ±1 nível; 23% ficam mais de 1 nível abaixo; só 3,4% ficam mais de 1 nível acima.
- NPS (desconhecidos, n = 55): diferença média de −0,88. Quanto menor a intimidade, maior o "desconto".
- **Confiança: média-alta** para a direção (é consistente nos 3 corpora). O Score do Jev tem regressão à média própria, o que
  reforça que a comparação relativa (humano × LLM, no mesmo juiz) é mais confiável que os valores absolutos.

### A4. Forma do turno de flerte e vocabulário

Estilo dos turnos FA × neutros (`a7_style.json`, teste de Fisher / Mann-Whitney):

| | maichat FA | maichat neutro | whatsapp FA | whatsapp neutro |
|---|---|---|---|---|
| mediana de caracteres | **21** | 26 (p < 0,001) | 57 | 43 (p < 0,001) |
| pergunta | **2,9%** | 10,2% (p < 0,001) | 24% | 25% (ns) |
| emoji | 10,6% | 3,4% | 13,7% | 3,8% |
| `<3` / xx / emoticon | 7,2% | 0,6% | 28,5% | 11,2% |
| apelido | 12% | 1,2% | 16% | 0,8% |
| riso | 3,8% | 4,7% (ns) | 18% | 14% (p = 0,06) |
| alongamento | 4,8% | 6,4% (ns) | 17% | 6,5% |
| "!" | 1,4% | 3,0% (ns) | 34% | 21% |

Emojis mais frequentes: no maichat FA, 🥺 (7), 👍 (6), 😇 (5), 😘 (4), 🥰 (3), 😌 (3), 😚 (2); no maichat neutro, 😭 (14), 😂 (12), 😱, 🤡, 💀.
No whatsapp FA: 😘 (12), ❤ (5), 🙈 (3); no neutro, 😂, 👍, 😑, 😅. **O afeto tem os seus emojis (🥺😘😇😌🥰😚) e o humor tem os dele
(😭😂💀🤡). Misturar os dois conjuntos soa errado.**

**Vocabulário distintivo** (log-odds com prior de Dirichlet informativo, Monroe et al. 2008; prior = o corpus inteiro; mínimo de 3 ocorrências;
entre parênteses, a contagem):

- **maichat FA (40):** love (18), u (30), 👍 (6), proud (5), miss (5), 😇 (5), sweet (5), you (38), aww (4), tho (7), eat (7), hey (4), too (13), im (15), 😌 (3), btw (7), cute (5), will (9), <3 (8), i'll (3), call (3), okay (5), kinda (4), 🥺 (7), text (3), our (4), ur (5), xx (5), sleep (3), pls (3), us (4), baby (3), bye (5), say (3), babe (3), look (3), week (3), 😘 (4), go (6), him (3)
- **maichat neutro (40):** the, yeah, it, not, no, had, is, to, on, a, like, haha, also, need, doing, lol, been, going, them, really, dog, all, an, and, have, idk, in, watch, what, time, i've, cat, school, gonna, videos, hmm, was, type, there, does
- **whatsapp_nl FA (40):** xx, je, bent, ;), ‹alongamento›, schat, :d, 😘, slapen, snel, weer, lekker, leuke, geval, lief, slaap, jou, balen, vinden, 🎉, uit, ieder, reis, sterkte, succes, goed, aah, met, x, hihi, liedje, namens, hoor, lang, 👍, komt, ;p, jongen, heel, gezellig
- **whatsapp_nl neutro (40):** heb, de, ik, ja, die, nee, in, ze, heeft, nog, had, zal, geen, ben, of, dus, oh, zijn, dan, dacht, staat, anders, week, denk, n, prima, vanavond, volgens, oké, over, twee, bij, vragen, nu, thuis, deze, idd, precies, geweest, oke
- **NPS flerte × não flerte (40, EN, desconhecidos):** me, pm, chat, any, m, girls, guys, u, wanna, tryin, sexy, f, hugs, girl, with, babe, ladies, :), honey, want, ;-), hot, gurls, bi, ;), if, females, please, hugss, love, sweet, r, wana, tx, fl, lez, canada, aww, heyy, <3.
  No lado neutro: the, lol, i, is, was, it, her, lmao, up, in, no, at, not, play, oh, ok, one, but, pick, look… (a lista de flerte é **parcialmente
  circular**, porque os candidatos saíram de uma regex com sexy/wanna/pm).

**Leitura:** o léxico do afeto real é **pronome de segunda pessoa + abreviação + palavra de cuidado + emoji**: "u/ur/tho/pls/kinda/btw",
"eat/sleep/proud/miss/call/text", "you/our/us", "<3/xx/🥺". Adjetivos floreados não aparecem (nenhum "gorgeous", "amazing", "wonderful"
nos turnos FA do maichat). Curiosamente, "haha"/"lol" pertencem ao lado **neutro**: o flerte entre conhecidos não é marcado por riso.
Em holandês, o afeto aparece nas despedidas e nos votos ("slaap lekker", "succes", "sterkte", "goede reis") e no "xx".

**Confiança:** alta para os marcadores (apelido, xx, emoji; p < 10⁻⁵); média para as listas de palavras (o maichat FA tem só 1.230 tokens).

### A5. Timing e digitação do flerte (maichat)

| | responde a FA | responde a neutro | p |
|---|---|---|---|
| latência (fim da msg do outro → envio), mediana | **18,0 s** | 13,2 s | 0,0002 |
| ocioso antes de começar a digitar, mediana | **11,2 s** | 4,4 s | < 10⁻⁵ |
| tempo de composição, mediana | 3,4 s | 5,0 s | 0,0003 |
| s/caractere | 0,186 | 0,213 | 0,01 |
| turnos com alguma deleção | 29% | 39% | 0,007 (razão) |
| maior pausa durante a digitação, mediana | 0,22 s | 0,33 s | 0,001 |
| mediana de caracteres | 21,5 | 26 | 0,01 |

O próprio turno de flerte também é escrito mais rápido (composição mediana de 3,3 × 5,0 s; menos deleções, p = 0,009). Por tipo de
resposta: devolver o afeto leva 13 s; ignorar, 14 s; ack, 19 s; provocar, 22,5 s; desinflar com humor, 30 s.
**Leitura:** a hesitação acontece *antes* de digitar, na hora de decidir o que dizer; o texto sai de uma vez, curto e sem revisão.
**Confiança: média.** A latência inclui leitura e as conversas do maichat são induzidas (as pessoas estão "fazendo a tarefa").

### A6. LLM × humano nos mesmos 80 contextos reais

Mesmo contexto (até 14 turnos, turnos do outro como `user`), persona mínima ("You are Sam, chatting with Alex on a messaging app.
Reply as Sam."), temperatura 0,8.

| | humano | gemini baseline | gpt-4o-mini baseline | gemini + briefing |
|---|---|---|---|---|
| mediana de caracteres | **23** | 62 | 46,5 | 13 |
| razão de tamanho em relação ao humano do mesmo contexto (mediana) | 1 | 2,16 | 1,88 | 0,48 |
| mesmo balde de tamanho do humano | — | 21% | 25% | 40% |
| tem pergunta | **9%** | 42,5% | 36% | 4% |
| tem emoji | **14%** | 60% | 24% | 16% |
| "!" por mensagem | **0,09** | 0,49 | 1,45 | 0,03 |
| riso (haha/lol/lmao) | **6%** | 31% | 5% | 6% |
| termina com pontuação | **10%** | 39% | 81% | 6% |
| "reação + \n\n + gancho" | **0%** | 45% | 0% | 0% |
| movimento = ignora ou ack | **40%** | 16% | 11% | 22,5% |
| movimento = engaja com entusiasmo | 7,5% | 15% | **44%** | 0% |
| movimento = devolve provocação | 14% | **31%** | 9% | 25% |
| movimento igual ao do humano | — | 29% | 30% | 24% |
| intensidade R − T (Jev) | **−0,49** | −0,12 | −0,22 | −0,30 |
| Jev: sobrevalida | 0,15 | 0,20 | 0,24 | **0,10** |
| Jev: clichê | 0,13 | 0,16 | 0,21 | 0,18 |
| Jev: genérica | 0,67 | 0,61 | 0,78 | 0,73 |
| Jev pareado: humano escolhido como "a pessoa real" | — | **25%** | 66% | 41% |

Exemplos lado a lado (contexto → humano | gemini | gpt):

- "yes pls / i miss ur face" (teste isolado) → humano: *"its the same face as yesterday"* | gemini: *"Aww stop it u are making me blush 🥺🙈 cant wait to see u later tho fr!!"* | gpt: *"Aww, I miss you too! Can't wait to see you again. When do you want to hang out?"*
- "😇😇😇😇😇 / hahahhah… / ur so funny" → humano: *"something funny? actually a couple of people have said that recently"* | gemini: *"stopppp u are making me blush 🥰 seriously tho, was it actually good or are u just easily impressed?"* | gpt: *"thanks! I try to keep things fun! Glad I could make you laugh! 😄"*
- "Hello my love" → humano: *"How are you tonight"* | gemini: *"Hey babe! 🥰 How's your day going?"* | gpt: *"Hey there! How's your day going?"*
- "will you be my valentine / now i hate you instead / oh" → humano: *"let me check my diary"* | gemini: *"jk jk of course ill be ur valentine dummy ❤️"* | gpt: *"kidding! Of course I'll be your Valentine. 💖"*
- "wait thats kinda sweet / dont get used to it / too late <3" → humano: *"finish at least 5 pages tonight"* | gemini: *"ugh whatever see you later 🙄"* | gpt: *"guess I'm stuck with you now!"*

**Os dois vícios que o usuário apontou existem, cada um num modelo:**
1. **"Gíria forçada" (gemini):** mais "lmaooo", "😭", "🥹", "stopppp", "u are making me blush", riso em 31% e emoji em 60%. Imita o registro
   de chat, mas **exagera** e acrescenta um segundo bloco com pergunta ou novo gancho.
2. **"Cordial/assistente" (gpt-4o-mini):** frase completa com maiúscula e ponto, "!" em quase tudo, "Glad I could…", "What other… do
   you have?" e entusiasmo automático (44% "engaja com entusiasmo").
3. **O ponto comum:** as duas **sempre respondem ao afeto de forma direta e calorosa**. A resposta humana típica (seguir o assunto,
   "hehe", responder o "já comeu?" literalmente, desinflar com uma piada seca) quase não aparece.

**Confiança:** alta para as diferenças de forma (tamanho, pergunta, emoji, "!"), porque são grandes e determinísticas. Média para o
movimento, que depende do Choice do Jev. Baixa para o juízo "soa humano" do Jev (ver A7).

### A7. O Jev como juiz: bom para classificar, ruim para julgar "realismo"

- No pareado, o Jev escolheu o gemini como "a pessoa real" em 75% dos contextos (probabilidade média do humano de 0,37) e como "mais
  adequado ao momento" em 89%. O gpt-4o-mini, mais formal, foi detectado (o humano venceu em 66%).
- Nos Nouls absolutos, "soa humano" deu 0,84 para o humano, 0,88 para o gemini, 0,81 para o gpt e 0,88 com briefing. **Não discrimina.**
- **Interpretação:** o Jev (e, por extensão, qualquer juiz de "qualidade") premia respostas coerentes, engajadas e pertinentes.
  A resposta humana real é muitas vezes "pior" por esse critério: ignora o carinho, muda de assunto, responde "hehe".
  **Consequência para o sistema:** não use "isso soa humano?" como filtro de qualidade. Use o Jev para **classificar** (movimento,
  intensidade, se é flerte) e compare com as **distribuições humanas** medidas aqui.

### A8. Protótipo do briefing (gemini + briefing montado com o Jev antes da resposta)

- **Ganhos:** tamanho, pergunta e emoji muito mais próximos do humano (tabela A6), nenhum "reação + gancho", a menor sobrevalidação
  do grupo e o humano vencendo em 41% no pareado, contra 25% do baseline. Ou seja, o briefing ficou mais difícil de distinguir do humano.
- **Problemas:**
  (a) **encurtou demais** (mediana de 13 contra 23), porque o nível de tamanho previsto foi obedecido de forma literal;
  (b) **copiou os exemplos** ("miss u too 😚" ×4, "ugh ur sweet" ×3, "keep dreaming" ×3, "dont expose me" ×2);
  (c) **errou o movimento:** a previsão pré-resposta (Choice) bateu com o humano em só 27,5%. Ela prevê demais "devolve provocação" (24)
  e "devolve afeto" (20), e de menos "ignora" (8 contra 25 no humano). Juntos, (b) e (c) geraram respostas incoerentes, como
  "The babies are my favorite / You know that" → "miss u too 😚".
- **Lição:** o movimento não tem uma única resposta certa (vários são válidos). Por isso o código deve **amostrar** a partir de uma
  mistura entre a distribuição do Jev e o prior empírico por gatilho (A2), e não usar o argmax. Os exemplos do briefing precisam ser de
  **estilo** (outro conteúdo, mesmo registro), com a instrução "não copie", e um Noul de coerência deve checar a resposta antes do envio.

---

## 3. Padrões "invisíveis" (o que as pessoas fazem sem perceber)

1. **Receber carinho e seguir em frente.** Um terço das respostas ao afeto ignora o carinho e continua o assunto ("quality over quantity
   babe" → "deadline today right"). Entre pessoas íntimas, isso não é frieza: o afeto fica implícito. Um bot que "agradece" todo carinho soa carente ou robótico.
2. **O "já comeu?" é o "eu te amo" do dia a dia.** O cuidado prático é o 2º movimento de afeto mais comum (14,5%) e se responde
   **literalmente** ("i had noodles", "coffee counts right", "ummm / define eat").
3. **Carinho é curto e não pergunta.** Os turnos de afeto são mais curtos e têm 3,5× menos perguntas que os neutros (no maichat).
4. **Pausa antes, rajada depois.** Diante de um flerte, a pessoa demora 2,5× mais para começar a digitar e depois escreve rápido e sem apagar.
5. **Devolver o afeto é rápido; a provocação é lenta.** "Love you too" sai em ~13 s; uma resposta espirituosa leva 22–30 s.
6. **A devolução é simétrica e às vezes atrasada.** "love you <3" → "love you too <3" (mesmas palavras, mesmo emoji, mesmo tamanho). Ou então
   a pessoa termina o fio anterior e só no turno seguinte manda "I miss you too😚😚".
7. **Encabular-se é o movimento mais "vivo".** "ugh ur sweet", "dont expose me", "HELP", "shut up": curtíssimo (mediana de 13 caracteres),
   levemente defensivo, e o outro continua flertando em 6/7 dos casos.
8. **Desinflar com humor seco.** "its the same face as yesterday", "i live with you. i have to.", "dont get used to it", "or just tired".
   Rebaixa o romantismo sem rejeitar e mantém a cumplicidade.
9. **O "te amo" também vira provocação.** "love u tho" → "love u more idiot"; "love u too / now go eat real food" → "yes sir / bye before u
   assign me chores". Afeto + insulto carinhoso ("dummy", "idiot", "brat", "Sukkel") é uma assinatura de intimidade.
10. **Os emojis de afeto e os de humor não se misturam.** 🥺😘😇😌🥰😚 × 😭😂💀🤡. E o emoji é espelhado: se o outro usa, a chance de
    usar também sobe de 10% para 24%.
11. **O afeto se concentra nas bordas da conversa.** No whatsapp, 38% dos turnos FA estão nos 3 primeiros turnos da sessão e 39% nos 3
    últimos (saudação com apelido, "slaap lekker xx"). No maichat, 5,8% dos turnos FA encerram a sessão, contra 1,1% dos neutros.
12. **O flertador insiste.** Quando a resposta desinfla muito, o outro volta a flertar em 47% dos casos. Um "deixa pra lá" não encerra o jogo.

---

## 4. Tradução para o sistema (Jev + código + briefing)

> Convenção: *U* = último turno do usuário; *H6* = últimos 6 turnos da sessão (texto juntado por turno); os níveis de intensidade são
> os da escala 0–4 usada aqui.

### J1. Detector de "modo flerte" + intensidade do usuário (em todo turno do usuário)
- **Noul** `flirt_now`: "Is the user flirting or being affectionate/romantic in their last message?"
- **Score** `user_int` (5 níveis): 0 nada · 1 carinho leve · 2 flerte ou afeto claro · 3 afeto romântico forte (amor, saudade, apelido) · 4 intenso/sexual.
- **Choice** `user_move` (as 15 classes de A1, mais "none").
- **State:** H6 + U. **Quando:** junto com as outras perguntas D, na mesma chamada, assim que U chega.
- **Código:** `flirt_mode = flirt_now ≥ 0,5 ou user_int ≥ 1,5`. Guardar `user_int` numa média móvel (EMA) de 3 turnos, que dá a
  "temperatura" da cena. Se a confiança do Score for < 0,5, usar `round(user_int)` − 0,5 (conservador).

### J2. Nível de intimidade da relação (variável lenta)
- **Score** `intimacy` (0 desconhecidos · 1 conhecidos · 2 amigos · 3 crush/flerte · 4 casal), com state = resumo da relação +
  últimos 20 turnos. Roda a cada ~10 turnos ou a cada evento de afeto.
- **Código:** define o **teto** de intensidade (`cap = intimacy`; nível 4 sexual só com opt-in explícito e intimidade 4) e os
  apelidos permitidos (intimidade ≥ 3: "amor", "bb", "vida"; intimidade 2: "bobo(a)", "idiota" carinhoso; abaixo disso, nenhum).
  O NPS mostra que, entre desconhecidos, as pessoas descontam ~0,9 nível.

### J3. Escolha do MOVIMENTO de resposta (Choice + prior empírico + amostragem)
- **Choice** `reply_move` com as classes: `continue_topic` (seguir o assunto sem comentar o afeto) · `minimal_ack` · `tease_back` ·
  `reciprocate` · `flustered_accept` · `deflate_humor` · `engage_plan` · `mock_offense` · `escalate_half_step` · `soft_dodge`.
  Instrução: "Which reply move would this character most naturally make next?" State: H6 + U + persona (1 linha) + `intimacy`.
- **Código:** `p_final = 0,5·p_jev + 0,5·prior[user_move]`, com o prior tirado de A2. Exemplos: após provocação, {tease_back .30,
  continue .25, ack .15, engage .08, deflate .07, flustered .05, …}; após "já comeu?", {continue/resposta literal .40, tease .18,
  ack .15, thanks .10, deflate .07}; após "I miss you"/"te amo", {reciprocate .35, continue .30, tease .15, deflate .10, mock .05}.
  **Amostrar** de `p_final` (não usar o argmax). Proibir `escalate` se `user_int` estiver caindo nos últimos 2 turnos ou se a intimidade for < 3.
  Se o usuário tiver ignorado o afeto do bot 2 vezes seguidas → `soft_dodge`/`continue_topic` e nada de afeto por N turnos.

### J4. Regra de calibração da intensidade
- `target = clamp(round(user_int − 0,5), 0, min(user_int + 1, cap))`, que imita o humano (−0,5 em média, 73% dentro de ±1).
- **Meio passo acima** (`user_int + 0,5` a `+1`) só quando: `flirt_mode` nos últimos 2 turnos do usuário **e** o Noul
  `user_invites_more` ("Is the user inviting more flirting?") ≥ 0,6. Com essa condição, sorteia-se `escalate_half_step` com p ≈ 0,2. É o regime que
  mais mantém o flerte vivo (80% "vivo", 67% de continuidade), mas acontece em só ~8% das respostas humanas.
- **Nunca** acima de `user_int + 1` (humanos: 3,4%).

### J5. Briefing para a "boca" (LLM)
Conteúdo mínimo, montado pelo código:
1. `move` (1 linha, em linguagem natural; ex.: "encabule-se de leve, sem agradecer").
2. `intensidade alvo` (nível + 1 frase: "tão carinhoso quanto ele, não mais").
3. `tamanho`: `clamp(0,7–1,0 × chars(U), 4, 80)` caracteres. Para afeto, o humano é simétrico: mediana R/T = 1,0; nunca menos de ~0,5× U.
4. `pergunta`: **não**, por padrão. Só se `reply_move = engage_plan` ou se o Noul P `p_question` ≥ 0,5 (humano: 5–9%).
5. `emoji`: só se U tiver emoji (então, 1 emoji do **conjunto de afeto**: 🥺😘😚😇😌🥰) ou se o `p_emoji` do P ≥ 0,4. **Nunca** 😭💀😂 em resposta a carinho.
6. **2–3 exemplos de ESTILO** de outros contextos (mesmo movimento, conteúdo diferente) + "não copie, é só o tom". Em PT-BR (inferência minha):
   - devolver o afeto: "tb te amo" · "saudade tb 🥺" · "vc tb dorme bem"
   - encabular-se: "para kkkk" · "aff vc" · "não me expõe" · "sai daqui 🙈"
   - desinflar com humor: "é a mesma cara de ontem" · "café conta?" · "vc que é carente"
   - devolver provocação: "vc queria" · "olha quem fala" · "não se acostuma"
   - seguir o assunto: responder o conteúdo literal ("comi miojo") ou puxar o fio anterior.
7. **Palavras proibidas** (vícios medidos): "aww/awww", "that's so sweet / que fofo(a) você", "you're making me blush / tô corando",
   "can't wait / mal posso esperar", "I'd love to / eu adoraria", "that means a lot / significa muito", "you always know how…",
   "meu coração", "borboletas", "honestly/sinceramente", "Glad I could…/fico feliz em…", ❤️💖✨ em série, "hahaha" + 😭 no mesmo turno, e o
   formato "reação + parágrafo novo + pergunta".
8. **Checagens pós-geração, em código:** contar "?", "!", emojis, caracteres, "\n\n" e HTML; se passar dos limites, regenerar ou truncar.
   Depois, **Noul de coerência** ("Does Sam's reply make sense as a reply to Alex's last message?") ≥ 0,7 e **Noul de cópia** (similaridade
   com os exemplos do briefing, feita em código, com Jaccard > 0,6 → regenerar).

### J6. Entrega (delivery)
- Resposta a um turno de flerte: **atraso antes do "digitando…"** ~1,5–2,5× o normal (humano: 11 s × 4 s de ocioso), depois
  "digitando…" curto (composição de ~0,19 s/caractere) e **1 bolha**. Em `reciprocate`: resposta rápida (é o mais rápido).
  Em `tease_back`/`deflate_humor`: mais lenta.
- **Devolução atrasada:** se o turno do usuário tiver afeto + outro conteúdo, responder primeiro o conteúdo e mandar o
  "saudade tb 😚" numa bolha separada logo em seguida, como no conv001 (sorteado com p ≈ 0,3, não sempre).
- **Bordas da sessão:** permitir o apelido na saudação e o afeto na despedida ("dorme bem bb", "bjs") mesmo quando a sessão foi neutra.
  É onde o afeto se concentra no WhatsApp.

### J7. Iniciativa de afeto do bot (quando o bot começa)
- **Noul** `care_opportunity` ("Did the user mention a need, effort or state — tired, hungry, exam, work — that a close person would
  check on?") → no turno seguinte ou na próxima sessão, um check-in curto ("já comeu?", "e a prova?", "orgulho de vc"). É o movimento mais
  humano e barato.
- Limite de frequência: FA em ≤ ~10% dos turnos do bot (humanos: 7–8%), episódios de 1–2 turnos e **nunca** 3 turnos seguidos de afeto
  sem conteúdo.

### J8. Detector de "flerte não correspondido" (para recuar)
- **Noul** `user_deflects`: "Is the user deflecting, ignoring or declining the bot's affection/flirting?", com state = último turno do bot
  + U. Rodar sempre que o último turno do bot for FA.
- **Código:** ≥ 0,6 → intensidade alvo = 0 por 5 turnos, movimento `continue_topic` e nada de pirraça. Duas vezes seguidas → reduzir `intimacy` em 1.
  Os humanos quase nunca recusam explicitamente; o sinal é a mudança de assunto. O bot deve ler isso e recuar em silêncio.

---

## 5. Limitações

- **maichat:** são só 42 conversas, **induzidas** (pares de conhecidos conversando para um estudo; "we're supposed to chat for the task
  btw"). Poucos pares são românticos (conv068, conv092, conv049, conv007, conv025, conv090), e conv092/068/090 respondem por ~1/3 dos turnos FA.
  O gênero dos falantes é desconhecido.
- **whatsapp_nl:** é holandês (o Jev é melhor em inglês) e foi coletado em 2012–2014, antes do emoji generalizado. A análise por relação é dominada
  por um único casal (wa_M_003, 8.654 turnos), que também concentra quase toda a insinuação sexual. A latência tem resolução de minuto.
- **NPS:** salas públicas de 2006, com muito assédio e anúncio. A amostra de flerte foi pré-selecionada por regex (recall estimado: 4,3% de
  flerte na amostra de controle sem regex), então o vocabulário do NPS é parcialmente circular. O alinhamento T→R no grupo é heurístico
  (menção ao apelido + próxima fala do alvo) e só 55/87 respostas foram confirmadas como dirigidas.
- **Rótulos do Jev** (movimento, intensidade, "vivo") não foram validados contra anotação humana. A confiança média do Choice fica em ~0,7 e
  as classes vizinhas se confundem. As classes raras (encabulada, escala, recusa) têm n = 4–7: as tendências são indicativas.
  A "continuidade" (T+2 ainda FA) só existe onde a janela do jev_base cobre o turno seguinte (n = 309/413).
- **Juiz Jev:** mostrou viés para respostas engajadas (A7); por isso as conclusões de "realismo" se apoiam na estilometria.
- **LLM:** uma amostra por contexto (seed 0, temperatura 0,8), persona mínima com nomes neutros (Sam/Alex) e dois modelos baratos. O briefing
  é um protótipo único, sem ajuste fino dos limiares. Os modelos maiores/"de roleplay" podem ter outros vícios.
- **PT-BR:** todos os equivalentes em português ("kkk", "vc", "bjs", "já comeu?", "para kkkk", "não me expõe") são **inferência minha** por
  analogia de função. Nenhum dado aqui é brasileiro. Recomendo repetir as medidas A2–A4 com um corpus de WhatsApp/Telegram em PT-BR antes de
  fixar os priors.
