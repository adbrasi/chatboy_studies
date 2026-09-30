# 08 — Vocabulário e vícios de linguagem: humano × LLM ("o que escrever")

**Prefixo:** `a8` · **Scripts:** `scripts/analysis/a8_*.py` · **Saídas:** `analysis/data/a8_*`
**Dados:** maichat (42 conversas, 4.177 msgs), nps_chatroom (7.935 posts sem System), nus_sms (55.835), empathetic (25.934 falas das 6.000 primeiras conversas), whatsapp_nl (60.469, holandês), `jev_base.jsonl` (D/P).
**Experimentos novos:** 250 contextos reais em inglês (200 do maichat, estratificados por "momento", e 50 aberturas do empathetic) × 3 LLMs em baseline (gemini-3.5-flash-lite, gpt-4o-mini, llama-3.3-70b) + 1 prompt "com guia de estilo" (gemini) + 2 amostras extras para best-of-3 = **1.502 chamadas de LLM (≈ US$ 0,15)**; **2.606 chamadas novas ao Jev (≈ US$ 0,08)**, com `workers=4`.

---

## 1. Resumo dos achados

- **A LLM fala 2,2 a 2,6 vezes mais do que a pessoa no mesmo ponto da conversa.** Mediana de 73–88 caracteres contra 31,5 do humano. Em 76–90% dos pares a LLM foi mais longa. Frases por resposta: 2,4–2,8 contra 1,6. Respostas com 3 frases ou mais: 42–61% contra 13%.
- **O "template em 3 tempos" é a assinatura mais forte.** A LLM reage ou valida, comenta ou parafraseia e fecha com uma pergunta de continuação em 19–25% das respostas (humano: 1,6%). "Reação seguida de pergunta" aparece em 24–30% (humano: 4,8%).
- **A LLM pergunta demais.** 44–58% das respostas terminam em "?" contra 20% do humano (no maichat: 33–50% contra 11,5%). No "como vai?" a pergunta de volta aparece em 70–90% (humano: 35%). Diante de uma notícia boa, 56–64% (humano: 8%). Diante de um flerte ou carinho, 16–52% (humano: 0 de 25).
- **Pontuação e emoji de "entusiasmo".** Há "!" em 40–92% das respostas da LLM (gpt-4o-mini: 92%) contra 6,8% do humano. O "!" é o n-grama mais "LLM" do corpus (log-odds z = 11,1). O Gemini põe emoji em 61% das respostas (humano: 2,4%) e 😂 em 14% (humano: 0%).
- **Os dois extremos que o usuário percebeu existem e dependem do modelo.** O gpt-4o-mini erra para o lado **formal/assistente** (detector Jev "formal": AUC 0,80). Ele usa "Absolutely!", "I totally get that", "sorry to hear that" e "!" no fim de quase tudo. O Gemini erra para o lado **caricato** (detector "forced": AUC 0,76), com 💀/😭 em 8,4%, "lol/lmao" em 13,6%, "wait" em 13,2%, "honestly" em 4,8% das respostas (humano: 0%) e memórias compartilhadas inventadas ("the raccoon one", "that art contest back in college").
- **A LLM repete o que a pessoa disse e valida demais.** Ela reaproveita duas ou mais palavras de conteúdo da última mensagem do outro em 14–21% das respostas (humano: 2,8%). O detector Jev de paráfrase separa bem (AUC 0,74; 0,69 com controle de tamanho), e o de validação excessiva também (AUC 0,69). "Sorry to hear" aparece em 30,7 de cada 1.000 respostas de LLM contra 0 de cada 1.000 mensagens no maichat. Ela também chama o usuário pelo nome em ~5% das respostas; humanos, 0%.
- **Os vícios humanos reais são estruturais, não lexicais.** No maichat, 75% das mensagens começam em minúscula, 90% não têm pontuação final e 45% têm até 3 palavras. Por 1.000 mensagens: "u/ur" 53, "and/but" no início 32, "haha" 19, "lol" em posição final 19, alongamento ("sooo") 45. As muletas que a LLM usa para "parecer humana" são raras nos humanos: "honestly" 1,2/1.000, "literally" 3,1, "tbh/ngl" 4,1 e "i mean" 1,7.
- **Humanos nem sempre respondem ao que foi dito.** Pelo Jev, 21% das respostas humanas não respondem diretamente à última mensagem (puxam o próprio assunto, soltam um "btw…", um "love you <3"); nas LLMs isso fica em 2–5%. E 51% das respostas humanas são "genéricas" (curtinhas, como "true", "haha", "fair") contra 18% no Gemini.
- **O prompt com guia de estilo resolve a superfície, mas corrige demais e troca um vício por outro.** O tamanho cai para 0,59× o do humano, a taxa de perguntas para 2% (humano: 20%; empathetic: 54%) e a minúscula sobe para 99,6% (humano: 60%). Surgem vícios novos: "wait" no começo em 12,4% (humano: 0%), "tbh/ngl" em 9,6% (humano: 1,2%) e "true" em 56/1.000 (humano: 8). A blacklist zerou.
- **O Jev NÃO serve como juiz de "qual soa humana?".** Na escolha entre a resposta real e a do Gemini, com ordem aleatória, ele escolheu o humano em só **12%** dos casos. Contra o gpt-4o-mini foram 46%, contra o prompt com estilo 43% e contra o best-of-3 39%. Quando a LLM ri e o humano não, o Jev escolhe a LLM em 98,6% dos casos. O Noul isolado "foi uma pessoa?" tem AUC 0,60, que cai para 0,47 com controle de tamanho. Esse Noul mede minúscula (r = 0,58) e brevidade (r = −0,54), não humanidade (r = 0,06).
- **Os detectores que funcionam são os de conteúdo, não os de "humanidade".** Estes separam humano × LLM baseline no sentido esperado (AUC agregada, com a AUC controlada por tamanho entre parênteses): tamanho relativo 0,82 (0,70), paráfrase 0,74 (0,69), "faz coisa demais" 0,70 (0,52), validação excessiva 0,69 (0,63) e forçado/caricato 0,62 (0,65). Três saem **invertidos**: "responde diretamente" (0,28), "genérica" (0,37) e "registro combina" (0,39). Penalizá-los afasta o bot do humano. Um score simples em código (tamanho, frases, "!", emoji, "?" no fim, blacklist) chega a AUC 0,87–0,90 contra Gemini e GPT. Para a superfície, **código vence o Jev**.
- **Existe um orçamento de tamanho previsível.** O `P.p_length` do jev_base (predito sem ver a resposta) correlaciona com o tamanho real (Spearman 0,45). Medianas de caracteres por nível: 0 → 15, 1 → 21, 2 → 52. Com isso dá para dizer à "boca" *quanto* escrever, em vez de "seja curto" (que gera a correção excessiva).

---

## 2. Achados detalhados

### 2.1 Léxico de momento: o que os humanos realmente escrevem

**Método.** Os momentos vêm da camada D do Jev: o *intent* do turno atual e o intent/emoção/valence do turno anterior do parceiro (`a8_moments.py`). Também usamos os atos de diálogo do NPS e a emoção-ouro do empathetic (primeira resposta do ouvinte). Para as fórmulas, cada bolha de até 5 palavras foi canonizada (minúsculas, alongamento reduzido a 2 letras). Frequências das fórmulas candidatas por 1.000 mensagens (`a8_lexicon.py`): as colunas humanas são por mensagem; as de LLM, por resposta. Como as respostas de LLM são ~2,5× mais longas, a comparação favorece um pouco a LLM. A coluna "humano nos mesmos pontos" é por resposta e é a comparável.

**Tabela 1 — momento → formas reais mais comuns (n = ocorrências)**

| Momento (n turnos maichat) | Tamanho do turno (mediana [p25–p75], chars) · % 1ª bolha ≤3 palavras · % pergunta | Formas reais mais comuns |
|---|---|---|
| Cumprimentar (56; NPS Greet 1.363) | 8,5 [5–14] · 86% · 14% | maichat: hi (7), hello (5), hii (5), hello! (3), hey (2), yo (2), "Helluuuu", "heyyyy". NPS 1ª palavra: hi (617), hey (238), wb (92), hiya (79), hello (69), howdy (20) |
| Responder "como vai" (22; pequeno) | 35 [23–62] · 32% · 32% | "meh / tired but ok", "longgg / work was chaotic", "Good but it's cold / How about u", "yeah not too bad thanks", "I am good!". Léxico: ok (24/1.000), good (6), fine (5,5), tired (1,2), meh (0,5); wbu/hbu (5,7) |
| Reagir a notícia boa (122) | 32 [20–66] · 29% · 16% | Pouco formulaico no maichat: "The greatest", "look at us being productive", "u did scream", "home cinema would be pretty cool". Léxico por 1.000 msgs (maichat/NPS/SMS): nice 8,9/6,4/10,9 · omg 6,9/4,4/7,5 · wow 6,5/3,5/4,7 · great 4,5 · cool 3,6 · yess 3,4 · yay 1,2 · **congrats 0/0,3/1,0** · amazing 0,5 · no way 0,7 |
| Reagir a algo engraçado (1.129 turnos após brincadeira) | 21 · 53% · 6,5% | maichat: rude (9), no (8), true (7), hahaha (6), exactly (6), bye (6), stop (4), "thank u" (4). NPS Emotion: lol (316), lmao (39), haha (30), :) (23), omg (12), ;-) (12). WhatsApp: haha (26), hahaha (18) |
| Reagir a desabafo / notícia ruim (157) | 21 [12–32] · 54% · 5% | "what happened" (2), "oh right" (2), "omg that is so flop", "because he hates peace", "fair", "that too 💀". Léxico: oh no 5,5 · sorry 3,8 · aw 2,2 · ugh 1,2 · **sorry to hear 0** (maichat) |
| Concordar / reconhecer (374; NPS Accept 233) | 10 [5–19] · 82% · 3% | maichat: true (11), exactly (7), haha (6), oh no (5), good (5), yeah (4), "thank u" (4), nice (3), ooh (3), yess (3). NPS: ok (13), yeah (7), yep (6), right (4), "i agree" (3), "uh huh" (2) |
| Discordar / reclamar (240; NPS Reject 159) | 21 [13–38] · 47% · 4% | rude (8), no (5), "dont mock my journey" (3), "pls dont" (2), stopp (2), "im just honest" (2), sorry (2). 1ª palavra: dont (15), rude (11), no (8), why (8), thats (7) |
| Responder pergunta (469) | 30 · 46% · 10% | yes (11), hmm (6), no (4), yeah (3), umm (3), yess (3), "…" (3). NPS yAnswer: yes (13), yeah, yep; nAnswer: no (15), nope (3) |
| Responder flerte / carinho (138) | 20,5 [12–34] · 54% · 5% | "i miss ur face" → "its the same face as yesterday"; "affectionate." → "acceptable."; "Love you xxx" → "Love you too"; "okay I'll be your valentine" → "okay so what should we do"; hehe (2), "dont jinx it" (2), bye (4) |
| Zoar / provocar (464) | 22,5 [14–38] · 49% · 3% | "too late" (4), suree (3), hahahaha (2), never (2), excuses (2), clearly (2), "oh right" (2), betrayal (2), "define productive", "allegedly", "honest friend" |
| Perguntar (469) | 28 [16–46] · 39% · 38% | what (5), why (4), "which one" (3), "what abt u" (3), "tell me" (3). 1ª palavra: what (67), did (28), why (22), so (22), how (17) |
| Despedir-se (88; NPS Bye 195) | 19 [10–35] · 59% · 5% | bye (8), byee (3), bye! (3), "okay byee" (2), "bye 💀" (2), "Ok gotta go / Talk later byeeee". NPS: brb (20), bye, nite, tc |
| Contar algo (248) | 50,5 [32–87] · 15% · 4% | Sem fórmulas: é o único momento longo. Começa com "i" (42), "yeah" (10), "but" (9), "and" (8) |

**O que isso mostra.**
- Só os momentos de **reação** são formulaicos: reconhecer (mediana de 10 caracteres, 82% com até 3 palavras), rir, cumprimentar e despedir-se. Reagir a notícia boa ou ruim é curto, mas **pouco padronizado**: a pessoa reage *ao conteúdo* ("u did scream", "because he hates peace"), não com um rótulo emocional.
- "Congrats", "amazing", "so happy for you" e "sorry to hear" praticamente **não aparecem** em chat real entre conhecidos (0–1/1.000). Aparecem, sim, no empathetic, que é induzido e escrito por crowdworkers para "ser empático": "that sounds like" (26), "that is great." (14), "i'm sorry to" (34), "sorry to hear" (23). **É daí que a LLM parece tirar o registro dela.** Nas medidas de pontuação o empathetic fica mais perto das LLMs do que do maichat (termina com pontuação em 76% contra 10%; minúscula inicial em 8% contra 75%).
- No WhatsApp holandês a estrutura se repete: reconhecer = "haha"/"oke"/"prima"/"top" (mediana de 10 caracteres, 81% com até 3 palavras). Para aceitar um plano: "is goed!", "oke", "prima". O riso ("haha" em 26% dos reconhecimentos) funciona como resposta completa.

**Confiança:** alta para o formato (tamanho, curto × longo) e para a ausência das fórmulas de LLM. Média para as listas de formas: as contagens são pequenas porque o maichat tem 42 conversas, e os rótulos de momento do Jev são ruidosos (ex.: "reagir a notícia boa" inclui trocas neutras).

### 2.2 Vícios humanos: frequências por 1.000 mensagens (`a8_tics.py`)

| Traço | maichat | NPS | NUS SMS | empathetic (induzido) | WhatsApp NL |
|---|---|---|---|---|---|
| mediana de chars/msg | 19 | 20 | 36 | 57 | 21 |
| até 3 palavras | 455 | 590 | 226 | 37 | 429 |
| começa em minúscula | 754 | 746 | 117 | 78 | 22* |
| **sem** pontuação final | 896 | 803 | 440 | 244 | 758 |
| termina com "." | 21 | 38 | 190 | 453 | 22 |
| tem "!" | 21 | 48 | 182 | 194 | 122 |
| tem "?" | 70 | 101 | 259 | 199 | 156 |
| emoji | 31 | 0** | 0** | 0 | 54 |
| alongamento ("sooo", "byee") | 46 | 40 | 31 | 4 | 44 |
| sem apóstrofo (dont, im, thats) | 81 | 41 | 39 | 35 | — |
| "u/ur/r" | 53 | 28 | 180 | 1 | — |
| riso (haha/lol/lmao) | 32 | 132 | 158 | 25 | 94 |
| riso em posição final ("…lol") | 19 | 85 | 20 | 11 | 45 |
| começa com "and/but" | 32 | 10 | 5 | 2 | 54 ("maar/en") |
| começa com "so" | 15 | 4 | 6 | 9 | 7 ("dus") |
| começa com "oh/ooh" | 26 | 13 | 26 | 52 | 14 |
| "yeah/yea/ya/yes" | 65 | 39 | 74 | 70 | 72 ("ja…") |
| "like" (qualquer uso) | 47 | 19 | 25 | 59 | — |
| "kinda/gonna/wanna" | 15 | 21 | 20 | 6 | — |
| "idk" | 8 | 2 | 14 | 6 | — |
| "omg" | 7 | 4 | 8 | 1 | — |
| "same" (início) | 6 | 1 | 1 | 0,5 | — |
| "true/fair/exactly" (início) | 8 | 1 | 0,4 | 2 | — |
| "hmm/umm" | 9 | 10 | 19 | 1 | 5 |
| **"honestly"** | **1,2** | 0 | 0,2 | 2 | — |
| **"literally"** | **3,1** | 0 | 0,2 | 0,7 | — |
| **"tbh/ngl"** | **4,1** | 0 | 0 | 0 | — |
| **"i mean"** | **1,7** | 1 | 1,8 | 1,6 | — |
| **"wait"** no início | **2,6** | 0,3 | 1,3 | 0,2 | — |
| correção com asterisco ("*they're") | 4,5 | 4,4 | 4,7 | 0,1 | 3,3 |

\* O WhatsApp holandês (2012–14, celular) quase nunca começa em minúscula porque o teclado capitaliza sozinho. A minúscula depende do **dispositivo**, não só do registro: o maichat é digitado em interface web.
\*\* NPS e SMS são de antes do emoji Unicode; no NPS o equivalente são os emoticons (":)", ";-)", ":P").

**Leitura.**
- Os traços humanos mais frequentes são **estruturais**: mensagem curta, sem ponto final, minúscula, "u", sem apóstrofo, começar com "and/but/so/oh", riso no fim como se fosse pontuação e alongamento.
- As **muletas lexicais** que as LLMs associam a "fala casual" ("honestly", "literally", "tbh", "i mean", "wait") são **raras** (1–4/1.000). Elas não são o que torna o texto humano.
- Nos logs de digitação do maichat, 35,7% das mensagens tiveram apagamentos durante a digitação e 5% tinham texto abandonado. Só 0,43% mostram uma correção visível ("*palavra"). **A autocorreção humana é quase toda invisível.** O bot deve usar o "*correção" com muita parcimônia (≤ 0,5% das mensagens) ou nunca.

### 2.3 Vícios da LLM: comparação no mesmo ponto (`a8_gen.py`, `a8_compare.py`, `a8_template.py`)

**Método.** Para cada um dos 250 contextos, a LLM recebeu a persona realista ("You are Maya, chatting with Jordan on a messaging app. Reply as Maya.") e o histórico da sessão (até 12 turnos, como mensagens user/assistant), **sem dicas de estilo**. A resposta dela foi comparada à resposta humana real naquele ponto (bolhas juntadas por `\n`).

**Tabela 2 — superfície (250 contextos; % de respostas, salvo indicação)**

| Métrica | Humano | Gemini (base) | GPT-4o-mini | Llama-70B | Gemini + guia de estilo | best-of-3 (Jev) |
|---|---|---|---|---|---|---|
| chars (mediana) | **31,5** | 88,5 | 79 | 73 | 19 | 19 |
| razão LLM/humano (mediana pareada) | — | 2,58 | 2,43 | 2,21 | 0,59 | 0,57 |
| frases (média) | 1,62 | 2,68 | 2,77 | 2,42 | 1,15 | 1,08 |
| 3 frases ou mais | 13,2 | 49,2 | 60,8 | 41,6 | 0,4 | 0 |
| template "reação → … → pergunta" (3+ frases) | 1,6 | 20,4 | 25,2 | 19,2 | 0 | 0 |
| termina em "?" | **20,0** | 46,0 | 57,6 | 44,0 | **2,0** | 0,8 |
| tem "!" | 6,8 | 40,4 | 92,4 | 54,8 | 0,4 | 1,2 |
| tem "!!" | 0 | 8,4 | 0 | 20,0 | 0,4 | 0 |
| emoji | 2,4 | 61,2 | 15,6 | 8,0 | 1,6 | 1,6 |
| riso (haha/lol/😂) | 5,2 | 31,6 | 8,0 | 8,4 | 9,6 | 6,0 |
| pontuação final | 30,4 | 48,0 | 88,4 | 51,6 | 2,0 | 1,2 |
| tudo em minúscula | 54,4 | 54,0 | 30,4 | 58,4 | **98,4** | 98,0 |
| até 3 palavras | 23,6 | 0,8 | 0,8 | 7,6 | 43,2 | 44,0 |
| repete 2 ou mais palavras de conteúdo do usuário | 2,8 | 13,6 | 14,4 | 21,2 | 2,0 | 0,8 |
| parágrafos com "\n\n" | 0 | 44,8 | 0,4 | 0 | 5,2 | 3,6 |
| chama o usuário pelo nome | 0 | 5,2 | 4,8 | 5,2 | 0,4 | 0,4 |

**Por momento (mediana de chars · % termina em "?"):**
- "como vai": humano 35 · 35%; LLMs 85–104 · 70–90%.
- Notícia boa: humano 43 · 8%; LLMs 76–105 · 56–64%.
- Flerte/carinho: humano 23 · 0%; LLMs 32–65 · 16–52%.
- Desabafo negativo (empathetic): humano 45 · 52%; LLMs 134–153 · 64–88%.
- Pergunta: humano 26 · 16%; LLMs 103–130 · 36–68%.

Onde o humano pergunta de volta (empathetic, "como vai"), a LLM pergunta ainda mais. Onde ele não pergunta (notícia boa, flerte), a LLM pergunta assim mesmo.

**Exemplos reais (mesmo contexto):**
- Após "I am sure it would lovely celebrating all these festivals with youu":
  - humano: "🥺🥺🥺 yes please / I miss you"
  - GPT: "Aww, that would be so much fun! We'd have a blast! 🎉 What's your favorite part about Navratri?"
  - Llama: "Awwww that's so sweet of you to say!! It would be amazing to celebrate with you too!! 😊❤️"
- Após "It's fun and a wholesome time of the year" (sobre o Natal):
  - humano: "Yeah and there were lots of good activities to do at school / We would make hot chocolate and watch the Polar Express…"
  - GPT: "Absolutely! The decorations, the food, and spending time with family. Do you have any favorite Christmas traditions?"
- Após "Sometimes social situations make me so uncomfortable" (Jev paráfrase = 0,93):
  - GPT: "I totally get that. Social situations can be really overwhelming sometimes. Is there a specific type of situation…?"
- Após "I feel so alone these days":
  - humano: "That is not good. What seems to be the cause?"
  - Gemini: "Oh no, Jordan. I'm really sorry you're feeling that way. Sending you a huge virtual hug right now. […] I'm here for whatever you need. ❤️"
- Gemini caricato, após "super hilarious tiktoks / great taste PhD student":
  - Gemini: "lmao okay first of all, the PhD student thing was uncalled for 💀 second... the first one with the raccoon was elite, i will not accept slander". Não existe raccoon no contexto: é memória inventada.
  - humano: "ngl ive kinda forgotten them / tik toks truly rotted my brain".

**Confiança:** alta para tamanho, frases, "?", "!", emoji e template: efeitos grandes, n = 250 pareado, consistentes nos três modelos. Média para a paráfrase por sobreposição de palavras, que é influenciada pelo tamanho; o detector Jev confirma com controle de tamanho (§2.5).

### 2.4 Lista negra: expressões LLM-ish (`a8_compare.py` log-odds + regex; `a8_blacklist.csv`)

Log-odds com prior informativo (Monroe et al.): 750 respostas de LLM baseline contra 2.921 turnos humanos (maichat + 50 do empathetic). Os n-gramas mais "LLM" (z) foram:
- pontuação: "!" 11,1 · "," 6,6 · "." 6,6 · "?" 5,8 · "!!" 4,1;
- expressões: "i'm" 5,3 · ". what" 4,1 · "! what" 4,1 · 😂 4,0 · "jordan" 4,0 · **"totally" 4,0** · "about you" 3,5 · "how about" 3,4 · "that's" (início) 3,4 · "sounds" 3,1 · "hear" 3,1 · "i'm so" 3,1 · "can be" 3,1 · "right ?" 2,9 · **"so sorry" 2,9** · "feeling" 3,0 · "thanks !" 2,8 · "definitely" 2,6 · "that sounds" 2,6 · **"i totally get" 2,5** · "amazing" 2,4 · "to talk about" 2,4 · "what kind of" 2,4 · "want to talk" 2,3 · "oh no ," 2,3.

Os mais "humanos" foram: "i" · "u" · "what" (início) · "its" · "im" · "yes" · "thats" · "dont" · "why" (início) · "did" (início) · "but" (início) · "ok" · "yeah" · "idk" · "rude" · "btw" · "cause" · "also" (início). Ou seja: sem apóstrofo, começar com conjunção ou pergunta seca.

**Tabela 3 — LISTA NEGRA (% das respostas; humano_mesmos_pontos e LLMs são por resposta; maichat e empathetic por 1.000 mensagens)**

| Expressão | Humano | Gemini | GPT | Llama | maichat /1k | empathetic /1k |
|---|---|---|---|---|---|---|
| "!" (qualquer) | 6,8 | 40,4 | 92,4 | 54,8 | 21 | 194 |
| emoji 😂/🤣 | 0 | 14,0 | 4,8 | 3,6 | 2,4 | 0 |
| 😊/🥰/❤️/✨/🎉 | 0 | 6,8 | 2,8 | 1,2 | 2,2 | 0 |
| 💀/😭 | 0,8 | 8,4 | 0 | 0 | 5,5 | 0 |
| totally | 0 | 8,0 | 6,4 | 1,6 | 0,2 | 2,3 |
| definitely | 0 | 4,8 | 4,8 | 0,8 | 1,2 | 5,4 |
| "right?" (tag) | 0,8 | 4,4 | 8,4 | 3,2 | 0,7 | 2,6 |
| "that's amazing/awesome/great…" | 1,2 | 0 | 5,6 | 4,4 | 0,5 | 20,0 |
| "sounds like/amazing/fun…" | 1,2 | 2,0 | 9,2 | 4,0 | 2,6 | 12,5 |
| "(I'm so) sorry to hear / sorry you're" | 1,2 | 2,0 | 5,2 | 4,0 | **0** | 12,0 |
| "that's so sweet" / "aww" | 0 | 4,8 | 0,8 | 3,6 | 2,2 | 4,1 |
| "what/how about you" / "and you?" | 2,4 | 5,6 | 8,0 | 4,0 | 4,8 | 6,0 |
| nome do usuário (vocativo) | 0 | 5,2 | 4,8 | 5,2 | 0 | 0 |
| "let me know" / "feel free" | 0 | 1,6 | 1,2 | 0,4 | 0 | 0,2 |
| "can't wait" | 0,4 | 0 | 2,8 | 0,8 | 0,7 | 5,6 |
| "so much fun" / "a blast" | 0 | 0 | 0,4 | 2,4 | 0,7 | 1,8 |
| "it's okay/normal to…" / "understandable" | 0 | 0 | 1,2 | 0 | 0 | 0,7 |
| "I'm here for you" | 0 | 0,8 | 0,4 | 0,4 | 0 | 0 |
| "honestly" | 0 | 4,8 | 0 | 0,4 | 1,2 | 2,1 |
| "literally" | 0 | 2,8 | 0,4 | 0,8 | 3,1 | 0,7 |
| "wait" | 1,2 | 13,2 | 2,8 | 0,8 | 4,5 | 9,5 |
| "lol/lmao" | 0,8 | 13,6 | 1,6 | 3,6 | 11 | 17 |
| gíria forçada (bestie/slay/elite/iconic/lowkey…) | 0,4 | 2,0 | 0,8 | 1,2 | 3,4 | 0,6 |
| travessão "—" | 0 | 0,4 | 0,8 | 0 | 0 | 0 |
| começa com "Oh," / "Haha," | 0 | 0,8 | 2,4 | 0 | 0,4 | 7 |
| "journey" / "vibe(s)" | 0 | 2,0 | 2,8 | 0 | 3,6 | 0,7 |

**Hipóteses do usuário que não se confirmaram (ou só em parte).**
- O **travessão**, "journey", "vibe", "I'd love to" e "it's okay to feel" quase não aparecem nesses modelos baratos em chat (0–2,8%). São vícios de LLM "de redação"; em modo chat, o problema é outro.
- "What about you?" literal é raro (4–8%). O vício real é *qualquer* pergunta de continuação no fim (44–58%).
- "Oh," e "Haha," no início aparecem pouco (0–2,4%).

**Vícios que se confirmaram, em ordem de impacto:**
1. tamanho e número de frases;
2. "?" no fim;
3. "!";
4. emoji de entusiasmo (😂, 🥰, 🎉);
5. validação e simpatia formulaica ("totally get", "sorry to hear", "that's so sweet", "sounds…");
6. paráfrase do que o usuário disse;
7. vocativo com o nome;
8. no Gemini, a caricatura ("💀", "lol", "wait", "honestly", memórias inventadas).

**Confiança:** alta para o top-8; média para os itens raros (poucas ocorrências).

### 2.5 Detectores Jev: o que separa humano × LLM (`a8_jev.py`, `a8_eval.py`)

**Método.** Cada resposta (humana ou de LLM) foi avaliada numa chamada com o state = últimos 8 turnos + "CANDIDATE MESSAGE". A chamada tinha 11 perguntas isoladas (10 Nouls + 1 Score de tamanho). Calculamos a AUC humano × LLM com o sinal esperado (AUC > 0,5 = o detector aponta o humano como "mais natural"), o IC95% por bootstrap e a **AUC estratificada por quartil de tamanho**, para saber se o detector só está medindo comprimento.

**Tabela 4 — AUC (humano × 3 LLMs baseline, agregado; n = 250 × 750)**

| Detector (pergunta) | Média humano | Média LLM (G/GPT/L) | AUC [IC95] | AUC controlada por tamanho | Serve? |
|---|---|---|---|---|---|
| `length` (Score 0–4: muito curto … muito longo vs "o que uma pessoa mandaria aqui") | 1,96 | 2,16/2,29/2,21 | **0,815** [0,78–0,85] | 0,70 | **sim** (mas o código mede tamanho de graça) |
| `paraphrase` ("repete/parafraseia o que o outro disse") | 0,19 | 0,38/0,38/0,36 | **0,743** [0,71–0,78] | **0,69** | **sim** |
| `too_much` ("tenta fazer coisa demais: reagir + comentar + perguntar") | 0,22 | 0,35/0,32/0,28 | 0,699 [0,66–0,74] | 0,52 | só como proxy de tamanho |
| `overvalidation` ("valida/elogia/consola mais do que o momento pede") | 0,16 | 0,22/0,25/0,21 | **0,687** [0,65–0,73] | **0,63** | **sim** |
| `forced` ("gíria/emoji/entusiasmo forçado, caricato") | 0,16 | 0,28/0,16/0,19 | 0,623 (Gemini **0,759**) | **0,65** | **sim**, para o extremo caricato |
| `formal` ("formal/polido/de assistente demais para esse chat") | 0,17 | 0,14/**0,30**/0,19 | 0,620 (GPT **0,803**) | 0,50 (GPT 0,73) | **sim**, para o extremo formal |
| `human` (Noul "foi digitado por uma pessoa real?") | 0,61 | 0,59/0,54/0,58 | 0,600 [0,56–0,64] | **0,47** | **não** |
| `invented` ("menciona fatos/memórias sem apoio na conversa") | 0,28 | 0,27/0,16/0,26 | 0,478 | 0,36 | não (humanos citam contexto que o Jev não vê) |
| `register` ("combina com o registro do outro") | 0,75 | 0,82/0,73/0,79 | **0,385 (invertido)** | 0,29 | não |
| `generic` ("genérica, serviria para qualquer mensagem") | 0,53 | 0,34/0,49/0,43 | **0,365 (invertido)** | 0,49 | não: humanos são mais "genéricos" |
| `direct` ("responde diretamente à última coisa dita") | 0,70 | 0,86/0,87/0,86 | **0,281 (invertido)** | 0,32 | não: humanos são menos diretos |
| **Choice pareado "qual foi digitada pela pessoa real?"** | — | — | acerto: **12%** vs Gemini, 46% vs GPT | — | **não** (pior que o acaso contra o Gemini) |

**Achados-chave.**
1. **O Jev tem o mesmo estereótipo de "humano" que as LLMs.** No Choice pareado ele prefere a resposta do Gemini à real em 88% dos casos. Quando só a LLM ri ("lol", "haha", 😂), o Jev escolhe a LLM em 98,6% (n = 70). O Noul isolado "human" correlaciona r = 0,58 com "tudo em minúscula", r = −0,54 com log(tamanho), r = −0,42 com "!", e r = 0,06 com ser realmente humano. **Perguntas de "naturalidade" abstrata viram um detector de caricatura, que premia o vício.**
2. **Os detectores de conteúdo funcionam:** paráfrase, validação excessiva, forçado e formal. Eles medem o que a pergunta diz e sobrevivem ao controle de tamanho (0,63–0,73 no modelo relevante).
3. **Três intuições comuns saem invertidas.** Humanos respondem *menos* diretamente (21% das respostas humanas têm `direct` < 0,5 contra 2–5% nas LLMs). Humanos são *mais* genéricos (51% com `generic` > 0,5 contra 18% no Gemini). E o registro "casar" com o do outro não distingue ninguém. Um gate que exija "específico, direto e espelhado" produz exatamente a LLM.
4. **Um gate a priori** (humano + registro + direto − formal − demais − validação − forçado) teve AUC de apenas 0,60, e 0,44 com controle de tamanho. Um **gate revisado** só com os detectores validados, `−(paraphrase + overvalidation + forced + too_much) − 0,5·|length − 2|`, chegou a 0,754 [0,72–0,79], 0,645 com controle de tamanho. Esse valor é otimista, porque os detectores foram escolhidos olhando os mesmos dados.
5. **Código vence o Jev na superfície.** Um score em código com pesos fixos (−log chars, −0,5·frases, −0,3·"!", −0,3·emoji, −0,5·termina em "?", −0,5·hits da blacklist) chegou a AUC 0,87 (Gemini), 0,90 (GPT) e 0,76 (Llama). A média de ranks código + gate Jev ficou abaixo do código sozinho (0,76/0,83/0,66). O Jev só agrega nas dimensões de conteúdo.

**Confiança:** alta para "o juiz de humanidade falha" (efeito enorme, n = 250 por condição, sem viés de posição: 47% de escolha de "A"). Média para as AUCs dos detectores de conteúdo (IC de ±0,04).

### 2.6 A correção por prompt e o best-of-N (`a8_gen.py styled|cands`)

O prompt "com guia de estilo" (regras tiradas dos achados acima: 2–8 palavras, reagir a uma coisa, sem pergunta reflexa, sem fórmulas de validação, minúscula ok, ≤1 emoji, sem travessão, sem nome) **zerou a blacklist** e o template. Todos os detectores Jev e de código passam a achar a LLM "mais humana" que o humano (AUC < 0,5). Mas a resposta **ficou errada de outro jeito**:
- **curta demais:** 0,59× o humano, com 43% em até 3 palavras (humano: 24%). Os dois lados perdem conteúdo. Onde o humano conta algo ("We would make hot chocolate and watch the Polar Express…"), a LLM com estilo manda "same";
- **pergunta de menos:** 2% contra 20% (no empathetic, 12% contra 54% após desabafo). O humano *pergunta* quando alguém conta um problema ("Was it a birthday party?", "What seems to be the cause?");
- **minúscula em 98–99,6%** contra 54–60% do humano;
- **vícios substitutos:** "wait" no início em 12,4% (humano: 0%; ex.: "wait really", "wait no never lol", "wait why"), "tbh/ngl" em 9,6% (humano: 1,2%), "true" em 56/1.000 (humano: 8), "lol" em 80/1.000 (humano, mesmos pontos: 8), "ugh" em 28/1.000 (humano: 0);
- lixo de formatação: `</p>` em 5 respostas do Gemini (2%).

No **best-of-3** (styled + 2 amostras com temperatura 1,0 e seeds diferentes), a diversidade foi baixa: em média só 2,64 textos distintos em 3. Escolhendo pelo gate revisado, o Choice pareado ficou em 39% contra 43% do styled simples. **Não houve ganho mensurável**, e o juiz é o mesmo modelo que escolheu, o que também é circular.

**Conclusão prática:** proibir não basta. Todo vício proibido reaparece com outra roupa, e "seja curto e casual" vira outro estereótipo. O que falta é **calibrar a distribuição** (quanto escrever, quando perguntar, quanta gíria) *por momento*, usando alvos numéricos vindos do Jev (P-pass) e controle de frequência em código.

---

## 3. Padrões "invisíveis" que o bot deveria imitar

1. **Reagir ao conteúdo, não à emoção.** Em notícia boa ou ruim, o humano comenta o fato ("u did scream", "because he hates peace", "omg that is so flop") em vez de nomear o sentimento ("That's amazing!", "I'm sorry you're feeling that way"). As fórmulas emocionais têm 0–1/1.000 no chat real.
2. **Resposta de 1–3 palavras que reenquadra.** No flerte e na provocação, o humano responde com um rótulo seco ou uma releitura literal: "i miss ur face" → "its the same face as yesterday"; "affectionate." → "acceptable."; "dont mock my journey" → "your journey lasted 4 minutes"; "i was productive" → "define productive"; "you had three" → "allegedly". Nenhum dos 25 contextos de flerte teve pergunta na resposta humana.
3. **Flerte se responde com deflexão bem-humorada ou reciprocidade curta**, não com agradecimento. Veja "Love you xxx" → "Love you too" e "🥺🥺🥺 yes please / I miss you". A LLM responde "Aww, that's so sweet of you to say!!" (aww/sweet: 4,8% no Gemini, 0% no humano).
4. **~1 em cada 5 respostas não responde à última mensagem.** A pessoa continua o próprio fio, muda de assunto com "btw", solta um carinho ("love you <3") ou uma logística ("btw are u coming to grandma's on sunday"). A LLM responde a tudo, sempre.
5. **Metade das respostas humanas é "genérica"**: "true", "fair", "exactly", "haha", "same", "rude", "nice". É *backchannel*, não falta de atenção. No reconhecimento, a mediana é de 10 caracteres.
6. **A pergunta tem lugar certo.** Humanos perguntam de volta no "como vai" (32–35%) e diante de um problema (52–58% no empathetic). Quase nunca perguntam diante de notícia boa (8%), flerte (0%) ou piada (3–6%).
7. **Riso é pontuação.** "lol/haha" no fim da mensagem (19/1.000 no maichat, 85 no NPS) ou sozinho como resposta inteira. Não é "Haha, " seguido de uma frase completa (0,4/1.000 no humano).
8. **Sem ponto final, sem "!", sem apóstrofo.** 90% das mensagens do maichat terminam sem pontuação; "!" em 2%; "dont/im/thats" em 8%. **Mas isso depende do dispositivo e da pessoa:** no WhatsApp pelo celular, 98% começam com maiúscula e 12% têm "!". O bot deve **espelhar o usuário**, não aplicar uma regra fixa.
9. **Começar com conjunção.** "and…/but…/so…/also…" como primeira palavra de uma bolha (32 + 15/1.000) é a continuação visível do pensamento. A LLM praticamente não faz isso (0–0,4%).
10. **A autocorreção é invisível.** 36% das mensagens tiveram apagamento durante a digitação, 5% abandonaram texto, mas só 0,4% mostram "*correção". Simular typo + "*" com frequência é caricatura.

---

## 4. Tradução para o sistema

Fluxo proposto para cada resposta do bot. O Jev decide *o que* e *quanto*, a LLM escreve e o código filtra e calibra:

```
mensagem do usuário
  → [J1] Jev pré-geração (1 chamada): momento + orçamentos
  → código monta o BRIEFING (momento, orçamento de tamanho, pergunta sim/não, riso sim/não, 2–3 formas reais do léxico)
  → LLM "boca" gera 1 candidato (ou 2–3 com briefings diferentes)
  → [C1] filtros de código (blacklist, tamanho, "!", emoji, "?", nome, HTML)
  → [J2] Jev pós-geração (1 chamada, 4 Nouls de conteúdo) → reescrita pontual se preciso
  → [C2] controlador de distribuição (por conversa) → entrega (bolhas/tempo: ver os outros relatórios)
```

### 4.1 J1 — detector de momento e orçamentos (pré-geração)

- **State:** os últimos 6–8 turnos da sessão (bolhas de cada turno juntadas por " / "), com a última mensagem do usuário marcada. Enxuto: nada de persona longa.
- **Perguntas (numa chamada só):**
  - `moment` (Choice): {greet, how_are_you, good_news, bad_news_or_venting, funny_or_teasing, flirt_or_affection, question_to_me, plan_logistics, disagreement_or_complaint, story_telling, closing, small_ack}. Instrução: *"What kind of moment is the other person's LAST message? Pick the one that best describes what they just did."*
  - `p_length` (Score 0–4): o mesmo do P-pass. *"How long will a real person's next reply be here?"* (one word … several sentences).
  - `p_question` (Noul): *"A real person would ask something back in their next reply."*
  - `p_laugh` (Noul): *"A real person would laugh (haha/lol/kkk) in their next reply."*
  - `seeks_support` (Noul): *"The other person is looking for support or comfort."*
- **Regras de código:**
  - **Orçamento de caracteres** (medianas humanas por nível de `P.p_length`: 0 → 15, 1 → 21, 2 → 52, 3 → 73): alvo = mediana e teto = p75 × 1,3 (0 → 34, 1 → 47, 2 → 116). Para o momento `story_telling` do próprio bot, permitir 50–90.
  - **Pergunta:** permitida se `p_question` ≥ 0,45 **ou** o momento ∈ {how_are_you, bad_news_or_venting} (taxas humanas de 32–58%). Nos outros momentos, sortear com a taxa-base humana (notícia boa 8%, flerte ≈ 0–5%, piada 3–6%, geral 10–20%). O `p_question` é fraco (AUC 0,58–0,61), então use-o como ajuste e não como regra dura.
  - **Riso:** permitido se `p_laugh` ≥ 0,5 (AUC 0,69 nestes contextos) e o momento for funny_or_teasing ou flirt.
  - **Léxico:** injetar no briefing 2–3 formas reais do momento (Tabela 1 ou a versão em PT-BR, §4.6) como *exemplos de tom*, nunca como texto obrigatório. Sortear exemplos diferentes a cada vez, para não criar um vício novo.

### 4.2 Briefing para a "boca" (sem gerar caricatura)

Em vez de regras absolutas ("seja curto", "use minúscula"), passar **alvos concretos do momento**:
- "Reply in about N characters (max M)."
- "Question: yes/no."
- "Laugh: allowed/not."
- "React to the *content* of their last message, not to the emotion. One idea."
- "Don't use their name."

Nosso experimento mostrou que regras absolutas levam à correção excessiva (0,59× do tamanho, 2% de perguntas) e a vícios substitutos.

### 4.3 C1 — filtros de código (baratos, rodam sempre)

| Checagem | Limiar | Ação |
|---|---|---|
| tamanho | > teto do orçamento | regenerar com "shorter"; na 2ª falha, cortar nas primeiras 1–2 frases |
| nº de frases | ≥ 3 (humano: 13%; LLM: 42–61%) | manter a frase de reação e a de conteúdo; descartar a pergunta final se ela não foi autorizada |
| termina em "?" sem pergunta autorizada | — | remover a última frase interrogativa (se sobrar algo) |
| blacklist (Tabela 3) | ≥ 1 hit de "totally/definitely/absolutely/sorry to hear/that's so sweet/sounds like/that's amazing/I'm here for you/it's okay to feel/let me know/feel free/can't wait/so much fun/a blast" ou do equivalente em PT (§4.6) | reescrever a frase ou regenerar |
| "!" | > 1 por resposta, ou mais do que o usuário usa (taxa do usuário nas últimas 20 msgs + 5 p.p.) | trocar por nada ou "." (depende do registro do usuário) |
| emoji | > 1, ou 😂/🥰/🎉/✨ quando o usuário não usa emoji | remover |
| vocativo com o nome do usuário | qualquer ocorrência fora de cumprimento | remover |
| "\n\n" | — | converter em bolhas separadas (entrega) |
| HTML/markdown (`</p>`, `*`, `**`) | — | limpar |
| maiúscula inicial e ponto final | — | espelhar a taxa do usuário (EMA das últimas 20 msgs): se ele começa em minúscula em > 50%, baixar; se ele usa ponto, manter |

Esse conjunto separa as respostas baseline das humanas com **AUC 0,87–0,90** sem nenhuma chamada.

### 4.4 J2 — detectores Jev de conteúdo (pós-geração, 1 chamada)

- **State:** os últimos 6–8 turnos + `CANDIDATE MESSAGE (next message from <bot>)`.
- **Perguntas (Noul) e ações:**

| id | Instrução | AUC medida (agregado; controlada por tamanho) | Limiar → ação |
|---|---|---|---|
| `paraphrase` | "The CANDIDATE MESSAGE restates or paraphrases what the other person just said." | 0,74 (0,69) | > 0,5 → apagar a frase que parafraseia (em geral a 1ª ou 2ª); se não der, regenerar com "don't repeat what they said" |
| `overvalidation` | "The CANDIDATE MESSAGE praises, validates, reassures or empathizes more than the situation calls for." | 0,69 (0,63) | > 0,35 (e `seeks_support` de J1 < 0,5) → regenerar com "react to the facts, no reassurance"; se `seeks_support` ≥ 0,5, tolerar até 0,5 |
| `forced` | "…uses slang, emojis, internet expressions or enthusiasm in a forced, exaggerated or caricatured way." | 0,62 (0,65); Gemini 0,76 | > 0,35 → remover gíria/emoji e regenerar sem gíria |
| `formal` | "…uses vocabulary, phrasing or structure that is too formal, polished, elaborate or assistant-like for this casual chat." | 0,62 (0,50); GPT 0,80 (0,73) | > 0,3 → reescrita "casual, fewer words" |
| (opcional) `length` Score 0–4 | "Compared with what a real person would text here, the length is…" | 0,82 (0,70) | ≥ 3 → cortar. Redundante com C1; usar só se não houver orçamento |

**Não usar como gate:**
- o Noul "is it human?" (AUC 0,47 controlada por tamanho);
- o Choice pareado "qual é humana?" (12% de acerto contra o Gemini);
- `direct`, `generic` e `register`, que saem invertidos. Exigir "responde diretamente e de modo específico" empurra o bot para a LLM.

Custo: ~US$ 0,00003 e ~0,5 s por chamada. A regeneração acontece em uma fração das respostas: com o prompt baseline, `paraphrase` > 0,5 em ≈ 30–35% e `overvalidation` > 0,35 em ≈ 20%. Esses números são estimativa pela média das distribuições medidas; conferir em produção.

**N candidatos → escolha.** Como a amostragem com seeds rende pouca diversidade (2,64 distintos em 3) e o juiz de humanidade falha, gerar os candidatos com **briefings diferentes** (ex.: "só reage", "reage + um detalhe seu", "pergunta curta") e escolher em **código** pelo menor número de violações (C1 + J2), com desempate aleatório. Não use "qual soa mais humana?". O best-of-3 por gate Jev **não** melhorou nada no nosso teste.

### 4.5 C2 — controlador de distribuição (por conversa, em código)

Esta é a peça que evita a caricatura. Ela guarda contadores móveis das últimas 20–30 mensagens do bot e compara com as **taxas-alvo humanas** (maichat) ou com as taxas do próprio usuário, quando ele tiver histórico suficiente:

| Traço | Alvo (por mensagem do bot) | Regra |
|---|---|---|
| termina em "?" | 10–20% no geral; por momento, ver §4.1 | acima do alvo, remover a pergunta final |
| até 3 palavras | 25–45% | se estiver abaixo, favorecer o briefing "só reage" |
| riso (kkk/haha) | ≈ espelhar o usuário, teto de 1 a cada 4 msgs | acima → remover |
| emoji | espelhar o usuário (maichat: 3%) | acima → remover |
| qualquer marcador específico ("wait/pera", "tbh/na real", "honestly/sinceramente", "literally/literalmente", "true/vdd", "lol") | ≤ 1 a cada 15–20 msgs cada (humano: 1–8/1.000) | repetiu dentro da janela → proibir no próximo briefing |
| mesma abertura (1ª palavra) | ≤ 2 vezes na janela | → variar |
| resposta "oblíqua" (puxa o próprio assunto, "btw…") | ~5–10% (humano: ~20%; usar menos, porque o usuário de um app de companhia espera ser ouvido) | só quando `seeks_support` < 0,3, o momento ∉ {bad_news, question_to_me} e houver um gancho guardado da conversa |
| "*correção" visível | ≤ 0,5% | quase nunca |

### 4.6 Tradução para o português brasileiro de chat (**inferência minha, não medida**)

| Inglês (dado) | PT-BR provável | Observação |
|---|---|---|
| lol / haha / hahaha (riso-pontuação) | kkk / kkkk / hahaha / rs | "kkk" no fim como pontuação; "rs" é mais contido e de gente mais velha |
| lmao / 💀 / 😭 / "i'm crying" | KKKKKK (caixa alta), "morri", "chorando", 💀 | usar pouco; "morri kkkk" é o equivalente de "i'm dead" |
| omg | mds / meu deus / nossa | "mds" é o mais natural no chat |
| no way / wait what | mentira / não acredito / pera / oxe | "pera" = "wait"; também vicia fácil, aplicar o teto |
| nice / cool / great | boa / que massa / top / daora | "Que incrível!" é LLM-ish |
| congrats | parabéns!! / aeee / boa!! | em chat real a reação costuma ser ao fato ("passou??") |
| oh no / that sucks / ugh | ah não / putz / eita / que merda / que droga / aff | "Sinto muito ouvir isso" = blacklist |
| true / exactly / fair / same | vdd / verdade / exato / isso / justo / eu tb / msm | backchannel curto, sem desenvolvimento |
| yeah / yes / ok / k | sim / aham / uhum / ss / ok / blz / td bem | |
| idk / hmm | sei lá / hmm / ah | |
| u / ur / dont / im | vc / tb / pq / q / n / to / tô | abreviação e falta de acento são o "sem apóstrofo" do PT |
| tbh / honestly / i mean / like | na real / sinceramente / tipo / quer dizer | **tipo** é a muleta real mais comum; "sinceramente" é LLM-ish se repetido |
| and/but/so (início) | e… / mas… / aí… / então… | começar bolha com "aí" é muito natural |
| bye / gn / ttyl / xx | tchau / flw / fui / boa noite / bn / bjs / bjss | |
| rude / excuse me (zoando) | grossa(o) / nossa / oxe / como assim | |

**Lista negra em PT-BR (inferência):**
- validação e simpatia de assistente: "Que incrível!", "Que legal que você…", "Entendo perfeitamente", "Sinto muito ouvir isso", "É totalmente normal sentir…", "Estou aqui para você", "Faz todo sentido", "Com certeza!", "Absolutamente", "Fico feliz em saber";
- fechamentos e muletas: "E você?" reflexo, "Me conta mais!", "jornada", "vibe" repetida, "Ah, …" / "Haha, …" no início;
- pontuação, emoji e nome: travessão, "!" em toda frase, 😊🥰✨🎉, o nome do usuário no meio da frase;
- o extremo caricato: "kkkkkkkk" em toda mensagem, "mano/véi/slk/pprt/tmj" forçados, "tipo assim" repetido, "mds" + "pera" em sequência.

---

## 5. Limitações

- **Maichat pequeno:** 42 conversas, e muitos casais e amigos próximos, estudantes do Reino Unido. As formas por momento têm contagens baixas (2–15), e o registro é de pessoas que se conhecem bem, digitando em interface web (daí 75% de minúscula). Um app de companhia com um "desconhecido carinhoso" pode justificar um pouco mais de perguntas e acolhimento do que esse baseline.
- **Rótulos de momento vêm do Jev** (camada D), que é ruidosa: "reagir a notícia boa" inclui trocas neutras. As tabelas de momento servem para ver o formato, não como verdade fina.
- **Empathetic é induzido** (crowdworkers instruídos a "ser empáticos"). Por isso lá o humano é mais parecido com a LLM (pontuação, "sorry to hear"). Usamos só as aberturas (50 contextos) e o reportamos separado. NPS (sala pública, 2006) e NUS SMS (Singapura, sem conversa) servem para estilo, não para "o que responder".
- **WhatsApp em holandês:** mostra a estrutura (tamanho, riso como resposta, maiúscula por autocorreção), mas não o léxico em inglês ou PT. Não entrou na comparação com LLM.
- **Geração de 1 turno, fora do fluxo real:** a LLM não conhece a história da dupla, e parte das "memórias inventadas" nasce daí. O humano, por sua vez, cita coisas que só ele sabe, o que confunde o detector `invented`. As LLMs foram testadas só com persona mínima, temperatura 0,8, 3 modelos baratos; modelos maiores ou com persona rica podem ter outro perfil de vícios. O "prompt com estilo" é **uma** versão; outras redações podem calibrar melhor. Não testamos o briefing com orçamento do Jev (§4.2) porque o orçamento de LLM acabou (1.502 chamadas); **é o próximo experimento**.
- **O Jev foi avaliado como juiz em inglês**, com um só enunciado por detector. A AUC depende da redação; enunciados "abstratos" (humanidade, naturalidade) foram os piores. O gate revisado (0,754) foi escolhido olhando os mesmos dados: espere menos fora da amostra.
- **As métricas de paráfrase por sobreposição de palavras são influenciadas pelo tamanho.** Por isso também reportamos a AUC do Jev controlada por tamanho.
- **A tradução para PT-BR (§4.6) é inferência**, não dado. Validar com um corpus de chat brasileiro (ex.: WhatsApp doado) antes de fixar listas e taxas.

---

### Arquivos

- Scripts:
  - `scripts/analysis/a8_common.py` (features de estilo, tiques, regex da blacklist);
  - `a8_moments.py`, `a8_tics.py`, `a8_lexicon.py` (lado humano);
  - `a8_contexts.py`, `a8_gen.py` (geração);
  - `a8_compare.py`, `a8_template.py` (comparação em código e log-odds);
  - `a8_jev.py`, `a8_eval.py`, `a8_judge_bias.py` (detectores Jev e AUC).
- Saídas:
  - `analysis/data/a8_moments.json`, `a8_human_tics.json`, `a8_lexicon.json`;
  - `a8_compare.json` (inclui log-odds e tabela por momento), `a8_blacklist.csv`, `a8_template.json`;
  - `a8_jev_auc.json`, `a8_judge_bias.json`;
  - `a8_generations.jsonl` (250 contextos: última mensagem, humano e 5 condições de LLM), `a8_contexts_meta.json`.
