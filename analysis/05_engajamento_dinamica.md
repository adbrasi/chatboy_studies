# 05 — Engajamento e dinâmica da conversa

**Pergunta:** o que mantém uma conversa viva e o que a mata? Que "movimento" um bot deve fazer a cada turno para o usuário
continuar engajado, e como transformar isso em detectores Jev e regras de código (um "diretor de conversa")?

**Dados usados**
- `whatsapp_nl`: 33.896 turnos, 3.744 sessões (sessão nova após 3 h de silêncio), 60 chats. Mediana de 4 turnos por
  sessão (p75 = 10, p90 = 21). É o único corpus em que "a conversa morreu" tem sentido natural: um turno é o **último
  da sessão** quando o parceiro não responde em até 3 h.
- `maichat`: 42 conversas, 2.871 turnos, todos anotados. As sessões duram ~20–35 min por desenho do experimento, então o
  fim é imposto. Por isso uso o maichat para engajamento, tamanho e latência do turno seguinte, e não para "continuidade".
- `jev_base` (passes D e P): 5.968 turnos anotados (2.871 maichat + 3.097 whatsapp_nl).
- **Experimento Jev novo (a5_jev_moves):** 2.750 turnos (1.600 maichat + 1.150 whatsapp), 1 chamada por turno com 6
  perguntas: `callback` (Noul, com o trecho anterior da conversa no state), `transition` (Choice: continua / mudança suave /
  mudança abrupta / volta a tópico anterior / só reação), `personal_q`, `asks_back`, `compliment`, `story` (Nouls).
  Foram 2.743 chamadas novas, US$ 0,20 no total, com p50 de 0,49 s por chamada.
- Scripts: `scripts/analysis/a5_*.py`. Saídas: `analysis/data/a5_*.json|csv|jsonl`.

Convenções: "cont≥k" = a sessão ainda tem ≥ k turnos depois deste turno. `zch`/`zlat` = log do tamanho ou da latência
padronizado **dentro da mesma pessoa na mesma conversa**, o que compara a pessoa com ela mesma. "Engajamento" =
`D.engagement` (Score 0–4 do Jev). Os efeitos de "movimento" são observacionais: regressões com controles (engajamento
anterior do parceiro, fase, seriedade, posição na sessão, efeito fixo da conversa, erro padrão agrupado por conversa).

---

## 1. Resumo dos achados

- **Um turno sem gancho mata a conversa.** No whatsapp, depois de um turno só de "ok"/"ja"/emoji, o parceiro responde
  na mesma sessão em 75,8% dos casos; depois de uma pergunta, em 94,4%. Depois de "só risada" são 78,8%, depois de mídia
  sozinha 71,4% e depois de uma despedida 77,3%. Com controle de posição, hora e chat, a razão de chances (OR) de o
  parceiro responder é **0,34** para o turno mínimo e **0,36** para a risada pura, contra **2,99** para a pergunta.
- **`D.hook` do Jev é o melhor preditor único de continuidade** (OR 11,9 de 0→1, AUC 0,69, contra 0,56 de `has_q`).
  Com hook ≤ 0,2, a sessão continua em 61% dos casos e dura ≥ 5 turnos em 30%. Com hook > 0,5, continua em ~92% e dura
  ≥ 5 turnos em ~58%.
- **A pergunta segura o próximo turno, mas não a conversa longa.** A resposta a uma pergunta é 0,32 DP mais longa, e a
  uma pergunta **pessoal**, **0,49 DP** mais longa. A pergunta, porém, não sobe o engajamento do Jev (+0,02, n.s.), e
  P(cont≥10) é igual para pergunta e afirmação (43,7% × 44,0%). Nos chats com mais perguntas as sessões são mais curtas
  (ρ = −0,44, p < 0,001): conversa-interrogatório é conversa de logística.
- **Autorrevelação e história são o que mais eleva o engajamento do outro.** Depois de o falante revelar algo pessoal,
  o engajamento do parceiro no turno seguinte sobe **+0,15** (IC95 0,09–0,21). Depois de uma história ou informação, sobe
  **+0,18** (0,10–0,25). As duas também alongam a resposta (+0,17 e +0,24 DP). Em small talk, a história vale +0,38.
- **Brincar só funciona quando o momento é leve.** Com seriedade < 0,5, o humor soma +0,16 de engajamento (p = 0,008).
  Com seriedade ≥ 1, o efeito é nulo (+0,04) e a resposta tende a encurtar (−0,16 DP, p = 0,07). Isso confirma, de forma
  fraca, a hipótese do usuário.
- **A mudança abrupta de tópico custa engajamento; o callback não rende o que se esperava.** O "btw"/salto de tópico
  (5,6–7% dos turnos) reduz o engajamento do parceiro em **−0,22** (IC −0,37 a −0,08). Os callbacks aparecem em
  8,6% (maichat) e 13,4% (whatsapp) dos turnos, mas o efeito é **nulo** (+0,03 a +0,05, n.s.). **A hipótese de que
  callbacks aumentam o engajamento não se confirma.**
- **Devolver a pergunta é raro.** Em "e você?" depois de uma pergunta, a taxa é de 2,4% (whatsapp, n = 8.024) e 3,3%
  (maichat) pelo regex, e de 3,4–8,5% pelo Jev. Sobe para 10–25% só depois de "como vai?". Em geral uma pessoa faz
  60–67% das perguntas da conversa, e essa pessoa não é a que fala mais (a coincidência é de 52%, puro acaso). Quem
  abre tópicos novos é quem escreve mais (maichat ρ = 0,50, p < 0,001), não quem pergunta (ρ = 0,01).
- **A latência contamina e sinaliza o fim.** Minha latência acompanha a do outro (ρ = 0,30 no whatsapp). O turno final
  de uma sessão chega 0,37 DP mais lento que o normal da pessoa, contra −0,13 quando ainda faltam mais de 10 turnos.
  Quem volta depois de 1–3 h pergunta mais (31,5% × 19,9%) e manda menos "ok" (5% × 11%), mas o Jev o vê menos
  engajado (1,67 × 2,06).
- **O engajamento tem forma de serra, e o arco é pequeno.** Turno a turno o engajamento alterna (83% das conversas do
  maichat e 66% das sessões do whatsapp). Ele acompanha o tamanho do turno (ρ = 0,51–0,67). Na sessão há um arco suave
  (pico no meio: 2,27 no whatsapp) e uma **queda nos 20% finais** (1,98 no whatsapp; 1,85 no maichat).
- **Validação do passe P.** `p_end` funciona: AUC 0,75 no total (0,82 maichat, 0,66 whatsapp), bem calibrado por faixa.
  Somado a features de código, chega a AUC 0,80 (CV), contra 0,70 só com código. `p_question` é fraco (AUC 0,61–0,62) e
  superestima (média de 35% contra 16% reais). No maichat, a taxa de perguntas do falante até ali ganha dele (AUC 0,66).
  `p_topic_shift` é quase inútil (AUC 0,60 contra o rótulo D, 0,56 contra o rótulo novo de transição e 0,53 para mudança abrupta).
- **Padrões invisíveis.** 93,6% das sessões do whatsapp **terminam sem despedida** (6,4% com despedida), e 7,7% terminam
  numa pergunta que ficou sem resposta. 54% das sessões **abrem com uma pergunta** e só 10% com "oi". O tamanho do turno
  espelha o do outro (β = 0,10–0,12 DP por DP), e o número de bolhas também (ρ = 0,12–0,21).

---

## 2. Achados detalhados

### A1. O que mata e o que mantém a conversa (whatsapp_nl, todos os 33.896 turnos)

P(a sessão continua ≥ k turnos | tipo do turno), com IC95 de Wilson em k = 1:

| tipo do turno (código) | n | cont≥1 | IC95 | cont≥3 | cont≥5 | cont≥10 |
|---|---:|---:|---|---:|---:|---:|
| mídia (foto sozinha) | 304 | 0,714 | 0,66–0,76 | 0,50 | 0,43 | 0,26 |
| mínimo ("ok", "ja", 👍) | 2.383 | 0,758 | 0,74–0,78 | 0,63 | 0,53 | 0,39 |
| despedida | 1.061 | 0,773 | 0,75–0,80 | 0,49 | 0,37 | 0,23 |
| só risada ("hahaha") | 463 | 0,788 | 0,75–0,82 | 0,68 | 0,60 | 0,45 |
| afirmação curta/média | 18.874 | 0,889 | 0,88–0,89 | 0,73 | 0,62 | 0,44 |
| longo (≥100 chars) sem pergunta | 2.601 | 0,925 | 0,91–0,93 | 0,76 | 0,63 | 0,43 |
| pergunta | 8.210 | 0,944 | 0,94–0,95 | 0,80 | 0,66 | 0,44 |

- A regressão logística (P de o parceiro responder na sessão ~ tipo + log(posição) + faixa horária + efeito fixo do
  chat, n = 33.896), com a afirmação curta como referência, dá: pergunta OR **2,99** [2,68–3,34]; longo **1,86**;
  despedida 0,66; mídia 0,46; risada pura **0,36**; mínimo **0,34**.
- O efeito vale em todas as posições. No início da sessão (turnos 0–2), o turno mínimo deixa 62% de chance de resposta,
  contra 90% da pergunta. Com 10 turnos ou mais: 84% × 98%.
- O efeito é **local**. Em cont≥10 as categorias quase se igualam (~0,44), exceto despedida e mídia. A pergunta salva o
  próximo turno, não a sessão inteira.
- Exemplos de turnos que encerraram a sessão com hook < 0,2: "Ah ok.", "Leuk! Xx", "Hihi", "Dank je xxx", "Sterkte".
  Turnos que abriram ≥ 10 turnos: "Is die bezichtiging niets geworden?", "Goeed!! Wat ben je aant doen?".
- Confiança: **alta** para a direção, porque n é grande e o efeito é robusto a controles. A limitação é que é holandês e
  de 2012–14, e que "fim" = 3 h de silêncio.

### A2. O gancho (`D.hook`) prediz a continuidade melhor que qualquer regra de código

Janelas anotadas do whatsapp (n = 3.097):

| D.hook | n | cont≥1 | cont≥3 | cont≥5 |
|---|---:|---:|---:|---:|
| ≤ 0,2 | 359 | 0,61 | 0,41 | 0,30 |
| 0,2–0,5 | 1.157 | 0,86 | 0,62 | 0,49 |
| 0,5–0,8 | 628 | 0,92 | 0,74 | 0,60 |
| > 0,8 | 953 | 0,92 | 0,75 | 0,57 |

- Na regressão logística multivariada com efeito fixo do chat, o hook tem OR **11,9** (0→1) [7,1–19,9]. O `vulnerable`
  tem OR 2,16 [1,08–4,32]. O `topic_shift` tem OR **0,39** [0,23–0,65]. O turno mínimo (código) tem OR 0,42, mesmo depois
  de controlar o hook. O `playful` não tem efeito (OR 1,0).
- AUC para "o parceiro responde": `D.hook` 0,69; "não é mínimo" 0,57; `has_q` 0,56. O hook capta ganchos sem "?"
  ("Vertel!", "net thuis, verslag wordt wetenschappelijk artikel…").
- Por intenção (`D.intent`), cont≥1 varia de closing 0,66, react_acknowledge 0,72 e greet 0,71 até answer 0,91,
  share_feeling 0,91 e ask_question 0,92.
- No maichat (sem continuidade natural), o engajamento do parceiro no turno seguinte também sobe com o hook do turno
  anterior: 1,78 com hook < 0,2 (n = 129) contra 2,05–2,09 com hook > 0,5.
- Confiança: **alta**.

### A3. Perguntas: seguram o próximo turno, alongam a resposta, mas não "engajam" e não fazem a conversa durar

- Tamanho da resposta: a resposta a uma pergunta é +0,32 DP mais longa [0,25–0,40]. Em mediana, 38 × 24 caracteres
  (maichat) e 44 × 34 (whatsapp). Para uma **pergunta pessoal** (Jev `personal_q`, subamostra), são **+0,49 DP**
  [0,36–0,61]. Em small talk, a pergunta alonga a resposta em +0,55 DP.
- Engajamento do Jev no turno seguinte: +0,015 (n.s.). A pergunta pessoal dá +0,03 (n.s.).
- No nível do chat (whatsapp), a taxa de perguntas correlaciona **negativamente** com a duração média das sessões
  (ρ = −0,44, p = 0,0005). Os chats de "pergunta-resposta-ok" são logísticos.
- A pergunta **atrasa** a resposta: o parceiro responde no mesmo minuto em 34% dos casos, contra 41% sem pergunta. No
  maichat é o inverso, mas por pouco (58% × 52% em menos de 15 s).
- Jev sobre a composição: só 34% (whatsapp) e 47% (maichat) das perguntas são pessoais; o resto é logística ou
  confirmação.
- Exemplo (maichat): "What about youu? / Do you have a funny memory?" → "I do! When I was around 6 or 7 I would wait
  for my mom to come home from work…" (resposta longa e engajada).
- Confiança: **alta** para "pergunta → resposta mais longa e maior chance de resposta"; **média** para "não engaja".

### A4. Autorrevelação e histórias elevam o engajamento do outro

Regressão OLS do engajamento do parceiro no turno seguinte (n = 5.045 pares, controles acima):

| movimento do falante | n | Δ engajamento do parceiro [IC95] | Δ tamanho da resposta (DP) |
|---|---:|---|---|
| revelar algo pessoal (`vulnerable` > 0,5 ou share_feeling) | 673 | **+0,15** [0,09; 0,21] | +0,17 |
| contar história/informação (share_story_or_info) | 494 | **+0,18** [0,10; 0,25] | +0,24 |
| só reagir (ack / react_acknowledge) | 790 | +0,08 [0,02; 0,14] | +0,14 |
| flertar | 175 | +0,12 (p = 0,09) | n.s. |
| elogiar | 88 | +0,10 (n.s.) | +0,23 (subamostra Jev, p = 0,004) |
| perguntar | 1.159 | +0,02 (n.s.) | **+0,32** |
| brincar (`playful` > 0,5) | 2.964 | +0,01 (n.s.) | n.s. |
| mudar de tópico (`topic_shift` > 0,5) | 579 | −0,03 (n.s.) | n.s. |

- O efeito é consistente nos dois corpora: revelar dá +0,13 (maichat) e +0,17 (whatsapp); história dá +0,13 e +0,22.
- No whatsapp, revelar dá OR 2,16 de o parceiro responder (A2).
- Exemplos: "it really bothers me when you and grant go without me" → "we should go / im sorry you feel that way, do you
  want to discuss it further". "prof made eye contact with me once / i panicked" → "did he ask you something".
- O paradoxo do "só reagir": ele **não** reduz o engajamento do turno seguinte, que até sobe um pouco, porque o outro
  preenche o espaço. Mas ele **mata a continuidade** (OR 0,36 no whatsapp). Isso é um sinal de que o outro carrega a
  conversa sozinho.
- Confiança: **média-alta**. O efeito é observacional e o Jev que rotula o movimento é o mesmo que rotula o engajamento
  (os rótulos são independentes por turno, mas não por modelo).

### A5. Humor depende da seriedade (a hipótese do usuário se confirma, com efeito pequeno)

Efeito de "brincar" sobre o engajamento do parceiro, por estrato:

| estrato | n | Δ engajamento | Δ tamanho |
|---|---:|---|---|
| seriedade < 0,5 | 2.256 | **+0,16** (p = 0,008) | +0,13 (p = 0,11) |
| seriedade 0,5–1 | 1.860 | +0,02 (n.s.) | +0,01 |
| seriedade ≥ 1 | 929 | +0,04 (n.s.) | **−0,16** (p = 0,07) |
| fase profundo/tenso | 357 | −0,03 (n.s.) | −0,16 (p = 0,16) |

- No whatsapp, brincar aumenta a continuidade: OR 1,58 (cont≥1) e 1,64 (cont≥5).
- Na fase de brincadeira, revelar algo (+0,23) e flertar (+0,21) valem mais do que mais uma piada (+0,07).
- Na fase profundo/tenso **nenhum** movimento tem efeito significativo (n = 357 pares), porque falta poder estatístico.
- Confiança: **média**.

### A6. Tópicos: como mudam, o custo da mudança abrupta e o callback que não funcionou

Jev, `transition` (2.750 turnos):

| tipo | maichat | whatsapp | engajamento do parceiro depois | cont≥5 (whatsapp) |
|---|---:|---:|---:|---:|
| continua o tópico | 72% | 73% | 2,07 | 0,57 |
| só reação | 16% | 13% | 2,00 | **0,40** |
| mudança abrupta | 5,6% | 7,0% | **1,75** | 0,47 |
| volta a tópico anterior | 3,1% | 4,0% | 1,98 | 0,50 (n = 46) |
| mudança suave (associação) | 3,4% | 3,0% | 1,93 | 0,32 (n = 34) |

- Ajustado (engajamento anterior, fase, seriedade, tamanho do próprio turno): a mudança abrupta dá **−0,22** de
  engajamento [−0,37; −0,08] e −0,20 DP de tamanho (p = 0,09). A mudança suave dá −0,09 (n.s.).
- Só 23% das mudanças abruptas usam marcador explícito ("btw", "anyway", "trouwens", "wait"); as outras só pulam
  ("we need help", "what u doing tonight", "ok i should actually start cleaning"). Muitas mudanças "abruptas" são
  **saídas** disfarçadas: o assunto acabou.
- **Callback** (Noul com o histórico no state): 8,6% dos turnos no maichat e 13,4% no whatsapp. Contra um proxy lexical
  (palavra de conteúdo que aparece no trecho antigo mas não nos 3 turnos recentes), a AUC é 0,75, o que dá validade
  razoável. Efeito sobre o parceiro: engajamento +0,03 [−0,10; 0,15], tamanho −0,02, latência 0,00. Médias brutas:
  2,02 × 2,04. **Não há efeito detectável.** A "volta a tópico anterior" também não tem efeito (−0,08, n.s.).
- A mudança de tópico precede subidas "brutas" de engajamento (p(subida) 19% × 11%), mas isso é **regressão à média**:
  mudanças acontecem quando a energia já está baixa. Com controle do nível anterior, o efeito some (−0,03).
- Confiança: **média** para o custo do abrupto; **média** para o callback nulo (n = 234 callbacks, com poder para
  detectar ~±0,12).

### A7. Reciprocidade: quem pergunta, quem devolve, quem abre tópicos

- **Devolver a pergunta** (regex EN/NL do tipo "and you?", "wbu", "en jij?", "met jou?", "jij ook?"): acontece em 2,4%
  [2,0–2,7] das respostas a perguntas no whatsapp (n = 8.024) e em 3,3% [1,8–6,2] no maichat (n = 271). Depois de
  perguntas do tipo "como vai/o que está fazendo" sobe para 10,3% (whatsapp, n = 487) e 25% (maichat, n = 12).
- Pelo Jev (`asks_back`), depois de uma pergunta do outro: 3,4% no maichat (n = 145) e 8,5% no whatsapp (n = 307).
  Depois de uma pergunta **pessoal**: 8,1% e 14%. Depois de uma pergunta não pessoal: 0% e 8%.
- Quando há devolução, a resposta seguinte do outro é mais longa (+0,48 × −0,04 DP), mas n = 30, então é **fraco**.
- Exemplos: "Hoe gaat het met jou?" → "Goooed! En met jou? / Jaaaaaa zeker! Jij ook? :d"; "How's your day been today?"
  → "Good but it's cold / How about u".
- **Assimetria:** a pessoa que mais pergunta faz, na mediana, 66,7% (maichat) e 60,1% (whatsapp) das perguntas. Na fala
  a divisão é 53/47 e 56/44. Perguntar mais e falar mais são **independentes**: coincidem em 52% das conversas.
- **Equilíbrio × duração:** no maichat, a conversa mais desigual em caracteres é mais curta (ρ = −0,35, p = 0,02). No
  whatsapp não há efeito (ρ = 0,04). A assimetria de perguntas não prediz duração em nenhum dos dois (ρ = 0,12 e 0,01).
- **Quem abre tópicos:** quem mais inicia tópicos abre 65% (maichat) e 58% (whatsapp) deles. A proporção de tópicos
  iniciados acompanha a proporção de texto (maichat ρ = 0,50, p < 0,001; whatsapp ρ = 0,20, n.s.), e não a de perguntas
  (0,01 e 0,18, n.s.). O papel típico: um "narrador" que traz assunto e um "entrevistador" que pergunta.
- Confiança: **alta** para "devolver é raro"; **média** para os papéis.

### A8. Latência como sinal (whatsapp: resolução de minuto; maichat: segundos)

| latência do turno (whatsapp) | n | caracteres (mediana) | P(pergunta) | P("ok") | engajamento (Jev) | o parceiro responde |
|---|---:|---:|---:|---:|---:|---:|
| < 1 min | 10.261 | 31 | 0,20 | 0,11 | 2,06 | 0,94 |
| 1–2 min | 8.894 | 38 | 0,22 | 0,09 | 2,03 | 0,91 |
| 3–9 min | 4.823 | 40 | 0,23 | 0,09 | 2,08 | 0,88 |
| 10–59 min | 3.593 | 40 | 0,26 | 0,07 | 2,06 | 0,83 |
| 1–3 h | 1.117 | 40 | **0,32** | 0,05 | **1,67** | 0,77 |

- **Contágio de ritmo:** a latência z de A correlaciona com a latência z anterior de B em ρ = **0,30** (whatsapp,
  n ≈ 30k). Quando o outro demorou muito (z > 1), eu também demoro (mediana de 180 s contra 0 s quando ele foi rápido).
  No maichat, ao vivo, o efeito é quase nulo (ρ = 0,04).
- **Sinal de fim:** a latência z cresce até o último turno. Com mais de 10 turnos ainda por vir, z = −0,13; com 3 a 10,
  −0,04; com 1 a 3, +0,06; com 1, +0,23; no turno final, +0,37.
- **Quem responde rápido escreve mais ou menos?** Um pouco **menos**: dentro da pessoa, ρ(zlat, zch) = +0,09 no whatsapp
  e +0,11 no maichat (no maichat parte é mecânica, porque a latência inclui o tempo de digitar). Resposta rápida = curta
  e reativa; resposta demorada = mais longa, com pergunta, reabrindo o assunto.
- Latência × engajamento do Jev (mesma pessoa, controlando o tamanho): −0,10 por DP de latência no whatsapp
  (p < 10⁻¹⁵, n = 2.440); nulo no maichat.
- Confiança: **média**. A resolução de minuto é grosseira e a latência no whatsapp mistura interesse com
  disponibilidade (trabalho, sono).

### A9. Trajetórias de engajamento

- **Serra turno a turno:** o engajamento alterna de sinal a cada turno em 71% (mediana) das transições. 83% das
  conversas do maichat e 66% das sessões do whatsapp (sessões com ≥ 12 turnos anotados, n = 73) são "serra"
  (alternância > 0,6 e DP > 0,6). A serra vem do fato de o engajamento acompanhar o tamanho do turno
  (ρ = 0,51 maichat, 0,67 whatsapp) e de as pessoas alternarem entre turno longo e reação curta. O engajamento
  correlaciona mais com o próprio turno anterior (ρ = 0,39 / 0,29) do que com o do parceiro (0,26 / 0,18).
- **Formas no nível da sessão** (média móvel de 5 turnos). No maichat (n = 42): oscilante sem tendência 57%, sobe e cai
  21%, decrescente 19%, crescente 2%. No whatsapp (n = 73): oscilante 29%, decrescente 25%, crescente 19%, sobe e cai
  14%, cai e sobe 8%, plana 5%.
- **Curva média** (5 faixas de posição relativa): no whatsapp, 2,01 → 2,15 → **2,27** → 2,09 → 1,98; no maichat,
  2,05 → 2,04 → 2,12 → 2,08 → **1,85**. O arco tem pico no meio e a queda se concentra no último quinto.
- **O que precede uma subida** (o engajamento de S sobe ≥ 1 em relação ao turno anterior de S; a taxa base é 12%),
  pelo que o parceiro fez no meio: história 18,2% (lift 1,5), elogio/afeto 18,2% (lift 1,5, n = 88), share_feeling 14%,
  pergunta 14% (1,16), flerte 20% (n = 175). A piada fica na base (12,1%). **Responder** a uma pergunta é o que menos
  precede subida (6,7%, lift 0,56), assim como o conforto (6,9%). A pessoa que recebe a resposta tende a só reagir.
- Cuidado: a regressão à média é forte. O delta médio é +1,16 quando o nível anterior é 0 e −0,88 quando é 4. Por isso
  os efeitos causais-plausíveis estão em A4, que controla o nível anterior.
- Confiança: **média** para a serra e a queda final; **baixa** para a tipologia de formas (poucas sessões, e as janelas
  do whatsapp cortam sessões).

### A10. Validação dos preditores P (a posição exata do bot)

| preditor | alvo | n | taxa-base | AUC Jev | baseline de código | acurácia Jev@0,5 × maioria |
|---|---|---:|---:|---:|---:|---|
| `p_question` | turno tem "?" | 5.491 | 15,8% | 0,62 (maichat 0,61; whatsapp 0,61) | taxa de perguntas do falante até ali: 0,67 (maichat 0,66; whatsapp 0,57) | 0,77 × 0,84 |
| `p_topic_shift` | `D.topic_shift` > 0,5 | 5.491 | 11,1% | 0,60 | turno anterior foi "ok": 0,53 | 0,89 × 0,89 |
| `p_topic_shift` | transição = suave/abrupta/volta (a5_jev_moves) | 2.601 | 10,5% | 0,56 (só abrupta: 0,53) | turno anterior foi "ok": 0,55 | 0,89 × 0,90 |
| `p_end` | despedida, último turno ou closing | 5.491 | 10,6% | **0,75** (maichat 0,82; whatsapp 0,66) | posição 0,77 (maichat); despedida anterior 0,56 | 0,89 × 0,89 |
| `p_end` | último turno da sessão (whatsapp) | 2.665 | 12,6% | 0,66 | 0,54 | 0,87 × 0,87 |

- **Calibração de `p_end`:** boa. Por faixa, predito/observado: 0,04/0,03; 0,08/0,05; 0,14/0,10; 0,28/0,29; 0,50/0,48.
- **`p_question` superestima:** 0,30 predito contra 0,14 observado na faixa 0,2–0,4, e 0,67 contra 0,33 acima de 0,6.
  Não serve como probabilidade; no máximo como ranking.
- **Combinado** (regressão logística, CV agrupada por conversa): `p_end` + despedida anterior + posição dá AUC **0,80**,
  contra 0,70 só com código. `p_question` + taxa do falante dá 0,69, contra 0,67 só com código. **Use `p_end`; descarte
  `p_topic_shift`; `p_question` só junto com a taxa do usuário.**

### A11. Espelhamento, aberturas e fins

- **Tamanho:** o z do meu turno sobe 0,10 (maichat) a 0,12 (whatsapp) DP por DP do turno anterior do outro (p < 10⁻⁶),
  controlando meu próprio turno anterior. **Bolhas:** o nº de bolhas correlaciona com o do turno anterior do outro
  (ρ = 0,21 maichat, 0,12 whatsapp).
- **Fins:** só 6,4% das sessões do whatsapp terminam com despedida explícita. 7,7% terminam numa pergunta que ficou
  sem resposta por mais de 3 h (ex.: "Beestfeest volgende week donderdag?", "NEE :p / Ben je dr al?"). O último turno é
  sobretudo uma afirmação curta (56%) ou um turno mínimo (15%).
- **Aberturas:** 54% das sessões abrem com uma pergunta e só 10% com uma saudação. Quem fecha uma sessão abre a
  seguinte em 48% dos casos, ou seja, a reabertura é equilibrada entre os dois.

---

## 3. Padrões "invisíveis" (o que as pessoas fazem sem perceber e o bot deveria imitar)

1. **As conversas morrem por inanição, não por despedida.** 94% das sessões terminam sem "tchau". A última mensagem é
   um "ok"/"haha"/afirmação sem gancho, e a latência já vinha subindo (z +0,37). Um bot que sempre fecha com despedida
   formal soa artificial. Um bot que nunca deixa a conversa esfriar soa carente.
2. **Não se devolve a pergunta.** Humanos devolvem "e você?" em só 2–8% das respostas a perguntas (10–25% depois de
   "como vai?"). Um bot que sempre devolve parece um formulário. O certo é devolver **só** depois de pergunta pessoal ou
   de "como vai", e em ~1 de cada 5 vezes.
3. **Papéis complementares.** Um "entrevistador" (faz ~2/3 das perguntas) e um "narrador" (abre tópicos, escreve mais).
   Se o usuário é o entrevistador, o bot deve narrar (contar coisas, abrir assunto), e vice-versa.
4. **Serra, não linha.** O engajamento alterna turno longo ↔ reação curta. O bot não deve manter todo turno "alto": turnos
   curtos de reação no meio são naturais e não reduzem o engajamento do outro. O que mata é o turno curto **sem
   gancho** no fim de um tópico.
5. **Contágio de ritmo.** A latência de um espelha a do outro (ρ = 0,30), e o tamanho do turno também (β ≈ 0,1).
   Depois de um silêncio longo, a pessoa volta com pergunta (32%) e não com "ok".
6. **O assunto acaba antes da conversa.** As mudanças abruptas (sem "btw" em 77% dos casos) são muitas vezes saídas
   disfarçadas ("ok i should actually start my hw") e derrubam o engajamento do outro (−0,22).
7. **Contar algo de si é mais "engajante" que perguntar.** A pergunta rende uma resposta longa (+0,32 a +0,49 DP), mas
   a revelação e a história é que sobem o investimento do outro (+0,15 a +0,18).
8. **O que "puxa" a pessoa para cima é receber algo dela** (história, elogio, flerte), e não receber a resposta à própria
   pergunta: depois de receber uma resposta, a subida de engajamento é rara (lift 0,56).

---

## 4. Tradução para o sistema: o "diretor de conversa"

### 4.1 Fluxo

```
mensagem(ns) do usuário → [código] features (latência, tamanho, "?", ack, risada, taxa de perguntas do usuário, ritmo)
  → [Jev chamada ÚNICA "leitura" ~0,6 s] (perguntas D1–D10, em paralelo)
  → [código] estado da conversa (fase, seriedade, gancho do usuário, risco de fim, fios abertos)
  → [Jev chamada "movimento" (Choice M1), opcional] + regras de veto/limiar do código
  → briefing para a LLM (movimento, tamanho-alvo, gancho obrigatório?, devolver pergunta?, humor permitido?)
  → [código] entrega (nº de bolhas, atraso espelhando o ritmo)
```

### 4.2 Detectores Jev (chamada de leitura, a cada turno do usuário)

State: `{"conversation_so_far": últimos 8 turnos (bolhas juntadas por turno, com "replied: after N min" quando > 10
min), "user_turn": turno atual do usuário, "open_threads": até 5 fios abertos mantidos pelo código (texto curto)}`.
Em projetos em PT-BR, manter o state enxuto: o gancho e a fase dependem só dos últimos turnos.

| id | tipo | instrução (literal, atômica) | critérios/níveis | consumido por |
|---|---|---|---|---|
| D1 `user_hook` | Noul | "Does `user_turn` give the other person something easy to reply to (a question, a hook, an invitation)?" | — | A1/A2: se < 0,3, o bot **precisa** carregar (proibir resposta só-reação) |
| D2 `engagement` | Score | "How engaged/invested in the conversation is the speaker of `user_turn`?" | 0 disengaged … 4 very high | trajetória: média móvel de 5 turnos; queda ≥ 0,7 do pico → "modo resgate" |
| D3 `phase` | Choice | "Which phase is the conversation in at `user_turn`?" | opening, small_talk, deep_personal, playful_banter, logistics, conflict_or_repair, winding_down | estratificação dos movimentos (A4/A5) |
| D4 `seriousness` | Score | "How serious is this moment of the conversation?" | 0 playful … 3 very serious/emotional | vetar humor se ≥ 1 |
| D5 `user_disclosed` | Noul | "Is the speaker of `user_turn` disclosing something personal or vulnerable?" | — | se > 0,6: responder com validação + revelação recíproca, **sem** mudar de tópico |
| D6 `asked_personal` | Noul | "Does `user_turn` ask the other person a personal question about their life, feelings, opinions or experiences?" | — | responder com conteúdo pessoal (história); devolver a pergunta com prob. 0,2 |
| D7 `will_end` | Noul | "Will the speaker of `user_turn` start ending the conversation in their next turn?" (state = até o turno do usuário) | — | risco de fim (A10) |
| D8 `topic_exhausted` | Noul | "Has the current topic run out (only short reactions, nothing new being added in the last turns)?" | — | libera mudança de tópico (preferir suave) |
| D9 `joke_welcome` | Noul | "Would a joke be a good idea for the next reply?" | — | humor só se > 0,5 **e** D4 < 1 |
| D10 `thread_to_resume` | Choice | "Which of `open_threads` is the user most likely to want to talk about now?" | fios + "none" | callback só na reabertura de sessão ou se D8 > 0,7 (efeito nulo no meio da conversa) |

Pelas boas práticas, contagens, latências e taxas ficam no código, não no Jev: `user_q_rate`, `bot_q_rate`, `zlat`
(latência do usuário comparada com a média dele), `zch` e flags `ack`/`laugh_only` por regex.

### 4.3 Risco de fim (código combinando Jev + features)

`logit(end) = a·logit(D7) + b·[usuário mandou despedida no turno anterior] + c·log(posição) + d·[zlat do usuário > 1]
+ e·[user_turn é ack/risada]`. Os pesos iniciais saem de `a5_validate_P.py` (AUC CV 0,80 com Jev contra 0,70 sem).
Faixas:
- **< 0,15:** normal.
- **0,15–0,35:** "resgate". O próximo movimento **tem** de ter gancho: história curta + pergunta pessoal leve. Nada de
  mudança abrupta.
- **> 0,35 ou despedida do usuário:** não prender. Movimento `wind_down`: despedida curta e calorosa, com gancho para a
  próxima sessão ("depois me conta como foi X"). Guardar o fio em `open_threads`.

### 4.4 Choice de movimento (M1) e regras de código

Chamada Jev opcional, no mesmo state de leitura:
**M1 `best_move`** (Choice): "Which move would best keep the user engaged in the bot's next reply?"

| opção | descrição |
|---|---|
| `share_story` | conta algo que aconteceu com o bot (curto, concreto) |
| `self_disclose` | revela um sentimento, opinião ou fraqueza do bot |
| `ask_personal` | faz uma pergunta pessoal sobre a vida, os sentimentos ou os planos do usuário |
| `answer_and_ask_back` | responde e devolve a pergunta |
| `joke_tease` | brinca ou provoca de leve |
| `compliment` | elogia o usuário ou algo que ele fez |
| `react_briefly` | reação curta (haha, sério?!), sem conteúdo novo |
| `comfort_support` | acolhe, valida |
| `smooth_shift` | puxa um assunto novo a partir de algo que acabou de ser dito |
| `resume_thread` | retoma um fio aberto (callback) |
| `wind_down` | começa a encerrar com calor |

O código combina a distribuição do Jev com **pesos a priori estimados dos dados** e aplica vetos:
`score(m) = P_jev(m) × prior(m | fase, seriedade)`, com os vetos e limiares abaixo, e o bot escolhe o argmax.

Priors (multiplicadores iniciais, a partir de A1–A6):

| movimento | leve (ser < 0,5) | casual (0,5–1) | sério (≥ 1) / profundo | logística | encerrando |
|---|---|---|---|---|---|
| share_story | 1,3 | 1,3 | 1,3 | 1,2 | 1,3 |
| self_disclose | 1,3 | 1,1 | 1,2 | 1,0 | 1,2 |
| ask_personal | 1,1 | 1,1 | 1,0 | 1,0 | 0,8 |
| joke_tease | 1,2 | 1,0 | **0,3** | 1,0 | 1,0 |
| compliment | 1,1 | 1,1 | 1,0 | 0,8 | 1,0 |
| react_briefly | 0,9 | 0,8 | 0,7 | 0,8 | 1,0 |
| smooth_shift | 0,8 | 0,8 | **0,3** | 0,9 | 0,6 |
| resume_thread | 0,9 | 0,9 | 0,7 | 0,9 | 0,9 (1,3 na reabertura de sessão) |
| comfort_support | 0,5 | 0,8 | 1,5 | 0,5 | 0,8 |

Vetos e limiares (código):
1. **Gancho obrigatório:** se `user_hook` < 0,3 ou o usuário mandou ack/risada, `react_briefly` fica proibido **sozinho**.
   A reação vira prefixo de `share_story`/`ask_personal` (A1: OR 0,34 do turno mínimo).
2. **Humor:** `joke_tease` só se `seriousness` < 1 **e** `joke_welcome` > 0,5 **e** `anxious`/`tension` < 0,5 (A5).
3. **Mudança de tópico:** só `smooth_shift`, e só se `topic_exhausted` > 0,7. Mudança abrupta **nunca**, exceto na
   reabertura de sessão (A6: −0,22).
4. **Devolver a pergunta:** `answer_and_ask_back` só se `asked_personal` > 0,6, e em no máximo ~25% dessas vezes (A7).
   Senão, `answer` + `share_story`.
5. **Orçamento de perguntas:** se `bot_q_rate` (últimos 10 turnos do bot) > 0,5, multiplicar `ask_personal` por 0,5
   (A3: chats-interrogatório têm sessões mais curtas). Se o usuário já é o "entrevistador" (`user_q_rate` > 0,4),
   priorizar `share_story`/`self_disclose` (papéis complementares).
6. **Revelação recíproca:** se `user_disclosed` > 0,6, então `self_disclose` ou `comfort_support` (conforme a
   seriedade), sem mudar de tópico nem brincar.
7. **Confiança:** se a confiança do Choice M1 for < 0,35, usar só `prior × regras` (o Jev fica como desempate).
8. **Resgate:** se a média móvel de `engagement` cair ≥ 0,7 abaixo do pico da sessão **ou** o risco de fim estiver entre
   0,15 e 0,35, forçar `share_story` + `ask_personal` curta.

### 4.5 O que vai no briefing da LLM

```
move: share_story            # do diretor
hook_required: true          # termina com algo fácil de responder (pergunta leve OU afirmação convidativa)
ask_back: false              # devolver a pergunta?
humor: off|light|on          # por D4/D9
length_target: z_user*0.1 + base_persona   # espelha levemente o tamanho do usuário (A11)
topic: keep|smooth_shift|resume:"<fio>"    # nada de "btw" abrupto
tone_hint: phase, seriousness
avoid: "não despedir-se formalmente", "não fazer 2 perguntas seguidas", "não devolver 'e você?' mecanicamente"
```

### 4.6 Entrega (código) ligada à dinâmica

- **Atraso:** espelhar o ritmo do usuário (ρ = 0,30). `delay_bot = clamp(mediana_das_latências_do_usuário × U(0,6;1,2),
  piso, teto)`. Depois de o usuário sumir por muito tempo, **não** responder instantaneamente ao que ele mandou ao
  voltar.
- **Resposta rápida = curta:** turnos de reação saem rápido e em 1 bolha. História e revelação saem mais devagar e
  podem ter 2–3 bolhas (espelhando o nº de bolhas do usuário, ρ = 0,12–0,21).
- **Reabertura de sessão (o bot inicia):** abrir com uma **pergunta** (54% das aberturas humanas), de preferência
  retomando um `open_thread`. Saudação sozinha só em 10% dos casos.
- **Fim:** se o usuário parou de responder, não mandar "você sumiu?". Deixar a sessão morrer e reabrir depois com
  gancho.

---

## 5. Limitações

- **Tudo é observacional.** "Movimento → reação" pode refletir quem faz o movimento e quando (confusão por contexto).
  Controlei o engajamento anterior, a fase, a seriedade, a posição e a conversa, mas não há randomização. A regressão à
  média é forte nas subidas e descidas (A9).
- **Circularidade de rótulo:** movimento e engajamento vêm do mesmo modelo (Jev). O engajamento do Jev correlaciona
  0,51–0,67 com o tamanho do turno, então parte do "engajamento" é "tamanho". Para contornar, reportei também o tamanho
  (`zch`) e a latência (`zlat`), que são de código.
- **whatsapp_nl** é holandês, de 2012–14, com timestamps de minuto. O Jev é pior em holandês; por exemplo, a AUC de
  `p_end` cai de 0,82 para 0,66. "Fim" = 3 h de silêncio mistura desinteresse com indisponibilidade. As janelas
  anotadas (60 turnos por chat) cortam sessões, o que afeta a tipologia de formas.
- **maichat:** 42 conversas entre conhecidos, com duração imposta, então não serve para continuidade. As taxas de
  pergunta com "?" são baixas (9,4%), porque muitas perguntas vêm sem "?". Os efeitos por conversa (ρ) têm n = 42.
- **Amostras pequenas:** elogio (n = 88), devolução de pergunta (n = 30–44), fase profundo/tenso (n = 357 pares),
  "volta a tópico" (n = 92). Esses efeitos ficam como hipóteses.
- **Callback:** a validade do Noul foi checada só contra um proxy lexical (AUC 0,75), não contra rótulo humano. O nulo
  pode refletir ruído do detector. O estudo não avaliou callbacks entre sessões ("como foi a prova ontem?"), que podem
  funcionar de outro jeito.
- **Pessoas × bot:** os dados são entre humanos que já se conhecem. Um usuário falando com um bot pode tolerar mais
  perguntas e esperar mais iniciativa. Os priors do diretor devem ser recalibrados com logs próprios (A/B nos pesos).
- O Jev custou 2.743 chamadas novas (US$ 0,20). Reutilizei `jev_base` para todo o resto.
