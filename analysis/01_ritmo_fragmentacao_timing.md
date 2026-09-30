# 01 — Ritmo, fragmentação e timing (a "camada de entrega")

**Pergunta:** quantas bolhas as pessoas mandam, como quebram o texto, quanto tempo levam entre bolhas, para responder e
para digitar, e o que disso o bot deve imitar. **Dados:** `maichat` (42 conversas em inglês, ao vivo, 2.871 turnos,
4.177 mensagens, logs de digitação com resolução de ms) e `whatsapp_nl` (57 chats diádicos em holandês, 33.396 turnos,
timestamps com resolução de 1 min). As variáveis de "momento" (emoção, arousal, ansiedade, seriedade…) vêm do passe **D**
de `jev_base.jsonl` (2.871 turnos do maichat e 3.000 do whatsapp_nl; o Jev viu as bolhas **juntadas**, então não há
circularidade entre o rótulo e o número de bolhas). Experimentos novos de Jev: **2.716 chamadas (≈ US$ 0,10)**.
Scripts em `scripts/analysis/a1_*.py`; saídas em `analysis/data/a1_*`.

---

## 1. Resumo dos achados

- **Fragmentar é a regra, não a exceção.** 31% dos turnos do maichat e 47% do whatsapp_nl têm mais de uma bolha. Como
  esses turnos são justamente os mais longos, **53% e 71% de todas as mensagens** estão dentro de bursts. Distribuição (WA):
  1 = 52%, 2 = 29%, 3 = 11%, 4 = 4%, 5+ = 3%.
- **O número de bolhas é decidido principalmente pelo *quanto* a pessoa tem a dizer.** Só o log do tamanho total explica
  R² = 0,25–0,29 de "mandou >1 bolha"; a identidade do falante explica 0,13–0,19; todas as variáveis de momento do Jev
  juntas, 0,15–0,18. Até 20 caracteres, 87–93% vão em 1 bolha; acima de 160 caracteres, 44–45% vão em 4 ou mais.
- **A hipótese do usuário se confirma só em parte.** O motor da fragmentação "picotada" é a **agitação (arousal)**, não a
  ansiedade: controlando o tamanho, cada nível de arousal (0–4) soma +6 a +7 pp na chance de mandar várias bolhas e
  deixa cada bolha **32–36% mais curta**. A ansiedade aumenta o número de bolhas no WhatsApp (2,18 contra 1,58), mas
  porque o ansioso **escreve mais**: as bolhas dele são **mais longas** (37 contra 22 caracteres), não mais curtas.
  Ansioso também **não responde mais rápido** (latência igual ou maior). O "mais rápido" só aparece na velocidade de
  digitação: 5,4 contra 4,3 caracteres/s.
- **Seriedade → bolhas mais longas (+27–30% por nível), mas NÃO menos bolhas**, e mais reescrita: a chance de apagar
  sobe (OR 2,7 por nível de seriedade) e a de abandonar texto também (OR 2,7). Em momentos "sérios" (maichat), 11% das
  mensagens têm um rascunho apagado inteiro e reescrito, contra 3,4% na zoeira.
- **Gramática das bolhas** (Jev em 1.216 bursts de 2–4 bolhas): o padrão dominante é **conteúdo principal + continuação**
  (a mesma ideia partida em duas: 24% e 20% das transições). Depois vem **reação curta + conteúdo** ("That's true" / "But
  some cars are nice"). A reação abre o burst em 29–34% dos casos. Quando há uma única pergunta, ela fica **no fim** em
  59–69% das vezes (o acaso daria 43%). **Risada abre** o burst ("hahaha" / conteúdo: 835 contra 213 no fim, no WA),
  **emoji fecha**. A 1ª bolha é a mais curta (mediana de 16–17 caracteres contra 21–25 nas seguintes).
- **Tempo entre bolhas (ao vivo):** mediana de 8,7 s (p25 = 4,7, p75 = 16,5). A pessoa começa a próxima bolha quase na
  hora (a mediana do ócio é 1,7 s; 37% em menos de 1 s), e o intervalo cresce com o tamanho da bolha seguinte: 3,3 s
  (≤ 5 caracteres) → 17,7 s (> 60 caracteres). Uma reação ou risada vem em ~4,5 s; uma continuação, em ~10 s.
- **A latência de resposta quase não depende da emoção**: correlações dentro do mesmo falante |ρ| ≤ 0,08. O que manda
  é o **ritmo do outro**: a latência do parceiro prevê a minha com ρ = 0,43–0,44 (0,36 dentro do falante no WA).
  E no WA, **responder devagar está associado ao fim da conversa**: 6% dos turnos respondidos em menos de 1 min são os
  últimos da sessão, contra 23% dos respondidos 1–3 h depois.
- **Fórmula do "digitando…"** (mediana humana, digitação "limpa"): `t ≈ 0,15 s + 0,17 s × caracteres` (p25: 0,14 s/caractere;
  p75: 0,57 s + 0,23 s/caractere), o que dá ≈ 5–6 caracteres/s. Moduladores: seriedade ×1,17 por nível (p = 0,06); tensão ou
  irritação ×0,6 (digitam **mais rápido e sem editar**); mais pausas longas no sério (28% contra 14%).
- **"Digitou e desistiu" existe e tem gatilho claro:** 30% das respostas começam a ser digitadas **antes** de chegar a
  mensagem do outro. Nesses casos, 18% apagam o rascunho inteiro e recomeçam (contra 0,9%) e 75% editam (contra 15%).
- **Bolha extra depois ("ah, e…")**: 7,7% dos turnos do WA têm uma bolha enviada ≥ 2 min depois, sem resposta no meio
  (mediana de 10,5 min). Ela é **pergunta** em 29% dos casos, contra 16,5% nas outras bolhas: é uma reabertura.
- **Fim de turno do usuário** (para o bot não "atropelar"): depois de cada bolha do usuário há 32% (ao vivo) a 48% (WA)
  de chance de vir outra. 56% das continuações chegam em até 10 s, 83% em até 20 s e 92% em até 30 s. Bolha terminando
  em "?" é seguida por outra em 23–25% dos casos; risada, em 54–68%. O Jev (Noul "terminou?") tem AUC de 0,61–0,65,
  um pouco melhor que as regras de código (0,58–0,60).
- **Passe P do Jev para `p_n_msgs`: falhou.** Ele prevê "1 bolha" em ≈ 100% dos casos, com P(várias) média de 2% contra
  31–48% reais, e fica abaixo da taxa-base no Brier. A causa é em parte **um artefato do state** (as bolhas iam
  juntadas). Com as bolhas visíveis e o histórico do falante no state, a calibração se corrige (média de 42–51%) e a AUC
  sobe para 0,73 (maichat) e 0,62 (WA), **empatando** com a regra de código "taxa de fragmentação do próprio falante".
  Já o **`p_length` funciona**: é o melhor preditor isolado do tamanho do próximo turno (ρ = 0,44 e 0,33, contra 0,34 e
  0,26 do "espelhar o tamanho do parceiro") e soma quando combinado com código.

---

## 2. Achados detalhados

### A1. Distribuição de bolhas e peso do estilo pessoal
| | maichat | whatsapp_nl |
|---|---|---|
| turnos | 2.871 | 33.396 |
| média de bolhas por turno | 1,45 | 1,79 |
| % de turnos com > 1 bolha | 31,3% | 47,5% |
| % das mensagens que estão em bursts | 52,8% | 70,7% |
| 1 / 2 / 3 / 4 / 5+ | 69 / 22 / 6 / 2 / 1% | 52 / 29 / 11 / 4 / 3% |

Há um **estilo pessoal** forte: entre falantes com 20 turnos ou mais, a taxa de turnos com várias bolhas vai de 7% (p10)
a 57% (p90) no maichat e de 25% a 63% no WA. O ICC de "várias bolhas" é 0,17 no maichat e 0,07 no WA. Também existe
**espelhamento**: P(várias | o parceiro mandou várias) = 45% contra 26% (maichat) e 53% contra 44% (WA).
*Confiança: alta.*

### A2. O que explica o número de bolhas
Regressão com efeito fixo por falante e erro agrupado por conversa (`a1_bursts.py`):
- **Tamanho total** é o que mais pesa (R² isolado de 0,29 e 0,25). Tabela de consulta (`a1_split_table.json`):

| total de caracteres | maichat P(1) / média de bolhas | WA P(1) / média de bolhas | bolha mediana (nos bursts) |
|---|---|---|---|
| 1–20 | 93% / 1,08 | 87% / 1,14 | 7 |
| 21–40 | 71% / 1,34 | 58% / 1,50 | 14–15 |
| 41–80 | 38% / 1,85 | 34% / 1,98 | 24–25 |
| 81–160 | 16% / 2,57 | 21% / 2,65 | 39–41 |
| > 160 | 8% / 3,74 | 16% / 3,72 | 56–60 |

- **Engajamento** (D) é a variável de momento mais forte: ρ = 0,32–0,39 com o número de bolhas e 0,43–0,55 com o
  tamanho das bolhas. **Vulnerável**: ρ = 0,23–0,34 com o número de bolhas. Quem se abre escreve mais e em mais bolhas.
- **Intenção** (média de bolhas, maichat / WA): `share_story` 1,74 / 2,05, `answer` 1,67 / 1,88,
  `make_plans` 1,65 / 1,95; no outro extremo, `react_acknowledge` 1,19 / 1,28 e `compliment_affection` 1,20 / 1,65.
  **Fase**: `conflict_or_repair` 1,18 no maichat (discussão ao vivo = mensagem única e seca), mas 1,96 no WA;
  `deep_personal` 1,63 / 2,20 (o maior no WA); `winding_down` 1,35 / 1,47.
- **Tamanho da mensagem do parceiro**: ρ = 0,25 (maichat) e 0,11 (WA) com o número de bolhas. **Posição**: o primeiro
  turno quase nunca é fragmentado (13% no maichat); depois disso, efeito fraco (ρ = −0,04 a 0,05).
*Confiança: alta para tamanho, estilo e engajamento; média para intenção e fase (rótulos do Jev, sobretudo em holandês).*

### A3. Teste direto da hipótese "ansiedade/agitação → mais bolhas, mais curtas e mais rápidas; calma/seriedade → menos e mais longas"

| grupo (rótulo D) | n | média de bolhas | % multi | bolha mediana (caracteres) | total mediano | latência mediana |
|---|---|---|---|---|---|---|
| **maichat** ansioso ≥ 0,5 | 291 | 1,41 | 27% | 19 | 22 | 17,0 s |
| maichat não ansioso < 0,2 | 1.727 | 1,46 | 32% | 20 | 25 | 12,3 s |
| maichat arousal ≥ 3 | 162 | **1,86** | **48%** | **17** | 30,5 | 11,2 s |
| maichat arousal ≤ 1,5 | 1.550 | 1,33 | 26% | 19 | 22 | 14,2 s |
| **WA** ansioso ≥ 0,5 | 365 | **2,18** | **58%** | **37** | 71 | 60 s |
| WA não ansioso < 0,2 | 1.621 | 1,58 | 39% | 22,5 | 34 | 60 s |
| WA arousal ≥ 3 | 167 | **2,22** | **56%** | 27 | 47 | 60 s |
| WA arousal ≤ 1,5 | 1.520 | 1,51 | 36% | 25 | 35 | 60 s |

Com controle de tamanho total, falante, engajamento, valência e contexto (probabilidade linear de "várias bolhas"):
- **arousal**: +7,2 pp por nível (p = 3e-7) no maichat e +5,7 pp (p = 0,003) no WA. No modelo do tamanho da bolha
  (log), −0,38 e −0,44 por nível, ou seja, **bolhas ×0,64–0,68 por nível**. Dentro da mesma faixa de tamanho, "energético"
  manda mais bolhas: 41–100 caracteres, 2,20 contra 1,80 (maichat) e 2,09 contra 1,63 (WA). **Confirmado, efeito moderado.**
- **ansiedade**: +17 pp (de 0 a 1 no Noul, p = 0,001) no maichat; no WA, +6,5 pp (p = 0,34, não significativo). Nas
  bolhas, −0,22 (p = 0,002) e −0,19 (p = 0,13) no log. Sem controles, porém, o ansioso do WA escreve bolhas **mais
  longas**. Exemplo real (WA, ansioso): *"Ohhh toch wel!" / "Oké nee ja dat kan. Fijn dat jullie het allebei hadden" /
  "Maar de vorige keer vond je 'm nog wel leuk toch? …"*: são mais bolhas, e compridas. **Parcialmente confirmado:
  "mais bolhas" sim, "mais curtas" não.**
- **"mais rápidas"**: a latência de resposta não muda com ansiedade ou arousal (ρ dentro do falante entre −0,05 e 0,03);
  o intervalo entre bolhas também não (ρ entre −0,03 e 0,08). O que acelera é a **digitação**: ansiosos a 5,4 caracteres/s
  contra 4,3, e com metade das edições (22% contra 41% apagam algo). **Não confirmado para latência; confirmado para
  velocidade de digitação.**
- **seriedade**: bolhas **+27–30% por nível** (p < 0,001), sem efeito no número de bolhas depois de controlar o tamanho
  (−0,01, p = 0,7). No WA, os turnos sérios têm até **mais** bolhas (2,14 contra 1,71), porque são longos. **"Mais
  longas" confirmado; "menos bolhas" não.** Exemplo (maichat, vulnerável): *"gosh im so stupid" / "i really hate myself
  sometimes"*.
- **brincadeira (playful)**: com o tamanho fixo, a zoeira gera **mais bolhas e mais curtas** que o casual e o sério
  (41–100 caracteres, maichat: 2,09 bolhas de 29,5 caracteres na zoeira contra 1,65 de 48 no sério). Picotar é marca da zoeira.
- Exemplos de alta agitação (maichat): *"wait what" / "ohhhhh" / "hahahhahahhahah" / "i got scared there"*;
  *"omg a fridge!!!!" / "like the guy who is running a marathon with a fridge" / "^^^ yes drawers would be a slay"*.

*Confiança: alta para arousal e seriedade → tamanho da bolha; média para ansiedade (o rótulo é ruidoso: no maichat "anxious"
pega muita provocação de amigos). No maichat quase não há momentos sérios (13 turnos com seriedade ≥ 2).*

### A4. Como as pessoas quebram as bolhas
Heurística de código em todos os bursts de 2–4 bolhas (866 do maichat e 14.875 do WA) mais **Jev (Choice, 10 funções)**
em 1.216 bursts: todos os do maichat e 350 do WA. A concordância entre Jev e heurística foi de 73% e 83%; a confiança
mediana do Jev, 0,86 e 0,82.

| função (Jev) | maichat | WA | onde aparece |
|---|---|---|---|
| continuação (add_on) | 29% | 23% | **última bolha em 51% / 41%** |
| conteúdo principal | 27% | 27% | 1ª bolha em 49% / 45% |
| reação curta | 17% | 19% | **1ª bolha em 29% / 34%** |
| pergunta | 13% | 15% | última em 17% / 25% |
| ponto novo depois ("ah, e…") | 4,4% | 5,7% | última em 7% / 9% |
| risada solta | 2,5% | 4,5% | 1ª em 4% / 9% |
| emoji solto | 1,4% | 3,1% | última em 1,5% / 5% |
| suavizador ("jk", "anyway") | 2,2% | 1,0% | |
| correção ("it*") | 0,9% | 0,1% | última |

Sequências mais comuns (Jev, maichat): `main+add_on` (188), `reaction+add_on` (93), `main+question` (37),
`reaction+question` (34), `main+add_on+add_on` (29), `main+afterthought` (23).
Exemplos reais: *"i used to, mainly for cute animal videos, but it started to get too distracting" / "so i uninstalled
it"* (conteúdo + continuação); *"That's true" / "But some cars are nice"* (reação + conteúdo); *"Almost certainly - its
always popular." / "do you want brussels sprouts?"* (conteúdo + pergunta); *"hahaha" / "The guy at the cafe thought it
was so cool I paid for my muffin with my watch"* (risada + conteúdo); *"Also how long is it gonna take us to walk
there's" / "*there?"* (correção).
- Com uma só pergunta no burst, ela fica **no fim** em 59% (maichat) e 69% (WA) dos casos, contra 43% ao acaso.
  Com uma só reação, ela fica **no começo** em 63% e 69%, contra 43–45% ao acaso.
- Risada **abre** o burst: no WA, 835 vezes na 1ª posição contra 213 na última; no maichat, 20 contra 8. Emoji solto
  **fecha** no WA (665 contra 430).
- Tamanho mediano por posição (WA, bursts de 4): 14 → 21 → 26 → 25 caracteres. A **1ª bolha é a mais curta**: é a
  "abertura rápida".
*Confiança: alta para posições e sequências (repetem nos dois corpora e nos dois métodos); média para a taxonomia fina.*

### A5. Tempo dentro do burst
- **Maichat** (1.306 intervalos): p10 = 2,7 s, p25 = 4,7 s, **mediana = 8,7 s**, p75 = 16,5 s, p90 = 28 s. O intervalo é
  quase todo gasto **digitando a próxima bolha**: o ócio entre enviar uma e começar a outra tem mediana de 1,7 s (37% abaixo
  de 1 s, e 12% começam a próxima antes de enviar a anterior). O intervalo cresce com o tamanho da bolha seguinte
  (ρ = 0,49): ≤ 5 caracteres 3,3 s; 5–15 5,0 s; 15–30 8,6 s; 30–60 12,3 s; > 60 17,7 s.
- Por função da bolha seguinte (Jev, maichat): reação 4,5 s (p25–p75: 2,7–6,9), risada 4,1 s, suavizador 4,8 s,
  correção 5,0 s, emoji 6,8 s, continuação 9,8 s (6,1–18,2), ponto novo 11,5 s, pergunta 11,8 s.
- **WA** (26.396 intervalos, resolução de minuto): 71% na mesma hora-minuto, 20% no minuto seguinte, 4,7% entre 2 e 9 min
  e 5% acima de 10 min.
- A emoção quase não mexe no intervalo entre bolhas (|ρ| < 0,09).
- **Invisível:** respostas rápidas vêm **mais** fragmentadas no maichat: 40% têm várias bolhas quando a latência é ≤ 5 s,
  contra 20% quando é de 30–60 s. A pessoa manda uma 1ª bolha curta logo e completa depois.
*Confiança: alta (maichat), mas é chat de laboratório; no WA o intervalo não é mensurável.*

### A6. Latência de resposta
- **Maichat**: p10 = 2,8 s, p25 = 6,2 s, **mediana = 13,8 s**, p75 = 27 s, p90 = 38 s. **WA**: 36% no mesmo minuto,
  67% em até 2 min, mediana de 60 s, p75 = 5 min, p90 = 21 min e 3,9% acima de 1 h (dentro da sessão).
- **Por emoção** (maichat, agregado): surpresa 8,6 s, curiosidade 9,9 s e alegria 10,0 s, contra frustração 19,0 s e
  tédio 21,0 s. Isso, porém, é sobretudo diferença **entre pessoas e conversas**: dentro do mesmo falante, as correlações
  com valência, ansiedade e arousal ficam entre −0,01 e 0,01. No WA, quase tudo tem mediana de 60 s. Os turnos muito
  sérios (nível 3) têm mediana de 360 s, mas **n = 16**.
- **Engajamento × latência**: fraco. No WA, ρ dentro do falante = −0,08 (mais engajado, um pouco mais rápido); no
  maichat, ≈ 0.
- **Depois de uma pergunta**: no maichat, 12,3 s contra 14,0 s (p = 0,08); no WA, **mais lento**, 120 s contra 60 s
  (p = 6e-7), porque as perguntas do WA são de logística e respondê-las dá trabalho. **Depois de uma confissão
  vulnerável do parceiro**: 15,1 s contra 13,4 s (maichat) e 120 s contra 60 s (WA). As pessoas **não** respondem mais
  rápido a uma confissão; demoram um pouco mais e escrevem mais.
- **Espelhamento de ritmo** (o maior efeito): ρ(latência do parceiro, minha latência) = 0,43 (maichat) e 0,44 (WA);
  dentro do falante, 0,07 e 0,36. Duas pessoas convergem para um tempo comum.
- **Lentidão e fim da conversa** (WA): a fração de turnos que são os últimos da sessão sobe com a latência: < 1 min 6,0%;
  1 min 8,6%; 2–5 min 10,8%; 6–30 min 15,3%; 31–60 min 18,5%; 1–3 h 22,9%. A resposta seguinte do parceiro também
  demora mais (mediana de 0 → 300 s). É correlação, e a causalidade pode ir nos dois sentidos, mas o sinal é claro.
*Confiança: alta para espelhamento e fim de sessão; baixa para os efeitos de emoção.*

### A7. Digitação (maichat, logs brutos; `a1_typing.py` e `a1_typing2.py`)
- Composição (1ª tecla → enviar): mediana de 3,9 s (p25 = 2,3, p75 = 9,0). Velocidade: **4,6 caracteres/s** (p25 = 2,8,
  p75 = 6,9); 5,0 sem as pausas de mais de 2 s. Celular 5,1 contra desktop 4,3. Quem digita no desktop edita mais
  (42% contra 23% apagam algo).
- **Fórmula (regressão quantílica, 3.046 mensagens "limpas": digitadas depois de a mensagem do outro chegar e sem reset):**
  - mediana: `t = 0,15 + 0,173 × caracteres` (s); p25: `0,00 + 0,138 × caracteres`; p75: `0,57 + 0,234 × caracteres`;
  - por faixa (mediana e p25–p75): ≤ 5 caracteres 1,5 s (0,9–2,2); 5–15 2,0 s (1,6–2,7); 15–30 3,3 s (2,7–4,5);
    30–60 7,1 s (5,1–10,4); 60–120 17,3 s (13,7–22,7). Mensagens longas crescem mais que linearmente, porque têm pausas.
- **Hesitação por momento** (mensagem individual):

| momento | n | caracteres/s | apaga algo | pausa > 2 s | rascunho reescrito |
|---|---|---|---|---|---|
| zoeira (seriedade 0) | 2.326 | 5,0 | 28% | 14% | 3,4% |
| casual (1) | 1.724 | 4,0 | 45% | 26% | 6,8% |
| sério (2–3) | 90 | 4,4 | 53% | 28% | 11,1% |
| tensão ≥ 0,5 | 546 | **6,8** | **14%** | 9% | 1,6% |
| frustração/raiva | 256 | 6,5 | 17% | 8% | 2,3% |
| tédio | 90 | 6,7 | 6% | 7% | 2,2% |
| ansiedade/insegurança | 95 | 4,1 | 33% | 26% (**18% > 5 s**) | 9,5% |

  Modelo logístico, controlando tamanho e aparelho: **seriedade** OR 2,7 por nível para apagar (p = 3e-5) e para
  abandonar texto (p = 4e-4); **tensão** OR 0,12 para apagar (p = 0,003). No log do tempo de composição: seriedade
  ×1,17 por nível (p = 0,06), tensão ×0,60 (p = 6e-6), brincadeira ×0,69. "Vulnerável" **não** tem efeito próprio depois
  de controlar o tamanho (OR 1,15, p = 0,64): a hesitação vem da seriedade do momento, não da confissão em si.
- **"Digitou e desistiu"**: 5% das mensagens têm um rascunho de 8 caracteres ou mais apagado inteiro e reescrito, e 2,3%
  terminam com texto abandonado (o pico de tamanho superou o final em 3 caracteres ou mais). Exemplos (rascunho → enviado):
  *"he was growing up around the time elvis" → "he had quite a complicated relationship with his mom and dad"*;
  *"I'm trying to prepare for your return." → "OK"*; *"wowow i forgot you have the mac" → "do you have the mac"*.
- **Gatilho da desistência**: 30% das respostas começam a ser digitadas **antes** de chegar a mensagem do parceiro.
  Nessas, 18,1% têm reset do rascunho (contra 0,9%) e 75% apagam algo (contra 15%). A nova bolha do outro "atropela" o
  que a pessoa ia dizer, e ela refaz.
- **Tempo de "pensar"** (mensagem do outro chega → 1ª tecla): mediana de 14,6 s (p25 = 5,6, p75 = 28,6). **Não cresce
  com o tamanho da mensagem do outro** (ρ = −0,09; após mensagens de mais de 60 caracteres, 7,5 s). Não há um "tempo de
  leitura" proporcional ao tamanho. Depois de uma fala vulnerável, 18,0 s contra 14,3 s.
*Confiança: média-alta. São 42 conversas de universitários em plataforma de pesquisa, e a digitação no desktop pesa.*

### A8. A bolha extra depois ("ah, e…") e o double text
- WA: 2.562 bolhas enviadas ≥ 2 min depois da anterior do mesmo falante, sem resposta no meio, em **7,7% dos turnos**.
  Intervalo: p25 = 3 min, **mediana = 10,5 min**, p75 = 37 min. São perguntas em **29%** dos casos, contra 16,5% nas
  outras bolhas não iniciais. Exemplos: *"Oké rond half zes gaan?" … (26 min) … "Of later?"*;
  *"Slaap lekker xx" … (4 min) … "Jij bent lief"*; *"Mooi!" … (21 min) … "Hey zusje!"* (reabertura).
- Ao vivo (maichat), 10% dos intervalos dentro do burst passam de 28 s.
*Confiança: média, porque não dá para saber se o outro leu a mensagem.*

### A9. Fim do turno do usuário (quando o bot pode começar)
- P(a próxima mensagem é do mesmo falante): 32% (maichat) e 48% (WA). Por função da bolha (WA / maichat): pergunta
  26% / 23%; conteúdo 48% / 30%; reação 60% / 38%; risada 68% / 54%; mídia 69% (WA); reticências 61% (maichat).
- Risco ao longo do silêncio (maichat): P(ainda vem outra bolha) = 32% em 0 s, 29% após 5 s, 25% após 10 s, 19% após
  20 s e 16% após 30 s. Das continuações, 28% chegam em até 5 s, 56% em até 10 s, 72% em até 15 s, 83% em até 20 s e
  92% em até 30 s.
- Regras de código (logística agrupada por conversa): AUC de 0,60 e 0,61. **Jev** (Noul "o falante terminou o turno?",
  250 mensagens por corpus, bolhas visíveis no state): AUC de **0,65** (maichat) e 0,61 (WA), com Brier de 0,188 contra
  0,199. A média Jev + código chega a 0,67 no maichat. O sinal é fraco mas útil: a espera é o que decide.
*Confiança: alta para as curvas de tempo; média para os detectores.*

### A10. Validação do passe P (o Jev prevê o próximo turno?)
Turnos com P e pelo menos 1 turno anterior do mesmo falante: 2.786 no maichat e 2.559 no WA (`a1_pvalid.py`).

| | maichat | WA |
|---|---|---|
| acurácia de 1/2/3/4+: Jev / sempre 1 / repetir o próprio anterior / espelhar o parceiro / média do próprio histórico | 0,686 / 0,686 / 0,583 / 0,586 / 0,597 | 0,525 / 0,525 / 0,404 / 0,425 / 0,420 |
| Spearman com o número real de bolhas: Jev (valor esperado) / média do próprio histórico / tamanho do parceiro | 0,21 / **0,35** / 0,26 | 0,10 / **0,28** / 0,14 |
| AUC de "várias bolhas": Jev / histórico próprio | 0,61 / **0,71** | 0,55 / **0,64** |
| Brier: Jev / taxa-base / histórico próprio | 0,300 / 0,215 / **0,198** | 0,453 / 0,249 / **0,239** |
| P(várias) média prevista / real | 0,018 / 0,314 | 0,024 / 0,475 |
| **p_length**: Spearman com os caracteres (Jev / tamanho do parceiro / próprio anterior / média do próprio histórico) | **0,44** / 0,34 / 0,27 / 0,41 | **0,33** / 0,26 / 0,18 / 0,27 |
| CV agrupada, AUC de "várias": código / Jev / código + Jev | 0,730 / 0,674 / **0,736** | 0,666 / 0,570 / 0,667 |
| CV agrupada, Spearman do tamanho: código / Jev / código + Jev | 0,404 / 0,434 / **0,457** | 0,266 / 0,327 / **0,330** |

- **`p_n_msgs` é inútil do jeito que foi perguntado:** o argmax é "1" em ≥ 99,9% dos casos, a calibração está muito
  errada (quando o Jev dá ≤ 5%, o real é 30–47%) e a confiança fica ≥ 0,8 em 97–98% dos casos, então não serve como
  filtro. **Experimento novo** (`a1_pexp.py`, 250 turnos por corpus, 1.000 chamadas) com o state trazendo as bolhas
  separadas (V1) e, além disso, uma linha de código com o histórico do falante (V2):

| AUC / Brier / P média prevista | P base (bolhas juntadas) | V1 Noul | V2 Noul | só código (taxa própria) | taxa-base |
|---|---|---|---|---|---|
| maichat (taxa real de 29%) | 0,64 / 0,277 / 1% | 0,70 / 0,214 / 46% | **0,73 / 0,196 / 42%** | 0,71 / **0,192** / 32% | – / 0,207 |
| WA (taxa real de 47%) | 0,51 / 0,453 / 3% | 0,53 / 0,257 / 50% | **0,62 / 0,239 / 51%** | 0,61 / 0,249 / 45% | – / 0,249 |

  Conclusão: mostrar as bolhas **corrige a calibração**. Mesmo assim, prever quantas bolhas o *usuário* vai mandar é
  difícil (teto de AUC em torno de 0,7), e o Jev só empata com a regra de código.
- **`p_length` funciona** e é o melhor preditor isolado; o viés é pequeno (+0,2 nível no maichat). É útil para o bot
  **espelhar o tamanho** da resposta.
*Confiança: alta.*

---

## 3. Padrões "invisíveis" (o que as pessoas fazem sem perceber)

1. **Abertura rápida e depois o resto.** A 1ª bolha é a mais curta e muitas vezes é uma reação ("true", "wait what",
   "hahaha"). O resto vem 5–10 s depois. Respostas rápidas vêm mais fragmentadas (40% contra 20%).
2. **A mesma frase partida no meio.** A continuação (add_on) é a função mais comum: *"satisfying both parts of the brain
   lol" / "yes non fiction"*. Não é uma ideia por bolha, é **pensamento em voz alta**.
3. **Risada na frente, emoji no fim, pergunta por último.** A pergunta separada no fim devolve a vez ao outro.
4. **Zoeira picota, seriedade alonga.** Para o mesmo tamanho total, a zoeira usa mais bolhas curtas e o momento sério
   usa bolhas longas. Ansiedade no WhatsApp gera **muito texto**, não bolhas curtinhas.
5. **Briga se digita rápido e sem apagar** (6,8 contra 4,2 caracteres/s; 14% contra 43% editam). Momento sério se
   digita com pausa e reescrita (11% resetam o rascunho).
6. **Tempo espelhado.** Cada pessoa responde no ritmo em que é respondida (ρ ≈ 0,4). E quem demora muito tende a
   encerrar a conversa (6% → 23%).
7. **Resposta atropelada.** Um terço das respostas começa enquanto o outro ainda está falando. Se chega uma nova bolha,
   a pessoa apaga e refaz (18% contra 1%).
8. **O "pensar" não depende do tamanho do que se leu.** A mediana é de ~15 s ao vivo, e é até menor depois de mensagens
   longas.
9. **Correção com asterisco** ("it*", "*there?") em 0,4–0,6% das bolhas, ~5 s depois do erro. É um detalhe raro, mas
   muito humano.
10. **"Ah, e…" depois de minutos** (7,7% dos turnos no WA), quase sempre para puxar assunto de volta (pergunta em 29%).

---

## 4. Tradução para o sistema: especificação da camada de entrega

### 4.1 Fluxo (em ordem de execução)
```
usuário manda bolha ─► [E1 fim de turno? código + Jev opcional] ─► espera (debounce)
                        │ nova bolha no meio? volta a E1 (e, se o bot já "digitava", aplica E7)
                        ▼
  cérebro (Jev: roteamento, briefing) ─► LLM gera o texto T
                        ▼
  [E2 perfil do momento sobre T + contexto]  (Jev, 1 chamada, ~0,5 s, em paralelo)
                        ▼
  [E3 nº de bolhas]  (código: tabela A2 × moduladores)  ─► [E4 divisão do texto] (código + Jev opcional)
                        ▼
  [E5 latência inicial] ─► para cada bolha: [E6 "digitando…" + envio] ─► intervalo ─► próxima
                        ▼
  [E8 bolha extra "ah, e…"]  (temporizador depois do silêncio do usuário)
```

### 4.2 Detectores e regras

**E1: fim do turno do usuário (não atropelar).**
- *Código*: `p_mais = base_usuário` (taxa de fragmentação do usuário nesta conversa, com prior de 0,35), ajustada pela
  última bolha: termina em "?" ×0,6; risada, emoji solto ou reação ×1,4; reticências ×1,8; mídia ×1,4. Janela de
  espera `W = 4 s` se `p_mais < 0,25`, `8 s` se estiver entre 0,25 e 0,45, `12 s` se `> 0,45`. Referência (maichat): uma espera de 5 s
  captura ~28% das continuações, 10 s ~56%, 15 s ~72% e 20 s ~83%; o risco residual de vir outra bolha cai de 32% para
  25% (10 s) e 19% (20 s). É um compromisso entre não atropelar e não parecer lento. Se o
  cliente mostrar que o usuário está digitando, esperar até 20 s.
- *Jev (opcional, só na zona ambígua 0,3–0,5 e rodando **durante** a espera)*: **Noul**, instrução "Is `current_speaker`
  done with their turn, i.e. will they now wait for the other person to reply instead of sending another message right
  away?". State: as últimas 6 falas, com as bolhas separadas em listas, mais `current_speaker_messages_so_far`. Regra:
  `p = média(p_código, 1 − p_jev)`; se `p > 0,5`, estender a janela em +6 s. Medido: AUC de 0,61–0,65 e 0,67 combinado.

**E2: perfil de entrega da resposta (1 chamada ao Jev sobre o texto já gerado).**
State: as últimas 6 falas mais `bot_reply_draft` (texto da LLM) mais o briefing do "cérebro" (humor da persona). Perguntas:
- `arousal`: **Score** de 0 a 4 ("How energetic/excited is the reply?"; níveis: muito calmo, calmo, moderado, energético,
  muito agitado);
- `seriousness`: **Score** de 0 a 3 (zoeira, casual, meio sério, muito sério/emocional);
- `playful`, `tension` e `vulnerable_moment`: **Noul** ("Is the user sharing something vulnerable or is the moment
  emotionally serious?").
Esses rótulos, extraídos da mesma forma no passe D, têm os efeitos medidos em A3 e A7. Usar `confidence < 0,5` →
valores neutros (arousal 1,5; seriousness 1).

**E3: número de bolhas (código).**
1. Base pelo tamanho total `L` de T (tabela A2): `L ≤ 20` → 1; `21–40` → 1 (30–40% de chance de 2); `41–80` → 2 (35% de
   chance de 1 e 20% de 3); `81–160` → 2–3; `> 160` → 3–4, com teto de 5. Sortear a partir da distribuição da faixa,
   não usar o argmax, para não ficar robótico.
2. Moduladores: `+0,2 bolha por nível de arousal acima de 1,5`; `playful > 0,6` → +0,3 e bolhas mais curtas;
   `seriousness ≥ 2` → −0,3 e bolhas mais longas (nunca picotar um momento sério em pedaços de menos de 25 caracteres);
   `tension > 0,6` no chat ao vivo → preferir 1 bolha seca (fase `conflict_or_repair` = 1,18 bolha no maichat).
3. **Estilo da persona**: um parâmetro fixo `frag_style` entre 0,1 e 0,6 (a faixa humana vai de 7% a 63%).
4. **Espelhar o usuário**: se a taxa de fragmentação do usuário nos últimos 10 turnos for maior que 0,5, somar +0,15
   em P(várias); se for menor que 0,15, subtrair 0,15 (espelhamento medido: +9 a +19 pp).

**E4: como dividir T (código; o Jev opcional para rotular os segmentos).**
- Cortar em fronteiras de frase ou oração (". ", "? ", "! ", ", but", ", so", " lol", quebras de linha), buscando
  bolhas de 15–40 caracteres (60 no máximo, exceto nos momentos sérios).
- **Ordem canônica**: [reação curta ou risada] → [conteúdo principal] → [continuação(ões)] → [pergunta]. Emoji solto
  vai para o fim. Se a LLM pôs a pergunta no meio, movê-la para a última bolha quando isso não quebra o sentido.
- **1ª bolha curta**: se T começa com uma interjeição ou reação ("haha", "wait", "omg", "aww", "true"), ela vira a 1ª
  bolha sozinha (29–34% dos bursts humanos começam assim).
- Não deixar bolha com menos de 5 caracteres, a não ser que seja reação, risada ou emoji.
- *Jev opcional*: um **Choice** por segmento com as 10 funções de A4 (state = mensagem anterior do usuário mais
  `current_burst` numerado), para decidir junções: juntar `add_on` com menos de 12 caracteres à bolha anterior e
  garantir no máximo 1 `question` por turno, no fim. Concordância com a heurística: 73–83%. Vale só se a ordem da LLM
  vier ruim.
- *Correção com asterisco* (humanizador raro): no máximo 1 a cada 200 bolhas, só com `seriousness ≤ 0,5` e
  `playful > 0,5`. Enviar a palavra com erro e, ~5 s depois, "*palavra". Nunca num momento sério ou vulnerável.

**E5: latência antes de começar (código).**
- Alvo = mediana das últimas 5 latências do usuário × `k_ritmo` (espelhamento, ρ = 0,43), com piso e teto conforme
  o modo:
  - *chat ao vivo*: pensar = lognormal com mediana de 5 s (a mediana humana é 14,6 s; recomendo 0,3–0,5× o valor
    humano, porque ninguém quer esperar um bot) e limites de 1,5 a 20 s;
  - *assíncrono/WhatsApp*: mediana de 60 s, p75 de 5 min.
- **Não** escalar pelo tamanho da mensagem do usuário: não há efeito de leitura (ρ = −0,09).
- **Quando demorar um pouco mais**: `vulnerable_moment > 0,6` ou `seriousness ≥ 2` → ×1,3 (humanos: 15,1 s contra
  13,4 s ao vivo; 120 s contra 60 s no WA), e a resposta vem mais longa.
- **Quando NÃO demorar**: (a) o usuário está respondendo rápido (latência dele abaixo de 10 s ao vivo ou abaixo de 1 min
  no WA): espelhar; (b) a conversa está esfriando, isto é, a latência do usuário vem crescendo (no WA, respostas lentas
  precedem o fim da sessão, 6% → 23%). Nunca passar de 5 min no modo assíncrono, a não ser que a persona esteja
  "ocupada" de propósito; (c) respostas de reação curta (< 15 caracteres): começar em ≤ 3 s.
- **Abertura rápida**: se E3 der 2 ou mais bolhas e a 1ª for uma reação, enviá-la com latência ×0,5.

**E6: "digitando…" e intervalo entre bolhas (código).**
- Duração do indicador para uma bolha de `n` caracteres: `t = 0,15 + 0,173·n` segundos, com ruído amostrado entre as
  retas de p25 (`0,138·n`) e de p75 (`0,57 + 0,234·n`) e limites de 0,8 a 25 s. Multiplicadores: `seriousness` ×1,17 por
  nível; `tension > 0,6` ×0,6; `playful > 0,6` ×0,7; tédio ou encerramento ×0,7.
- **Pausas visíveis** (indicador some e volta) com probabilidade de 14% na zoeira, 26% no casual e 28% no sério, e 18%
  de pausas maiores que 5 s em insegurança. **Reset visível** ("digitou, parou e recomeçou"): 3% na zoeira, 7% no
  casual e 11% no sério. Nunca em tensão (1,6%).
- Intervalo entre bolhas = ócio (mediana de 1,7 s; amostrar lognormal com mediana de 1,5 s e p75 de 6 s) mais o
  "digitando…" da bolha seguinte. Referência empírica: reação 4,5 s, continuação 10 s, pergunta 12 s.

**E7: o usuário manda algo enquanto o bot "digita" (código + E1).**
Fazer o que os humanos fazem nesse caso (18% resetam, 75% editam): parar o indicador, esperar a janela de E1, **refazer**
o plano com a nova bolha (o "cérebro" roda de novo) e recomeçar a digitar. As bolhas que já foram enviadas ficam. Não
mandar o texto antigo se ele ficou incoerente com a nova bolha: o Jev pode checar com um **Noul** "Does `draft_reply`
still make sense as a reply after `new_user_message`?"; se `p < 0,5`, regenerar.

**E8: bolha extra depois ("ah, e…").**
- Gatilho de código: o usuário não respondeu, o último turno do bot **não** terminou em pergunta, e já passaram
  T minutos (assíncrono: amostrar entre 3 e 30 min, mediana de 10 min; ao vivo: 25–45 s).
- *Jev*: **Noul** "Is there a natural short follow-up (a question or a new small point) that would add to the bot's
  last message without seeming needy?" mais **Noul** "Is the moment tense or emotionally serious?". State: as últimas 6
  falas e o tempo decorrido (calculado em código). Regra: mandar se `p_follow > 0,6`, `p_tense < 0,4`, e no máximo 1 vez
  seguida (sem double text em série). Dar preferência a uma **pergunta** (29% das bolhas extras humanas).
- Frequência-alvo: ≤ 8% dos turnos do bot. *Parâmetro empírico; o detector não foi validado.*

**E9: tamanho da resposta (insumo para o briefing da LLM).**
Usar o `p_length` do Jev (Score 0–4, state com o contexto até o último turno do usuário) combinado com código (tamanho
do usuário no último turno e média do histórico dele) numa regressão linear simples: Spearman de 0,46 no maichat e 0,33
no WA, contra 0,40 e 0,27 só com código. Passar à LLM uma faixa de caracteres. **Não usar `p_n_msgs`** para nada; o
número de bolhas é decisão de E3.

### 4.3 Parâmetros resumidos (para `delivery_config`)
| parâmetro | valor (fonte) |
|---|---|
| janela de fim de turno | 4 / 8 / 12 s conforme `p_mais` (A9) |
| bolhas por tamanho | tabela A2 (sorteio), com teto de 5 |
| arousal | +0,2 bolha por nível; bolha ×0,66 por nível (A3) |
| seriedade | bolha ×1,28 por nível; composição ×1,17 por nível (A3, A7) |
| digitação | 0,15 + 0,173 s/caractere (p25 0,138, p75 0,57 + 0,234) (A7) |
| ócio entre bolhas | mediana de 1,7 s, 37% abaixo de 1 s (A5) |
| latência ao vivo | humano: mediana 13,8 s (p25 6,2, p75 27); bot recomendado: ~5 s × espelhamento (A6) |
| latência assíncrona | mediana 60 s; 67% em ≤ 2 min; teto de 5 min (A6) |
| bolha extra | ≤ 8% dos turnos, mediana de 10 min (async), 29% pergunta (A8) |
| correção com "*" | ≤ 0,5% das bolhas, ~5 s depois, só na zoeira (A4, A5) |

---

## 5. Limitações

- **Maichat** tem só 42 conversas e 84 falantes; são universitários conhecidos entre si, numa plataforma de pesquisa e
  com tarefa de conversar. As latências ficam comprimidas (quase não há respostas acima de 60 s) e 65% das mensagens
  vêm do desktop. Quase não há momentos sérios: 13 turnos com seriedade ≥ 2 e 90 mensagens na faixa 2–3.
- **whatsapp_nl** está em holandês, com timestamps de **minuto** (71% dos intervalos dentro do burst são "0") e 1.461
  latências negativas descartadas por problemas de ordem. Os rótulos D do Jev em holandês são menos confiáveis. São
  dados de 2012–2014, com emojis em códigos privados (``), então a detecção de emoji é incompleta.
- A definição de turno junta bolhas separadas por qualquer intervalo sem resposta; no WA, 5% dos intervalos "dentro do
  burst" passam de 10 min. As bolhas extras (A8) estão incluídas nesses turnos.
- Os rótulos D (ansioso, tensão…) são do Jev, não de humanos. No maichat, "tension" e "anxious" pegam muita provocação
  entre amigos, e é provável que o efeito "briga = digitação rápida" venha em parte da zoeira rápida.
- Correlação não é causalidade (sobretudo em "latência lenta → fim da sessão"). Os efeitos de momento são **fracos
  diante do tamanho do texto e do estilo pessoal**.
- As funções de bolha classificadas pelo Jev não foram validadas por humanos; a concordância com a heurística foi de
  73–83%. Os detectores E7 e E8 são propostas com parâmetros empíricos, mas **não** foram testados como detectores.
- O `p_n_msgs` da camada base ficou prejudicado pelo formato do state (bolhas juntadas); a reavaliação usou uma amostra
  de 250 turnos por corpus.

---

### Arquivos
- Scripts: `scripts/analysis/a1_load.py` (carga e enriquecimento), `a1_bursts.py`, `a1_split_table.py`, `a1_bubbles.py`
  (mais `a1_heur.py`), `a1_timing.py`, `a1_typing.py`, `a1_typing2.py`, `a1_pvalid.py`, `a1_pexp.py`, `a1_turnend.py`.
- Saídas: `analysis/data/a1_bursts.json`, `a1_split_table.json`, `a1_bubbles.json`, `a1_bubble_functions.jsonl`,
  `a1_timing.json`, `a1_typing.json`, `a1_typing2.json`, `a1_typing_messages.csv.gz`, `a1_pvalid.json`, `a1_pexp.json/.csv`,
  `a1_turnend.json`, `a1_turnend_sample.csv`.
- Chamadas novas ao Jev: 1.216 (funções de bolha) + 1.000 (P com bolhas visíveis) + 500 (fim de turno) = 2.716,
  ≈ US$ 0,10, com workers = 4.
