# 12 · Arquiteturas de chamadas do Jev para decidir "o que escrever" + Briefing v2 (rodada reduzida)

> Prefixo `b3`. Scripts em `scripts/analysis/b3_*.py`. Saídas pequenas em `analysis/data/b3_*`; as grandes em
> `data/processed/b3_*` (fora do git). Corpus: **maichat** (inglês). Modelos de LLM usados, conforme a regra nova do usuário:
> `openai/gpt-6-luna` (2º rotulador, referência e ator), `google/gemini-3.5-flash-lite` (ator). Nenhum claude, gpt-4o ou
> gpt-4o-mini foi chamado nesta rodada; a condição D (haiku) aparece só como dado histórico do a9.
>
> **Status: parcialmente completo.** Os créditos acabaram durante a execução e o coordenador mandou encerrar. Rodaram
> **13 arquiteturas + 12 misturas** de previsão de movimento (dev 240 e teste 400 pontos), além de tom, subtexto e elemento
> reagido. **Não rodaram**: as candidatas geradas com Nouls atômicos (`cand`, código pronto), o retrieval com k=5 e o
> retrieval sem rótulos (`rag5` e `ragtxt`, código pronto). O **Briefing v2 rodou em versão reduzida**: 40 pontos, 2 atores
> (flash-lite e gpt-6-luna) e 4 condições (A, B1, N2, B2). As condições K2 (código + corpus) e O2 (oráculo) foram
> construídas, mas não geradas, e mercury e deepseek não rodaram.

---

## 1. Resumo com números

- **O movimento humano é mais previsível do que a 1ª rodada sugeria, mas o teto é baixo, e nenhuma arquitetura rompe esse
  teto.** No teste (400 pontos, 35 conversas), a Choice plana da 1ª rodada, agora com o ouro rotulado uma resposta por
  chamada, acerta **37,5%** [32,1–42,6] no top-1. A classe majoritária acerta 19,8%. O **teto** medido pela concordância
  entre dois rotuladores independentes (Jev × gpt-6-luna, os dois vendo a resposta real) é de **73,5%** [68–78]. Então a
  arquitetura fica a cerca de metade do caminho entre o acaso e o teto.
- **A melhor arquitetura escolhida no dev é o retrieval de casos humanos misturado com a Choice plana** (`rag20 + flat`, pesos
  0,5/0,5, em log). Ela fez **38,8%** [34,0–43,4] de top-1, 66,8% de top-3 e 45% de "acerto em algum movimento que o humano fez"
  (rótulo multi-rótulo do luna). Contra a Choice plana, o top-1 sobe **+1,3 pp (IC −1,2 a +4,0), sem significância**, e o
  log-loss melhora **−0,16 (IC −0,22 a −0,11), com significância**. **Leitura honesta:** o ganho está na **calibração da
  distribuição** (útil para sortear), não no argmax.
- **O retrieval "few-shot dinâmico" sozinho empata com a Choice plana** (rag20 38,5%; rag10 38,0%). O que tornou o retrieval
  útil foi a similaridade: TF-IDF puro recuperava casos ruins (o voto kNN acertava 24–28% no dev). Com um **"embedding Jev"**
  (os rótulos D do turno do parceiro: emoção, intenção e fase em one-hot, mais os Nouls), o voto subiu para 35%. Sem nenhuma
  chamada online ao Jev, o **voto kNN puro faz 30,2%** no teste. Com o Jev lendo os casos, sobe para 38,5% (+8,3 pp, IC +3,7 a
  +12,8).
- **As cascatas pioraram.** Com a leitura do momento, em rótulos, no state de um 2º Jev e mais o guia de movimentos: 33,2%, ou
  −4,3 pp contra a plana (IC −8,1 a −0,5). Com o prior empírico por gatilho junto: 33,0% (−4,5 pp, IC −8,0 a −0,5), e o log-loss
  piora muito (+0,44). O **guia no state** fez 34,5%, o **estado do personagem e da relação** 35,8%, os **critérios ricos**
  35,2%, o **hierárquico** 36,8% e os **16 Nouls** 26,0%. Nenhuma dessas passou da Choice plana. **O Jev decide melhor com o
  contexto cru do que com rótulos intermediários ou regras em prosa.** Quando o 2º passo recebe os rótulos do 1º, ele herda os
  erros dos rótulos e ancora nas taxas do guia.
- **A LLM "como cérebro" é pior que o Jev.** O gpt-6-luna, prevendo o movimento com raciocínio, fez 32,5% no top-1, contra
  37,5% da Choice plana do Jev (que é ~10× mais barata e responde em 0,5 s).
- **O portão de confiança funciona.** Com confiança ≥ 0,7 (29% dos pontos do teste), o top-1 da arquitetura escolhida é de
  **64%**. Abaixo de 0,55, fica em 25–30%. **Ditar o movimento só com confiança alta** é a regra certa.
- **Sortear em vez de usar o argmax.** A coincidência esperada de um sorteio na distribuição do Jev é de 32% (contra 38,8% do
  argmax). Em troca, a variedade de movimentos se aproxima da humana: o argmax cobre 3,2 bits de entropia, contra 3,5 bits dos
  humanos, e o movimento mais previsto fica com 23% das previsões.
- **O Jev acha a palavra que o humano pegou.** Em 69% das respostas humanas o ouro aponta uma palavra específica da última
  mensagem. Entre essas, a Choice "a que elemento reagir" do Jev acerta **61,6%** [55–69], contra 30% do acaso, **42,7%** da
  regra "última palavra" e 28,7% da "palavra mais longa". Contra o ouro do luna, o acerto é de 56% (acaso: 30%).
- **O tom é previsível e o subtexto, pouco.** O tom acerta 44,8% [39–49], contra 29% da classe majoritária (teto entre
  rotuladores: 72,5%). A previsão de "deixar implícito" tem AUC de só 0,62.
- **Briefing v2 (reduzido, 40 pontos, n pequeno, ICs largos).** Os dois atores ficaram perto do humano na superfície, com
  qualquer briefing. **Mas o conteúdo do Jev no v2 não melhorou a coincidência de movimento**: flash B2 30% contra N2 (só
  código) 42% (Δ −12,5 pp, IC −25 a 0); luna B2 35% contra N2 35%. O que continua a fazer o grosso do trabalho é o
  **código** (A→N2: erro de tamanho −0,38 no flash, com IC excluindo 0; distância do perfil de traços −0,024, com
  significância). E apareceu um **vício novo criado pelo briefing**: a ordem "greet back + one concrete thing about your
  moment" gerou "hey drinking coffee / just making coffee" em 5 das 6 aberturas.

**Resposta à pergunta central, com o que foi medido:** com instruções em tempo real, a LLM (barata ou "forte") fica **muito
mais humana na forma**. Pergunta, "!", riso, emoji, tamanho, abertura performática e o molde de fazer três coisas de uma vez
caem para perto da taxa humana. A maior parte desse ganho vem do **código calibrado**, e isso vale para os dois atores. Na
**decisão do que fazer**, o Jev acerta o movimento em ~39% (64% quando está confiante) e a palavra a que reagir em ~62%.
**Mas, neste teste pequeno, passar essa decisão para a LLM como ordem ainda não aproximou a resposta do humano.**

---

## 2. Desenho

- **Pontos de decisão:** todos os 2.826 turnos do maichat que respondem a um turno do parceiro na mesma sessão (mesma definição
  do a9). **Dev** = as 7 conversas fora do teste do a9 (240 pontos). **Teste** = as 35 conversas de teste do a9 (400 pontos,
  incluindo os 119 do a9). Todas as escolhas (arquitetura, pesos, limiares) foram feitas no dev. O retrieval e os priors usam
  sempre **outras conversas** (deixando a própria de fora).
- **Ouro:**
  - (1) o Jev rotula a resposta **real** (uma por chamada, sem candidatas misturadas no state) com a taxonomia de 16 movimentos
    da 1ª rodada, 7 famílias, tom, subtexto e "pega palavra específica". Foram 2.818 chamadas, que servem de ouro e de base de
    casos;
  - (2) o gpt-6-luna rotula de forma independente os 640 pontos de avaliação (movimento principal, **todos** os movimentos
    presentes, tom e elemento).
  - Concordância: movimento 73,5%, movimento do Jev ∈ movimentos do luna 82,5%, família 79,8%, tom 72,5%, elemento 77%.
    **Esse é o teto prático.** Meu ouro concorda em 80% com o rótulo do humano no a9, que usava outro formato de state.
- **Métricas:** top-1 e top-3 contra o ouro do Jev; top-1 contra o ouro do luna; "hit" (o previsto ∈ os movimentos do luna);
  log-loss (com suavização de 3%); coincidência esperada do sorteio; família; diversidade. ICs por bootstrap **por conversa**,
  com 2.000 reamostras.
- **Custo desta tarefa:** ≈9.000 chamadas ao Jev (≈US$ 0,54) e ≈1.600 chamadas de LLM (≈US$ 0,17).

---

## 3. Arquiteturas testadas (todas, inclusive as que falharam)

Notação: `S` é o state padrão (7 turnos anteriores + `last_message`). Cada caixa é uma chamada ao Jev. As perguntas dentro de uma
chamada são independentes (fan-out).

```
(0) majority / prior_trigger / knn20   [SEM Jev online: código]
    prior_trigger: P(mov | intenção D do parceiro), tabela LOCO
    knn20: 20 casos mais parecidos (TF-IDF 0,5 + "embedding Jev" 0,5) → voto ponderado

(1) flat (1ª rodada)        S ─► [Choice 16 movimentos] ─► argmax
(2) flat_rich               S ─► [Choice 16, critérios com "quando humanos fazem" + exemplos]
(3) nouls                   S ─► [16 Nouls "um amigo responderia <mov>?"] ─► normaliza em código
    flat_x_nouls            flat × nouls (produto em código)
(4) hier                    S ─► [Choice família (7)] + [Choice fino dentro de cada família] (mesma chamada)
                                 ─► P(m) = P(fam)·P(m|fam) em código
(5) guide                   S + move_guide{mov: "quando / ~x% das respostas" (taxas do dev)} ─► [Choice 16]
(6) casc  (2 chamadas)      S ─► [18 Nouls/Choices de leitura: teasing, direct_q, vulnerable, wants,
                                   seriousness, energy, story, opinion, topic_done…]
                                 ─► código transforma em rótulos ("Alex asked Sam a direct question"…)
                            S + moment_reading + move_guide ─► [Choice 16]
    cascp                   casc + how_people_usually_reply_here (prior LOCO por gatilho, top-5 com %)
(7) char                    S + character{relação, intimidade, humor atual de Sam (D do turno anterior dele),
                                  engajamento, último ato de Sam, estilo} ─► [Choice 16]
(8) rag10 / rag20           S + similar_situations_from_other_chats[k]{3 turnos de contexto, resposta real,
                                  movimento} ─► [Choice 16]
    (rag5, ragtxt = sem rótulos: NÃO RODARAM)
(9) luna_top3 (referência)  gpt-6-luna com raciocínio: top-3 de movimentos (sem Jev)
(10) misturas               média ponderada de log-P de 2–7 arquiteturas; pesos na grade do dev
(11) cand (NÃO RODOU)       LLM escreve 1 candidata por família ─► S + candidates ─► [5 Nouls atômicos × 7]
                            (responde ao que foi dito? tom combina? mais intensa? amigo faria? faz demais?)
                            ─► soma em código ─► família da melhor
(12) elemento               S + last_message_elements{e1..en} ─► [Choice "a que elemento reagir"]
                                                                 + [Choice "que palavra um amigo pegaria"]
(13) subtexto / tom         S ─► [Noul "deixar implícito?"] [Choice explícito/insinuar/pular] [Choice tom]
```

---

## 4. Resultados

### 4.1 Movimento (teste, n = 400; IC 95% por conversa)

| arquitetura | chamadas Jev | top-1 (ouro Jev) | top-3 | log-loss | top-1 (ouro luna) | hit multi-rótulo | família |
|---|---|---|---|---|---|---|---|
| majority | 0 | 19,8 [15,5–24,1] | 37,8 | 2,44 | 18,5 | 19,5 | 27,5 |
| prior_trigger (código) | 0 | 28,0 [23,2–32,8] | 52,0 | 2,15 | 26,5 | 32,2 | 41,2 |
| knn20 (código + casos) | 0 | 30,2 [25,4–34,8] | 59,0 | 2,14 | 28,2 | 33,5 | 42,0 |
| **flat (1ª rodada)** | 1 | **37,5** [32,1–42,6] | 66,0 | 2,25 | 36,5 | 44,5 | 47,0 |
| flat_rich | 1* | 35,2 [31,1–39,5] | 63,5 | 2,36 | 32,5 | 38,8 | 48,8 |
| nouls | 1* | 26,0 [21,7–30,4] | 52,5 | 2,43 | 21,5 | 29,8 | 30,5 |
| flat_x_nouls | 1* | 37,2 [32,2–42,2] | 65,5 | 2,08 | 36,0 | 44,0 | 48,2 |
| hier | 1* | 36,8 [32,2–41,0] | 61,0 | 2,28 | 35,0 | 41,8 | 47,5 |
| guide | 1 | 34,5 [29,7–39,0] | 65,2 | 2,29 | 33,0 | 39,0 | 46,2 |
| casc | 2 | 33,2 [28,2–38,2] | 62,7 | 2,44 | 34,0 | 39,2 | 48,2 |
| cascp | 2 | 33,0 [28,2–37,7] | 61,5 | 2,70 | 32,2 | 38,8 | 45,0 |
| char | 1 | 35,8 [30,8–40,6] | 64,0 | 2,32 | 35,0 | 42,0 | 48,2 |
| rag10 | 1 | 38,0 [32,4–42,9] | 67,8 | 2,20 | 37,8 | 44,8 | 48,2 |
| rag20 | 1 | 38,5 [33,2–43,7] | 67,0 | 2,20 | 37,5 | 43,2 | 49,2 |
| luna_top3 (LLM, sem Jev) | 0 | 32,5 [27,5–37,5] | 63,2 | — | 36,2 | 43,8 | 44,0 |
| **rag20 + flat (escolhida no dev)** | 2 | **38,8** [34,0–43,4] | 66,8 | **2,09** | 37,5 | **45,0** | 49,2 |
| rag20 + cascp | 3 | 38,2 | **68,2** | 2,16 | **38,2** | 45,0 | **51,2** |
| flat + knn20 | 1 | 36,2 | 67,2 | **1,93** | 35,8 | 42,5 | 46,8 |
| ensemble de 7 variantes | 8 | 37,0 | 66,5 | 3,56 | 35,5 | 41,5 | 48,5 |

\* na mesma chamada da `flat` (fan-out de 46 perguntas, ≈0,48 s).

Diferenças pareadas no teste (top-1, ouro Jev): escolhida − flat **+1,3** (−1,2 a +4,0); rag20 − knn20 **+8,3** (+3,7 a +12,8);
casc − flat **−4,3** (−8,1 a −0,5); cascp − flat **−4,5** (−8,0 a −0,5); flat_rich − flat −2,3 (−6,1 a +1,5); hier − flat −0,8
(n.s.). Log-loss da escolhida − flat: **−0,16** (−0,22 a −0,11).

No **dev** a ordem foi parecida (escolhida: 43,3%; flat: 40,0%; rag20: 39,6%; casc: 36,2%; nouls: 28,3%). O ganho da mistura
encolheu no teste, o que é o esperado com escolha no dev.

**Por que as cascatas e os guias pioraram** (leitura dos erros): o 2º Jev dá peso demais aos rótulos em linguagem natural
("Alex asked a direct question" → `answer` mesmo quando o humano zoou) e às porcentagens do guia (puxa para os movimentos
frequentes). Com o contexto cru, o Jev usa pistas que os rótulos apagam. Isso reproduz o achado da 1ª rodada sobre emoção:
**cascata só ajuda quando o 2º passo traz informação nova de verdade**. Aqui, a informação nova que ajudou foram **casos
reais** (retrieval), não rótulos nem regras.

**Candidatas geradas + Nouls atômicos:** não rodou por falta de créditos. O script está pronto (`b3_arch.py cand`).

### 4.2 Confiança como portão (arquitetura escolhida)

| confiança (máx. P) | dev: fração · top-1 · hit | teste: fração · top-1 · hit |
|---|---|---|
| < 0,40 | 20% · 35% · 33% | 27% · 26% · 30% |
| 0,40–0,55 | 25% · 22% · 27% | 24% · 30% · 44% |
| 0,55–0,70 | 20% · 49% · 49% | 21% · 31% · 39% |
| **≥ 0,70** | **35% · 61% · 67%** | **29% · 64% · 64%** |

A faixa ≥ 0,7 é estável entre dev e teste, e é onde vale ditar o movimento. A faixa 0,55–0,7 não se sustentou no teste.

### 4.3 Sorteio × argmax

| | argmax | sorteio da distribuição |
|---|---|---|
| coincidência com o humano (escolhida) | 38,8% | 31,8% (esperada) |
| entropia dos movimentos produzidos | 3,2 bits (humanos: 3,5) | ≈ a da distribuição |
| movimento mais frequente | 23% das respostas | — |

O argmax desta arquitetura já é bem variado (16 movimentos, nenhum acima de 23%), bem diferente do "sempre 1 bolha" da 1ª
rodada. O sorteio custa ~7 pp de coincidência. **Recomendação:** usar o argmax quando a confiança for ≥ 0,7 e sortear (ou não
ditar nada) abaixo disso.

### 4.4 Elemento reagido ("responde à palavra, não ao tema")

| teste (n = 185 respostas cujo ouro é uma palavra; 2 a 12 elementos) | acerto |
|---|---|
| **Jev, "a que elemento o Sam vai reagir?"** | **61,6%** [55,0–68,8] |
| Jev, "que palavra um amigo pegaria?" | 54,6% |
| regra: última palavra | 42,7% |
| regra: palavra mais longa | 28,7% |
| regra: primeira palavra | 17,3% |
| acaso (1/n) | 30,0% |
| Jev contra o ouro do luna (n = 225) | 56,4% (acaso: 30,0%) |

Humanos pegam uma palavra específica em 69% (ouro Jev) a 84% (luna) das respostas. **Esta é a arquitetura com o sinal mais
forte da rodada**, com a vantagem de ser uma ordem concreta e barata ("reaja a 'Wilf'").

### 4.5 Tom e subtexto

- Tom: Jev 44,8% [39,4–49,4] contra a majoritária (`matter_of_fact`) 29,0%. Teto entre rotuladores: 72,5%.
- Subtexto: humanos deixam o ponto implícito em 26% das respostas (ouro Jev). O Noul preditivo "deixar implícito?" tem AUC
  **0,62** (fraco). A Choice explícito/insinuar/pular quase só responde "explícito". **Não serve** para dirigir o briefing
  nessa formulação.

### 4.6 Briefing v2: rodada reduzida (40 pontos de teste do a9, 2 atores)

**v2 = correções do relatório 09** (`b3_brief2.py`):
- pergunta, "!", riso e emoji **sorteados** com a taxa humana: global no N2, local (20 casos parecidos, encolhida para a global)
  no B2, com os mesmos números aleatórios entre as condições;
- tamanho como **alvo com folga** ("About N words, a bit shorter or longer is fine"), vindo do Score de tamanho do Jev e da
  mediana dos vizinhos;
- linha positiva de início ("Start plainly… or go straight to the content");
- **elemento concreto** a que reagir (se P ≥ 0,3);
- **movimento só com confiança ≥ 0,525** (limiar do dev) ou família com P ≥ 0,575. Isso ditou movimento ou família em 66% dos
  pontos;
- tom só com confiança ≥ 0,5.

N2 é o mesmo formato **sem Jev** (taxas globais, sem momento, movimento, elemento nem tom).

| n = 40 | palavras (mediana) | pergunta | riso | emoji | "!" | LLM-ish | abertura perf. | erro de tamanho | ρ tamanho | movimento = humano | família | tom | distância do perfil de traços |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **humano** | 6,5 | 25% | 7% | 3% | 17% | 7% | 7% | — | — | — | — | — | — |
| D haiku (histórico) | 21 | 65% | 45% | 20% | 47% | 42% | 35% | 1,22 | 0,46 | 45% | 45% | 35% | 0,200 |
| flash A | 20 | 70% | 53% | 53% | 40% | 33% | 57% | 1,24 | 0,37 | 33% | 40% | 33% | 0,201 |
| flash B1 (v1) | 7 | 3% | 0% | 0% | 0% | 3% | 10% | 0,88 | 0,45 | 42% | 40% | 42% | 0,185 |
| flash N2 (só código) | 6 | 0% | 12% | 0% | 3% | 3% | 10% | 0,85 | 0,21 | 42% | 50% | 30% | **0,177** |
| flash B2 (v2 + Jev) | 7,5 | 3% | 7% | 0% | 0% | 0% | 10% | **0,81** | 0,35 | 30% | 38% | 45% | 0,187 |
| luna A | 15 | 45% | 38% | 72% | 38% | 35% | 20% | 1,01 | 0,29 | 38% | 42% | 35% | 0,213 |
| luna B1 (v1) | 8 | 7% | 0% | 3% | 0% | 10% | 0% | 0,78 | 0,43 | 40% | 45% | 40% | 0,196 |
| luna N2 (só código) | 6,5 | 7% | 7% | 0% | 0% | 3% | 3% | **0,77** | 0,25 | 35% | 47% | 33% | 0,190 |
| luna B2 (v2 + Jev) | 11 | 15% | 5% | 0% | 3% | 3% | 0% | 0,84 | 0,28 | 35% | 42% | 45% | 0,202 |

(O "movimento = humano" aqui usa o **mesmo rotulador** Jev nas duas respostas. A distância do perfil é a média de |P(traço)
condição − P(traço) humano| em 12 traços atômicos do Jev, por ponto: responde direto, pega palavra, parafraseia, valida demais,
faz coisa demais, mais intensa, gíria forçada, soa assistente, genérica, inventa fatos, fala de si, implícita.)

Diferenças pareadas (n = 40, ICs largos):
- **A → N2 (código):** erro de tamanho −0,38 no flash (IC −0,76 a −0,01) e −0,24 no luna (n.s.); distância do perfil −0,024
  no flash (−0,039 a −0,010) e −0,023 no luna (−0,036 a −0,012). **É aqui que está o ganho.**
- **N2 → B2 (Jev online):** movimento −12,5 pp no flash (−25 a 0) e 0 no luna; família −12,5 e −5 pp; perfil +0,010 e +0,012
  (n.s., na direção errada). **O Jev no briefing não melhorou nada mensurável nesta amostra.**
- **B1 → B2:** movimento −12,5 pp no flash (−29 a +2) e −5 pp no luna (−16 a 0).
- Traços (flash A → B2): "faz coisa demais" 0,74 → 0,32 (humano: 0,39); "mais intensa que o outro" 0,78 → 0,40 (humano: 0,52);
  "gíria forçada" 0,36 → 0,16 (humano: 0,22). O B2 corrige a caricatura. O luna B2 **parafraseia mais** (0,46 contra 0,24 do
  humano): o "React to '<palavra>'" virou repetição da palavra.
- Aderência: pergunta sorteada cumprida em 80–93%. O flash ignora o "you *can* end with a question" (3% de pergunta com 21%
  autorizadas). O luna obedece mais (15%). **Permissão não é ordem:** para acertar a taxa, a instrução precisa ser "End with…".

**Por que o B2 não ganhou do N2** (exemplos, "/" separa bolhas):

| o outro disse | humano | flash A | flash B2 | luna B2 | luna N2 |
|---|---|---|---|---|---|
| "Hello" | "Hi / Did you get my message" | "Hey Alex! How's it going?" | "hey drinking coffee" | "hi, just making coffee" | "hey, nice to hear from you" |
| "Hey hon" | "heya!! / did you watch the videos" | "Hey babe! ❤️ How's your day going?" | "hey just making coffee" | "hey hon, grabbing coffee rn" | "hey, good to see you" |
| "right onto pets / i have a fat dog / called Wilf" | "how fat is he" | "OMG Wilf is such a good name for a dog 😭 is he act…" | "fat wilf is a strong image / does he actually roll around" | "wilf is an elite name for a fat dog / does he waddle everywhere?" | "wilf is an elite dog name" |
| "yeah, its nice. / i have been trying to learn some coding… / its so hard :(" | "did u try watching videos on coding tht could possibly help u" | "aww hang in there! what language…" | "yeah coding is really hard" | "yeah, coding can be really hard" | "yeah, coding can be a lot at first" |
| "really should start keeping up with news" | "so plausibly they will have to pay for compensation / yesss u…" | "lol yeah probably, it's pretty wild once you dive…" | "oh news is a wild concept for you i know" | "yeah, apparently the news has been keeping you very busy avoiding it" | "yeah, it's easy to miss these things" |
| "she posted a tiktok… / i am an influencer" | "interesting i know who youre talking about but…" | "lmao sure you are. what's your follower count, 12?" | "right / so educated" | "oh, so you're educated / just not on your competition apparently" | "fair, you're already in the trenches…" |
| "Rn? / At uni in a rented flat, defo not a dog…" | "v responsible" | "that's super responsible of u tbh…" | "fair enough, what kind of dog would u get eventually" | "yeah, fair to the dog. when do you think you'll be somewhere longer-term?" | "yeah, better to wait till you're settled" |
| "I LOVE THIS IDEA / what horoscope are you…" | "aries - I'm so firey / wbu" | "im a gemini / what about u" | "im a potato so my horoscope said eat fries" | "i'm a libra, my horoscope says i always win board games" | "i'm a mildly cursed libra" |
| "ok speak soon" | "Oh yeah, they're picky as hell" (não se despediu) | "bye! ❤️" | "yeah see you soon love" | "yeah, talk soon then" | "yeah, talk to you soon" |
| "I think we have done like 25 mins / Yep!!!!!" | "dammm i could keep going" | "time flies honestly lol / catch u later!" | "oh 25 minutes see ya" | "yeah, 25 mins, bye for now" | "yeah we got heaps done today" |
| "Ooo yeah" | "or a cardboard box" | "that's actually a pretty smart life hack 😂…" | "yeah that actually holds so much stuff do you have one" | "we need to make this happen soon" | "we could fit snacks in there too" |
| "Is this you?" | "ah yes / the search has appeared on the laptop" | "Wait, what do you mean? Is what me?" | "that is me" | "yeah, that's me" | "yeah, that's me" |

Leitura:
1. **Uma ordem de movimento vira estereótipo.** "Greet back + one concrete thing" produziu "coffee" em série. O mesmo aconteceu
   com o "hey there" do v1. Toda ordem fixa precisa de exemplos sorteados e de um controlador de repetição.
2. **O elemento certo ajuda o conteúdo** (Wilf, winehouse, dating show), mas "React to 'X'" também faz a LLM **repetir** a
   palavra. A instrução deveria ser "pegue 'X' e faça algo com ela, sem repeti-la ao pé da letra".
3. **Ditar o movimento** "tease back" ou "answer" produz a versão prototípica do movimento. O humano costuma ser mais oblíquo
   ("v responsible", "how fat is he"). Com n = 40, o efeito líquido foi negativo ou nulo.
4. Os dois atores reagem igual ao código: **a boca "forte" não dispensa o esqueleto de código**, e o luna A também tem
   emoji em 72%, "!" em 38% e riso em 38%.

---

## 5. Arquitetura recomendada (com o que os dados sustentam)

```
 mensagem do usuário
   │
   ├─► CÓDIGO: features (tamanho, "?", riso), "embedding Jev" do turno do usuário (rótulos D, já lidos)
   │           → kNN nos casos humanos (corpus de OUTRAS conversas, rotulado offline pelo Jev)
   │
   ├─► JEV chamada 1 (S, fan-out ~50 perguntas, ≈0,5 s):
   │     Choice movimento (16) · leitura do momento · Score tamanho · tom · Choice elemento
   │
   ├─► JEV chamada 2 (em paralelo, ≈0,5 s):  S + 20 casos parecidos {contexto, resposta real, movimento}
   │     → Choice movimento
   │
   └─► CÓDIGO: P(mov) = normaliza(√(P_flat · P_rag))
         conf ≥ 0,70 → dita o movimento (≈30% dos turnos; ~64% de acerto)
         conf < 0,70 → NÃO dita o movimento (ou sorteia, se quiser variedade)
         elemento: P ≥ 0,5 → "pegue 'X'" (sem repetir a palavra)
         pergunta / "!" / riso / emoji: SORTEADOS com a taxa dos 20 vizinhos (encolhida para a global) → ordem firme
         tamanho: alvo com folga = f(Score do Jev, mediana dos vizinhos, persona)
         controlador de repetição por conversa (aberturas, "coffee", mesma estrutura)
```

- As duas chamadas ao Jev rodam **em paralelo** (a 2ª não depende da 1ª), então a latência continua em ≈0,5 s e o custo fica
  em ≈US$ 0,0002 por turno.
- **Não usar:** cascata "leitura → rótulos → 2º Jev" para o movimento; guia de movimentos em prosa com porcentagens; 16 Nouls
  por movimento; a LLM como cérebro.
- **A decisão mais valiosa para o briefing é o elemento, não o movimento:** 62% de acerto, ordem concreta, fácil de verificar
  em código.

## 6. Tradução para o sistema

| decisão | como | evidência | quando entra no briefing |
|---|---|---|---|
| movimento | `rag20 + flat` (2 chamadas paralelas) | top-1 38,8%, log-loss 2,09 | só com conf ≥ 0,7 (64% de acerto) |
| família | a mesma distribuição, somada | 49% | fallback com P ≥ 0,6 (não validado no v2) |
| elemento | Choice sobre as palavras numeradas | 61,6% (acaso 30%; última palavra 43%) | P ≥ 0,5; "pegue", não "repita" |
| tom | Choice | 44,8% (majoritária 29%) | só com conf ≥ 0,5 |
| pergunta / "!" / riso / emoji | código: taxa dos vizinhos + sorteio | aderência 80–100% quando é ordem | sempre como ordem firme, nunca como permissão |
| tamanho | Score do Jev + vizinhos + persona | erro de tamanho 0,77–0,85 (A: 1,0–1,24) | alvo com folga |
| subtexto | ✘ (AUC 0,62) | — | não usar |
| base de casos | corpus rotulado offline pelo Jev (movimento e tom da resposta real) | voto kNN 30%; lida pelo Jev, 38,5% | em PT-BR: rotular um corpus brasileiro |

## 7. Limitações

- **Encerrado antes do fim.** Não rodaram: candidatas + Nouls atômicos, rag5 e ragtxt, as condições K2 e O2 do v2, os atores
  mercury e deepseek, e o v2 completo (só 40 dos 119 pontos). As conclusões sobre o v2 têm **n = 40**, com ICs que chegam a
  ±25 pp.
- **O ouro é um rótulo de modelo.** O Jev (e o luna) rotulam a resposta humana, com 73,5% de concordância entre si. A
  arquitetura é o Jev e o ouro principal também, o que pode favorecer arquiteturas "parecidas com o rotulador". Por isso
  reportei também o ouro do luna e o multi-rótulo: as conclusões se mantêm nos três.
- **Uma resposta humana por ponto.** Muitas respostas diferentes seriam igualmente humanas. 39% de coincidência não significa
  61% de erro.
- **maichat:** 42 conversas, jovens britânicos, em tarefa de estudo. O dev tem só 7 conversas.
- **Juiz decomposto pelo Jev**, sem juízes humanos. Os traços atômicos já mostraram bom poder de separação antes (relatório 8),
  mas não há validação humana.
- O limiar de movimento do v2 (0,525) veio de uma regra de precisão no dev que, olhando a tabela 4.2, é permissiva. O limiar
  recomendado (0,7) não foi testado no v2.

### Arquivos

- Scripts: `b3_common.py`, `b3_points.py` (pontos e divisão), `b3_gold.py` (ouro Jev, elemento e luna), `b3_arch.py` (todas as
  arquiteturas), `b3_eval.py` (avaliação), `b3_brief2.py` (briefing v2: montagem e geração), `b3_brief2_eval.py` (rótulos e
  métricas do v2).
- Saídas: `analysis/data/b3_arch_results.json` (teste, ICs, pareadas, gating, tom, subtexto, elemento), `b3_arch_dev.json`,
  `b3_brief2_thresholds.json`, `b3_brief2_results.json`. Dados intermediários: `data/processed/b3_*.json(l)`.
