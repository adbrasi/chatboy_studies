# 11 — Arquiteturas Jev para a entrega (bolhas) e o caso "5 bolhas em menos de 1 minuto"

**Status: PARCIAL.** O trabalho foi interrompido a pedido (os créditos acabaram). A Parte A está completa. Na Parte B,
completei o cenário "texto conhecido", com 4 arquiteturas de Jev, 6 baselines de código, 1 regressão ordinal com o
Jev como fonte de features e 2 misturas. Ficaram **incompletos**, sem resultado:
- T5, os Nouls por fronteira de corte ("onde cortaria?");
- T6, a cascata binária;
- T9, a LLM dividindo o texto em bolhas;
- todo o cenário P (prever as bolhas antes de o texto existir);
- a latência e a bolha extra "ah, e…".

Os scripts desses experimentos estão prontos (`b2_nb_T.py` a partir de T5, `b2_nb_P.py`, `b2_after.py`, `b2_llm_split.py`)
e podem ser rodados depois.

Dados: maichat (42 conversas, timestamps em ms e logs de digitação) e whatsapp_nl (57 chats, resolução de minuto).
Custo: ≈ 8,5 mil chamadas ao Jev (≈ US$ 0,45) e ≈ 2,1 mil chamadas de LLM, todas nos 4 modelos autorizados (< US$ 0,3).

---

## 0. Resumo com números

**Parte A: "5 bolhas em menos de 1 minuto é normal?"**
- **Acontece, mas é raro, e concentrado em poucas pessoas.**
  - No chat ao vivo (maichat), 1,0% dos turnos têm 5 ou mais bolhas em menos de 60 s. Isso dá ≈ 1 a cada 9 horas de conversa. Só 23% das pessoas
    fizeram isso alguma vez, e os 10% mais "rajadeiros" produzem 62% dessas rajadas.
  - No WhatsApp, a taxa é de 1,8% dos turnos contando bolhas no mesmo minuto ou no minuto seguinte, e 0,8% contando só o mesmo minuto. Uma pessoa faz isso em
    7,8% dos seus dias ativos, e os 10% mais rajadeiros produzem 67% das rajadas.
  - Já **3 ou mais bolhas em menos de 60 s é comum**: 9% dos turnos no maichat e 16% no WhatsApp (11% no critério estrito).
    Isso soma 22–33% de todas as mensagens.
- **O motor principal é o tamanho** do que a pessoa tem a dizer. Com até 40 caracteres, quase nunca há uma rajada de 5 (0–0,3%).
  Acima de 160 caracteres, ela acontece em 15–31% dos casos.
- **Ansiedade, controlando o tamanho: não tem efeito próprio.** A razão de chances (OR) de uma rajada de 5 é 0,25 no maichat e
  0,91 no WhatsApp, sem significância.
- **Briga ao vivo quase nunca vira rajada.** No maichat, turnos de discussão ou acusação têm lift de 0,23 para rajadas de 3 e zero
  rajadas de 5: quem briga manda uma mensagem seca.
- O que realmente dispara a rajada de 5 (lift sobre a taxa-base, com a OR controlada pelo tamanho):
  - **"explicar demais"**: lift de 14,8× no maichat e 9,6× no WhatsApp;
  - **corrigir ou esclarecer a si mesmo**: 8,6× e 3,2× (OR 6,4 e 2,3);
  - **pensar em voz alta**: 4,0× e 3,2× (OR 17,5 e 4,2);
  - **contar uma história**: 2,2× e 4,1×;
  - **notícia ou empolgação**: 2,7× e 1,8×;
  - **defensivo / pego no flagra**: no WhatsApp, o detector composto dá lift de 4,2× (OR 5,9, p = 4e-5) e o Noul "pego" dá
    2,8× (OR 3,9, p = 1e-5). No maichat, o tipo "sério" da cascata dá lift de 4,3× (OR 3,7, p = 0,04), mas o n é pequeno.
- **Resposta ao usuário:** a rajada de 5 **não é sinal de ansiedade**. Ela é sinal de **alguém se explicando**: se justificando,
  corrigindo, pensando alto, contando uma história, dando uma notícia. **"Pego mentindo" é um dos gatilhos mais fortes por unidade**,
  mas é raro (4% dos turnos).
- **O outro responde mais rápido e rajada puxa rajada.**
  - No maichat, depois de uma rajada de 5 o outro responde em 5,3 s, contra 11,0 s num controle de mesmo tamanho. Em 31% das vezes
    ele responde com outra rajada, contra 14%, e quase não pergunta (4% contra 13%).
  - No WhatsApp, a resposta vem em 60 s contra 110 s, e o outro ri junto mais (20% contra 10%, pelo Jev).
  - Depois de uma rajada defensiva, o outro **ri junto em 39%** das vezes: a maioria das "defesas" é leve.
- **Detector "defensivo", validado contra 2 LLMs** (216 casos em que as duas concordam):
  - a versão **composta** (2 Nouls: "o outro cobrou/acusou?" E "está se justificando ou negando?", combinados com AND em código)
    teve precisão de 0,60, recall de 0,78 e κ de 0,62;
  - o Noul único teve precisão de 0,37. A cascata com guia teve precisão de 0,52, mas recall de 0,44;
  - o escore contínuo teve AUC de 0,97.

**Parte B: o Jev decidindo o número de bolhas, com o texto já escrito (teste, 400 turnos de 46 conversas não vistas)**
- **O fracasso da 1ª rodada foi de arquitetura, não do modelo.**
  - Com o state "cru" (T1), o Jev repete o erro: P(várias bolhas) média de 3% contra 34% real, e argmax "1" em 99,8% dos casos.
  - Com um **guia** no state (T2), a P(várias) média sobe para 23% e a AUC vai de 0,81 para 0,88.
  - **Encadeado** (T3: um 1º Jev lê o texto, e os rótulos dele entram no state do 2º) chega a **RPS 0,061, AUC 0,89, acurácia de 72%,
    ρ 0,68 e P(várias) de 28%**. É o melhor Jev "puro" e empata com a melhor tabela de código.
- **O melhor no geral é o Jev como fonte de features** mais uma regressão ordinal em código (T7). Foi escolhido no dev
  (RPS 0,046) e, no teste, deu **RPS 0,048 [0,036–0,063], AUC 0,92, acurácia de 75%, ρ 0,71 e distribuição prevista com TVD
  de 0,05** em relação à real.
  - Com os mesmos 400 turnos de treino, **o Jev melhora o código**: RPS 0,059 → 0,048 e AUC 0,89 → 0,92.
  - Mas um modelo **só de código treinado em 12 mil turnos** chega a RPS 0,052 e AUC 0,93. Os ICs se sobrepõem.
  - **Veredito:** o Jev acrescenta pouco sobre um código bem treinado. O ganho dele é real quando há pouco dado de treino
    e no chat ao vivo (maichat, RPS 0,042 contra 0,059).
- **O roteador por momento (T4) piorou** (RPS 0,089): taxas específicas com n pequeno confundiram o Jev.
- A receita do relatório 01 (tabela de tamanho × moduladores) superestima a fragmentação: P(várias) de 46% contra 34%.

---

## 1. Parte A: rajadas

### 1.1 Definições
Uma **rajada Rk** é um turno (bolhas seguidas do mesmo falante) com **k ou mais bolhas dentro de alguma janela de 60 s**.
- No maichat, a janela é medida em ms.
- No WhatsApp, a resolução é de minuto. O critério "leniente" aceita diferença de carimbo ≤ 1 min (mesmo minuto ou o seguinte;
  o tempo real fica abaixo de 120 s). O "estrito" aceita só o mesmo minuto (tempo real garantidamente abaixo de 60 s).

Os números abaixo usam o critério leniente, salvo indicação.

### 1.2 Frequência (código, `b2_bursts_stats.py` → `b2_bursts_stats.json`)
| | maichat | WhatsApp |
|---|---|---|
| turnos com R3 / R5 | 9,1% / 1,0% | 15,8% / 1,8% (estrito: 10,9% / 0,8%) |
| mensagens dentro de R3 / R5 | 22% / 4% | 33% / 6% |
| falantes (≥ 30 turnos) com ao menos 1 R5 | 23% | 68% |
| % das R5 feitas pelos 10% de falantes mais rajadeiros | 62% | 67% |
| taxa de R3 por falante (p10 / p50 / p90) | 0 / 5% / 24% | 3% / 12% / 29% |
| conversas (sessões ≥ 10 turnos) com ao menos 1 R5 | 33% | 27% |
| frequência | 1,0 R3 e 0,11 R5 por hora de chat | 0,99 R3 e 0,11 R5 por pessoa-dia; 7,8% dos dias com R5 |
| R3 por faixa de tamanho: ≤ 20 / 21–40 / 41–80 / 81–160 / > 160 caracteres | 0,2 / 2,9 / 18 / 47 / 66% | 0,9 / 6,5 / 20 / 41 / 53% |
| R5 com > 160 caracteres | 41% | 41% |

**Por fase** (rótulo de fase do Jev, passe D; lift de R3 sobre a média):
- abertura: 0,15 no maichat e 0,45 no WhatsApp;
- conversa pessoal/profunda: 1,36 e 1,97;
- discussão/reparo: **0,18 no maichat** (ao vivo, brigar é mandar uma mensagem seca) e 0,97 no WhatsApp;
- desacelerando: 0,9 e 0,53.

Na sessão, os 3 primeiros turnos têm menos rajadas (5% contra 9% no maichat).

### 1.3 O que está acontecendo nas rajadas (Jev, `b2_rajadas_jev.py`)
**Arquitetura de leitura (3 chamadas, com cascata condicional):**
```
turno T (bolhas JUNTADAS por espaço: o Jev não vê a fragmentação)
 └─ J1 leitura  state = {previous_turns[≤5] (bolhas visíveis), partner_last_turn, current_turn}
      ~31 perguntas: Choice main_context (17 opções), Score arousal, Score amount_to_say,
      17 Nouls de contexto (multirrótulo), 11 Nouls atômicos (outro_acusou, se_justifica,
      pego_no_flagra, nega, explica_demais, acusação_de_brincadeira, pensa_alto, urgente, raiva…)
 └─ código: detector acendeu? (se_justifica≥.3 | pego≥.3 | nega≥.3 | outro_acusou≥.5)
      └─ J2 cascata  state = J1.state + first_pass_labels (saídas de J1) + defense_guide (6 tipos descritos)
           Choice: mentira / erro-esquecimento / criticado / acusação de brincadeira / mal-entendido / não
 └─ J3 reação  state = J1.state + partner_next_turn → Choice (9 reações do outro)
 └─ código: detectores = (a) Noul único; (b) COMPOSTO = outro_acusou≥.5 E (se_justifica≥.5 ou nega≥.5);
            (c) "pego"≥.5; (d) tipo da cascata ∈ {mentira, erro, criticado}
```

**Amostra estratificada com pesos:**
- maichat: todas as 261 R3 e 450 turnos sem rajada;
- WhatsApp: 250 R5, 350 R3–4 e 600 turnos sem rajada;
- no total, 1.911 turnos (J1), 664 chamadas J2 e 1.149 chamadas J3.

**Lift sobre a probabilidade de rajada** (`b2_rajadas.json`). O lift é P(Rk | estado) / P(Rk), com IC por bootstrap de conversas.
A OR vem de uma regressão com log(tamanho) e erro agrupado por conversa. "—" = sem R5 nesse estado.

| estado (Jev) | prevalência mc / WA | lift R3 mc / WA | **lift R5** mc / WA | OR (R5, com tamanho) mc / WA |
|---|---|---|---|---|
| explica demais | 3% / 4% | 4,4 / 3,5 | **14,8 / 9,6** | 3,8* / 1,6 |
| corrige ou esclarece a si mesmo | 5% / 5% | 2,7 / 2,2 | **8,6 / 3,2** | 6,4** / 2,3** |
| pensa em voz alta | 24% / 28% | 3,3 / 2,5 | **4,0 / 3,2** | 17,5** / 4,2*** |
| conta uma história | 9% / 11% | 1,9 / 2,4 | 2,2 / **4,1** | 0,9 / 1,7* |
| notícia / empolgação | 14% / 17% | 2,2 / 1,7 | 2,7 / 1,8 | 2,4* / 1,7 |
| fofoca (3ª pessoa) | 5% / 8% | 1,4 / 1,5 | 2,0 / 2,4 | 1,1 / 1,8 |
| **defensivo: composto** | 9% / 2% | 0,23 / 1,7 | 0,4 / **4,2** | 1,1 / **5,9*** |
| **defensivo: "pego no flagra"** | 11% / 6% | 0,55 / 1,7 | 1,6 / **2,8** | 4,2* / **3,9*** |
| defensivo sério (cascata: mentira, erro, criticado) | 3% / 4% | 0,7 / 1,7 | **4,3 / 3,0** | 3,7* / 2,4* |
| acusação de brincadeira (cascata) | 23% / 12% | 0,7 / 1,7 | 0,9 / 2,2 | 2,6 / 2,6*** |
| ansiedade / insegurança | 5% / 9% | 0,8 / 1,5 | 0,6 / 1,9 | **0,25 / 0,91** (n.s.) |
| raiva | 12% / 7% | **0,33** / 1,6 | — / 1,8 | — / 1,0 |
| discutindo / acusando o outro | 15% / 3% | **0,23** / 1,2 | 0,2 / 0,9 | 0,5 / 0,7 |
| zoeira | 60% / 49% | 1,0 / 1,3 | 1,2 / 1,4 | 3,7*** / 1,9** |
| logística | 8% / 29% | 1,5 / 1,0 | 2,2 / 1,1 | 1,2 / 0,56*** |

\* p < 0,05; \*\* p < 0,01; \*\*\* p < 0,001.

**Leitura:**
1. **Ansiedade não causa rajada.** O ansioso escreve mais, e o tamanho explica tudo (OR controlada 0,25–0,91). Isso confirma o
   relatório 01.
2. **O gatilho é "ter de se explicar"**: explicar demais, pensar alto e se corrigir têm OR de 2 a 17 mesmo controlando o tamanho.
3. **Pego no flagra / defensivo gera rajada no WhatsApp** (OR 3,9–5,9). No chat ao vivo, **a acusação séria gera mensagem
   única e seca** (lift de R3 0,23); só o "pego" leve vira rajada. Exemplo real (maichat):
   - B: "I think you are lying to me";
   - A: "Okay u got me" / "ph may god" / "IT DEKETED MY MESSAGE" / "IM SO ANNOYED";
   - B: "omg that is so flop" (ri junto).
4. **Zoeira**: com o tamanho fixo, gera mais rajada (OR 1,9–3,7). É a zoeira "picotada" do relatório 01.

**Forma da rajada por estado** (turnos R3; o intervalo e a digitação só no maichat):

| estado | bolhas | caracteres por bolha (mediana) | total | intervalo entre bolhas | apaga algo | caracteres/s |
|---|---|---|---|---|---|---|
| todas as R3 (maichat) | 3,5 | 21,5 | 83 | **7,1 s** (contra 11,4 s nos bursts não-rajada) | 50% (contra 41% e 25%) | 4,0 |
| empolgação / notícia | 3,8 | **12,8** | 77 | 5,8 s | 48% | 3,0 |
| história | 3,9 | **36,5** | 169 | **13,3 s** | 60% | 3,7 |
| raiva | 3,1 | 18,5 | 64 | 8,6 s | **15%** | **6,2** |
| defensivo sério (WhatsApp) | **5,0** | 26 | 141 | — | — | — |
| ansioso (WhatsApp) | 4,6 | 26 | 146 | — | — | — |

- A rajada **começa mais rápido**: latência de 8,2 s contra 12–15 s.
- A pessoa começa a bolha seguinte quase na hora: ócio mediano de 1,4 s.
- Só 12–17% das bolhas terminam com pontuação.

**O que o outro faz depois** (controle com a mesma distribuição de tamanho):

| | maichat R5 / controle | WhatsApp R5 / controle |
|---|---|---|
| latência do outro (mediana) | **5,3 s / 11,0 s** | **60 s / 110 s** |
| responde com outra rajada (R3) | **31% / 14%** | 26% / 22% |
| pergunta | 3,6% / 13% | 22% / 24% |
| ri junto (Jev) | 14% / 8% | **20% / 10%** |

Depois de uma rajada defensiva (detector composto, WhatsApp), o outro **ri junto em 39%** das vezes, entra no conteúdo em 22% e
consola em 9%.

### 1.4 Detector "defensivo / pego no flagra": arquiteturas comparadas
Para checar o detector, usei 239 casos estratificados pelo escore do Jev, rotulados por 2 LLMs (gpt-6-luna e deepseek-flash). As duas
concordaram entre si com κ = 0,65; a comparação usa os 216 casos em que concordaram. Esse rótulo **não é ouro humano**.

| variante | precisão | recall | κ com as LLMs |
|---|---|---|---|
| Noul único "se justifica?" | 0,37 | 0,85 | 0,41 |
| Noul "pego no flagra?" | 0,37 | 0,74 | 0,39 |
| **composto: 2 Nouls + AND em código** | **0,60** | **0,78** | **0,62** |
| cascata J2 (tipo sério) | 0,52 | 0,44 | 0,41 |
| cascata J2 (qualquer tipo, incluindo brincadeira) | 0,27 | 0,93 | 0,27 |
| escore contínuo `max(se_justifica, pego, nega) × (0,5 + 0,5 × outro_acusou)` | AUC **0,97** | | |

**Conclusão:** a **decomposição atômica combinada em código** ganhou do Noul único e da cascata com guia. O Noul "o outro acusou?"
sozinho já tem AUC de 0,95.

---

## 2. Parte B: o Jev decidindo o número de bolhas, com o texto conhecido

**Tarefa.** Dado o contexto e o texto da resposta (a rajada inicial humana **juntada por espaço**), prever em quantas bolhas
(1, 2, 3, 4, 5+) a pessoa mandou. É a decisão da camada de entrega depois que a LLM escreve.
- Amostra: 200 turnos por corpus em cada split, separados **por conversa** (dev: 53 conversas; teste: 46).
- Alvo: as bolhas até o 1º intervalo de mais de 120 s.
- Métrica de escolha: **RPS** (ranked probability score, própria para distribuições ordinais), no dev.
- No teste: acurácia do argmax, Spearman, AUC de "2 ou mais bolhas", **calibração** (P(várias) média, ECE, TVD entre a
  distribuição prevista e a real).
- Script: `b2_nb_T.py`; avaliação em `b2_nb_eval_min.py` → `b2_nb_results.json`.

### 2.1 Arquiteturas
```
T1 bare     [J] state={conversation_so_far(8, bolhas visíveis), reply_speaker, reply_text}
                → Choice 1..5+ (opções secas)
T2 guide    [J] state = T1 + speaker_history (código: "X often splits… 36% as 1, 41% as 2…")
                + reply_length (código, em palavras: "long (81-160 chars), 1 sentence")
                + bubble_guide (taxas medidas NO DEV por faixa de tamanho; quando mais/menos bolhas; estilo pessoal)
                → Choice com critério descritivo por opção
T3 chain    [J-A] state=T1 → 14 rótulos do texto (energia, ansioso, zoeira, tensão, seriedade, empolgação, defensivo,
                quanto tem a dizer, nº de ideias, pensa alto, história, abre com reação, termina em pergunta, momento)
            → código converte em palavras ("energy: energetic", "joking: clearly yes")
            [J-B] state = T2 + reply_reading{…} → Choice 1..5+
T4 router   [J-A] (a mesma) → momento (8 classes)
            → código: taxas P(n | momento, faixa de tamanho) medidas no DEV
            [J-B] state = T1 + história + tamanho + moment_guide{regra do momento, taxas do momento} → Choice
T7 ordinal  código: regressão ordinal (logit) treinada no DEV sobre
                [código: log caracteres, nº de frases, maiúsculas no meio, pontuação, taxa própria, média própria,
                 bolhas e tamanho do parceiro, corpus] + [Jev: 11 features de J-A e o E[n] de T2]
C_*         só código: taxa-base; tabela de tamanho (P(n | faixa) no dev); histórico do falante;
            espelhar o parceiro; ordinal só de código (no dev inteiro, 12 mil turnos, ou na amostra dev de 400)
T8          receita do relatório 01 (tabela de tamanho × moduladores de arousal/zoeira/seriedade/tensão);
            mistura 50/50 de T2 com a tabela
NÃO CONCLUÍDOS: T5 (Nouls por fronteira: "começa nova bolha em reply_words[i]?", Poisson-binomial),
            T6 (cascata ">1?" → ">2?" com a decisão anterior no state), T9 (LLM divide o texto; 4 modelos)
```

### 2.2 Resultados (teste; IC 95% por bootstrap de conversas)
Distribuição real no teste: 66% / 22,5% / 7,5% / 2,2% / 1,8% (P(várias) = 34%).

| arquitetura | RPS dev | **RPS teste** [IC] | acurácia | ρ | AUC ≥ 2 [IC] | P(várias) média | ECE | TVD |
|---|---|---|---|---|---|---|---|---|
| T1 Jev cru | 0,135 | 0,116 [0,091–0,138] | 0,66 | 0,60 | 0,81 [0,78–0,84] | **0,03** | 0,31 | 0,31 |
| T2 Jev + guia | 0,074 | 0,070 [0,052–0,081] | 0,70 | 0,65 | 0,88 [0,84–0,92] | 0,23 | 0,11 | 0,11 |
| **T3 Jev encadeado** | 0,066 | **0,061** [0,046–0,071] | 0,72 | 0,68 | **0,89** [0,86–0,93] | 0,28 | 0,07 | 0,06 |
| T4 roteador por momento | 0,097 | 0,089 [0,068–0,104] | 0,67 | 0,61 | 0,85 [0,80–0,89] | 0,13 | 0,21 | 0,21 |
| C taxa-base | 0,104 | 0,094 | 0,66 | 0,20 | 0,61 | 0,37 | 0,06 | 0,03 |
| C histórico do falante | 0,098 | 0,088 | 0,66 | 0,35 | 0,70 | 0,38 | 0,05 | 0,04 |
| C espelhar o parceiro | 0,118 | 0,108 | 0,52 | 0,20 | 0,63 | 0,38 | 0,15 | 0,04 |
| C tabela de tamanho | 0,068 | 0,068 [0,053–0,081] | 0,68 | 0,58 | 0,83 [0,79–0,88] | 0,37 | 0,04 | **0,03** |
| C ordinal (400 do dev) | 0,057 | 0,059 [0,046–0,075] | 0,74 | 0,67 | 0,89 | 0,41 | 0,07 | 0,07 |
| C ordinal (dev inteiro, 12 mil) | 0,062 | 0,052 [0,039–0,071] | **0,81** | **0,72** | **0,93** [0,89–0,96] | 0,40 | 0,06 | 0,06 |
| **T7 ordinal código + Jev** (400 do dev) | **0,046** | **0,048** [0,036–0,063] | 0,75 | 0,71 | 0,92 [0,89–0,94] | 0,39 | 0,05 | 0,05 |
| T7 ordinal só Jev (+ tamanho) | 0,047 | 0,049 | 0,76 | 0,70 | 0,91 | 0,39 | 0,05 | 0,05 |
| T8 receita do relatório 01 | 0,071 | 0,069 | 0,68 | 0,61 | 0,85 | **0,46** | 0,12 | 0,12 |
| T8 mistura T2 + tabela | 0,066 | 0,064 | 0,70 | 0,64 | 0,88 | 0,30 | 0,06 | 0,04 |

**Por corpus (RPS no teste):**

| | maichat | WhatsApp |
|---|---|---|
| T7 | **0,042** | 0,054 |
| ordinal só de código (12 mil) | 0,059 | **0,045** |
| T3 | 0,050 | 0,072 |

**Leitura:**
1. **Guia, histórico e tamanho em palavras no state resolvem a calibração**, que era o defeito da 1ª rodada (P(várias) de 3% →
   23%). **Encadear rótulos de outro Jev melhora mais** (28%, ECE de 0,07). O argmax de T3 escolhe "2" em 24% dos casos,
   contra 0% no T1.
2. **O roteador por momento piorou.** Com taxas específicas estimadas em n pequeno no dev, o Jev "ancorou" em "1". Guia geral
   com os rótulos do momento (T3) > guia específico por momento (T4).
3. **Como decisor direto, o melhor Jev (T3) empata com a tabela de tamanho** (RPS 0,061 contra 0,068; ICs sobrepostos). Ele ordena
   melhor (AUC 0,89 contra 0,83; ρ 0,68 contra 0,58), mas subestima um pouco a fragmentação.
4. **O melhor uso do Jev é como fonte de features** para um modelo calibrado em código (T7). Esse modelo foi o melhor no dev e no
   teste. O ganho sobre o código treinado no mesmo n é real (0,059 → 0,048). Sobre o código treinado com 30× mais dado, o ganho
   é pequeno e não significativo (0,052). **O Jev acrescenta sobretudo quando há pouco dado do domínio**, o que é justamente
   o caso de um produto novo em PT-BR.
5. O "juntar por espaço" deixa pistas no texto (maiúscula no início de cada bolha, no WhatsApp). O ordinal de código as usa, via
   `caps_mid`. Num texto de LLM essas pistas mudam, então **o ordinal de código deve ser retreinado com textos de LLM** antes de
   ir para o sistema.

---

## 3. Recomendação e tradução para o sistema

**Número de bolhas (depois de a LLM escrever):**
```
texto da LLM ─► [J-A: 1 chamada, ~14 Nouls/Scores sobre o texto + o contexto] ─► features Jev
             └► código: tamanho, nº de frases, pontuação, estilo da persona, bolhas do usuário
                         └► regressão ordinal (T7) → distribuição P(1..5+) → SORTEIO (não argmax)
fallback (Jev fora do ar ou lento): tabela P(n | faixa de tamanho) × estilo da persona
```
- Custo: 1 chamada de Jev (~0,5 s, em paralelo com a checagem de estilo). A calibração vem da regressão em código, que se
  retreina com os logs do produto.
- **Não** usar o roteador por momento com taxas específicas (T4) e **não** perguntar "quantas bolhas?" com o state cru (T1).
- Se a regressão for inviável, use **T3** (J-A → J-B com guia e rótulos, e o sorteio na distribuição do Jev). É bem calibrado
  (TVD de 0,06), mas subestima um pouco a fragmentação: some +0,05 em P(várias) ou misture 50/50 com a tabela de tamanho (T8).

**Rajadas (quando o bot deve mandar 5 bolhas em menos de 1 min):**
- **Raramente**: no máximo ≈ 1–2% dos turnos, e só se a persona for "rajadeira" (parâmetro `frag_style` alto). Em 77% das
  pessoas isso nunca acontece.
- **Gatilhos** (J-A, código com AND): `pensa_alto`, `explica_demais` ou `se_corrige` ≥ 0,5; ou o **composto defensivo**
  (`outro_acusou` ≥ 0,5 E (`se_justifica` ou `nega` ≥ 0,5)) no modo assíncrono. Mais: `história` ou `notícia` com texto acima de
  100 caracteres.
- **Nunca** transformar ansiedade em rajada por si só. No chat ao vivo, **acusação séria → 1 mensagem seca**, sem rajada.
- **Forma:**
  - empolgação: bolhas curtas (≈ 13 caracteres), com intervalo de ≈ 6 s;
  - história: bolhas longas (≈ 35 caracteres), com intervalo de ≈ 13 s;
  - defesa: 4–5 bolhas de ≈ 25 caracteres;
  - raiva: digitação rápida (6 caracteres/s), sem hesitação visível;
  - em todos os casos: ócio de ≈ 1,4 s entre bolhas, poucas pontuações finais e 1ª bolha logo.
- **Do lado do usuário:** se ele mandar uma rajada, espere mais (a E1 do relatório 01) e **responda mais rápido** do que o normal
  (os humanos respondem em 5 s contra 11 s). Não faça pergunta (4% contra 13%). Rir junto é a reação mais humana a uma rajada
  defensiva leve (39%).

---

## 4. Limitações
- **Incompleto:** T5 (onde cortar), T6 (cascata binária), T9 (LLM dividindo o texto), o cenário P (antes do texto), a latência
  (responder rápido ou demorar) e a bolha extra "ah, e…" **não têm resultado**. O que existe sobre eles vem do relatório 01.
- Rótulos de estado vindos do Jev, não de humanos. O detector defensivo foi checado **contra 2 LLMs** (κ = 0,65 entre elas), não
  contra pessoas. "Pego mentindo" de verdade é raro: 21 casos no maichat e 63 no WhatsApp, e os ICs dos lifts de R5 são largos.
- WhatsApp com resolução de minuto: o critério leniente inclui rajadas de até ~2 min reais (o estrito corta a taxa de R5 pela
  metade). Os dados são em holandês e de 2012–14; o maichat é chat de laboratório entre universitários. Nada em PT-BR.
- No cenário T, o texto é humano juntado. Com texto de LLM, os resultados precisam ser remedidos (pontuação e maiúsculas
  diferentes). O teste tem 400 turnos de 46 conversas; os ICs de T3, T7 e do ordinal de código se sobrepõem.

### Arquivos
- Scripts: `scripts/analysis/b2_common.py`, `b2_bursts_stats.py`, `b2_rajadas_jev.py`, `b2_rajadas_analysis.py`,
  `b2_nb_common.py`, `b2_nb_T.py`, `b2_nb_eval_min.py` (avaliação usada), `b2_nb_eval.py` (versão completa, para
  quando T5, T6 e P rodarem), `b2_nb_P.py`, `b2_after.py`, `b2_llm.py`, `b2_llm_split.py`.
- Saídas: `analysis/data/b2_bursts_stats.json`, `b2_rajadas.json`, `b2_rajadas_examples.json`, `b2_rajadas_items.csv.gz`,
  `b2_nb_results.json`, `b2_nb_preds_test.csv.gz`.
