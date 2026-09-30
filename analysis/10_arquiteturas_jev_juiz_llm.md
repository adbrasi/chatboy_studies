# 10 · Arquiteturas Jev para detectar "cara de LLM" (o juiz) e dizer o que está errado

> Prefixo `b1`. Scripts: `scripts/analysis/b1_*.py`. Saídas: `analysis/data/b1_results.json`, `b1_question_auc.csv`,
> `b1_strong_actors_code.json`. **Relatório parcial:** a rodada foi interrompida por falta de créditos (ver §6). Das 13
> arquiteturas planejadas, 7 foram medidas de verdade. Outras 6 estavam implementadas, mas não chegaram a rodar.

## 1. Resumo com números

Dados: 1.951 respostas avaliadas pelo Jev: 429 humanas e 1.522 de LLM, em 429 pontos reais de decisão. Esses pontos vêm do
a8 (250) e do a9 (179). Os modelos são gemini-flash-lite, gpt-4o-mini, llama-70b e haiku-4.5, estes três últimos como dado
histórico. Há também as versões "com guia de estilo" e com o briefing do Jev (B). O **dev** e o **teste** foram separados
**por conversa** (hash do `conv_id`). Todas as escolhas foram feitas no dev e os números abaixo são do **teste** (211
humanas × 739 LLM). Os ICs são bootstrap por conversa. "AUC-tam" é a AUC dentro de faixas fixas de tamanho (0–12, 12–25,
25–45, 45–80, 80+ caracteres): um detector que só mede tamanho fica perto de 0,5 nela.

| detector (teste) | AUC | AUC-tam | AUC só nas saídas **instruídas** (estilo/briefing) | só o briefing B |
|---|---|---|---|---|
| só tamanho (código) | 0,665 | 0,55 | 0,39 | — |
| **só código** (31 features, regressão logística) | 0,865 [0,83–0,90] | 0,841 | 0,750 | 0,659 |
| **só Jev**: banco de 85 perguntas + regressão logística | 0,852 [0,83–0,87] | 0,838 | 0,665 | 0,519 |
| **código + Jev** (regressão logística) | **0,912 [0,885–0,935]** | **0,896 [0,87–0,92]** | **0,824 [0,77–0,87]** | **0,726** |
| código + Jev "enxuto" (16 perguntas escolhidas por L1 no dev) | 0,903 | 0,888 | 0,806 | — |
| rodada 1: "foi uma pessoa?" (Noul holístico) | 0,577 | 0,535 | — | — |
| "foi escrita por IA?" (Noul) / "o outro suspeitaria de IA?" | 0,513 / 0,449 | 0,433 / 0,370 | — | — |

- **O Jev é um bom diagnosticador e um juiz fraco sozinho. Combinado ao código, é o melhor juiz que medimos.** Com o
  mesmo desenho, as perguntas atômicas do Jev combinadas em código empatam com o detector de código (0,85 × 0,87) e,
  **somadas a ele, acrescentam +0,05 de AUC (+0,055 controlada por tamanho)**. O ganho é maior justamente no caso difícil:
  saídas de LLM **já instruídas** para parecer humanas, onde o código sozinho cai para 0,75 e código + Jev fica em 0,82.
- **Perguntas holísticas continuam falhando, qualquer que seja a formulação**: "foi uma pessoa?", "é IA?", "o outro
  suspeitaria?", "um amigo poderia ter mandado exatamente isso?" (esta última sai **invertida**, 0,29). Os contrafactuais
  também falharam (0,41–0,60, a maioria abaixo de 0,5).
- **Para dizer O QUE está errado, o Jev funciona muito bem traço a traço.** Contra os rótulos de código: termina em
  pergunta 0,98; interjeição performática no início 0,98 (precisão 0,85, recall 0,87); pergunta genérica de continuação
  0,96; repete palavras do outro 0,93; molde "reação → … → pergunta" 0,93; fórmula de validação 0,92; gíria da moda 0,90;
  "mais animado que o outro" 0,83 (contra "!"/emoji); "longo demais" 0,82 (contra ser mais de 2× a resposta humana real).
  **Mas a Choice "qual é o vício principal?" falha**: acerta o vício em 5,8% (o acaso é 11%) e responde "nada" em 72% das
  respostas de LLM. O diagnóstico tem de ser **um Noul por vício**, nunca uma Choice entre vícios.
- **Score com níveis ancorados > Noul**, no mesmo conceito: "texto de amigo … resposta de assistente" (0–4) dá AUC
  0,68/0,62 (tam.), contra 0,51/0,49 do Noul "parece assistente"; validação 0,67 × 0,62; energia 0,70 × 0,67.
- **O ensemble de paráfrases (4 formulações com média) não ganhou da melhor formulação única escolhida no dev.** Ele
  serve contra o azar da redação: a AUC-tam entre formulações variou 0,46–0,65 no conceito "faz coisa demais".
- **Os atores mais fortes** (gpt-6-luna e deepseek-flash, gerados nos 250 contextos; o Jev não chegou a rodar sobre eles)
  têm a mesma assinatura. O gpt-6-luna sem estilo põe emoji em 75% e pergunta no fim em 41%; o detector de código acha
  AUC 0,94. **Com o guia de estilo, o deepseek é o mais difícil de pegar em código** (AUC 0,77; 0,75 tam.) e o luna
  continua fácil (0,905; emoji em 26%).
- Custo: o banco inteiro (85 perguntas) é **1 chamada, US$ 0,000145, p50 de 0,51 s**. A versão enxuta (16 perguntas)
  deve custar ~1/2–1/3 disso (estimativa, não medido).

## 2. Arquiteturas testadas

`state` padrão (JSON): `{setting, conversation_so_far: [até 8 turnos {from, text}], candidate_message: {from, text}}`.

### A1 · Banco de atômicos + combinação aprendida em código (**executada**)
```
resposta candidata ──┬──> [código] 31 features (tamanho, frases, ?, !, emoji, caixa, apóstrofo, lista LLM-ish,
                     │              eco de palavras, interjeição, pergunta recíproca, molde, razão de tamanho vs. o outro)
                     └──> [Jev, 1 chamada] state padrão + 85 perguntas em paralelo:
                            53 Nouls atômicos (traços concretos) + 7 contrafactuais + 18 paráfrases + 6 Scores ancorados
                            + 2 holísticos + 1 Choice "vício principal"
          features ──> regressão logística (C escolhido por GroupKFold-5 no dev) ──> P(LLM)
                   └─> árvore/GBM rasos (comparação)
```
Resultados (teste): tabela do §1. Mais:
- Árvores e GBM rasos perderam para a regressão logística: GBM código + Jev 0,873; árvore de profundidade 3 só com Jev
  0,763.
- **Combinações a priori** (média dos traços com o sinal esperado, gate do a8, máximo dos vícios) ficaram em 0,64–0,68 e
  **invertem nas saídas instruídas** (0,32–0,43). O Jev só serve com pesos aprendidos.
- As perguntas mais úteis (AUC-tam no dev; teste entre parênteses):
  - `t_blunt` "seca/curta" (sinal humano): 0,685 (0,672);
  - `t_own_thread` "ignora a última mensagem e segue o próprio fio" (sinal humano): 0,668 (0,666; 0,63 nas instruídas);
  - `p_enthusiasm_1` "mais energia/exclamação que a conversa até aqui": 0,645 (0,686);
  - `s_energy_vs_other`: 0,63 (0,66);
  - `t_multi_acts` "faz 2+ coisas: reage, comenta, conta de si, pergunta": 0,595 (0,65);
  - `t_trendy_slang`: 0,61 (0,60; **0,65 nas instruídas**).
- **Invertidas** (o humano pontua mais):
  - `t_direct` 0,33;
  - `t_register` 0,35;
  - `t_invented` 0,36 (o humano cita o que o Jev não vê);
  - `t_reacts_to_fact` 0,42;
  - `cf_friend_same` 0,30.

  Na regressão logística elas viram sinal útil, com o peso negativo.
- **Lista enxuta** (L1 no dev, 16 perguntas): `t_template`, `t_answers_several`, `t_offers_support`, `t_emoji_decor`,
  `t_trendy_slang`, `t_polite_markers`, `t_blunt`, `t_own_thread`, `t_direct`, `cf_friend_same`, `p_echo_1`,
  `p_echo_2`, `p_enthusiasm_1`, `p_too_much_3`, `s_friend_vs_assistant`, `s_energy_vs_other`.

### A8 · Contrafactuais (**executada**, dentro da mesma chamada)
Mesmo state. "Se um amigo respondesse, seria mais curto?", "…teria parado na 1ª frase?", "…não faria essa pergunta?",
"…com menos emoção?", "o outro suspeitaria de IA?", "caberia em outra conversa?", "um amigo poderia ter mandado
exatamente isto?". **Falhou.** A AUC-tam ficou entre 0,29 e 0,60, e quase todas abaixo de 0,5, porque o Jev acha a
resposta da LLM mais "plausível para um amigo". Só "caberia em outra conversa?" ficou acima do acaso (0,60).

### A9 · Ensemble de paráfrases (**executada**, na mesma chamada)
São 6 conceitos com 4 formulações cada. Média das 4 contra a formulação única (AUC / AUC-tam no teste):
- validação: 0,63/0,59 contra 0,62–0,63/0,55–0,59;
- eco: 0,64/0,60 contra 0,60–0,65/0,57–0,62;
- entusiasmo: 0,68/0,65 contra 0,60–0,71/0,57–0,69;
- "faz coisa demais": 0,67/0,57 contra 0,59–0,71/0,46–0,65;
- "assistente": 0,56/0,51;
- pergunta de enchimento: 0,54/0,51.

A média fica no meio das formulações e não ganha da melhor do dev. **Recomendação:** testar 3–4 formulações no dev e
congelar a melhor. Não vale pagar a média em produção.

### A10 · Score ancorado × Noul (**executada**, na mesma chamada)
| conceito | Score ancorado (AUC / AUC-tam) | Noul equivalente |
|---|---|---|
| amigo ↔ assistente (0 = "texto de amigo: curto, uma reação" … 4 = "validação + explicação + pergunta") | **0,68 / 0,62** | "parece assistente" 0,51 / 0,49; "é IA" 0,51 / 0,43 |
| validação (nenhuma … a mensagem inteira) | **0,67 / 0,64** | fórmula de validação 0,62 / 0,58 |
| energia vs. a última mensagem do outro | **0,70 / 0,66** | "mais animado que o outro" 0,67 / 0,63 |
| quantas coisas faz | 0,68 / 0,59 | "faz 2+ coisas" **0,71 / 0,65** |
| polimento | 0,56 / 0,49 | 0,53–0,59 |

**Ancorar os níveis em exemplos concretos é o que salva uma pergunta "holística"**: o mesmo conceito, que como Noul
fica no acaso, vira um sinal útil.

### A11 · Hierarquia "é LLM?" → "qual vício?" → "qual frase?" (**parcial**)
```
[código+Jev LR] P(LLM) ──> se alto: 8 Nouls de vício (já na chamada do banco) ──> vícios com p > 0,5
                                    └─ (alternativa testada) Choice "vício principal" (8 + "nada")
                       ──> [Jev, 2ª chamada] Choice sobre frases numeradas "qual frase é a culpada"  (NÃO RODOU)
```
- **Nível 2, Nouls de vício contra rótulos de código** (AUC, precisão/recall com p > 0,5, só nas respostas de LLM):

  | vício | AUC | precisão / recall |
  |---|---|---|
  | termina em "?" | 0,98 | 0,78 / 0,99 |
  | interjeição performática | 0,98 | 0,85 / 0,87 |
  | pergunta genérica | 0,96 | 0,43 / 0,88 |
  | eco ("repete palavras") | 0,93 | — / 0,93 de recall |
  | molde | 0,91 | 0,50 / 0,92 |
  | validação | 0,92 | 0,44 / 0,71 |
  | gíria da moda | 0,90 | — |
  | entusiasmo | 0,83 | 0,67 / 0,88 |
  | longo demais | 0,80–0,82 (Score de tamanho) | 0,62 / 0,92 |

  A precisão baixa (0,4–0,5) em validação, pergunta genérica e molde vem de o Jev achar o traço também em casos que o
  regex não pega. Parte disso é acerto semântico que o código perde, mas isso não foi medido à mão.
- **Nível 2, Choice "vício principal": falhou.** Top-1 de 5,8% contra 11% do acaso e top-2 de 26%. O modo é "nada" (72%
  das respostas de LLM) e depois "tom de assistente". Uma Choice relativa entre vícios não acha o vício. O Noul absoluto
  por vício acha.
- **Nível 3 (frase culpada): implementado em `b1_sent.py`, mas não rodou.**

### A5 · Cascata por momento (**só o estágio 1 rodou**)
```
[Jev 1] state {conversa} → Choice momento (9 opções) + confiança
        └─> [Jev 2] state {momento, guia do momento (humano × IA, com nossas taxas), conversa, candidata}
                    → 5 Nouls específicos do momento + "faz o que o guia diz que a IA faz?" + "faz o que amigos fazem?"
```
O estágio 1 rodou em 379 contextos: brincadeira 113, pergunta 94, história/info 49, planos 37, flerte 26,
saudação/"como vai" 26, desabafo 17, despedida 11, notícia boa 6. A confiança média foi 0,71. **O estágio 2 não rodou.**

### Não executadas (código pronto em `b1_variants.py`, `b1_pairs.py`, `b1_sent.py`, `b1_auto.py`)
- **A2** rubrica com as nossas taxas no state;
- **A3a/A3b** few-shot com pares rotulados de outras conversas, e só com exemplos humanos;
- **A4** referência do próprio falante (10 mensagens reais dele no state + perguntas relativas);
- **A6** encadeamento (leituras do banco + medidas de código no state de um 2º Jev);
- **A7** pares decompostos por traço, nas duas ordens;
- **A12** frase a frase ("um amigo escreveria esta frase?"), com o experimento de corte;
- **A13** autoresearch (o gpt-6-luna propõe perguntas a partir dos erros do dev);
- o banco do Jev sobre as respostas do gpt-6-luna e do deepseek.

**Não tirar conclusão sobre elas.**

## 3. Arquitetura recomendada (com o que foi medido)

```
resposta da LLM ──┬─ código: 31 features (grátis, <1 ms)
                  └─ Jev, 1 chamada, ~16–25 perguntas (Nouls atômicos + 2 Scores ancorados), ~0,5 s em paralelo
        ─> regressão logística (pesos treinados em dados rotulados humano × LLM) ─> P(cara de LLM)
        ─> se P alta: os Nouls de vício com p > 0,5 dizem O QUE corrigir
           (pergunta no fim, pergunta genérica, interjeição, validação, eco, molde, entusiasmo, gíria, longo demais)
```
- **Juiz:** código + Jev. AUC 0,91 (0,90 controlada por tamanho) e 0,82 nas saídas já instruídas.
- **Diagnóstico:** um Noul por vício. Nunca uma Choice entre vícios, nunca "soa humano?".
- **O que o Jev acrescenta sobre o código:**
  1. +0,05 de AUC no geral e +0,07 a +0,1 nas saídas instruídas e no briefing B;
  2. sinais que o código não mede: "segue o próprio fio", "seca", "faz 2+ coisas", "energia acima da do outro", "gíria
     forçada", "oferece apoio";
  3. a leitura semântica do vício, útil para reescrever.

## 4. Tradução para o sistema

1. **Validação pós-geração (J2 do `RELATORIO_FINAL`):**
   - trocar os 4 Nouls do a8 pela lista enxuta do §2/A1, com pesos aprendidos;
   - o limiar de P(LLM) é calibrado nos logs do produto;
   - acima do limiar, os vícios com p > 0,5 viram **ações de código**: cortar a última frase se for pergunta genérica,
     tirar a interjeição inicial, tirar a fórmula de validação, baixar "!"/emoji, e regenerar com "uma ideia só" se o molde
     ou "faz 2+ coisas" dispararem.
2. **Não usar como gate:** o Noul "é IA/humano", a Choice "vício principal", os contrafactuais e `direct`/`register`/
   `invented` com o sinal ingênuo.
3. **Retreinar os pesos por ator:** o briefing B (0,73) e o deepseek com estilo (0,77 só em código) são os mais difíceis.
   Os pesos precisam ver saídas do ator escolhido.

## 5. Limitações

- **Interrompido por créditos.** 6 das 13 arquiteturas não rodaram, e o Jev não avaliou os atores fortes (só código).
- O "humano" é o maichat (jovens britânicos, amigos e casais) e 50 aberturas do empathetic. Não há dado em PT-BR.
- Os rótulos de vício vêm de regex (proxies). Não houve rotulagem humana dos vícios semânticos.
- O teste tem 211 humanas em ~21 conversas, e os ICs refletem isso. As features do Jev foram escolhidas vendo o a8, que
  está no mesmo corpus (há algum risco de otimismo, embora os pesos e a seleção L1 tenham sido feitos só no dev).
- Custo: 2.330 chamadas ao Jev (US$ 0,30) e 1.000 gerações de LLM (US$ 0,15).
