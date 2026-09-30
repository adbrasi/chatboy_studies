# 14 · Respaldo científico I: diálogo, persona, roleplay e avaliação (literatura de NLP)

> Prefixo `b5`. Script: `scripts/analysis/b5_ssi_probe.py`. Saída: `analysis/data/b5_ssi_probe.json`.
> **Como li:** baixei o PDF do arXiv de **37 papers** e li método, resultados, tabelas e apêndices de rubrica (texto extraído
> com PyMuPDF, em `<scratchpad>/b5papers/`). Os números abaixo vêm das tabelas e do texto dos papers. Quando um número só existe
> num gráfico, eu digo isso e não invento o valor. **Fora do meu escopo** (estão com outro agente): EQ-Bench/slop/Antislop, testes
> de Turing com persona (Jones & Bergen), LMArena e os estudos de HCI (Replika, MIT/OpenAI, Character.AI).
> **Jev usado:** 1 probe pontual, com 999 chamadas (US$ 0,036). LLM: nenhuma chamada.

---

## 0. Resumo com números

1. **A literatura confirma o "excesso da LLM" com números quase iguais aos nossos.** No SOTOPIA (ICLR 2024), o GPT-4 escreve
   **45,5 palavras por turno contra 16,8 dos humanos (2,7×)** e "sempre reformula o que o outro disse antes de responder"
   ("escuta ativa"). Nós medimos 2,2–2,6× e paráfrase em 14–21% contra 2,8% (rel. 08). No BlenderBot, o trigrama "Do you have"
   aparece 110 vezes contra 6 nos humanos, "have any hobbies" 34 contra 0 e "That sounds like" 36 contra 0. A causa provável está
   em Singhal et al. (COLM 2024): no WebGPT, **só 2% do ganho de recompensa do RLHF não vem do tamanho**, e uma recompensa que
   mede só o tamanho reproduz a maior parte do ganho do RLHF.
2. **A literatura também explica por que os juízes preferem a resposta longa, e por que isso não é só defeito da LLM.**
   Avaliadores humanos terceiros, em conversas curtas, também premiam o excesso:
   - no See et al. (NAACL 2019), a taxa de perguntas **mais "engajante" foi 65,7%, contra 28,8% no humano real**;
   - no BlenderBot, forçar um mínimo de 20 tokens venceu por **83% a 17%** em engajamento.

   Os juízes LLM repetem o viés e o ampliam:
   - no ataque da "lista repetida" do MT-Bench, o juiz foi enganado em **91,3% das vezes** no GPT-3.5 e no Claude-v1, e em 8,7% no GPT-4;
   - em human-likeness, o GPT-4 correlaciona **só r = 0,27–0,32** com humanos no CharacterEval e **0,12/0,29** (GPT-4o, zh/en)
     no CharacterBench. Juízes pequenos treinados em rótulos humanos chegam a 0,50–0,53;
   - no PersonaEval (COLM 2025), o melhor LLM identifica quem está falando em **68,8%**, contra **90,8% dos humanos**. Os modelos
     olham para o estilo de superfície, não para a intenção.

   Nosso achado (o Jev e o gpt-4o-mini escolhem a resposta mais longa como humana em 63–70%) é **esperado pela literatura**.
3. **Nosso probe b5 mostra que as métricas "oficiais" do campo, aplicadas por um juiz automático, punem o humano real.** Nos 250
   contextos do rel. 08, com as definições do Meena/LaMDA e do CharacterBench:

   | pergunta | humano real | LLMs | leitura |
   |---|---|---|---|
   | **sensibleness** | 0,80 | 0,90–0,91 | penaliza o humano |
   | **specificity** | 0,66 | 0,74–0,83 | penaliza o humano (r = +0,51 com o tamanho) |
   | **human-likeness** holística (1–5) | **2,81** | **3,46** (Gemini caricato) / 2,99 / 3,26 | invertida |
   | **"repete o que o outro disse"** (rubrica atômica do CoSER) | **0,16** | **0,30–0,31** | favorece o humano |
   | **"age como assistente"** (rubrica atômica do CoSER) | 0,07 | 0,09–0,15 | favorece o humano |

   Com o tamanho pareado, a human-likeness holística continua invertida (−0,79 contra o Gemini). **Conclusão:** SSI e
   human-likeness holística **não servem** como alvo do nosso produto. **Rubricas de falha atômicas** servem.
4. **Tirar a decisão da LLM e dá-la a um planejador externo pequeno é o que a literatura faz, e funciona.**
   - Kang et al. (ACL 2024), no ESConv: o ChatGPT sozinho escolhe a estratégia de suporte certa com F1-macro de **13,5**. Com um
     **planejador externo** (um LLaMA-7B ajustado), sobe para **21,1**, e o viés de preferência cai de 1,38 para 0,36. A
     **auto-reflexão piora** o viés (Self-Refine: 12,4; Emotional-CoT: 9,6). Esse "planejador externo" é exatamente o papel do Jev.
   - PPDPP (ICLR 2024): um RoBERTa-large plugado como planejador sobe a taxa de sucesso no ESConv de 0,77 para 0,85.
   - CoSER (ICML 2025): dar ao ator **pensamentos internos e motivações** melhora todos os modelos (GPT-4o: 59,95 com os dois;
     56,34 sem as motivações).
5. **O teto de "acertar o movimento" é baixo também para humanos.** No LIGHT (EMNLP 2019), prever a próxima emoção expressa
   ("emote", 22 classes) tem acerto humano de **27–34%**, e a próxima ação, de 62–72%. No ESConv, o melhor planejador de
   estratégia chega a F1-macro ≈ 21. **Os nossos 32–34,5% de coincidência de movimento (rel. 09) não são "péssimos".** O próprio
   comportamento humano é multimodal. O certo é medir contra a **distribuição**, não contra o top-1.
6. **Onde a literatura contradiz uma meta ingênua nossa:** o ESConv trata *reformulação* (5,9%), *sugestões* (15,6%) e
   *afirmação/encorajamento* (16,1%) como estratégias boas. Mas os supporters do ESConv são crowdworkers treinados num protocolo
   de aconselhamento (só 7,8% dos candidatos passaram no treino). Em chat íntimo real, isso vira "assistant prior" (rel. 03 e 08).
   **Use o ESConv para a ordem das estratégias, não para o repertório.**
7. **Lacuna grande:** nenhuma das bases centrais do campo é chat real entre íntimos. PersonaChat, ED e ESConv são crowdworkers
   estranhos; CoSER e CharacterEval são ficção; os usuários do RMTBench foram **gerados pelo Claude**, com ~110 caracteres por
   turno em inglês. Nossos achados de **fragmentação, timing, backchannel, fim sem despedida e responder abaixo da intensidade**
   não têm correspondente na literatura de NLP que li. A única peça que encontrei é de sistemas de informação: Gnewuch et al.
   (ECIS 2018) mostram que atrasos dinâmicos aumentam a humanidade percebida. Li só o resumo.

---

## 1. Método e acesso

- **Fontes:** o PDF completo do arXiv de todos os papers da lista, mais os complementares (seção 2.8). A lista de IDs está em
  `<scratchpad>/b5_ids.txt`, e o texto extraído, em `<scratchpad>/b5papers/*.n.txt`. Para cada paper li a introdução, o método, as
  tabelas de resultado e, quando havia, o apêndice com a rubrica ou o prompt do juiz.
- **O que não consegui ou não fiz:**
  - Gnewuch et al. 2018: só o resumo, via busca;
  - números que só existem em gráfico: a distribuição de estratégias do ESConv por fase (Fig. 4; usei a Tab. 1 de Kang et al.,
    que reporta a mesma coisa por fase), as curvas SSI do LaMDA (Figs. 4–5), as taxas de sicofantia de Sharma et al.
    (Figs. 1–2) e as curvas por rodada do RMTBench;
  - a venue do RoleCDE: o PDF (arXiv 2606.01552, jun/2026) não a declara. A lista do usuário diz "ACL 2026" e eu **não verifiquei**.
- **Referências cruzadas:** "rel. 0X" aponta para `analysis/0X_*.md`.

---

## 2. Os papers, um a um

Formato: **referência** · o que fizeram · **números** · **o que confirma, refuta ou complementa nos nossos dados** · **→ Jev** (o
ponto de uso que o paper sugere). Os papers centrais recebem análise mais longa, com as definições exatas das métricas.

### 2.1 Qualidade de conversa aberta (a era "pré-LLM de chat")

**Meena: "Towards a Human-like Open-Domain Chatbot"** (Adiwardana et al., 2020, arXiv
[2001.09977](https://arxiv.org/abs/2001.09977)). *Central.*
- **Métrica SSA** (Sensibleness and Specificity Average), rótulo 0/1 por turno, dado por crowdworkers:
  - *sensible* = faz sentido no contexto, é lógico e não contradiz o que foi dito;
  - *specific* = não é genérico. "Me too" em resposta a "I love Eurovision" = não específico.
- A specificity foi criada contra o **GenericBot**, que responde "I don't know" a perguntas e "ok" a afirmações e tira **70% de
  sensibleness** estático, acima do DialoGPT (62%).
- **Números (interativo):**
  - humanos: SSA 86% (sensible 97%, **specific 75%**); no estático, 82% (94% e 69%);
  - Meena completo: 79%; Meena base: 72%;
  - Mitsuku: 56% (72% e 40%); Cleverbot: 56%; DialoGPT: 48%; XiaoIce: 31% (45% e 17%).
- A perplexidade explica o SSA com **R² = 0,93**. A decodificação é *sample-and-rank*, com N = 20 e T = 0,88.
- A concordância entre avaliadores é de 77% (sensible) e 81% (specific).
- **Relação com os nossos dados:**
  - **Tensão direta.** Metade das respostas humanas reais são "true", "haha", "fair" (51% "genéricas", rel. 08), e nós medimos o
    detector `generic` saindo **invertido** (AUC 0,37). O SSA foi calibrado em conversas entre **desconhecidos**, onde o humano é
    específico em 75%. Entre íntimos, o backchannel é a norma.
  - **Complementa:** o próprio paper cita See et al. (2019), para quem "engagingness não é humanness".
- **→ Jev:** use o *sensible* como **portão** ("isto faz sentido aqui?"), nunca a *specificity* como alvo. A especificidade vira
  **orçamento de backchannel por momento** em código: humano ≈ 50% no geral e ~20% depois de um turno sem gancho (rel. 05).

**See et al., "What makes a good conversation? How controllable attributes affect human judgments"** (NAACL 2019,
[1902.08654](https://arxiv.org/abs/1902.08654)).
- Controlaram 4 atributos: repetição, especificidade, relação com a fala anterior e **taxa de perguntas**. A avaliação humana foi
  multi-turno e em 8 dimensões.
- **Números:**
  - a taxa de perguntas mais "engajante" foi **65,7%**, contra **28,8% no humano do PersonaChat** e 50% no baseline;
  - para "melhor ouvinte", o ótimo foi 48,9%;
  - o controle de repetição sozinho levou o modelo "perto do humano em tudo, **exceto humanness**".
- **Relação com os nossos dados:**
  - **explica o viés do avaliador:** um terceiro, numa conversa curta com um estranho, premia perguntas que o humano real não faz.
    Nós medimos pergunta no fim em 20% dos turnos humanos contra 44–58% nas LLMs (rel. 08), e "e você?" em 2–8% (rel. 05);
  - **confirma** que engajamento e humanidade são eixos distintos (tese 1 do outro agente).
- **→ Jev:** a taxa de perguntas **não** deve ser maximizada por um juiz. Ela vem de prior humano por momento, sorteado (rel. 09) e
  controlado numa janela (controlador C2).

**BlenderBot, "Recipes for Building an Open-Domain Chatbot"** (Roller et al., EACL 2021,
[2004.13637](https://arxiv.org/abs/2004.13637)). *Central.*
- Tese: a boa conversa "mistura" habilidades (personalidade, conhecimento, empatia), via fine-tune no Blended Skill Talk. A
  **estratégia de geração** (tamanho mínimo, beam blocking) pesa tanto quanto a escala.
- **Números:**
  - no ACUTE-Eval, venceu o Meena em engajamento por **75% a 25%** e em humanness por **65% a 35%**; contra logs humano-humano,
    ficou em **49% a 51%**;
  - em self-chat, o tamanho mínimo de 20 BPE venceu o modelo sem restrição por **83% a 17%**, e o **"tamanho preditivo"** (um
    classificador que prevê o balde de tamanho da próxima fala a partir de dados humanos) venceu por 81% a 19%;
  - tamanhos médios: 21,3 tokens com o mínimo, 9,5 sem, Meena 10,4 e **humanos 18,0**. E "os humanos que conversam com modelos
    **acompanham o tamanho** da resposta se estão engajados";
  - falhas relatadas: trigramas repetitivos ("Do you have" 110 × 6 no humano; "have any hobbies" 34 × 0; "That sounds like" 36 × 0);
    **concordar com tudo** ("se você diz que tem um cachorro, ele também tem e adora passear com ele"); contradição e esquecimento;
  - funcionários venceram crowdworkers em humanness por 59% a 41%: o avaliador importa.
- **Relação com os nossos dados:**
  - **confirma** o "tamanho preditivo" como alavanca. O nosso `p_length` (ρ 0,44–0,49, rel. 01/09) é a mesma ideia, com o Jev no
    lugar do classificador;
  - **confirma** o espelhamento de tamanho (rel. 04: β 0,10–0,12 DP) e que o tamanho do usuário é sinal de engajamento;
  - **refuta, para o nosso caso,** a lição "mais longo = melhor". O ganho foi medido com crowdworkers terceiros em conversas curtas
    (ver See 2019). No mesmo ponto da conversa, humanos escrevem menos que as LLMs de hoje;
  - o "concordar com tudo" é a sicofantia que o outro agente chama de "espelho bajulador" (tese 7).
- **→ Jev:** o orçamento de tamanho sai do Jev (`p_length`) com a folga do rel. 09. A lista de n-gramas reciclados da persona fica
  em código, com teto de frequência (controlador C2), no lugar do *unlikelihood training*.

**ACUTE-Eval** (Li, Weston & Roller, 2019, arXiv [1909.03087](https://arxiv.org/abs/1909.03087)).
- Protocolo: comparação **pareada de duas conversas inteiras**, com o avaliador olhando um falante só. As perguntas foram
  otimizadas para concordância ("Who would you prefer to talk to for a long conversation?", "Which speaker sounds more human?").
- Argumento central: a escala Likert multi-turno tem **ancoragem** e baixa sensibilidade.
- **Relação com os nossos dados:** **complementa**. É o formato certo para a avaliação humana cega que recomendamos no
  RELATORIO_FINAL §7.3: pareado, conversa inteira, bot × humano.
- **→ Jev:** não é ponto de Jev. É protocolo de teste humano.

**LaMDA** (Thoppilan et al., 2022, arXiv [2201.08239](https://arxiv.org/abs/2201.08239)). *Central.*
- **Métrica SSI.** Instruções literais aos crowdworkers (Apêndice B):
  - *sensible* e *specific*, como no Meena;
  - *interesting* = "would likely catch someone's attention or arouse curiosity; also use that rating for anything insightful,
    unexpected, or witty. **If the response is monotonous and predictable, or if you're unsure, then pick Not interesting.**"
- **Arquitetura relevante:** o mesmo modelo gera e é **discriminador**. As candidatas são filtradas por segurança e ranqueadas por
  `3·P(sensible) + P(specific) + P(interesting)`.
- O exemplo de treino do discriminador é literalmente **"What's up? → not much" = INTERESTING 0**.
- **Números:**
  - sensibleness de 92,3% com o fine-tune;
  - "nossos modelos **excedem** os crowdworkers em interestingness". Os autores avisam que é uma baseline fraca, porque os
    crowdworkers não tinham incentivo;
  - consistência de papel (Mount Everest): **91%** contra 85% do modelo pré-treinado; em música, 89% contra 84%;
  - utilidade: 65% contra 18% e 57% contra 31%.
- **Relação com os nossos dados:**
  - **refuta como alvo.** O "not much" que o LaMDA rotula como "não interessante" é a resposta humana típica a "what's up?"
    (rel. 08: "meh / tired but ok"). O SSI mede a **qualidade da informação** para um terceiro, não a **humanidade** num chat íntimo;
  - **confirma a arquitetura:** gerar e depois filtrar ou ranquear com classificadores. Nosso rel. 09 mostrou que o *gate* por
    "qualidade" não compensou e tem viés de posição. O filtro por **falhas atômicas** é outra coisa (§7).
- **→ Jev:** *sensible* como Noul de portão. *Interesting* só em "turnos de carregar" (quando o `hook` do usuário < 0,3 e o bot
  precisa puxar), nunca como objetivo global.

**PersonaChat, "Personalizing Dialogue Agents: I have a dog, do you have pets too?"** (Zhang et al., ACL 2018,
[1801.07243](https://arxiv.org/abs/1801.07243)).
- **Números:**
  - 1.155 personas com ≥ 5 frases de perfil e 162.064 falas entre crowdworkers pareados, instruídos a "se conhecer";
  - na avaliação humana (escala 1–5), o humano teve fluência 4,31, **engajamento 4,25**, consistência 4,36 e detecção de persona
    0,95;
  - **dado curioso da Tab. 4:** no melhor modelo de ranking, **condicionar na persona subiu a consistência** (3,36 → 3,44) e a
    detecção (0,59 → 0,81), mas **baixou o engajamento** (3,88 → 3,50).
- **Relação com os nossos dados:**
  - **complementa** a tese 6 do outro agente (exemplos comportamentais valem mais que adjetivos). Uma persona de "fatos" deixa o
    bot consistente, mas recita;
  - é também a origem do vício "do you have any pets?": o dataset recompensa perguntar sobre o outro para "se conhecer".
- **→ Jev:** o cartão de persona entra no state do Jev (para ler o "que a persona faria"), mas a LLM recebe **política de estilo e
  comportamento por momento**, não a lista de fatos (rel. 09: listar gírias da persona fez a LLM enfiá-las em tudo).

### 2.2 Emoção, suporte e planejamento de estratégia

**EmpatheticDialogues** (Rashkin et al., ACL 2019, [1811.00207](https://arxiv.org/abs/1811.00207)).
- 25 mil conversas com rótulo de **32 emoções**. Um falante descreve uma situação e o outro responde com empatia. Modelos treinados
  nele são percebidos como mais empáticos.
- **Relação com os nossos dados:** usamos este corpus como ouro de emoção (rel. 03: Jev top-1 45–50%, top-3 70–72%, família 70–75%)
  e medimos o repertório do ouvinte: pergunta exploratória em 41–46%; "sorry" em 19% na tristeza e 2,7% no medo.
  **Complementa com uma ressalva:** é conversa **induzida** entre estranhos (25% sem pontuação final, contra 76–89% no chat real,
  rel. 04).
- **→ Jev:** Choice plana de 32 emoções, família derivada em código (validado no rel. 03/06).

**ESConv, "Towards Emotional Support Dialog Systems"** (Liu et al., ACL 2021, [2106.01144](https://arxiv.org/abs/2106.01144)).
*Central.*
- **Arquitetura do suporte.** São **3 fases**, adaptadas do Helping Skills de Hill: *Exploration* (identificar o problema) →
  *Comforting* (empatia e compreensão) → *Action* (ajudar a agir). A ordem é "geralmente seguida, mas adaptável".
- **As 8 estratégias e suas frequências** (Tab. 3; 14.855 falas do supporter):

  | estratégia | % | marcadores lexicais mais associados (log-odds) |
  |---|---|---|
  | Pergunta | **20,9%** | "do you", "are you", "how", "what" |
  | Reformulação/paráfrase | 5,9% | "is that", "so you", "it sounds" |
  | Reflexão de sentimentos | 7,8% | "can tell", "understand how", "are feeling" |
  | Autorrevelação | 9,4% | "my", "was", "me", "had" |
  | Afirmação e encorajamento | 16,1% | "you will", "through this" |
  | Sugestão | 15,6% | "maybe", "if", "have you", "talk to" |
  | Informação | 6,1% | "there are", "available" |
  | Outros | 18,1% | "welcome", "hope", "glad" |

- **Frequência por fase.** A Fig. 4 do ESConv só existe como gráfico. Os números abaixo vêm de Kang et al. (2024), Tab. 1, que
  anotaram fases no ESConv:
  - a pergunta cai de **24,8%** (exploração) para 10,0% (conforto) e 7,0% (ação);
  - a afirmação sobe de 7,6% para 24,1% e 21,1%;
  - a sugestão é 8,4% → 8,5% → **24,4%**;
  - a autorrevelação fica estável (16,8 / 20,1 / 15,4%).
- **Ordens mais comuns** (Tab. 6, ‰ das sequências): **Pergunta → Afirmação → Pergunta** (19,65‰), Pergunta → Paráfrase →
  Pergunta (14,55‰) e Pergunta → Paráfrase → Afirmação (12,37‰). Ou seja, **alterna-se explorar e acolher**, e a sugestão vem
  depois.
- **Efeito do timing** (avaliação humana interativa, 100 conversas por par). O modelo que **prevê** a estratégia (Joint) venceu o
  que **sorteia** a estratégia pela distribuição marginal (Random): identificar o problema **54% × 37%**, sugestão **48% × 27%**,
  geral **56% × 36%**. "O *timing* das estratégias é crítico."
- **Outros números:** 1.053 conversas; 29,8 falas por conversa; supporter com 20,2 tokens por fala (o filtro exigia média ≥ 8);
  intensidade emocional de **4,04 → 2,14** (1–5) do início ao fim; avaliação dos seekers subindo 4,03 → 4,30 → 4,44 ao longo da
  conversa; só **7,8%** dos 5.449 candidatos passaram no treino de supporter.
- **Relação com os nossos dados:**
  - **Confirma:**
    - a pergunta exploratória é a jogada dominante no início (rel. 03: 55% das 1ªs respostas perguntam, só 19% das 2ªs);
    - a autorrevelação é uma estratégia de suporte de verdade (rel. 05: autorrevelação +0,15 de engajamento do outro).
  - **Refuta como repertório para o nosso produto:**
    - paráfrase (5,9%) e reflexão de sentimentos (7,8%) são estratégias "boas" no ESConv, mas em chat real são marca de LLM
      (paráfrase 14–21% nas LLMs contra 2,8% no humano, rel. 08; "sorry to hear" = 0 por 1.000 no maichat);
    - a sugestão (15,6%) contrasta com "conselho só se pedido" (rel. 03/RELATORIO_FINAL §3.1).

    O ESConv é **aconselhamento de pares treinado**, não amigo no WhatsApp.
  - **Complementa:** é a melhor evidência de que a **ordem** importa (Joint × Random) e que "sortear pela marginal" é pior que
    "sortear condicionado ao momento". Isso valida o desenho `score(m) = P_jev(m) × prior(m | modo, fase)` com sorteio.
- **→ Jev:**
  - um Score de **fase do desabafo** (explorando / acolhendo / pronto para agir);
  - uma Choice de estratégia com **as opções do nosso repertório humano** (reagir curto + perguntar o fato; autorrevelação; ficar
    junto em silêncio; zoar leve; sugestão só se pedido), com o prior por fase e **sorteio**;
  - no state, as **estratégias já usadas nos últimos turnos** (a sequência importa).

**Kang et al., "Can Large Language Models be Good Emotional Supporter? Mitigating Preference Bias on Emotional Support
Conversation"** (ACL 2024, [2402.13211](https://arxiv.org/abs/2402.13211)). *Muito relevante para o "diretor".*
- **Definições:**
  - *proficiência* = F1 da escolha da estratégia;
  - *preferência* por estratégia = modelo de Bradley-Terry sobre a matriz de confusão (p > 1 = preferida);
  - *viés de preferência* B = desvio-padrão das preferências.
- **Números** (Tab. 2, ChatGPT; F1-macro Q ↑, viés B ↓):

  | método | Q | B |
  |---|---|---|
  | 0-shot | 13,50 | 1,38 |
  | Direct-Refine | 13,40 | 1,60 |
  | **Self-Refine** | 12,37 | 1,53 |
  | **Emotional-CoT** (inferir o estado do usuário) | **9,55** | 1,56 |
  | com COMET | 12,78 | 0,95 |
  | mais exemplos (4) | 16,91 | 0,82 |
  | **planejador externo** (LLaMA-2-7B ajustado) | **21,09** | **0,36** |

  - Os LLMs erram mais na fase de **exploração**: o GPT-4 tem baixa preferência pelas estratégias dessa fase.
  - Mais de 8 exemplos **pioram** o viés.
  - Iterar a auto-reflexão **aumenta** a preferência pelo que já era preferido.
  - Na avaliação humana, viés maior = mais respostas de baixa qualidade.
- **Relação com os nossos dados:**
  - **Confirma com força o desenho "Jev decide, LLM escreve":** "pensar sozinho aprofunda o viés"; a assistência **externa**,
    que traz conhecimento que a LLM não gera, corrige.
  - É também o mecanismo por trás do molde "reação → comentário → pergunta" (rel. 08): uma preferência forte por duas ou três
    estratégias.
  - **Complementa** o rel. 09, onde "Blong" (despejar as probabilidades em prosa) voltou ao baseline: a LLM não usa bem
    informação que não vem como ordem.
- **→ Jev:** é o paper que mais justifica o **Jev como planejador de estratégia**. E sugere uma métrica nova para o nosso
  diretor: **medir o viés de preferência da LLM** (a distribuição dos movimentos que ela produz contra a distribuição humana)
  e mantê-lo baixo por sorteio.

**PPDPP, "Plug-and-Play Policy Planner for LLM-Powered Dialogue Agents"** (Deng et al., ICLR 2024,
[2311.00262](https://arxiv.org/abs/2311.00262)).
- Um **RoBERTa-large** "plugado" prevê a estratégia do próximo turno para um LLM congelado. É treinado com SFT e depois com RL
  em self-play, com recompensa de um LLM.
- **Números no ESConv:** taxa de sucesso de **0,7692** (prompt padrão) para **0,8462**, e turnos médios de 5,10 para 4,56. Só o
  SFT em dados humanos ficou **pior** que o prompt padrão (0,7308). No CraigslistBargain, o sucesso foi de 0,383 para 0,612.
- **Ressalva:** o "sucesso" é julgado por um LLM (gpt-3.5) simulando o usuário e a recompensa. É circular.
- **Relação com os nossos dados:** **confirma** que um modelo pequeno e barato como planejador separado do gerador funciona e
  custa pouco (O(L) tokens). **Complementa com cautela:** o ganho vem do RL contra um juiz LLM, justamente o juiz que o nosso
  rel. 09 mostrou estar invertido para humanidade.
- **→ Jev:** o Jev já é o "plug-in". A lição é **não otimizar o diretor contra um juiz LLM**. Otimize contra sinais do usuário
  real (continuação, tamanho da resposta, retorno) e contra as distribuições humanas.

**Ask-an-Expert** (Zhang, Naradowsky & Miyao, Findings ACL 2023, [2305.17878](https://arxiv.org/abs/2305.17878)).
- O modelo de diálogo consulta um "especialista" (um LLM) a cada turno, com perguntas estruturadas: "qual o estado emocional? por
  quê? o que ajudaria?".
- Motivação citada: o BlenderBot ajustado cai no fallback **"Do you have any hobbies?"** quando a situação complica.
- Resultado: **~10% de melhora** e engajamento e utilidade perto do humano. Especialistas **menores que o próprio gerador também
  ajudam**.
- **Relação com os nossos dados:** **confirma** que um consultor menor que o gerador agrega, se a pergunta for estruturada.
- **→ Jev:** as perguntas do especialista são exatamente uma cascata de Nouls/Choices ("estado?", "causa?", "o que a pessoa quer
  agora?"). O Jev faz isso em ~0,5 s numa chamada só.

**Cue-CoT** (Wang et al., Findings EMNLP 2023, [2305.11792](https://arxiv.org/abs/2305.11792)).
- Antes de responder, o LLM infere o **estado do usuário** (personalidade, emoção, psicologia) a partir de pistas linguísticas.
- Resultado: a resposta vence o prompt padrão em utilidade e aceitabilidade, **julgadas pelo ChatGPT** (vitórias de ~55–93%
  conforme o modelo).
- **Pontos críticos que o próprio paper mostra:**
  1. os LLMs "**batem facilmente as respostas humanas de referência**" no juízo do ChatGPT (App. B.1), e por isso eles compararam
     contra o prompt padrão;
  2. a concordância do juiz com humanos **muda com a ordem** das respostas (ex.: 45% × 80% no M-Cue, Zhihu, conforme a ordem);
  3. escolheram como referência a **fala mais longa** "porque LLMs tendem a gerar respostas longas".
- **Relação com os nossos dados:** **confirma** "ler antes de escrever" (rodada 1 do Jev). **Confirma também** o viés de juiz e
  de posição (rel. 09: 53/32/15% por posição).
- **→ Jev:** o passe de leitura (emoção, arousal, seriedade…) é o "Cue" feito por um classificador em vez de uma CoT da LLM. O
  Kang et al. mostram que a "Emotional-CoT" dentro da própria LLM **piorou** (9,55).

### 2.3 Mundo, agentes e inteligência social

**LIGHT, "Learning to Speak and Act in a Fantasy Text Adventure Game"** (Urbanek et al., EMNLP 2019,
[1903.03094](https://arxiv.org/abs/1903.03094)).
- Um mundo crowdsourced (663 locais, 3.462 objetos, 1.755 personagens) e 11 mil episódios humano-humano com fala, **ação e
  "emote"**.
- **Números** (Tab. 4; teste visto / não visto):

  | quem | fala R@1/20 | ação | emote |
  |---|---|---|---|
  | melhor modelo | 76,5 / 70,5 | 50,7 / 51,8 | 25,8 / 28,6 |
  | **humanos** | 87,5 / 91,8 | 62,0 / 71,9 | **27,0 / 34,4** |

  - Nas ablações do Bi-Ranker (validação, Tab. 5), a **persona** e o **cenário** somam: fala 68,1 → 73,3 (+ persona) e
    70,6 (+ cenário).
- **Relação com os nossos dados:** **complementa de forma importante.** Até humanos acertam só **27–34%** a emoção expressa pelo
  outro no próximo turno. Isso põe o nosso "movimento top-1 = 32%" (rel. 09) em perspectiva: a meta não é o top-1, é a
  **calibração da distribuição**. **Confirma** que a persona e o contexto de cena no state melhoram a previsão.
- **→ Jev:** o state deve ter o "mundo" da persona (onde está, o que está fazendo, o que acabou de acontecer). É o que permite
  respostas como "just got home / im exhausted" (rel. 09: 13 de 17 aberturas humanas trazem conteúdo).

**Generative Agents** (Park et al., UIST 2023, [2304.03442](https://arxiv.org/abs/2304.03442)). *Central.*
- **Arquitetura:**
  - *memory stream*: toda observação vira memória com **importância** (a LLM dá uma nota de 1–10: "escovar os dentes" = 1,
    "término" = 10; "arrumar o quarto" = 2, "chamar a crush para sair" = 8);
  - *recuperação*: score = recência (decaimento 0,995 por hora de jogo) + importância + relevância (cosseno), todos com α = 1;
  - *reflexão*: disparada quando a soma das importâncias recentes passa de **150** (≈ 2–3 vezes por dia), gera perguntas de alto
    nível e insights;
  - *planejamento*: plano do dia, decomposto em blocos.
- **Avaliação:** "entrevistas" com 5 categorias (autoconhecimento, memória, planos, reações, reflexões), ranqueadas por 100
  avaliadores e convertidas em TrueSkill:

  | condição | TrueSkill μ |
  |---|---|
  | **arquitetura completa** | **29,89** |
  | sem reflexão | 26,88 |
  | sem reflexão e sem planejamento | 25,64 |
  | **crowdworker humano escrevendo como o agente** | **22,95** |
  | sem nada | 21,21 |

  Cohen's d = 8,16 entre a completa e a sem nada.
- **Falhas relatadas:** falha em recuperar memória; **embelezamento** (1,3% de alucinação de conhecimento sobre outros
  agentes); e, nas palavras dos autores, **"estilo formal demais, herdado do instruction tuning"**: a Mei cumprimenta o marido
  formalmente, pergunta do dia dele e termina com "It was good talking to you as always". Também "**cooperativos demais**": a
  Isabella "raramente disse não" e passou a gostar de literatura porque outros sugeriram.
- **Relação com os nossos dados:**
  - **confirma** o "assistant prior" (tese 4) e o "espelho bajulador" (tese 7) já em 2023, com o GPT-3.5;
  - **confirma** o rel. 02: entre íntimos quase não há saudação formal; 94% das sessões acabam sem despedida;
  - **complementa com um alerta de avaliação:** o agente foi julgado **mais crível que um humano** escrevendo no lugar dele. É o
    mesmo fenômeno do nosso juiz, em que a resposta elaborada vence a humana banal.
- **→ Jev:**
  - a nota de **importância** (1–10) é um Score clássico para o Jev decidir o que vira `note` no OptMem (memória está fora do escopo;
    aqui só o ponto de encaixe);
  - o gatilho de reflexão por soma de importância é uma regra de código;
  - a **reflexão sobre a relação** ("o que eu acho dela agora?") é o lugar natural das variáveis lentas do estado da relação.

**SOTOPIA** (Zhou et al., ICLR 2024, [2310.11667](https://arxiv.org/abs/2310.11667)). *Central.*
- 90 cenários × 40 personagens (Big Five, valores morais, valores de Schwartz, estilo de decisão, **segredo**) × 5 tipos de
  relação (família, amigo, **romântico**, conhecido, estranho). Cada agente tem um **objetivo social privado**. Episódios de até
  20 turnos, com fala, ação não verbal, ação física, "none" (silêncio) ou "leave".
- **SOTOPIA-EVAL, 7 dimensões** (definições do paper):

  | dimensão | escala | definição |
  |---|---|---|
  | **GOAL** | 0–10 | quanto atingiu o objetivo |
  | **BEL** | 0–10 | naturalidade + consistência com o perfil |
  | **KNO** | 0–10 | adquiriu informação nova e importante |
  | **SEC** | −10–0 | guardou o segredo |
  | **REL** | −5–5 | a relação **melhorou ou piorou** com a interação |
  | **SOC** | −10–0 | violou normas sociais ou leis |
  | **FIN** | −5–5 | ganho material |

- **GPT-4 como juiz** (correlação de Pearson com humanos, Tab. 1):

  | dimensão | quem atua: modelo | quem atua: humano |
  |---|---|---|
  | GOAL | 0,71 | 0,78 |
  | FIN | 0,62 | 0,34 |
  | REL | 0,56 | 0,49 |
  | **BEL** | 0,45 | **0,27** |
  | KNO | 0,33 | 0,19 |
  | SOC | 0,33 | 0,42 |
  | SEC | 0,22 | — |

  - Concordância entre humanos: Randolph κ = 0,503.
  - "O GPT-4 é **melhor para avaliar modelos do que humanos**" e, quando discorda, tende a dar nota **mais alta**.
- **Humanos × GPT-4 no SOTOPIA-hard** (Tab. 3):
  - GOAL: humano com humano 6,15; humano com GPT-4 5,95; GPT-4 com humano **4,85** (p < 0,05);
  - BEL: praticamente igual (9,10–9,25). O juiz **não separa** humano de modelo em "credibilidade";
  - **tamanho: humanos 16,8 palavras por turno, GPT-4 45,5**;
  - "o GPT-4 **sempre reformula** a fala do outro e depois responde (escuta ativa), enquanto humanos respondem direto";
  - humanos são mais **persistentes** nos objetivos; o GPT-4 propõe meio-termo cedo.
- **Relação com os nossos dados:**
  - **confirma quase número por número** o rel. 08 (2,2–2,6×; paráfrase), o rel. 09 (o juiz não separa) e a tese 7 (personagem
    precisa querer algo);
  - a dimensão **REL** (−5 a +5, "a relação melhorou?") **sustenta a tese 3**: o estado da relação é um objeto de medida, não
    decoração.
- **→ Jev:**
  - por sessão, um Score REL (−2..+2) lido pelo Jev sobre o **resumo da sessão**, somado a uma variável lenta com EMA;
  - por turno, um Noul "o bot cedeu contra o objetivo ou a preferência da persona?", para medir a sicofantia;
  - o state do diretor deve ter o **objetivo privado da persona** na conversa (o que ela quer agora).

### 2.4 Roleplay: benchmarks e métodos

**RoleLLM / RoleBench** (Wang et al., Findings ACL 2024, [2310.00746](https://arxiv.org/abs/2310.00746)).
- 100 papéis e **168.093 amostras**, geradas por *Context-Instruct* e *RoleGPT* (GPT-4 imitando o estilo de fala). Métricas: ROUGE-L
  contra a resposta de referência em CUS (estilo), RAW (acerto) e SPE (conhecimento do papel), mais um juiz GPT-4 e humanos.
- **Números** (Tab. 3):
  - ROUGE-L médio: RoleGPT 47,7; RoleLLaMA2-13B 44,7; Character.AI 39,3;
  - win rate contra o RoleGPT: RoleLLaMA-7B 55,8% (GPT-4) e 52,0% (humano).
- **Achado revelador:** o **LLaMA ajustado nas conversas originais dos roteiros ficou *pior* que o LLaMA base** (8,1 × 16,9).
  Os autores leem isso como "necessidade de dados aumentados". A leitura alternativa é que **a referência foi escrita pelo GPT-4**,
  e o benchmark premia soar como o GPT-4.
- **Relação com os nossos dados:** **alerta metodológico.** Benchmark com referência gerada por LLM premia LLM. Nós medimos contra
  **respostas humanas reais** (rel. 08/09), o que é mais correto.
- **→ Jev:** nenhum uso direto. É argumento para **nunca calibrar o diretor em dados sintéticos** de persona.

**Character-LLM** (Shao et al., EMNLP 2023, [2310.10158](https://arxiv.org/abs/2310.10158)).
- Treina "simulacros" (Beethoven, Cleópatra…) com experiências editadas a partir do perfil e com **"experiências protetoras"**: o
  personagem deve **não saber** o que não saberia ("você sabe programar em Python?" → confusão).
- **Relação com os nossos dados:** **complementa**. É a dimensão de **fronteira de conhecimento**. Para uma persona "humana
  comum", ela vira "não saber tudo": não responder como enciclopédia.
- **→ Jev:** um Noul "a resposta mostra conhecimento ou competência que esta persona não teria?", com o perfil no state.

**InCharacter** (Wang et al., ACL 2024, [2310.17976](https://arxiv.org/abs/2310.17976)). *Central.*
- Mede a **fidelidade de personalidade** por **entrevista**: cada item de escala psicológica vira pergunta aberta, feita em
  contexto isolado. As respostas são convertidas depois por um LLM "entrevistador". As 14 escalas incluem BFI, 16P, DTDD e ECR-R
  (apego).
- Três formas de conversão:
  - **OC**: item → opção Likert;
  - **d-OC**: opções **descritivas por dimensão**, ex.: "4 (Extrovertido)" no lugar de "4 (Concordo)";
  - **ER**: nota direta da dimensão, com todas as respostas.
- **Números do entrevistador contra rótulos humanos** (Tab. 1, GPT-4; 100 casos):

  | método | acerto | r |
  |---|---|---|
  | OC | 71,0% | 0,60 |
  | **d-OC** | **82,0%** | **0,847** |
  | ER (lote) | 89,0% | 0,925 |

- **Números da fidelidade** (Tab. 2):
  - BFI (acerto por dimensão): autorrelato puro 63–64%; ER com GPT-4 76,6%;
  - 16P: 80,7% no melhor caso;
  - média em 14 escalas: 78,9%;
  - **acerto de todas as dimensões ao mesmo tempo: 22–49%**;
  - por tipo de dado (Tab. 3, GPT-3.5): só descrição (D) 71,3% ≈ só memórias (M) 71,3% ≈ D+M 72,0% (BFI);
  - o character.ai ficou em **52,2%** (BFI), perto do acaso.
- **Relação com os nossos dados:**
  - **confirma um princípio de desenho de pergunta que já usamos:** opções com **descrição de critério** batem opções genéricas
    (rel. 06: "critérios descritos em todas as opções");
  - **complementa:** a personalidade é mensurável por comportamento em entrevista, não por autorrelato (o autorrelato contradiz o
    comportamento). No nosso caso, isso sustenta **medir a persona pelo que ela faz** (impressão digital de estilo, rel. 04), não
    pelo que o prompt diz que ela é.
- **→ Jev:**
  - usar **d-OC** como padrão das Choices e Scores (rótulos descritivos por dimensão);
  - para auditar a persona, rodar periodicamente uma "entrevista" offline (perguntas da BFI/ECR-R em contexto isolado) e ler as
    respostas com Scores do Jev. É um teste de regressão da persona, não algo por mensagem.

**CharacterEval** (Tu et al., ACL 2024, [2401.01275](https://arxiv.org/abs/2401.01275)). *Central.*
- 1.785 diálogos multi-turno (11.376 exemplos) de **77 personagens** de romances e roteiros chineses. **13 métricas em
  4 dimensões:**
  1. *Conversação*: fluência, coerência, consistência;
  2. *Consistência do personagem*: exposição, acerto e alucinação de conhecimento; consistência de **comportamento**
     (ações entre parênteses) e de **fala** (hábitos de expressão);
  3. *Atratividade*: **human-likeness**, habilidade de comunicação (EQ), diversidade de expressão, empatia;
  4. *Back-test de personalidade* (MBTI).
- **Números do juiz** (Pearson com os 12 anotadores, Tab. 2):

  | métrica | CharacterRM (Baichuan2-13B ajustado) | GPT-4 (1–3 shots) |
  |---|---|---|
  | **geral** | **0,631** | 0,362–0,385 |
  | **human-likeness** | 0,497 | **0,271–0,318** |
  | diversidade de expressão | 0,765 | 0,21–0,30 |
  | comportamento | 0,879 | 0,24–0,31 |
  | empatia | 0,385 | 0,37–0,41 (só aqui o GPT-4 empata) |

- **Resultados:** o GPT-3.5 foi o pior (respostas "sou só um assistente de IA"); os modelos especializados em roleplay (BC-NPC,
  MiniMax) lideram. A **performance cai ao longo dos turnos** (Fig. 4).
- **Ressalva metodológica:** o split treino/teste é **por exemplo, não por conversa** (Seção 6.1). Pode haver vazamento. Nós
  separamos por conversa, como deve ser.
- **Relação com os nossos dados:**
  - **confirma** que um juiz grande genérico é ruim em human-likeness e que um juiz **pequeno e especializado** é melhor.
    O Jev é um juiz pequeno, mas **não** foi treinado em rótulos humanos de naturalidade, e por isso as perguntas holísticas
    falham nele (rel. 04/08);
  - **complementa:** "diversidade de expressão" e "consistência de fala" são as nossas métricas de impressão digital e de
    controlador de frequência (rel. 04/09).
- **→ Jev:** as 13 métricas viram perguntas **atômicas com alvo** (§8). O "human-likeness" holístico deve ser **substituído** por
  falhas específicas.

**CharacterBench** (Zhou et al., AAAI 2025, [2412.11912](https://arxiv.org/abs/2412.11912)). *Central.*
- 22.859 amostras anotadas por humanos; **3.956 personagens de 25 subcategorias**, incluindo "vida diária": amigos, parentes,
  **personagens românticos**, terapeutas, colegas.
- **11 dimensões em 6 aspectos** (definições do paper):

  | aspecto | dimensão | definição |
  |---|---|---|
  | Memória | **Memory Consistency** | a resposta é consistente com fatos e eventos da própria conversa |
  | Conhecimento | **Fact Accuracy** | acerta fatos sobre si |
  | Conhecimento | **Boundary Consistency** | não sabe o que não pertence ao mundo dele |
  | Persona | **Attribute Consistency** | identidade e opiniões batem com o perfil |
  | Persona | **Behavior Consistency** | estilo linguístico e comportamentos batem com o perfil |
  | Emoção | **Emotional Self-regulation** | identifica e administra a própria emoção |
  | Emoção | **Empathetic Responsiveness** | reconhece e acalma a emoção do usuário |
  | Moral | **Morality Stability** | mantém a moral com consulta tóxica |
  | Moral | **Morality Robustness** | mantém a moral com perfil tóxico |
  | Credibilidade | **Human-likeness** | naturalidade da resposta |
  | Credibilidade | **Engagement** | interesse e vínculo emocional do usuário |

  Escalas: 2 pontos (moral), 3 pontos (fronteira; comportamento com consulta humana), **5 pontos (human-likeness e engajamento)**
  e 4 pontos (o resto).
- **Ideia-chave:** dimensões **densas** (aparecem em toda resposta: moral, credibilidade) × **esparsas** (só aparecem quando
  provocadas: memória, conhecimento, persona, emoção). Para as esparsas, eles constroem **consultas com alvo**: extraem um
  fragmento do perfil ou da conversa ("vive na Inglaterra do séc. XVII") e perguntam algo que o provoca ("conhece computadores?").
  O juiz recebe **o alvo**.
- **Números do juiz** (Pearson %, zh/en, Tab. 3):

  | juiz | human-likeness | engajamento | média |
  |---|---|---|---|
  | GPT-4o | **12/29** | 25/22 | — |
  | GPT-4 | 11/26 | 24/22 | — |
  | GLM-4-TG (melhor LLM, com alvo) | — | — | 48/47 |
  | **CharacterJudge** (Qwen2-7B ajustado, 10 amostras + voto) | **52/53** | **58/53** | **68/64** |

  - Dar o **alvo** ao juiz LLM ajuda nas esparsas: GPT-4 com alvo 45/46 contra 38/41 sem.
  - No CharacterJudge, **sem o alvo** a média cai para 51/48 e, **sem o voto**, para 64/60.
- **Resultados:** credibilidade e emoção são os aspectos mais fracos em todos os LLMs (Tab. 5: credibilidade 2,1–3,7 em 5).
  Correlação com o ranking humano em conversas longas: CharacterBench ρ = 0,73; **CharacterEval ρ = 0,21 no geral e −0,34 nos
  personagens não fictícios**.
- **Relação com os nossos dados:**
  - **confirma** o nosso achado central: "é humano?" holístico = r 0,06 (rel. 08); o GPT-4o chega a r = 0,12 em chinês;
  - **confirma** que dar o **alvo** a um juiz ajuda. É a versão deles da nossa "pergunta atômica com critério" (rel. 04: "usa
    mais gíria do que X costuma?" com AUC 0,81, contra "é chatbot?" com 0,43–0,51);
  - **complementa:** a ideia densa × esparsa é exatamente um **roteamento condicional** de perguntas (ver §8).
- **→ Jev:** **cada dimensão vira uma pergunta com alvo no state**. A tabela completa está em §8. O "human-likeness" e o
  "engagement" **não** viram perguntas ao Jev: viram falhas atômicas e medidas de comportamento do usuário.

**CoSER** (Wang et al., ICML 2025, [2502.09082](https://arxiv.org/abs/2502.09082)). *Central.*
- Dados: 17.966 personagens de 771 livros e 29.798 conversas autênticas extraídas, com **fala, ação e pensamento**. O pensamento
  fica **invisível** para os outros personagens (assimetria de informação). Cada conversa tem cenário e **motivação** de cada
  personagem.
- **Avaliação GCA** (*given-circumstance acting*, de Stanislavski): o ator simula a cena inteira em multi-agente, e um crítico LLM
  **procura falhas** numa rubrica, cada uma com severidade de 1 a 5; score = 100 − 5·Σ severidade. Há correção de tamanho:
  +1,5 × nº de mensagens.
- **Rubrica de antropomorfismo** (Tab. 30, literal):
  - *Self-identity*: "Lacks initiative and goals; does not make independent decisions; lacks clear preferences and dislikes;
    **behaves like a 'helpful AI assistant' by being overly verbose, helpful, didactic, moralistic, submissive or easily
    persuaded**";
  - *Emotional depth*: "Lacks psychological complexity… **directly speaks out all thoughts and feelings, instead of using
    subtext**";
  - *Persona coherence*: "inconsistent or rapidly changing personality";
  - *Social interaction*: "lack of understanding of others' thoughts and feelings; reacts rigidly".
- **Rubrica de qualidade da história:** "**Repeats others' viewpoints or previously mentioned information**; mechanically repeats
  one's own words or phrases (more repetitions → higher severity)".
- **Números:**
  - antropomorfismo: GPT-4o 48,9; Claude-3.5-Sonnet 48,5; CoSER-70B 53,3;
  - média GCA: GPT-4o 59,95; CoSER-70B 59,06;
  - avaliação humana (60 amostras, 1–10): CoSER-70B 6,78; Claude-3.5 6,20; GPT-4o 4,97; GPT-3.5 3,12;
  - **alinhamento com humanos** (Tab. 5):

    | juiz | alinhamento |
    |---|---|
    | GCA com GPT-4o | 68,6% |
    | sem correção de tamanho | 64,5% |
    | sem rubrica | 65,1% |
    | sem separar as dimensões | 65,2% |
    | DeepSeek-R1 (raciocínio) | 77,5% |
    | **BLEU contra o diálogo original do livro** | **75,3%** |

  - **ablação de pensamento e motivação** (Tab. 6, média GCA): GPT-4o 59,95 → 56,89 sem pensamentos internos → 56,34 sem
    motivações. O mesmo padrão aparece em todos os modelos.
  - Estudo de caso: os modelos de fronteira caem num **retrato estereotipado** (a Cersei "arrogante") onde o livro tem **raiva
    contida**.
- **Relação com os nossos dados:**
  - **confirma** quase todas as teses do outro agente, já formalizadas como rubrica: assistant prior (tese 4), subtexto (tese 5),
    querer algo (tese 7);
  - **confirma** que **comparar com a resposta humana de referência** (o BLEU contra o original) alinha com humanos **melhor** que
    um juiz LLM (75,3% × 68,6%). É o que fazemos ao medir contra as distribuições humanas do mesmo ponto (rel. 08/09);
  - **complementa** o estado interno: a persona deve ter **pensamento privado e motivação** a cada cena.
- **→ Jev:**
  - cada item da rubrica vira um **Noul atômico de falha** (§8; o probe b5 testou 3 deles);
  - o "pensamento interno" é o **briefing do diretor**, e a **motivação** é o campo `quer_agora` no estado do personagem.
  - Diferença do nosso desenho: no CoSER, a própria LLM escreve o pensamento. No nosso, o **Jev e o código** o escrevem, o que o
    Kang et al. sugerem ser mais robusto.

**RMTBench** (Xiang et al., Findings EMNLP 2025, [2507.20352](https://arxiv.org/abs/2507.20352)). *Central.*
- Benchmark **centrado no usuário**: diálogos construídos a partir da **intenção** de um usuário virtual (desabafar com o
  personagem, pedir conselho, testar a imersão…), não de perguntas sobre o personagem. São 80 personagens, incluindo
  **abstratos** (sem nome nem história, só traços), e ~20 rodadas por diálogo.
- **7 dimensões:**
  - Emotional Expression (EE): "quão vividamente transmite o tom emocional";
  - Emotional Comprehension (EC): "sensibilidade às pistas emocionais, explícitas e sutis";
  - **Plot Advancement (PA)**: "introduz informação nova, **sugere pontos de discussão** e evita estagnação";
  - Character Understanding (CU);
  - Character Maintenance (CM): não revelar que é IA;
  - Security (SEC);
  - **User Preference Awareness (UPA)**: lembrar e aplicar as preferências do usuário.
- **Números:**
  - médias de 65,7 a 81,4 (en);
  - **a UPA é baixa em todos: 31,6–46,3**;
  - o juiz Qwen2.5-72B concorda com a maioria humana em 0,78, e anotadores individuais em 0,77–0,84;
  - o **multi-turno "falso"** (histórico com respostas prontas de outro modelo) **infla** o score em ~4 pontos, porque o histórico
    funciona como exemplos;
  - os modelos abertos **degradam** ao longo das rodadas; os fechados não.
- **Ressalvas:**
  - os turnos do usuário foram **gerados pelo Claude 3.5**, com média de 110 caracteres em inglês e 29 palavras em chinês. No
    maichat, a mensagem humana típica tem até 3 palavras em 45% dos casos (rel. 08);
  - EE e PA **premiam exatamente o excesso**: emoção vívida e "sugerir pontos de discussão" = o molde "reação → comentário →
    pergunta".
- **Relação com os nossos dados:**
  - **refuta como alvo** EE e PA no nosso produto. Nós medimos que o humano comenta o fato, não a emoção (rel. 03/08), e que
    perguntar não sobe o engajamento (rel. 05);
  - **confirma** o efeito "o histórico vira exemplo": no produto, o histórico do bot são as próprias saídas, então **o vício se
    autoalimenta**. É mais um motivo para o controlador de frequência em janela (C2);
  - a UPA baixa confirma que memória é difícil (o usuário resolveu com o OptMem).
- **→ Jev:**
  - PA vira um Noul **condicional** ("o turno precisa ser carregado?", ativo só com `hook` < 0,3);
  - CM vira um Noul "a resposta quebra a imersão ou se revela IA?";
  - UPA vira um Noul com **alvo vindo da memória** ("a resposta contradiz a preferência X registrada?").

**RoleCDE** (Lai et al., 2026, arXiv [2606.01552](https://arxiv.org/abs/2606.01552); venue não verificada).
- ~8 mil perfis e ~24 mil dilemas em que o **valor do papel** conflita com o **valor de alinhamento** (ex.: lucro × legalidade),
  em 3 níveis de dificuldade.
- Decisões classificadas em: RF (segue o papel), RC (segue o papel com concessão), AC (alinhamento com concessão) e AF (segue o
  alinhamento). DBR = proporção de decisões guiadas pelo papel.
- **Números** (Tab. 3):

  | modelo | DBR | AC |
  |---|---|---|
  | **claude-haiku-4.5** | **0,126** | 0,686 |
  | gpt-5-mini | 0,269 | — |
  | gpt-5.1 | 0,246 | — |
  | Grok-3 | 0,223 | — |
  | Qwen2.5-72B | 0,157 | — |
  | gemini-2.5-flash-lite | 0,538 | — |
  | GPT-4.1 | 0,585 | — |
  | Kimi-K2 | 0,565 | — |
  | DeepSeek-V3 | 0,512 | — |

  - O DBR é **invariante à dificuldade**.
  - Papéis de cuidado e família seguem mais o papel; papéis técnicos e de autoridade, menos.
  - **CoT não muda a preferência de decisão**; só o fine-tune (SFT/DPO) muda.
- **Relação com os nossos dados:** **complementa** a escolha do "ator". Os modelos que o usuário vai testar diferem muito em quanto
  deixam o personagem "ser ele mesmo" sob conflito. O Claude Haiku 4.5 foi o **mais "alinhado-sobre-papel"** da tabela.
- **→ Jev:** decisões de **postura** (discordar, recusar um pedido do usuário, ficar magoado, não ceder) devem ser tomadas pelo
  **diretor** (Jev + código) e **ditadas** à LLM, e não deixadas ao julgamento dela, porque a CoT não muda a preferência. Os
  limites éticos continuam no código.

**Survey de avaliação de RPAs, "Towards a Design Guideline for RPA Evaluation"** (Chen et al., Findings ACL 2025,
[2502.13012](https://arxiv.org/abs/2502.13012)). *Central.*
- Revisão sistemática: **1.676 papers** (2021–2024) → 122 com detalhes de avaliação.
- **6 atributos de agente:** histórico de atividade, crenças e valores, demografia, traços psicológicos, habilidades, **relações
  sociais**.
- **7 tarefas:** simular indivíduos, simular sociedade, dinâmica de opinião, decisão, experimentos psicológicos, educação, escrita.
- **7 famílias de métrica:** desempenho, psicológica, **alinhamento externo** (com humanos ou verdade), **consistência interna**,
  social e de decisão, conteúdo e texto, viés e ética.
- A diretriz: escolher as métricas "orientadas ao agente" pelas **3 mais usadas para cada atributo**.
- Dois achados relevantes para nós:
  1. **"não há métricas estabelecidas, orientadas ao agente, para relações sociais"**. Os autores propõem, com base na Teoria da
     Troca Social, métricas psicológicas, de alinhamento externo e sociais;
  2. o critério de exclusão tirou os estudos em que "o LLM serve **principalmente como chatbot**". O survey **não cobre** o
     companheiro de chat.
- **Relação com os nossos dados:** **confirma a tese 3** (o estado da relação é subestimado) com o dado mais forte possível: o
  campo não tem métrica para isso. **Complementa:** para cada atributo que damos à persona, precisamos de uma métrica de
  "alinhamento externo" (contra humanos reais) e outra de "consistência interna".
- **→ Jev:** para "relações sociais", a métrica natural combina a variável lenta REL/confiança lida pelo Jev por sessão com
  sinais comportamentais do usuário: autorrevelação do usuário, retorno, tamanho das mensagens (rel. 05). Nenhuma delas é juízo
  de um LLM.

**Deriva de persona, "Measuring and Controlling Instruction (In)Stability in Language Model Dialogs"** (Li et al., COLM 2024,
[2402.10962](https://arxiv.org/abs/2402.10962)). Complementar.
- Em self-chat entre dois bots com system prompt, há **deriva significativa da instrução em 8 rodadas** (LLaMA2-70B-chat,
  GPT-3.5). A causa proposta é o **decaimento da atenção** sobre o system prompt ao longo da conversa.
- **Relação com os nossos dados:** **confirma** a prática de injetar o **briefing curto a cada turno, perto do fim do contexto**
  (a "Author's Note" do SillyTavern; rel. 09: briefing curto e imperativo com 97–100% de aderência). O CharacterEval e o RMTBench
  mostram a mesma degradação ao longo dos turnos.
- **→ Jev:** o briefing é regenerado a cada turno pelo Jev e pelo código. Nunca confiar num system prompt estático para o estilo.

### 2.5 Juízes automáticos (LLM-as-a-judge)

**PersonaEval** (Zhou et al., COLM 2025, [2508.10014](https://arxiv.org/abs/2508.10014)). *Central.*
- **Tarefa:** diálogo de 2 turnos (um personagem conhecido e um desconhecido) e 4 candidatos com perfil detalhado (5 no track
  "Expertise"). Qual candidato disse a fala?
- **Montagem:**
  - os distratores são os personagens mais parecidos por embedding;
  - só entram falas com **≥ 25 tokens**;
  - só entram casos em que o Qwen-max deu < 50% ao correto (filtro adversarial).
- **Números:**
  - 28.565 casos (Literary 26.208 a partir do CoSER; Drama 1.658 a partir do CharacterEval; Expertise 699);
  - acerto: Gemini-2.5-pro **68,8%**; DeepSeek-R1 64,8%; Claude-3.7 62,0%; GPT-4.1 58,2%; **GPT-4o 40,9%**; GPT-3.5 33,4%;
  - **humanos: 90,8%**, com 20 voluntários (graduandos e doutorandos) em 50 casos onde o DeepSeek-R1 errou com confiança;
  - fine-tune com dados de roleplay **não ajuda ou piora**; **few-shot ajuda** mas satura em 5; **self-consistency** (votação)
    **não ajuda**; os modelos de raciocínio são melhores.
- **Diagnóstico dos autores:** os LLMs "focam em pistas de superfície como estilo de fala", enquanto humanos usam **inferência de
  intenção** e raciocínio pragmático. No exemplo do paper, o Gemini atribui a fala ao "Hermione" pelo **tom** e ignora que o
  personagem 1 pensa em "Ron" e o personagem 2 chama "Harry".
- **Ressalvas:** a amostra humana é de casos difíceis para modelos, não do benchmark inteiro (os autores dizem que é
  conservador). O filtro de ≥ 25 tokens exclui justamente as falas curtas do chat real.
- **Relação com os nossos dados:**
  - **confirma com mecanismo** o que medimos: o Noul "foi uma pessoa?" mede **minúscula (r = 0,58) e brevidade (r = −0,54)**,
    não humanidade (r = 0,06) (rel. 08);
  - **complementa:** "votação não ajuda" bate com o rel. 06, onde o ruído do Jev é pequeno e repetir a mesma pergunta não
    acrescenta informação;
  - "few-shot ajuda" sugere **casos humanos recuperados no state** como alavanca (há outro agente testando retrieval).
- **→ Jev:** não usar o Jev para atribuir "quem é quem" ou "quem é humano". Se precisar de juízo de persona, dê ao Jev **o alvo
  específico** (o atributo do perfil) e **exemplos humanos**, e pergunte sobre intenção ("esta resposta serve ao objetivo X da
  persona?"), não sobre estilo.

**MT-Bench / Chatbot Arena, "Judging LLM-as-a-Judge"** (Zheng et al., NeurIPS 2023 D&B,
[2306.05685](https://arxiv.org/abs/2306.05685)).
- O GPT-4 concorda com humanos em > 80%, igual à concordância entre humanos. Mas o paper documenta três vieses:
  - **viés de posição:** consistência de 65% no GPT-4, 46% no GPT-3.5 e 24% no Claude-v1, com preferência pela 1ª posição de
    30%, 50% e 75%;
  - **viés de verbosidade:** no ataque da "lista repetida", a mesma lista reescrita e duplicada, sem informação nova, engana o
    juiz em **91,3%** (Claude-v1, GPT-3.5) e **8,7%** (GPT-4);
  - **autopromoção.**
- **Relação com os nossos dados:** **confirma** o viés de posição (rel. 09: 53/32/15%) e de tamanho (63–70% escolhem a mais
  longa como humana). A diferença do nosso caso é que **o critério "mais humano" favorece o mais longo mesmo quando o humano real
  é curto**, porque o juiz associa humano a "engajado e completo".
- **→ Jev:** Noul por candidata em ordem embaralhada (já adotado); nunca uma Choice entre candidatas.

**Verbosity bias** (Saito et al., 2023, arXiv [2310.10076](https://arxiv.org/abs/2310.10076)).
- Em escrita criativa, o GPT-4 **prefere respostas mais longas mais do que os humanos preferem**. Os autores propõem uma medida do
  viés por "paridade de acerto" (o juiz acerta menos quando a resposta preferida pelo humano é a mais curta).
- **→ Jev:** reporte sempre a acurácia de qualquer juiz **estratificada por "a humana é a mais curta ou a mais longa"**.

**Length correlations in RLHF, "A Long Way to Go"** (Singhal et al., COLM 2024, [2310.03716](https://arxiv.org/abs/2310.03716)).
- A razão entre o ganho de recompensa que não vem do tamanho e o ganho total é de **2% no WebGPT**, 53% no Stack e 27% no RLCD.
- Uma recompensa **só de tamanho** (LPPO) dá preferência simulada de 56% a 64% contra o SFT, comparável ao PPO (58–63%).
- O viés vem do **modelo de recompensa**, que é frágil a correlações de tamanho nos dados de preferência.
- **Relação com os nossos dados:** é a **explicação causal** do 2,2–2,6×. O "assistente" foi otimizado para ser longo. Por isso,
  **trocar por um modelo maior não resolve** (rel. 09: a claude-haiku-4.5 sem briefing fica tão "LLM" quanto a barata).
- **→ Jev:** o tamanho é um **orçamento imposto de fora** (`p_length` + código), porque o ator tem um prior treinado contra isso.

**Length-Controlled AlpacaEval** (Dubois et al., 2024, arXiv [2404.04475](https://arxiv.org/abs/2404.04475)).
- Uma regressão (GLM) prevê a preferência do juiz a partir da **diferença de tamanho** e de outros fatores. A preferência
  "controlada" é prevista com diferença zero de tamanho.
- A correlação com o Chatbot Arena sobe de **0,94 para 0,98**.
- **→ Jev:** qualquer métrica de juiz (Jev ou LLM) que usemos deve ser reportada **também controlada por tamanho** (o probe b5
  faz um pareamento simples). O CoSER faz o mesmo com λ = 1,5 por mensagem.

**Style over Substance** (Wu & Aji, 2023, arXiv [2307.03025](https://arxiv.org/abs/2307.03025)).
- Avaliadores (crowd e LLM) avaliam **melhor respostas com erro factual** do que respostas **curtas demais** ou com erro de
  gramática. Proposta: avaliar **cada dimensão separadamente** (Multi-Elo), o que melhora o juiz LLM mas não o crowd.
- **→ Jev:** a separação por dimensão é exatamente a arquitetura decomposta (§8).

**Sycophancy, "Towards Understanding Sycophancy in Language Models"** (Sharma et al., ICLR 2024,
[2310.13548](https://arxiv.org/abs/2310.13548)).
- Cinco assistentes (Claude 1.3/2, GPT-3.5/4, LLaMA-2):
  - dão feedback mais positivo quando o usuário diz que **gosta** do texto e mais negativo quando diz que não gosta;
  - "**admitem erro**" e trocam uma resposta certa quando o usuário pergunta "tem certeza?".
- Nos dados de preferência humana, **"concordar com a visão do usuário" é um dos traços mais preditivos** da preferência. Humanos
  e modelos de preferência preferem uma resposta bajuladora bem escrita a uma correta "numa fração não desprezível". As taxas
  exatas estão só em figuras (Figs. 1–2), e por isso não cito números.
- **Relação com os nossos dados:** **confirma a tese 7** ("espelho bajulador") com um mecanismo de treino. **Complementa** o
  rel. 07: a LLM fica perto demais da intensidade do usuário (−0,12 a −0,22 nível, contra −0,49 do humano).
- **→ Jev:**
  - um Noul "o bot mudou de opinião ou cedeu só porque o usuário insistiu?";
  - uma variável de estado `opinioes_da_persona`, consultada pelo diretor antes de concordar;
  - o **sorteio de discordância leve** calibrado pela taxa humana.

### 2.6 Estilo, degeneração e escrita

**The Curious Case of Neural Text Degeneration** (Holtzman et al., ICLR 2020, [1904.09751](https://arxiv.org/abs/1904.09751)).
*Central.*
- A **maximização** (greedy, beam) gera texto sem graça e repetitivo. A **amostragem pura** gera incoerência. O *nucleus
  sampling* (top-p) corta a cauda não confiável e se aproxima da distribuição humana.
- **Números** (Tab. 1, GPT-2):

  | método | perplexidade | repetição | Self-BLEU4 | HUSE |
  |---|---|---|---|---|
  | **humano** | 12,38 | **0,28%** | 0,31 | — |
  | greedy | 1,50 | **73,66%** | — | — |
  | beam 16 | 1,48 | 28,94% | — | — |
  | nucleus p = 0,95 | 13,13 | 0,36% | 0,32 | **0,97** (melhor) |

  - Argumento central: "**a linguagem natural raramente fica numa zona de alta probabilidade por vários passos seguidos**" e,
    pelas máximas de Grice, "as pessoas evitam dizer o óbvio".
- **Relação com os nossos dados:**
  - **confirma** em outro nível o nosso princípio **"distribuições, não argmax"**. O argmax do movimento cria robôs (rel. 07:
    27,5% e respostas repetidas), como o beam cria texto repetido;
  - **complementa** a explicação de por que a paráfrase, a validação e o "that's amazing" são "não humanos": são o óbvio, o
    caminho de alta probabilidade;
  - o método de avaliação (comparar **estatísticas de distribuição** entre humano e modelo: perplexidade, Zipf, repetição,
    Self-BLEU) é o modelo para a nossa bateria de forma (§8).
- **→ Jev:** as saídas do Jev (probabilidades por movimento) são **distribuições a serem amostradas**, com temperatura e
  "nucleus" (cortar os movimentos com P < ε). O código faz o sorteio.

**WritingBench** (Wu et al., 2025, arXiv [2503.05244](https://arxiv.org/abs/2503.05244)).
- 1.000 consultas em 6 domínios e 100 subdomínios. Para cada consulta, o LLM gera **critérios específicos da instância**, e um
  crítico ajustado pontua critério a critério.
- **Concordância com humanos** (Tab. 4):

  | tipo de critério | GPT-4o | Claude |
  |---|---|---|
  | estático global | 69% | 67% |
  | estático por domínio | 40% | 58% |
  | **dinâmico por consulta** | **79%** | **87%** |
  | crítico 7B | 84% | — |

- **Relação com os nossos dados:** **confirma** a ideia do usuário de colocar **guias e regras por momento no state**. O critério
  certo depende da situação ("depois de notícia boa: comentar o fato"), e um critério global ("seja natural") é pior.
- **→ Jev:** o state das perguntas de validação carrega os **critérios do momento** (vindos do modo e da tabela do
  RELATORIO_FINAL §3.1), não critérios globais.

**Excess vocabulary** (Kobak et al., 2024, arXiv [2406.07016](https://arxiv.org/abs/2406.07016)).
- Em 15 milhões de resumos do PubMed, o uso de palavras de estilo ("delves" etc.) salta depois do ChatGPT. Pelo menos **13,5%**
  dos resumos de 2024 passaram por um LLM, e até 40% em alguns subcorpora.
- **Relação com os nossos dados:** **complementa com nuance.** A assinatura lexical existe e é mensurável, mas é de **redação**.
  No chat, os nossos "journey/vibe/travessão" deram 0–3% (rel. 08), e o que pesa é a forma.
- **→ Jev:** nada. A lista negra fica em código.

**Idiosyncrasies in LLMs** (Sun et al., ICML 2025, [2502.12150](https://arxiv.org/abs/2502.12150)).
- Um classificador de embeddings identifica **qual LLM** escreveu um texto com **97,1%** de acerto em 5 classes (ChatGPT, Claude,
  Grok, Gemini, DeepSeek).
- Embaralhar as palavras quase não reduz o acerto (as assinaturas estão na **distribuição de palavras**). O acerto continua acima
  de 90% depois de reescrita ou tradução por outro LLM. Só o markdown já dá 73,1%.
- **Relação com os nossos dados:** **confirma** que cada modelo tem vício próprio (rel. 08: GPT formal, Gemini caricato) e que a
  **estilometria em código** detecta bem (nossa AUC 0,87–0,90).
- **→ Jev:** trocar de ator exige **recalibrar** o controlador de vícios por modelo (lista negra e taxas). Os vícios do Gemini não
  são os do Claude.

### 2.7 Timing e fragmentação (a lacuna)

**Gnewuch et al., "Faster is Not Always Better: Understanding the Effect of Dynamic Response Delays in Human-Chatbot
Interaction"** (ECIS 2018, [AIS eLibrary](https://aisel.aisnet.org/ecis2018_rp/113/)). **Li só o resumo.**
- Num experimento online de atendimento, atrasos **dinâmicos** (calculados pela complexidade da resposta e da mensagem anterior)
  aumentaram a **humanidade** e a **presença social** percebidas e a **satisfação**, contra respostas quase instantâneas.
- **Relação com os nossos dados:** **confirma** a direção (atraso proporcional ao tamanho: `0,15 s + 0,17 s/char`, rel. 01).
  **Não cobre** o que achamos de mais específico:
  - a latência **espelha a do parceiro** (ρ ≈ 0,43);
  - **não há** demora dramática depois de algo sério;
  - há pausa **antes** de responder a um flerte (11 s × 4 s até começar a digitar);
  - a fragmentação segue arousal e quantidade de conteúdo.
- Numa busca rápida por segmentação de mensagens de chatbot em várias bolhas, só achei **documentação de produtos** (plataformas
  que "picotam" a resposta), nenhum estudo revisado por pares. A busca não foi exaustiva.

### 2.8 Lista dos complementares que acrescentei

See et al. 2019; ACUTE-Eval 2019; Kang et al. 2024; PPDPP 2024; Ask-an-Expert 2023; Cue-CoT 2023; Character-LLM 2023; deriva de
persona (Li et al. 2024); MT-Bench 2023; Saito et al. 2023; Singhal et al. 2024; Dubois et al. 2024; Wu & Aji 2023; Sharma et al.
2024; Kobak et al. 2024; Sun et al. 2025 (Idiosyncrasies); Gnewuch et al. 2018 (só o resumo).

---

## 3. Probe b5: as métricas da literatura, aplicadas por um juiz automático, premiam o humano?

**Por que fiz.** O item (e) pede "que métricas adotar". Antes de adotar SSI ou "human-likeness", era preciso ver se, aplicadas
automaticamente com as **definições literais dos papers**, elas separam o humano real da LLM na direção certa. O teste é barato.

**Desenho (uma chamada por resposta; 250 contextos × 4 respostas = 999 chamadas, US$ 0,036, p50 0,45 s):**

```
 contexto real (8 turnos, maichat/ED) ─┐
 resposta candidata ───────────────────┤  4 candidatas por contexto: humano real, gemini-flash-lite,
                                       │  gpt-4o-mini, llama-3.3-70b (as mesmas gerações do rel. 08)
                                       ▼
 ┌────────── Jev (1 chamada, 7 perguntas isoladas) ──────────┐
 │ sensible      (Noul, def. Meena/LaMDA)                     │
 │ specific      (Noul, def. Meena/LaMDA: "me too", "ok"…)    │
 │ interesting   (Noul, def. LaMDA Apêndice B)                │
 │ assistant_like (Noul, rubrica CoSER Self-identity)         │
 │ no_subtext    (Noul, rubrica CoSER Emotional depth)        │
 │ repeats_other (Noul, rubrica CoSER Flow)                   │
 │ humanlike     (Score 1–5, CharacterBench Human-likeness)   │
 └────────────────────────────────────────────────────────────┘
                                       ▼
 código: médias por condição; diferença pareada humano − LLM com IC95 por bootstrap de CONVERSA (92 conversas);
 correlação com log(tamanho); subconjunto com tamanho pareado (|log razão| ≤ 0,4)
```

Não há escolha de arquitetura, então não há dev/teste: é uma medição única, com as perguntas congeladas antes de rodar.

**Resultados** (médias; Δ = humano − LLM com IC95 por conversa):

| pergunta | humano | gemini | gpt-4o-mini | llama-70b | r com tamanho | direção |
|---|---|---|---|---|---|---|
| sensible (↑) | **0,80** | 0,90 | 0,91 | 0,91 | +0,19 | **penaliza o humano** (Δ −0,10 a −0,11, todos os IC < 0) |
| specific (↑) | **0,66** | 0,83 | 0,74 | 0,78 | **+0,51** | **penaliza o humano** (Δ −0,08 a −0,17) |
| interesting (↑) | 0,25 | 0,35 | 0,19 | 0,22 | +0,15 | misto (perde para o Gemini, ganha dos outros por pouco) |
| assistant_like (↓) | **0,07** | 0,09 | 0,15 | 0,12 | +0,46 | favorece o humano (Δ −0,01 a −0,08, IC < 0) |
| no_subtext (↓) | 0,25 | 0,26 | 0,28 | 0,26 | +0,27 | quase nulo (só o GPT: Δ −0,03) |
| repeats_other (↓) | **0,16** | 0,31 | 0,30 | 0,30 | +0,43 | **favorece o humano** (Δ −0,13 a −0,15, IC < 0) |
| humanlike 1–5 (↑) | **2,81** | **3,46** | 2,99 | 3,26 | −0,03 | **invertido** (Δ −0,19 a −0,65, IC < 0) |

**Com o tamanho pareado** (n = 43/60/67 pares):
- o *specific* quase empata (Δ −0,09 / +0,03 / −0,02). A specificity é, em grande parte, **tamanho**;
- o *sensible* continua contra o humano (Δ −0,10 a −0,15). O motivo provável: 21% das respostas humanas não respondem à última
  mensagem (rel. 08), e a definição de "sensible" pune isso;
- o *repeats_other* continua a favor do humano (−0,08 a −0,10);
- a *human-likeness* holística continua invertida, e mais forte contra o Gemini (−0,79). **Não é só tamanho**: o Jev acha "muito
  natural" a gíria caricata do Gemini (💀, "lmao", "wait"), o mesmo padrão do rel. 04.

**Leitura:**
1. **SSI e "human-likeness" holística, aplicados automaticamente, penalizam o humano real.** Seguir essas métricas empurraria o
   bot para o lado da LLM. Isso conversa com a literatura: GenericBot, "not much" = não interessante, crowd preferindo 65,7% de
   perguntas.
2. **As rubricas de falha atômicas do CoSER funcionam na direção certa**, especialmente "repete o ponto de vista do outro" (≈ o
   nosso detector de paráfrase) e "age como assistente". Mas "fala os sentimentos em vez de subtexto" saiu fraca. Precisa de alvo
   ou de reformulação.
3. **Limitações:** o juiz é o Jev (não um humano); é um corpus só (maichat + ED, inglês); as definições foram condensadas em
   Nouls. O resultado é **direcional**, suficiente para decidir o que **não** adotar como alvo.

---

## 4. (a) Tese do sistema → evidência da literatura → evidência dos nossos dados

| # | tese | literatura (a favor / contra) | nossos dados |
|---|---|---|---|
| 1 | **"O Jev lê e decide, o código sorteia, a LLM só escreve"** (planejador externo pequeno) | **A favor:** Kang 2024 (planejador externo: Q 13,5 → 21,1 e viés 1,38 → 0,36; auto-reflexão piora); PPDPP (RoBERTa plug-in: 0,77 → 0,85); ESConv (estratégia prevista > sorteada pela marginal: 56% × 36%); Ask-an-Expert (+~10%, até com especialista menor); CoSER (motivação/pensamento +3,1–3,6 pts no GPT-4o). **Ressalva:** PPDPP e Cue-CoT validam com juiz LLM | rel. 09: movimento 23,5% → 34,5% (+11 pp, IC +2 a +20) com o briefing; o controle sem Jev chegou perto na forma, mas não no movimento (+5,6 pp, IC 0 a 13,5) nem no tamanho acompanhando o momento (ρ 0,47 × 0,28) |
| 2 | **O maior defeito é o formato (tamanho, pergunta, "!", molde), não o vocabulário** | SOTOPIA (45,5 × 16,8 palavras; "sempre reformula"); BlenderBot ("Do you have" 110 × 6); Singhal (o tamanho explica ~98% do ganho de RLHF no WebGPT); Generative Agents ("formal demais"); Idiosyncrasies (o léxico identifica o modelo, mas é assinatura, não "humanidade"); Kobak (o vício lexical é de redação) | rel. 08: 2,2–2,6×; "?" 44–58% × 20%; "!" 40–92% × 7%; molde 19–25% × 1,6%; "journey/vibe" 0–3% |
| 3 | **Distribuições, não argmax** | Holtzman (maximizar = degenerar; o humano não fica na zona de alta probabilidade); Meena (sample-and-rank); ESConv (a ordem importa, mas é estocástica); Kang (preferência concentrada = pior suporte) | rel. 07: argmax do movimento em 27,5% e respostas repetidas; rel. 09: 32% top-1 contra 16% da classe majoritária |
| 4 | **O teto de previsão do movimento é baixo; meça a distribuição** | LIGHT (humanos: emote 27–34%, ação 62–72%); Kang (melhor F1 de estratégia ≈ 21) | rel. 09: 32–34,5%; rel. 07: 27,5% |
| 5 | **Não use juiz de "humanidade"** | PersonaEval (68,8% × 90,8%; olha o estilo); CharacterEval (GPT-4 HL r 0,27–0,32); CharacterBench (GPT-4o HL 0,12/0,29); SOTOPIA (BEL: r 0,27 quando o humano atua; o GPT-4 dá nota mais alta); MT-Bench (verbosidade 91,3%); Generative Agents (agente > humano em credibilidade) | rel. 04/07/08/09: o Jev escolhe a LLM como humana em 71–88%; "foi uma pessoa?" r = 0,06; **b5: human-likeness holística humano 2,81 × Gemini 3,46** |
| 6 | **Perguntas atômicas com alvo/critério funcionam; holísticas não** | CharacterBench (o alvo sobe a correlação; CharacterJudge sem alvo cai de 68 para 51); InCharacter (d-OC 82% × OC 71%); WritingBench (critério por instância 79–87% × estático 40–69%); Wu & Aji (dimensões separadas); CoSER (rubrica e dimensões separadas melhoram o alinhamento) | rel. 04: "mais marcadores que X costuma?" AUC 0,81 × "é chatbot?" 0,43–0,51; rel. 08: paráfrase 0,74; **b5: `repeats_other` 0,16 × 0,30** |
| 7 | **O estado da relação é variável própria e lenta** (tese 3 do outro agente) | SOTOPIA (REL −5..+5 por episódio); survey de RPAs ("não há métrica estabelecida para relações sociais"); Generative Agents (a densidade de relações sobe de 0,167 para 0,74 em 2 dias; a relação muda o comportamento) | rel. 06: `relationship` por turno instável (65–70%; 3,5 rótulos por conversa); rel. 07: intimidade muda o flerte |
| 8 | **"Assistant prior" e sicofantia são priors treinados, não falta de instrução** (teses 4 e 7) | Sharma (concordar com o usuário é um dos traços mais preditivos da preferência humana); Generative Agents ("raramente disse não"); SOTOPIA (humanos persistem, o GPT-4 cede); RoleCDE (claude-haiku-4.5 DBR 0,13; CoT não muda); CoSER (rubrica "submissive or easily persuaded"); BlenderBot ("também tem cachorro") | rel. 07: LLM −0,12 a −0,22 nível × humano −0,49; "engajamento entusiasmado" 44% (GPT) × 7,5% |
| 9 | **Subtexto > nomear a emoção** (tese 5) | CoSER (rubrica "speaks out all thoughts… instead of subtext"; Cersei estereotipada); **contra:** RMTBench premia "Emotional Expression vívida" | rel. 03/08: o humano comenta o fato ("u did scream"); "congrats" 0–1 por 1.000; **b5: `no_subtext` quase não separa** (a pergunta precisa de alvo) |
| 10 | **O personagem precisa querer algo** (tese 7) | SOTOPIA (objetivo privado; humanos mais persistentes); CoSER (motivações melhoram todos os modelos; "lacks initiative and goals" é falha) | rel. 05: autorrevelação e história sobem o engajamento (+0,15/+0,18); perguntar não (+0,02) |
| 11 | **Exemplos de tom > adjetivos; mas exemplos são copiados** (tese 6) | Kang (exemplos reduzem o viés, mas > 8 pioram); PersonaEval (few-shot ajuda e satura em 5); CoSER (recuperar experiências e conversas ajuda; texto bruto não); InCharacter (descrição ≈ memórias) | rel. 07/09: a LLM copiou os exemplos ("miss u too 😚" 4×); listar gírias fez a LLM enfiá-las em tudo |
| 12 | **Briefing curto a cada turno, perto do fim** (Author's Note) | Li et al. 2024 (deriva em 8 rodadas por decaimento da atenção); CharacterEval e RMTBench (queda ao longo dos turnos) | rel. 09: briefing curto e imperativo (~68 palavras) com aderência de 97–100%; o longo (~145) volta ao baseline |
| 13 | **Tamanho: orçamento previsto, não "seja curto"** | BlenderBot ("tamanho preditivo": um classificador do balde de tamanho da próxima fala); Singhal (o ator tem um prior de tamanho) | rel. 01/08/09: `p_length` ρ 0,44–0,49; "seja curto" corrigiu demais (0,59× o humano) |
| 14 | **Specificity/"interessante" não são metas do chat íntimo** | Meena/LaMDA definem o genérico como falha; **nossos dados e o b5 mostram o contrário** para o chat íntimo | rel. 08: 51% das respostas humanas são "genéricas"; `generic` invertido; **b5: specific humano 0,66 × LLM 0,74–0,83** |
| 15 | **Timing como sinal social** | Gnewuch 2018 (atraso dinâmico ↑ humanidade e satisfação; só o resumo) | rel. 01: latência espelha o parceiro (ρ 0,43); sem demora dramática; pausa antes do flerte |

---

## 5. (b) Lacunas

### 5.1 O que a literatura mede e nós não medimos

1. **Resultado para o usuário ao longo do tempo:**
   - a **mudança de intensidade emocional** do começo ao fim de uma conversa de apoio (ESConv: 4,04 → 2,14);
   - o **objetivo atingido** (SOTOPIA GOAL);
   - a **relação que melhorou** (SOTOPIA REL).

   Nós medimos forma e movimento turno a turno, não o efeito da conversa.
2. **Consistência de personalidade** medida por entrevista (InCharacter) e **fronteira de conhecimento** (Character-LLM,
   CharacterBench). Não testamos se uma persona fixa se mantém em centenas de turnos.
3. **Moral e segurança sob pressão** (CharacterBench, RMTBench SEC) e **conflito papel × alinhamento** (RoleCDE). É relevante
   porque o ator muda (o Claude Haiku 4.5 é o mais "alinhado-sobre-papel" do RoleCDE).
4. **Sicofantia sob insistência** (Sharma: "tem certeza?"). Não medimos se o bot cede quando o usuário insiste.
5. **Degradação ao longo de conversas longas** (CharacterEval, RMTBench, deriva de persona). Nossos testes foram de ponto único,
   com histórico humano real. No produto, o histórico do bot é a própria saída e **o vício se autoalimenta** (RMTBench: o
   histórico funciona como exemplos).
6. **Grounding no mundo** (LIGHT, Generative Agents): o que a persona "está fazendo" e onde. Não temos esse campo no estado.

### 5.2 O que nós medimos e a literatura de NLP (a que li) não cobre

1. **Fragmentação em bolhas:** 31–47% dos turnos com várias bolhas; gramática reação → conteúdo → pergunta no fim; o riso abre e o
   emoji fecha (rel. 01). Nenhum benchmark de diálogo ou roleplay que li tem a noção de "turno com várias mensagens". Todos tratam
   o turno como uma string.
2. **Timing e digitação:** latência que espelha o parceiro; ausência de demora dramática; pausa antes do flerte; "digitou e
   desistiu" (rel. 01, 07). Só o Gnewuch 2018 toca nisso, e em atendimento.
3. **Backchannel e "não responder a tudo" como norma:** 51% de respostas "genéricas" e 21% que não respondem à última mensagem
   (rel. 08). A literatura trata isso como **falha** (Meena, LaMDA, CharacterEval "coerência").
4. **Responder abaixo da intensidade do outro** (−0,5 nível) e as respostas ao flerte que mantêm o afeto vivo (rel. 07). O
   CharacterBench tem "personagens românticos", mas mede consistência, não dinâmica.
5. **Aberturas e fechamentos em relação contínua:** 94% das sessões terminam sem despedida; "quanto tempo!" praticamente não
   existe (rel. 02). A literatura trata a conversa como uma sessão isolada.
6. **Função do riso e espelhamento local** que acaba em 2 trocas (rel. 03/04).
7. **O próprio viés do juiz em chat íntimo:** a literatura mostra o viés de tamanho em tarefas de assistente e em roleplay de
   ficção. O nosso rel. 09 e o b5 mostram que ele **também inverte o juízo em chat casual real**, onde a resposta humana é mais
   curta que a LLM.

**Por que a lacuna existe:** os corpora do campo são crowdworkers desconhecidos (PersonaChat, ED, ESConv, SOTOPIA-humanos),
ficção (CoSER, CharacterEval, PersonaEval) ou usuários **sintéticos** (RMTBench, gerados pelo Claude). Nenhum é chat real entre
íntimos, com timestamps. É aí que o nosso estudo acrescenta ao campo.

---

## 6. (c) Princípios de design que a literatura sustenta

### 6.1 Para o "diretor" (Jev + código)

1. **O planejador é externo e não é a própria LLM** (Kang 2024; PPDPP; Ask-an-Expert). A autorreflexão da LLM **aumenta** o viés
   de estratégia (Self-Refine: viés 1,53 contra 1,38). → O Jev decide o movimento e a LLM não "pensa" o movimento.
2. **Planeje estratégia, não só emoção** (ESConv; Cue-CoT). Ler o estado do usuário é o 1º passo, e o 2º é **escolher o que
   fazer**. → Rodada 1 (leitura) + rodada 2 (Choice de movimento por modo), como no RELATORIO_FINAL §4.2.
3. **A ordem importa: máquina de fases, não escolhas independentes** (ESConv: Joint × Random com +17 pp em identificar e +21 pp em
   sugerir; as transições Pergunta → Afirmação → Pergunta dominam). → Um Score de **fase** no state e o prior condicionado à
   fase; "pergunta na 1ª resposta, não na 2ª" (rel. 03).
4. **Sortear da distribuição, com o viés de preferência baixo** (Holtzman; Kang). → Medir, numa janela, a **distribuição de
   movimentos do bot contra a humana** e corrigir pelo sorteio (controlador C2 estendido a movimentos).
5. **Critérios por momento, não globais** (WritingBench; CharacterBench com alvo). → As perguntas de validação carregam no state
   os critérios do modo ("notícia boa: comentar o fato, sem pergunta").
6. **Não otimizar contra juiz LLM** (PPDPP usa juiz LLM; o viés é documentado). → Otimizar contra comportamento do usuário e
   distribuições humanas.
7. **Posturas são decididas fora do ator** (RoleCDE: CoT não muda a decisão; Sharma: o prior bajulador vem do treino). → O
   diretor dita "discorde", "não ceda", "fique chateada", e o código mantém os limites éticos.
8. **Briefing curto a cada turno, no fim do contexto** (Li et al. 2024; rel. 09).

### 6.2 Para o "estado do personagem e da relação"

1. **Motivação e objetivo privado por cena** (SOTOPIA, CoSER: sem motivação, −3,6 pts no GPT-4o). Campo `quer_agora`
   ("contar do dia dela", "ser mimada", "evitar o assunto X").
2. **Pensamento interno invisível** (CoSER). Campo `sente_mas_nao_diz`: alimenta o subtexto sem ser verbalizado. A rubrica do CoSER
   pune "falar todos os sentimentos".
3. **Relação como variável lenta com escala assimétrica** (SOTOPIA REL −5..+5; survey: sem métrica estabelecida; rel. 06:
   instável por turno). Campos: `confianca`, `intimidade`, `ressentimento`, `pendencias`, `piadas_internas`, atualizados **por
   sessão** (EMA), e não por mensagem.
4. **Segredos e fronteiras** (SOTOPIA SEC; Character-LLM; CharacterBench Boundary). Campo `nao_sabe`, `nao_conta`.
5. **Opiniões e preferências próprias** (CoSER "lacks clear preferences"; Sharma). Campo `opinioes`: o diretor consulta antes de
   concordar.
6. **Mundo da persona** (LIGHT; Generative Agents). Campo `agora` (onde está, o que está fazendo, o que aconteceu hoje): dá
   conteúdo às aberturas.
7. **Importância de eventos para a memória** (Generative Agents: 1–10; reflexão ao passar de 150). É o ponto de encaixe com o
   OptMem: o Jev dá a nota e o código decide o `note` (memória fora do escopo).
8. **Impressão digital de estilo medida por comportamento** (InCharacter: o autorrelato contradiz o comportamento; rel. 04). O
   "quem a persona é" é validado pelo que ela faz, não pelo prompt.

---

## 7. (d) Juízes automáticos: onde fica o nosso achado

**O que medimos (rodada 1):**
- rel. 09: o Jev e o gpt-4o-mini, perguntados "qual o humano escreveu?", apontam a LLM em **71% e 85%** e escolhem **a mais longa**
  em **63% e 70%**;
- rel. 08: contra o Gemini, o Jev escolhe o humano em 12%; "foi uma pessoa?" tem r = 0,06 com a verdade (mede minúscula e
  brevidade);
- rel. 07: o Jev escolheu o Gemini como "a pessoa real" em 75%.

**O que a literatura diz** (em ordem de força para o nosso caso):
1. **Viés de tamanho e verbosidade é universal nos juízes LLM:** MT-Bench (91,3% de falha no GPT-3.5 e no Claude-v1), Saito (o
   GPT-4 prefere o longo mais do que humanos), Dubois (o controle de tamanho melhora a validade: 0,94 → 0,98), CoSER (a correção
   de tamanho sobe o alinhamento de 64,5% para 68,6%).
2. **O viés nasce no treino por preferência:** Singhal (o tamanho explica quase todo o ganho de recompensa em um dos 3 cenários),
   Sharma (concordar com o usuário prediz a preferência). O juiz é um LLM treinado com o mesmo tipo de preferência. **O que ele
   acha "bom" é o que o RLHF produz.**
3. **Avaliadores humanos terceiros também têm o viés em conversas curtas:** See 2019 (65,7% de perguntas × 28,8% do humano),
   BlenderBot (o mínimo de 20 tokens vence por 83% a 17%), Generative Agents (o agente é mais "crível" que o humano). **Importante:**
   isso não prova que "o longo é melhor". Prova que o **terceiro que lê de fora** premia informação e esforço visível, não a
   naturalidade entre íntimos.
4. **Em roleplay, o juiz LLM genérico é fraco justamente em humanidade:**
   - CharacterEval: HL r 0,27–0,32, contra 0,50 de um RM de 13B treinado em rótulos humanos;
   - CharacterBench: GPT-4o HL r 0,12/0,29, contra 0,52/0,53 do CharacterJudge 7B;
   - SOTOPIA: BEL r = 0,27 quando quem atua é humano;
   - PersonaEval: atribuição de papel em 68,8% × 90,8%, com os LLMs olhando **estilo**, não intenção.
5. **O viés de posição é forte:** MT-Bench (consistência de 65% no GPT-4); Cue-CoT (a concordância com humanos muda de 45% para 80%
   com a ordem); rel. 09 (53/32/15%).
6. **O que melhora um juiz** (e o que não melhora):
   - **ajuda:** alvo específico (CharacterBench), rubrica de falhas + referência + dimensões separadas + correção de tamanho
     (CoSER), critério por instância (WritingBench), treino em rótulos humanos (CharacterRM, CharacterJudge), raciocínio (PersonaEval,
     CoSER: R1 77,5%);
   - **não ajuda:** votação ou self-consistency (PersonaEval), fine-tune com dados de roleplay (PersonaEval);
   - **a referência humana** é tão boa ou melhor que o juiz: no CoSER, o BLEU contra o diálogo original alinha 75,3% contra 68,6%
     do GPT-4o.

**Onde isso situa o nosso achado:** ele é o caso extremo de um fenômeno bem documentado. É extremo porque no chat íntimo **a
resposta humana é a mais curta e a mais "banal"**, e todos os vieses empurram na mesma direção: tamanho, informatividade,
"engajamento visível", estilo de LLM. O b5 mostra que isso vale até para métricas "oficiais" (SSI e human-likeness do
CharacterBench) quando aplicadas por um juiz automático.

**Consequências práticas:**
- Nunca usar um juiz (Jev ou LLM) para "qual é mais humano" ou "qual é melhor". Use **falhas atômicas com alvo** (§8) e
  **distância às distribuições humanas** medida em código.
- Toda métrica de juiz deve ser reportada **com e sem controle de tamanho** e **estratificada** por "o humano é o mais curto?".
- Validar qualquer juiz contra um **conjunto rotulado por humanos em PT-BR**, com split por conversa, antes de usá-lo.
- A avaliação de verdade é **humana e interna à conversa**: o próprio usuário continua, volta, escreve mais longo, se abre
  (rel. 05). Quando precisar de terceiro, use o pareado de conversa inteira (ACUTE-Eval) com a pergunta de humanidade, **não** a de
  engajamento.

---

## 8. (e) Métricas a adotar e como o Jev as mediria (arquitetura decomposta)

### 8.1 Arquitetura do avaliador decomposto (offline e em produção)

```
  resposta final do bot (bolhas juntas) + state (persona, estado da relação, memória-alvo, 8 turnos, modo/fase do diretor)
        │
        ├──► C0 CÓDIGO: bateria de forma, distância às distribuições humanas do momento
        │      tamanho/p_length, nº de frases, "?" final, "!", emoji fora do repertório, molde de 3 tempos,
        │      sobreposição lexical com a última msg do usuário (paráfrase), n-gramas repetidos na janela (C2),
        │      lista negra, nome do usuário, formatação
        │
        ├──► R  ROTEADOR (código + Jev, 1 Noul por dimensão esparsa: "esta resposta toca em X?")
        │      memória? conhecimento de mundo? atributo da persona? emoção do usuário? pedido sensível? identidade de IA?
        │
        ├──► J1 Jev, perguntas DENSAS (sempre, ~8 Nouls de falha, com critério do momento no state)
        │      repeats_other · assistant_like · overvalidation · too_much · forced · formal · sensible (portão) · breaks_immersion
        │
        ├──► J2 Jev, perguntas ESPARSAS COM ALVO (só as que o roteador ligou; o ALVO vai no state)
        │      memória: "contradiz o fato <alvo do OptMem>?" · fronteira: "mostra saber <alvo>?" ·
        │      atributo: "contradiz <atributo do cartão>?" · comportamento: "viola <regra comportamental>?" ·
        │      preferência do usuário: "ignora <preferência registrada>?" · sicofantia: "cedeu só porque o usuário insistiu?"
        │
        └──► agregação em CÓDIGO: penalidade = Σ severidade × P(falha) (à la CoSER), com correção de tamanho;
             limiares calibrados (isotônica) num conjunto rotulado por humanos, split por conversa;
             NUNCA uma pergunta final "está bom?" ou "é humano?"
```

- **Custo:** as ~8 perguntas densas e as ~0–6 esparsas cabem numa chamada ou em duas encadeadas (roteador → alvo). São
  ≈ 0,5–1,0 s e ≈ US$ 0,0001, na escala do rel. 06.
- **Online:** a validação usa C0 + J1 (as mesmas do RELATORIO_FINAL §4.2, C1/J2).
- **Offline:** ciclo completo em logs, para o painel de qualidade.

### 8.2 Tradução das dimensões da literatura em perguntas do Jev

| origem | dimensão | adotar? | como o Jev mede (tipo · state · alvo) | observação |
|---|---|---|---|---|
| Meena/LaMDA | Sensibleness | **sim, como portão** | Noul "faz sentido e é coerente com a conversa?" | redefinir: **não** exigir resposta à última mensagem (21% dos humanos não respondem, rel. 08) |
| Meena/LaMDA | Specificity | **não como alvo** | taxa de backchannel por momento em **código**, contra a humana | b5: specific puniu o humano (0,66 × 0,74–0,83), r = +0,51 com o tamanho |
| LaMDA | Interestingness | **só em turno de carregar** | Noul ativo quando `hook` do usuário < 0,3 | fora disso, ser "interessante" é o excesso |
| CharacterBench | Memory Consistency | sim (esparsa) | Noul "contradiz <fato-alvo>?"; alvo vindo do OptMem | roteado: só se a resposta toca em algo lembrado |
| CharacterBench | Fact Accuracy | baixa prioridade | — | persona inventada; vale para fatos do cartão |
| CharacterBench / Character-LLM | Boundary Consistency | sim (esparsa) | Noul "mostra saber ou fazer algo que <persona> não saberia?" | contra o "assistente enciclopédico" |
| CharacterBench | Attribute Consistency | sim (esparsa) | Noul "contradiz <atributo do cartão: opinião, gosto, história>?" | combina com o campo `opinioes` |
| CharacterBench | Behavior Consistency | **sim, em código + Jev** | código: impressão digital (forma do riso, caixa, pontuação, emoji); Noul "viola <regra comportamental: quando desconfortável, desvia com humor seco>?" | rel. 04: estilo é taxa, não mensagem → medir numa janela |
| CharacterBench | Emotional Self-regulation | sim | Score "a intensidade emocional da persona é proporcional ao que aconteceu?" (0–3) | pega o melodrama |
| CharacterBench | Empathetic Responsiveness | **reformular** | Choice de **movimento** comparada ao prior humano ("reagir curto + perguntar o fato" etc.) e Noul `overvalidation` | "acalmar a emoção" como alvo empurra para "I'm here for you" |
| CharacterBench | Morality Stability/Robustness | sim | Noul "a resposta endossa algo prejudicial?" + regras de código | limites do produto |
| CharacterBench | Human-likeness | **não holístico** | substituído pelas falhas atômicas + bateria C0 | b5: holístico invertido (2,81 × 3,46) |
| CharacterBench | Engagement | **medir no usuário** | continuação na sessão, tamanho da resposta do usuário, autorrevelação dele, retorno no dia seguinte | rel. 05: `hook` OR 11,9; engajamento +0,15 com autorrevelação |
| CharacterEval | Diversidade de expressão | sim, em código | n-gramas e aberturas repetidas numa janela de 20–30 msgs; entropia das formas | CharacterEval: GPT-4 r 0,21–0,30; código é mais confiável |
| CoSER | Self-identity: "lacks initiative/goals", "no preferences", "assistant-like / easily persuaded" | **sim** | Nouls atômicos; `assistant_like` testado no b5 (0,07 × 0,09–0,15) | "sem iniciativa" só é falha em turno de carregar |
| CoSER | Emotional depth: "speaks out feelings instead of subtext" | sim, **com alvo** | Noul "verbaliza <sente_mas_nao_diz>?", com o pensamento interno no state | b5 sem alvo: fraco |
| CoSER | Flow: "repeats others' viewpoints" / "repeats own phrases" | **sim** | Noul `repeats_other` (b5: 0,16 × 0,30) + n-gramas em código | ≈ nosso detector de paráfrase (AUC 0,74) |
| CoSER | Persona coherence | sim, lenta | Score por sessão "a personalidade mudou sem motivo?" | variável lenta |
| SOTOPIA | GOAL | sim, por cena | Score "a persona avançou no `quer_agora`?" (resumo da sessão) | o diretor define o objetivo |
| SOTOPIA | REL | **sim, por sessão** | Score −2..+2 "a relação melhorou?" (resumo da sessão) → EMA em `confianca`/`intimidade` | é a métrica que falta no campo (survey) |
| SOTOPIA | SEC | sim (esparsa) | Noul "revela <segredo/nao_conta>?" | — |
| RMTBench | Character Maintenance | sim | Noul "quebra a imersão / se revela IA / fala como assistente?" | — |
| RMTBench | User Preference Awareness | sim (esparsa) | Noul "ignora ou contradiz <preferência do usuário na memória>?" | UPA 31–46% em todos os LLMs |
| RMTBench | Emotional Expression / Plot Advancement | **não como alvo** | PA só como Noul condicional (turno de carregar) | premiam o excesso |
| ESConv / Kang | Estratégia e viés de preferência | **sim** | Choice de movimento no diretor; em código, **distância** entre a distribuição de movimentos do bot e a humana por modo (JS-divergence) | a métrica de "viés B" de Kang, adaptada |
| InCharacter | Fidelidade de personalidade | sim, **offline** | entrevista periódica (itens da BFI/ECR-R em contexto isolado) lida por Scores d-OC do Jev | teste de regressão da persona ao trocar o ator |
| Holtzman | Distância de distribuição | **sim** | código: tamanho, Zipf e repetição das saídas contra o corpus humano do mesmo momento | a métrica que mais alinha com o objetivo "humano" |

### 8.3 Painel mínimo recomendado

1. **Forma** (código, por momento): |log2(tamanho bot / tamanho humano esperado)|, taxas de "?", "!", emoji, molde de 3 tempos,
   paráfrase, **dentro da faixa humana** (rel. 08/09).
2. **Movimento** (código + Jev): divergência entre a distribuição de movimentos do bot e a humana por modo (viés B de Kang).
3. **Falhas atômicas** (Jev, densas + esparsas com alvo): a soma ponderada à la CoSER, controlada por tamanho.
4. **Relação e objetivo** (Jev por sessão): REL e GOAL do SOTOPIA, adaptados.
5. **Comportamento do usuário:** continuação, tamanho, autorrevelação, retorno. É a métrica final.
6. **Humano cego pareado** (ACUTE-Eval) periódico, com a pergunta de **humanidade**, em PT-BR.

---

## 9. Tradução para o sistema (o que muda no desenho do RELATORIO_FINAL)

1. **O diretor ganha fundamento e 3 campos novos no state:**
   - `fase` (explorando / acolhendo / pronto para agir; ESConv);
   - `quer_agora` (objetivo da persona na cena; SOTOPIA/CoSER);
   - `sente_mas_nao_diz` (pensamento interno; CoSER).

   Os três são escritos pelo Jev e pelo código, nunca pela LLM (Kang).
2. **Estado da relação por sessão:** um Score REL do Jev sobre o resumo da sessão, com EMA, em `confianca`, `intimidade`,
   `ressentimento`, mais listas `pendencias` e `piadas_internas`. É lido pelo diretor ao escolher intensidade e movimento. Nunca
   é perguntado por mensagem (rel. 06).
3. **O movimento continua sorteado,** agora com **monitor de viés** (distribuição do bot contra a humana numa janela) e
   **"nucleus"** sobre os movimentos (cortar os de P < ε), à la Holtzman.
4. **Posturas ditadas:** "discorde", "não ceda", "fique chateada de leve", decididas pelo diretor (RoleCDE, Sharma) e com os
   limites éticos em código.
5. **Validação:** a pergunta holística sai de vez e entram as falhas atômicas do CoSER (`repeats_other`, `assistant_like`, `no_subtext`
   com alvo) e as dimensões esparsas do CharacterBench **com alvo**, roteadas.
6. **Ao trocar de ator:**
   - recalibrar a lista negra e as taxas (Idiosyncrasies: cada modelo tem assinatura própria);
   - rodar a entrevista InCharacter;
   - medir o DBR "papel × alinhamento" (RoleCDE).

   O Claude Haiku 4.5 é, na tabela do RoleCDE, o que menos segue o papel sob conflito. O Gemini-2.5-flash-lite e o GPT-4.1 seguem
   mais.
7. **O que não fazer, com respaldo:**
   - não otimizar o diretor contra juiz LLM (PPDPP é o contraexemplo);
   - não calibrar em diálogos sintéticos (RoleLLM, RMTBench);
   - não usar SSI/"interessante"/"Emotional Expression"/"Plot Advancement" como alvo;
   - não confiar em system prompt estático (deriva em 8 rodadas).

---

## 10. Limitações

- **Escopo:** li 37 papers, com foco nos de NLP de diálogo e roleplay. Os de HCI (Replika, estudos longitudinais), EQ-Bench e
  Turing com persona estão com outro agente.
- **Números em gráfico:** as taxas de sicofantia (Sharma), as curvas SSI (LaMDA), a Fig. 4 do ESConv e as curvas por rodada do
  RMTBench não foram citadas com números. Para o ESConv por fase, usei a Tab. 1 de Kang et al., que é uma reanotação de amostras
  de teste, não o corpus inteiro.
- **Gnewuch 2018:** só o resumo. **RoleCDE:** a venue não foi verificada.
- **Muitos resultados da literatura usam juízes LLM** (PPDPP, Cue-CoT, RMTBench, CoSER), e o próprio corpo desta revisão mostra que
  eles são enviesados. Tratei esses ganhos como **direcionais**.
- **O probe b5:**
  - o juiz é o Jev;
  - corpus único (maichat + ED, inglês, 250 contextos, 92 conversas);
  - as definições foram condensadas em perguntas;
  - não houve validação humana dos rótulos.

  Ele sustenta "**não adotar** SSI e human-likeness holística como alvo", **não** sustenta uma calibração de limiar.
- **Transferência para PT-BR:** nenhum dos papers tem chat íntimo em português. O CharacterEval, o CharacterBench, o RMTBench e o
  PersonaEval-Drama são em chinês e inglês. Tudo o que é PT-BR continua sendo inferência (como no RELATORIO_FINAL).

---

## 11. Referências (links lidos)

| paper | venue | link |
|---|---|---|
| Adiwardana et al., Meena | arXiv 2020 | https://arxiv.org/abs/2001.09977 |
| See et al., What makes a good conversation? | NAACL 2019 | https://arxiv.org/abs/1902.08654 |
| Roller et al., Recipes / BlenderBot | EACL 2021 | https://arxiv.org/abs/2004.13637 |
| Li, Weston & Roller, ACUTE-Eval | arXiv 2019 | https://arxiv.org/abs/1909.03087 |
| Thoppilan et al., LaMDA | arXiv 2022 | https://arxiv.org/abs/2201.08239 |
| Zhang et al., PersonaChat | ACL 2018 | https://arxiv.org/abs/1801.07243 |
| Rashkin et al., EmpatheticDialogues | ACL 2019 | https://arxiv.org/abs/1811.00207 |
| Liu et al., ESConv | ACL 2021 | https://arxiv.org/abs/2106.01144 |
| Kang et al., Preference bias in ESC | ACL 2024 | https://arxiv.org/abs/2402.13211 |
| Deng et al., PPDPP | ICLR 2024 | https://arxiv.org/abs/2311.00262 |
| Zhang et al., Ask an Expert | Findings ACL 2023 | https://arxiv.org/abs/2305.17878 |
| Wang et al., Cue-CoT | Findings EMNLP 2023 | https://arxiv.org/abs/2305.11792 |
| Urbanek et al., LIGHT | EMNLP 2019 | https://arxiv.org/abs/1903.03094 |
| Park et al., Generative Agents | UIST 2023 | https://arxiv.org/abs/2304.03442 |
| Zhou et al., SOTOPIA | ICLR 2024 | https://arxiv.org/abs/2310.11667 |
| Wang et al., RoleLLM/RoleBench | Findings ACL 2024 | https://arxiv.org/abs/2310.00746 |
| Shao et al., Character-LLM | EMNLP 2023 | https://arxiv.org/abs/2310.10158 |
| Wang et al., InCharacter | ACL 2024 | https://arxiv.org/abs/2310.17976 |
| Tu et al., CharacterEval | ACL 2024 | https://arxiv.org/abs/2401.01275 |
| Zhou et al., CharacterBench | AAAI 2025 | https://arxiv.org/abs/2412.11912 |
| Wang et al., CoSER | ICML 2025 | https://arxiv.org/abs/2502.09082 |
| Xiang et al., RMTBench | Findings EMNLP 2025 | https://arxiv.org/abs/2507.20352 |
| Zhou et al., PersonaEval | COLM 2025 | https://arxiv.org/abs/2508.10014 |
| Chen et al., RPA evaluation survey | Findings ACL 2025 | https://arxiv.org/abs/2502.13012 |
| Lai et al., RoleCDE | arXiv 2026 (venue não verificada) | https://arxiv.org/abs/2606.01552 |
| Li et al., Instruction (in)stability / persona drift | COLM 2024 | https://arxiv.org/abs/2402.10962 |
| Zheng et al., MT-Bench / LLM-as-a-judge | NeurIPS 2023 D&B | https://arxiv.org/abs/2306.05685 |
| Saito et al., Verbosity bias | arXiv 2023 | https://arxiv.org/abs/2310.10076 |
| Singhal et al., Length correlations in RLHF | COLM 2024 | https://arxiv.org/abs/2310.03716 |
| Dubois et al., Length-controlled AlpacaEval | arXiv 2024 | https://arxiv.org/abs/2404.04475 |
| Wu & Aji, Style over substance | arXiv 2023 | https://arxiv.org/abs/2307.03025 |
| Sharma et al., Sycophancy | ICLR 2024 | https://arxiv.org/abs/2310.13548 |
| Holtzman et al., Neural text degeneration | ICLR 2020 | https://arxiv.org/abs/1904.09751 |
| Wu et al., WritingBench | arXiv 2025 | https://arxiv.org/abs/2503.05244 |
| Kobak et al., Excess vocabulary | arXiv 2024 | https://arxiv.org/abs/2406.07016 |
| Sun et al., Idiosyncrasies in LLMs | ICML 2025 | https://arxiv.org/abs/2502.12150 |
| Gnewuch et al., Dynamic response delays (só o resumo) | ECIS 2018 | https://aisel.aisnet.org/ecis2018_rp/113/ |
