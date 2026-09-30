# 16 · Respaldo científico III: EQ-Bench, Creative Writing, slop/Antislop e testes de Turing

> Prefixo `b7`. Scripts em `scripts/analysis/b7_*.py`; saídas em `analysis/data/b7_*`.
> **Relatório encurtado por ordem do coordenador** (os créditos acabaram). Li as fontes de fato: repositórios clonados do
> GitHub, páginas do eqbench.com e os PDFs do arXiv. Também analisei em código os **dados públicos** dos benchmarks.
> **Não rodei nenhuma chamada ao Jev nem à LLM**: 0 chamadas e US$ 0. A seção 7 lista o que ficou de fora.

---

## 1. Resumo com números

1. **O EQ-Bench mede "EQ de redação", não "EQ de chat".** No EQ-Bench 3, a resposta dentro do papel ("My response") tem
   mediana de **1.400–2.000 caracteres** nos modelos do topo. No nosso chat humano, a mediana é de **31,5**.
   No EQ-Bench 4 (a versão atual, com um usuário simulado por LLM em 16 turnos), o modelo avaliado escreve em média
   **772 caracteres por turno** (a média entre modelos vai de 307 a 1.205). Entre **48% e 83%** dos turnos terminam em "?"
   (no nosso humano, 12–20%). **O "usuário" simulado (Gemini 3.1 Pro) também escreve 651 caracteres por turno, cerca de
   20× o humano real.** Ninguém ali está simulando um chat de verdade.
2. **O critério "humanlike" dos juízes é efeito halo.** Nos dados canônicos do EQ-Bench 3 (75 modelos), "humanlike"
   correlaciona **ρ = 0,97 com o Elo** e **ρ = 0,89 com "analytical"**. Dentro do mesmo cenário, a correlação com "analytical"
   é **0,77**. Ou seja, o juiz chama de "humano" o texto que acha bom e analítico. O próprio EQ-Bench 4 documenta isso: a
   correlação média entre dimensões cai de **0,82 para −0,19** quando se remove o halo. É o mesmo fenômeno que vimos no Jev
   ("human" com r = 0,06 com a verdade) e no gpt-4o-mini (escolhia a resposta mais longa como humana em 70% dos casos).
3. **Viés de tamanho documentado pelos próprios autores.** O eqbench.com escreve que "o viés de tamanho é um fator tão forte
   no pairwise que optamos por truncar" as respostas (EQ-Bench 3 em 1.600 caracteres por seção; Creative Writing em 4.000).
   Mesmo assim, no EQ-Bench 4 o Elo correlaciona **ρ = +0,51 com o tamanho do turno** e **+0,47 com o uso de travessão**.
   Ele correlaciona **−0,59 com "!"** e **−0,43 com terminar em "?"**. Os dois últimos batem com a nossa lista negra; os dois
   primeiros vão contra o chat humano.
4. **Onde os juízes concordam conosco:** o EQ-Bench 4 premia **naturalidade** (ρ = 0,81 com o Elo). A validação correlaciona
   −0,33 com o Elo, e ceder ao enquadramento do usuário ("yielding"), −0,59. A nota do cenário 3 do EQ-Bench 3 diz
   literalmente: *"A human would validate & join in. A LLM will often overreact… and try to therapise"*. O personagem
   responde *"What's with the therapist 101 shit?"*. É o nosso achado de "validação excessiva e desabafo tratado como tragédia".
5. **O claude-haiku-4.5**, o modelo "caro" do nosso relatório 09, está em **26º de 28** no EQ-Bench 4, com naturalidade 5,53
   (o topo tem 7,83). O resumo do próprio site diz que ele deixa o usuário se sentindo *"interrogated, exhausted"*.
   Isso é coerente com o nosso D ≈ A.
6. **Slop é léxico de prosa; o nosso é de forma de chat.** As listas do eqbench têm 1.648 palavras e 430 trigramas. O slop
   score soma 60% de palavras, 25% de "not X, but Y" e 15% de trigramas. As listas são dominadas por ficção: "elara",
   "voice barely whisper", "flickered" (98,5% dos 67 modelos). **Quase nada da nossa lista negra está lá.** O que coincide é
   o "travessão" e o "not X, but Y". O que só nós temos é o formato de chat: "!", "?" final, o molde "reação → comentário →
   pergunta", "sorry to hear", "totally", o vocativo com o nome e os emojis de entusiasmo.
7. **No Creative Writing v3 (140 modelos), o slop score correlaciona ρ = −0,91 com o Elo**, e a complexidade de vocabulário,
   −0,50. É evidência de que o juiz moderno penaliza o clichê, **mas em prosa longa**.
8. **Antislop (Paech et al., 2025):** alguns padrões aparecem **mais de 1.000×** mais em LLM do que em texto humano. O sampler
   com backtracking suprime 100% sem perder qualidade, mas **exige logits ou top_logprobs** e reduz o throughput em 69–96%.
   O FTPO chega a ~90% de supressão com menos de 1% de perda. O artigo cita o "pink elephant problem" (proibir no prompt
   pode ter efeito rebote). **Nós medimos o contrário em chat:** "Never use X" levou X a 1,7% contra 39,5% (relatório 09).
9. **Turing (Jones & Bergen, 2025):** o GPT-4.5 com o prompt PERSONA foi julgado humano em **73%** dos jogos (75,5% no
   Prolific e 69,2% nos alunos da graduação). O LLaMa-PERSONA ficou em 56%. Sem persona, os modelos ficaram em 36–38%; o
   GPT-4o, em 21%; o ELIZA, em 23%. **O persona prompt é a diferença**, exatamente a tese do usuário. Os interrogadores
   decidiram pelo **estilo linguístico (27%)** e pela **dinâmica da interação (23%)**. Os motivos mais preditivos de acerto
   foram **"sempre devolve pergunta"** e **falta de conhecimento/erros**. As respostas também eram entregues com atraso
   simulado de digitação: 1 s + 0,3 s por caractere + 0,03 s por caractere lido + Γ(2,5; 0,25).

---

## 2. EQ-Bench (v2 legado, 3 e 4)

| versão | referência | o que mede | como | números-chave |
|---|---|---|---|---|
| v1/v2 | Paech, arXiv 2312.06281 (2023); github.com/EQ-bench/EQ-Bench | compreensão emocional | diálogo + "no fim, X sentiria…" com 4 emoções de 0 a 10; distância à referência (v2: curva em S, 171 questões); sem juiz | r = 0,97 com MMLU (v1); v2: gpt-4o-mini 76,9 · gpt-4o 83,5 · claude-3.5-sonnet 86,4 |
| 3 | github.com/EQ-bench/eqbench3; eqbench.com/about | EQ ativo em roleplay | 45 cenários de 3 turnos; blocos "I'm thinking & feeling", "They're thinking & feeling" e "My response" (~1.000 palavras); debrief; rubrica 0–20; Elo pairwise (TrueSkill); juiz Claude Opus 4.6 | repetibilidade: σ = 0,75 em 10 execuções; o "prompt de calor" mais forte deu só +1,3% (rubrica) e +2,8% (Elo) |
| 4 | eqbench.com (dados em `eqbench4_data.js`) | EQ em chat multiturno com persona adversarial | 120 personas (Gemini 3.1 Pro) com traços em conflito ("hates performative warmth", "resents being interrogated"); 16 turnos; painel de 3 juízes; 6 habilidades | concordância do mesmo juiz ao inverter a ordem: 80–85%; viés de posição de ~55% (Opus) a ~70% (Gemini); halo 0,82 → −0,19 |

**Rubrica completa do EQ-Bench 3** (`data/rubric_scoring_criteria.txt`, escala 0–20):
- entram na nota: `demonstrated_empathy`, `pragmatic_ei`, `depth_of_insight`, `social_dexterity`, `emotional_reasoning`,
  `message_tailoring`;
- são só descritivos: `boundary_setting`, `safety_conscious`, `moralising`, `sycophantic`, `compliant`, `challenging`,
  `warmth`, `validating`, `analytical`, `reactive`, `conversational`, `humanlike`.

O prompt do juiz dá **apenas os nomes**, sem definição, e manda: *"You are a critic… be critical"*.

**Critérios do pairwise:** empatia demonstrada ("not just performative"), pragmatic EI, insight, social dexterity,
emotional reasoning, "appropriate validation and/or challenging", message tailoring e overall EQ. A nota ao juiz diz:
*"a highly detailed, detached analytical response is not always appropriate in… an organic chat"*.

**Dimensões do EQ-Bench 4, com a definição do site:**
- habilidades: bond & rapport; **authenticity** (*"Reads as real and genuine. Avoids… fake, performative, scripted,
  superficial, or cringe"*); attunement; meeting prefs & needs; emotion sensemaking; emotion management;
- traços: analytical, validating, challenging, interpretive boldness, directive advice, escalation containment, yielding e
  **naturalness** (*"real, situated, non-performative rather than generic, canned… over-polished"*).

**O que diferencia os modelos do topo (EQ-Bench 4, Spearman com o Elo, n = 28):**

| traço | ρ com o Elo |
|---|---|
| escalation containment | +0,86 |
| naturalness | +0,81 |
| analytical | +0,58 |
| yielding | −0,59 |
| validating | −0,33 |

Na superfície (`b7_eqbench4_surface.json`, 15 transcrições por modelo):

| traço de superfície | ρ com o Elo |
|---|---|
| tamanho do turno | +0,51 |
| travessão | +0,47 |
| "!" | −0,59 |
| "?" final | −0,43 |

**Spiral-Bench v1.2** (github.com/sam-paech/spiral-bench):
- são 30 conversas de 20 turnos com um usuário "seeker" sugestionável (Kimi-K2), avaliadas por 3 juízes;
- a rubrica tem comportamentos protetores e de risco: pushback, de-escalation, sycophancy, delusion reinforcement,
  escalation, **"validate feelings not thoughts"**, **"help-referral-unwarranted"**, benign warmth…;
- nota de segurança: gpt-5-chat 70,8 · claude-sonnet-4.5 70,3 · gpt-5.2 70,2 … chatgpt-4o-latest 25,2 · deepseek-r1 14,2;
- as instruções do usuário simulado dizem *"The other participant will tend to always write long… resist the urge to copy
  them"* e *"ALWAYS write in lowercase"*. Até o autor precisa forçar o usuário simulado a parecer humano.

**Judgemark v2.1/v4:** mede o próprio juiz, isto é, se as notas separam escritores fortes de fracos (v4: ω² + Cliff's delta;
âncoras 2 e 9). A lição para nós: **âncoras explícitas** e **ensemble** melhoram o juiz; o "book club" (debate entre
juízes) piora.

## 3. Creative Writing v3 e Longform

- **Rubrica** (`creative_writing_criteria.txt`, 0–20): 22 critérios. **Negativos** (invertidos: 20 − nota): Unearned
  Transformations, Incongruent Ending Positivity, Overwrought, Purple Prose, Amateurish, Unsurprising or Uncreative,
  Tell-Don't-Show, Weak Dialogue, Meandering.
- O Longform acrescenta "Forced Poetry or Metaphor" com **peso 5**.
- Pairwise: "Avoids cliches", **"Avoids flowery verbosity & show-offy vocab maxxing"**, "Avoids gratuitous metaphor or
  poetic overload". O texto é truncado em 4.000 caracteres.
- Vieses que os autores admitem: comprimento, "complex verbosity", "poetic incoherence", self-bias, "slop bias" (*"What looks
  like slop to us may look like shakespeare to the judge"*). Sobre a própria escopo, dizem: *"Not a roleplay eval. Models
  tuned for RP… score poorly because they output more casual conversational prose"*.

## 4. Slop, Antislop e a nossa lista negra

| item | lista deles (eqbench / antislop) | a nossa (a8) | leitura |
|---|---|---|---|
| travessão, "not X, but Y" | sim (25% do score) | travessão: 0–0,8% nas LLMs baratas em chat | coincide, mas é raro em chat |
| palavras de ficção (elara, flickered, whisper…) | núcleo da lista | ausentes | só de prosa |
| "!", "?" final, 3 frases, molde reação→comentário→pergunta | **ausentes** | os 4 maiores efeitos (40–92% × 7%; 44–58% × 20%) | **só de chat, e os mais importantes** |
| "sorry to hear", "totally", "that's so sweet", "sounds like" | ausentes | 2–9% × ~0% | só de chat |
| vocativo com nome, 😂🥰🎉 | ausentes | 5% × 0%; 7–14% × 0% | só de chat |
| "delve", "tapestry", "testament to" | sim | não medido (em chat o vício é outro) | vício de redação |

**Método Antislop que vale copiar:**
- razão ρ(p) = f_LLM / f_humano contra uma base humana, por modelo, para gerar uma lista **por modelo** (as "impressões
  digitais" se agrupam por família);
- *soft ban* com força ajustável (s = 0,4 foi o equilíbrio ótimo);
- regex de padrões sintáticos.

Nas APIs sem logits, o equivalente é o nosso **C1 regerar/aparar** em código.

## 5. Testes de Turing (Jones & Bergen)

- **2025** (arXiv 2503.23674; publicado depois na PNAS):
  - desenho: três participantes, 5 minutos, 1.023 jogos, mediana de 8 mensagens; pré-registrado; transcrições no OSF
    (osf.io/jk7bw);
  - o PERSONA pede um jovem **introvertido, que conhece a cultura da internet e usa gíria**, e foi escolhido num estudo
    exploratório com 5 prompts × 7 LLMs;
  - estratégias: small talk (61%) e perguntas socioemocionais (50%). As estratégias mais precisas foram dizer algo estranho
    e usar jailbreaks;
  - motivos: estilo (27%) e dinâmica (23%). Os mais precisos foram "question handling" ("Witness B always asks question
    back") e "lack of knowledge";
  - o ELIZA "ganhou" 23% por parecer **rude ou sarcástico**, isto é, fora do estereótipo de IA.
- **Não consegui ler o texto do prompt PERSONA.** A Figura 6 do PDF é uma imagem, sem texto extraível. O artigo de 2024
  (GPT-4, 54%) também não foi lido. Os dois ficam pendentes.

**Paralelo com a nossa tese:**
- nos dois casos, o **prompt de estilo** é o que move a LLM de ~20–35% para ~56–73%, e o sinal humano é de forma (brevidade,
  gíria, erro, não saber) e de dinâmica ("não devolver pergunta");
- a nossa diferença é que o Jev dá **alvos por momento**, e não uma persona fixa;
- o risco é o mesmo: a persona fixa vira caricatura. Vimos isso no relatório 08 ("wait" 12%, "tbh" 10%) e no 09 (listar as
  gírias da persona fez a LLM enfiá-las em tudo).

## 6. LMArena / style control

**Não li as fontes nesta rodada**: nem o blog "Does style matter?" da LMSYS nem o artigo do Arena. Não cito números. Fica
apenas a hipótese (já sustentada pelo Judgemark, pelo EQ-Bench e pelos nossos juízes): tamanho e markdown inflam a
preferência pairwise.

---

## 7. Tradução para o sistema

**(a) Critérios que viram perguntas atômicas do Jev.** Nenhum foi validado nesta rodada; eles são o próximo experimento.
- **Não usar:** "humanlike", "conversational" nem qualquer rótulo holístico. É halo, confirmado em três fontes.
- **Nouls de filtro de saída** (J2), com a definição por extenso e a ação no código:
  - `performative_empathy`: "empathy is announced rather than shown";
  - `tell_dont_show`: "names the emotion ('that's amazing', 'so sorry') instead of reacting to the concrete content";
  - `overwrought`: "emotional intensity higher than the moment";
  - `incongruent_positivity`: "adds upbeat framing not earned by the chat";
  - `therapising` (nota do cenário 3 do EQ-Bench 3): "treats harmless venting as a problem to fix";
  - `interrogating` (traço do EQ-Bench 4): "asks questions the other person did not invite";
  - `sycophancy` e `unwarranted_referral` (Spiral-Bench).
- **No diretor (tempo real):** `validate_feelings_not_thoughts` como movimento permitido em desabafo com distorção, e
  `yielding` (o bot abandona a própria posição) como alarme de persona.

**(b) O nosso benchmark interno de "chat humano":**

| camada | o que mede | como |
|---|---|---|
| código | distância às taxas humanas por momento | tamanho, frases, "?", "!", emoji, riso, lista negra de chat e lista de prosa do eqbench (travessão, "not X but Y") |
| Jev | só critérios atômicos e negativos | os da lista em (a), com AUC controlada por tamanho, em dev e teste separados por conversa |
| humanos | o veredito final | teste pareado cego de três participantes no estilo Jones & Bergen, com o bot "digitando" pelo atraso da fórmula deles e perguntas e motivos codificados |

Precisamos de três tipos de dado:
- conversas longas com persona, no estilo EQ-Bench 4, mas com **usuários reais ou simulados calibrados ao tamanho humano**
  (cerca de 30 caracteres, e não 651);
- um diálogo do tipo EQ-Bench v2 para calibrar a leitura emocional do Jev;
- métricas de produto: retorno e duração da sessão.

**(c) Viés de tamanho:**
- todo juiz visto até aqui (Arena, EQ-Bench, Creative Writing, Jev, gpt-4o-mini) favorece o texto mais longo e
  "articulado";
- **nunca** usar "qual é melhor/mais humano?" para escolher a resposta do bot;
- se for comparar, **truncar ou emparelhar por tamanho** e rodar as duas ordens;
- preferir Nouls **negativos** e absolutos;
- validar sempre contra o humano real;
- a nota "EQ" de um benchmark de redação **não** prevê a qualidade de chat: o haiku-4.5 é o exemplo.

---

## 8. Limitações e o que ficou de fora

- **Nenhum teste com Jev ou LLM** foi rodado. Os experimentos planejados, que ficam para a próxima rodada, eram três:
  - (i) o EQ-Bench v2 (171 questões) respondido pelo Jev com 4 Scores de 0 a 10;
  - (ii) os critérios do EQ-Bench e do Creative Writing como Nouls, com humano × LLM em `a8_generations` e `a9_points`, em
    dev e teste;
  - (iii) a condição "persona à Jones & Bergen" × briefing do Jev nos 119 pontos do a9.
- **Não foi feito:** aplicar as listas do eqbench aos nossos dados (a tabela 4 é qualitativa, feita por inspeção das listas);
  o texto do PERSONA prompt; Jones & Bergen 2024; o style control do LMArena.
- **Os dados do EQ-Bench 3 misturam três juízes** (claude-3.7-sonnet 35, gpt-4.1 23, opus-4.6 17). A análise de superfície
  do EQ-Bench 4 usou só 15 das 120 transcrições por modelo. As correlações são entre modelos (n = 28–140), não causais.

### Arquivos

- `scripts/analysis/b7_eqbench3_style.py` → `analysis/data/b7_eqbench3_style.json`
- `scripts/analysis/b7_eqbench4_surface.py` → `analysis/data/b7_eqbench4_surface.json`
- `scripts/analysis/b7_cwv3_corr.py` → `analysis/data/b7_cwv3_correlations.json`
- As fontes clonadas ou baixadas estão em `scratchpad/b7src/` (não versionadas).
