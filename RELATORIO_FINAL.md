# Jev como sistema nervoso de um chatbot conversacional: estudo com conversas humanas reais

> **Pergunta do estudo:** o que torna uma conversa de chat realista e engajante, inclusive os padrões que ninguém percebe,
> e como usar o Jev (TypeSafe, "System One") para que uma LLM rápida e barata escreva, a cada momento, exatamente o que
> uma pessoa escreveria?
>
> **O que foi feito:** 5 corpora de conversas humanas reais (≈238 mil mensagens), 6 mil turnos anotados por uma camada
> base do Jev, e 9 análises paralelas, cada uma com experimentos próprios de Jev e de LLM. Foram ≈35 mil chamadas ao Jev
> (≈US$ 1,90) e ≈3.500 gerações de LLM (≈US$ 0,31). Os relatórios detalhados estão em `analysis/01…09`; este
> documento é a síntese e a proposta de sistema.

## ★ 2ª rodada: a arquitetura das chamadas muda o veredito (relatórios 10–16)

> Resposta ao seu feedback de que "a arquitetura das chamadas é tão importante quanto o modelo". **Você tinha razão.**
> Onde a 1ª rodada concluiu "o Jev não serve", a culpa era da forma de perguntar. Os créditos acabaram no meio da
> rodada: os relatórios 10–13 são **parciais** (listam o que rodou e o que não rodou), mas os resultados abaixo são do
> **teste**, separado do dev por conversa.
> Atores testados: só `mercury-2.5`, `deepseek-flash-latest`, `gpt-6-luna` e `gemini-3.5-flash-lite`.

**1. Juiz de "cara de LLM" (relatório 10): de AUC 0,51 para 0,91.**

| arquitetura | AUC humano × LLM | controlando o tamanho |
|---|---|---|
| pergunta holística "é IA?" (1ª rodada) | 0,51 | 0,43 |
| só código (regressão logística sobre estilometria) | 0,865 | 0,841 |
| **banco de 85 perguntas atômicas do Jev numa chamada → regressão logística no código** | 0,852 | 0,838 |
| **código + banco do Jev** | **0,912** [0,885–0,935] | **0,896** |
| código + Jev enxuto (16 perguntas) | 0,903 | 0,888 |

- **Com saídas de LLM já instruídas** (o caso difícil): código 0,75, código + Jev **0,82**. No briefing B do relatório 09,
  que "enganava" o código, a combinação sobe de 0,66 para 0,73.
- **O Jev como diagnosticador por traço** (um Noul por vício) acerta muito, contra os rótulos de código: termina em
  pergunta 0,98; interjeição performática 0,98; pergunta genérica 0,96; eco do usuário 0,93; validação 0,92; molde
  "reação → comentário → pergunta" 0,91; gíria 0,90; entusiasmo 0,83; longo demais 0,82. **Ele diz exatamente o que
  corrigir.**
- **O que continua não funcionando:**
  - a pergunta holística e o contrafactual ("um amigo mandaria isso?": 0,29, invertido);
  - o Choice "qual é o vício principal?" (5,8%): cada vício precisa ser um Noul próprio;
  - o Score com níveis ancorados ("amigo ↔ assistente") foi melhor que o Noul (0,68 × 0,51).
- **Custo do banco:** 1 chamada, US$ 0,000145, 0,51 s.

**2. Número de bolhas (relatório 11): o "sempre 1" era problema de arquitetura.**

| arquitetura | P(várias) prevista × real (34%) | AUC ≥ 2 bolhas | RPS (↓) |
|---|---|---|---|
| state cru (1ª rodada) | 3% | — | — |
| com um **guia** de 1–5 bolhas no state | 23% | 0,88 | — |
| **encadeado**: os rótulos de um 1º Jev entram no state de um 2º com guia | 28% | 0,89 | 0,061 |
| roteador por momento → Choice específico | — | — | 0,089 (pior) |
| **o Jev como features → regressão ordinal no código** (escolhido no dev) | ≈ real (±0,05) | **0,92** | **0,048** |
| só código treinado com 12 mil turnos | — | 0,93 | 0,052 |

- É exatamente a arquitetura que você descreveu (guia + rótulo "ansioso" de outro Jev + Choice 1–5), e ela funciona: AUC
  0,89, empatando com a tabela de tamanho em código.
- O melhor é o Jev dando features e o código decidindo. Com pouco dado (um produto novo em PT-BR) e no chat ao vivo, o Jev
  **melhora** o código (RPS 0,059 → 0,048; ao vivo, 0,059 → 0,042). Com muito dado, empata. **Sorteie o número de bolhas
  pela distribuição prevista.**

**3. "5 bolhas em menos de 1 minuto é normal?" (relatórios 11 e 15)**
- **Normal, mas raro e concentrado em poucas pessoas:**
  - 5+ bolhas em < 60 s: 1,0% dos turnos ao vivo e 0,8–1,8% no WhatsApp;
  - os 10% de pessoas mais "rajadeiras" fazem 62–67% dessas rajadas;
  - **3+ bolhas em < 60 s é comum** (9–16% dos turnos);
  - quase não existe abaixo de 40 caracteres e aparece em 15–31% dos turnos acima de 160.
- **Os gatilhos** (lift sobre a taxa-base):
  - **explicar demais** (10–15×), **corrigir a si mesmo** (3–9×), **pensar em voz alta** (3–4×), contar uma história (2–4×) e dar uma notícia (2–3×);
  - **defensivo / "pego no flagra" / se justificando**: 2,8–4,2× no WhatsApp, e **OR 3,9–5,9 mesmo controlando o tamanho**
    (p < 10⁻⁴). A sua intuição se confirma nos dados.
- **A ansiedade sozinha não causa rajada** quando se controla o tamanho (OR 0,25–0,91, n.s.). A **briga ao vivo** vira uma
  mensagem **seca** (zero rajadas de 5).
- **A literatura de deception** não tem estudo sobre "rajada de quem foi pego". Quem mente demora ~10% mais, edita mais e,
  em conversa livre, escreve ~28% mais palavras (relatório 15).
- **O outro reage à rajada:** responde 2× mais rápido, responde com outra rajada (31% × 14%) e quase não pergunta. Depois de
  uma defesa, ri junto em 39%.
- **Detector "defensivo"** (2 Nouls com AND no código: "o outro acusou?" E "está se justificando ou negando?"): precisão
  0,60, recall 0,78 e AUC 0,97 do escore contínuo, medidos contra rótulos de LLM, não humanos. É melhor que o Noul único
  (0,37) e que a cascata com guia (0,52).

**4. Controle de saída (relatório 13, só dev, 60 pontos, 4 atores).** D = distância às taxas humanas em 10 vícios.
- **Sem controle:** D = 3,90 (flash-lite), 3,68 (luna), 1,93 (mercury), 1,64 (deepseek). O mercury e o deepseek já saem
  mais "humanos"; o luna põe emoji em 75%.
- **Com controle, todos convergem para D ≈ 1,15–1,35:**
  - o **normalizador em código** sozinho corta 25–47% do D, de graça;
  - o **briefing com alvos** ("~N palavras, uma ideia", pergunta e "!" sorteados) é a maior alavanca e zera o molde de 3
    tempos;
  - o melhor é **gerar 4 com o briefing e escolher em código** pela menor violação (D 1,15–1,21), a 4× o custo e +0,2–0,4 s.
- **Os controles de API quase não servem nesses atores:**
  - `max_tokens` justo corta a frase no meio em 33–90% dos casos;
  - `logit_bias` é ignorado ou não existe;
  - ninguém aceita prefill;
  - o flash-lite ignora a temperatura.
- **A poda de frases pelo Jev** ganha pouco da regra "1ª frase" (1,24 × 1,38).
- **Continuam sem solução:** a pergunta sobra sem controle e some com controle (25–50% → 2–5%, humano 15%); o eco do
  usuário no luna (17% × 5%). **Pipeline provisório:** leitura do Jev → orçamento em código → briefing com alvos → 4
  candidatas → escolha em código pelo banco do Jev (item 1) → normalizador.

**5. O que escrever (relatório 12): 13 arquiteturas + 12 misturas, teste com 400 pontos.**
- **O teto é baixo por natureza.** Dois rotuladores independentes que **veem** a resposta real concordam no movimento em
  73,5%. Prevendo antes de vê-la:

| arquitetura | top-1 do movimento |
|---|---|
| classe majoritária | 19,8% |
| voto kNN de casos parecidos (sem o Jev online) | 30,2% |
| **Choice plana do Jev** | **37,5%** |
| **retrieval de 20 casos humanos parecidos + Choice plana (escolhida no dev)** | **38,8%** (top-3 66,8%) |

  - O retrieval melhora de forma significativa a **distribuição** (log-loss −0,16), que é o que importa para sortear, mas não
    o argmax.
  - A similaridade que funcionou foi um **"embedding Jev"** (os rótulos D do turno do parceiro); TF-IDF puro recuperava
    casos ruins.
- **As cascatas pioraram a decisão do movimento:** rótulos no state de um 2º Jev + guia 33,2%; + prior empírico 33,0%; guia
  no state 34,5%; 16 Nouls 26%. **Para o "o que fazer", o Jev decide melhor com o contexto cru** do que com rótulos
  intermediários: o 2º passo herda os erros dos rótulos e ancora nas taxas do guia. (Nas bolhas, que são uma decisão
  estrutural, o encadeamento ajudou. **A arquitetura certa depende do tipo de decisão.**)
- **A LLM como cérebro é pior que o Jev:** o gpt-6-luna prevendo o movimento com raciocínio acertou 32,5%, contra 37,5%, a
  ~10× o custo.
- **O portão de confiança funciona muito bem:** com confiança ≥ 0,7 (29% dos pontos), o top-1 sobe para **64%**. Abaixo de
  0,55, fica em 25–30%. **Dite o movimento só com confiança alta; senão, dite só a forma.**
- **A que palavra reagir (especificidade): o Jev acerta 61,6%**, contra 42,7% da regra "última palavra" e 30% do acaso. É o
  sinal de conteúdo mais forte que achamos e ataca o "responder à palavra, não ao tema" e a falta de especificidade (Meena).
- **Tom:** 44,8% (majoritária 29%, teto 72,5%). **Subtexto** ("deixar implícito?"): AUC 0,62, fraco.
- **Briefing v2** (reduzido: 40 pontos, 2 atores, ICs largos):
  - os dois atores ficaram perto do humano na forma com qualquer briefing, e de novo quem faz o grosso é o **código**;
  - passar o movimento do Jev como ordem **não** melhorou a coincidência de movimento nesse n (flash 30% contra 42% só
    código; luna 35% contra 35%);
  - novo vício criado pelo próprio briefing: "greet back + one concrete thing" gerou "hey drinking coffee" em 5 de 6
    aberturas. **Cada ordem fixa vira um vício; as ordens precisam variar.**

**6. Respaldo científico (relatórios 14, 15 e 16).**
- **A literatura mede o mesmo excesso que nós:**
  - no SOTOPIA, o GPT-4 escreve 45,5 palavras por turno contra 16,8 dos humanos (2,7×; nós medimos 2,2–2,6×) e "sempre
    reformula a fala do outro";
  - no BlenderBot, "Do you have" aparece 110× contra 6× nos humanos;
  - no EQ-Bench 4, os modelos terminam em "?" em 48–83% dos turnos, com 772 caracteres por turno;
  - causa provável: o RLHF recompensa tamanho (Singhal et al. 2024).
- **Juízes premiam o texto longo, e isso é conhecido:**
  - a verbosidade engana juízes LLM em até 91% dos casos;
  - no EQ-Bench 3, o critério "humanlike" correlaciona 0,97 com o Elo geral (efeito halo), e os autores truncam respostas
    por causa do viés de tamanho;
  - GPT-4 como juiz de "parecer humano" concorda pouco com humanos (r 0,12–0,32);
  - no PersonaEval, o melhor LLM identifica quem fala em 69%, contra 91% dos humanos;
  - no nosso teste, as métricas-padrão SSI (sensibleness/specificity) e a nota geral de "parecer humano" favorecem a LLM.
    **Não use SSI nem nota holística como meta; use falhas atômicas** (item 1).
- **Planejador externo > a LLM decidir sozinha:** com planejador externo, a escolha da estratégia de apoio sobe de 13,5 para
  21,1 F1 e o viés cai de 1,38 para 0,36. Auto-reflexão ou CoT da própria LLM **pioram** (9,6–12,4 F1). É o respaldo direto
  para "o Jev decide, a LLM atua".
- **Teto baixo para prever o movimento:** no LIGHT, humanos acertam a próxima emoção do outro em 27–34%. Os nossos 32–34,5%
  são razoáveis; avalie contra a **distribuição**.
- **Turing com persona** (Jones & Bergen 2025): o GPT-4.5 com prompt de persona foi julgado humano em **73%**; sem persona,
  36–38%. Os juízes decidem pelo estilo e pela dinâmica, e os motivos que mais acertam são "sempre devolve pergunta" e
  "falta de conhecimento". **É a prova de que "instruções em tempo real" movem a percepção de humanidade.**
- **As listas de slop do eqbench são de prosa de ficção** e coincidem pouco com a nossa. Os vícios de **chat** ("!", "?"
  final, o molde de 3 tempos, "sorry to hear", o nome do usuário) não estão lá; a nossa lista negra é específica e nova.
- **CMC confirma os nossos números:**
  - a regra de latência de Kalman (82,7% das respostas dentro da latência média de quem responde);
  - a forma única de riso por pessoa;
  - o alongamento concentrado no afeto;
  - o estilo **não** converge ao longo da conversa entre humanos, enquanto o GPT-4o se acomoda 1,8× mais que o usuário já
    no 1º turno;
  - a LLM se alinha no conteúdo (paráfrase) e o humano no estilo.
- **Psicologia:** a pergunta de seguimento sobre o que o outro disse aumenta a simpatia, e a pergunta-espelho não
  (Huang 2017). Autorrevelação: d = 0,28. Só 2% das conversas terminam quando os dois querem (Mastroianni 2021).
- **Lacuna:** não há trabalho de NLP sobre fragmentação em bolhas, timing, backchannel, fim sem despedida ou responder
  abaixo da intensidade do outro. **Os nossos achados de forma são contribuição original.**
- **Ética e produto** (MIT/OpenAI 2025; Nature Human Behaviour 2026):
  - mais uso voluntário se associa a mais solidão e dependência;
  - em 37% das despedidas, os apps de companhia tentam segurar o usuário com culpa ou medo de perder algo (até 14× mais
    engajamento e mais vontade de largar o app);
  - requisitos: não otimizar minutos de uso; na despedida, espelhar e deixar ir; nunca expressar carência dirigida ao
    usuário; ser honesto sobre ser IA; ter protocolo de crise.
- **OptMem** (relatório 15): o `wake` (até ~8k tokens) vira o campo `memory` do state do Jev. O Jev decide o que vira nota
  e de que tipo (fato, pendência, piada interna, limite), e a LLM redige a nota fora do caminho crítico. As pendências
  alimentam as reaberturas ("e a prova, foi?"), e o bot guarda o que já contou para não repetir histórias.
  - **Observação:** pelo README e pelo código, o OptMem guarda **notas de uma linha** numa árvore de resumos, não o histórico
    bruto. O uso descrito no texto do Taelin (o chat inteiro como entradas) é uma forma de usar a mesma ferramenta.

**Novos princípios de arquitetura (substituem as ressalvas da 1ª rodada):**
1. **Nunca uma pergunta holística; sempre um banco de atômicas** (dezenas numa chamada), combinadas em código com pesos
   aprendidos. Isso vale para julgar, diagnosticar e decidir.
2. **Encadeamento com critério:** rótulos de um Jev, mais um **guia** com as regras e taxas humanas, no state de um 2º Jev
   recuperam decisões **estruturais** (bolhas: 3% → 28% de calibração). Mas **pioram** as decisões de conteúdo (movimento:
   37,5% → 33%), em que o contexto cru mais o **retrieval de casos humanos parecidos** funciona melhor. Teste as duas
   opções para cada decisão.
3. **Detectores compostos com AND no código** (ex.: defensivo = "foi acusado" ∧ "está se justificando") são mais precisos
   que um Noul único.
4. **O Jev como features, o código como decisor** (regressão logística ou ordinal), sempre que houver algum dado rotulado.
   Com pouco dado, o Jev dá o maior ganho.
5. **Um Noul por vício ou traço, nunca um Choice "qual o problema?".**
6. **O portão de confiança é o que torna o Jev confiável para ditar conteúdo:** com confiança ≥ 0,7, ele acerta o movimento em
   64% dos casos; abaixo disso, dite só a forma (tamanho, pergunta sim/não, riso, a palavra a que reagir).

## ★ Próxima camada: técnicas de geração e estado vivo do personagem

> Propostas do usuário, integradas à arquitetura. Ainda **não foram medidas** neste estudo; cada uma vem com o desenho
> do teste. Onde há evidência nossa ou da literatura, ela é citada.

### A. Geração com variações e probabilidade (Verbalized Sampling)
**Ideia:** em vez de "escreva a resposta", pedir "escreva 3–5 variações, cada uma com a probabilidade de ser a resposta
certa". O artigo é *Verbalized Sampling* (Zhang et al., 2025, arXiv 2510.01171): o alinhamento colapsa a LLM na resposta
"típica" (*mode collapse*), e pedir a **distribuição verbalizada** recupera a diversidade do pré-treino. Uma variação que o
modelo rotula com 0,3 tende a ser algo que ele **não** escreveria por padrão.

**Por que encaixa aqui:**
- O vício da LLM é justamente a resposta modal: longa, validando e terminando em pergunta.
- A resposta humana típica é "improvável" para ela: "hehe", seguir o assunto, "é a mesma cara de ontem".
- O relatório 12 mostrou que sortear o movimento aproxima a variedade humana (3,2 contra 3,5 bits).

**Como usar no sistema:**
1. A LLM gera N variações com probabilidade verbalizada, numa chamada só.
2. O código descarta as que violam o orçamento (tamanho, "?", "!", lista negra).
3. O **banco atômico do Jev** (relatório 10, AUC 0,91) pontua cada uma.
4. O código sorteia com um peso que favorece as de probabilidade verbalizada **média ou baixa** que passaram nos filtros,
   o que evita a moda sem cair no aleatório.

Isso também substitui o "gerar 4 com seeds diferentes", que teve pouca diversidade (2,64 textos distintos em 3, relatório
08) e o flash-lite ignorando a temperatura (relatório 13).

**Teste proposto:** as mesmas condições do relatório 13 (D = distância às taxas humanas), com VS-3 e VS-5 contra 4 seeds.
Medir também a coincidência de movimento e a diversidade.

### B. Restrições como fatos do personagem, não como ordens
**Ideia:** "Alice nunca fala sobre o pai" guia melhor que "não fale sobre o seu pai". A negação no imperativo põe o tema no
foco da atenção (o problema do "elefante rosa"; ver *Suppressing Pink Elephants with Direct Principle Feedback*,
Castricato et al. 2024). Um **traço de identidade** muda o comportamento sem nomear a ação proibida como tarefa.

**Evidência nossa (mista):**
- No relatório 09, a lista "Never use: aww, totally…" **não** causou efeito rebote (itens proibidos caíram de 39,5% para
  1,7%).
- Mas **listar o vocabulário da persona** ("palavras que você usa: idk, omg") fez a LLM enfiá-lo em tudo.
- Ou seja: proibir léxico no imperativo funcionou, e *afirmar* traços léxicos virou caricatura.
- O caso do usuário é outro, de **temas e comportamentos** ("nunca fala do pai", "nunca pede desculpa primeiro", "nunca
  manda textão"). Aí o formato de fato de identidade é o mais provável de funcionar, e vale testar.

**Desenho:**
- A ficha do personagem ganha uma seção **"Alice nunca…" / "Alice raramente…"**, com fatos de identidade em 3ª pessoa.
- Proibições **situacionais** geradas pelo diretor a cada turno (ex.: sem pergunta agora) continuam como ordens curtas no
  briefing, que funcionaram.
- O Jev vigia: um Noul por "nunca" ("A resposta menciona o pai de Alice?"), no banco pós-geração.

**Teste proposto:** a mesma ficha em três formatos (imperativo "não faça" × identidade "ela nunca" × nenhum), medindo a taxa
de violação e o efeito colateral (menção indireta ao tema, rigidez).

### C. Gerar dentro de um formato de chat real (o "schema" de exportação)
**Ideia:** em vez de deixar a LLM escrever livre, fazê-la **completar um log de conversa** no formato de exportação de uma
plataforma. Isso ativa a memória de chats reais do pré-treino, e ela passa a escrever "como no WhatsApp" sem precisar de
instruções de estilo:

```
[30/09/2026, 18:31:02] Cesar: mano
[30/09/2026, 18:31:05] Cesar: pior q pensei nisso
[30/09/2026, 18:31:08] Cesar: ontem
[30/09/2026, 18:31:14] Cesar: mas sla
[30/09/2026, 18:31:17] Cesar: acho q n vai dar
[30/09/2026, 18:32:40] Marcela:
```

**Por que é promissor:**
- Ataca a raiz: o *assistant prior* (reconhecer → validar → perguntar) é um **formato**, e trocar o formato troca o prior.
- O teste de Turing com persona (Jones & Bergen 2025, 73%) mostra quanto o enquadramento muda a percepção.
- É consistente com o nosso achado de que instruções abstratas de estilo corrigem demais ou criam vícios novos.
- **Bônus de entrega:** o formato com timestamps faz a LLM propor as **bolhas** (uma por linha) e os **intervalos**. O
  código trata isso como sugestão, validada pela camada de entrega (relatórios 01 e 11), e o Jev confere com o modelo de
  bolhas (AUC 0,92).
- O histórico real da conversa já pode ser apresentado nesse formato, o que dá continuidade de estilo.

**Riscos a medir:**
- A LLM pode continuar a conversa **pelos dois lados**, inventando a fala do usuário (a Gemini já fez isso no relatório 09).
  Mitigação: parar na próxima linha que não seja do personagem, com `stop` ou corte em código.
- Pode copiar o estilo de um usuário real do histórico em vez do estilo da persona.
- Pode perder a instrução do diretor. Mitigação: o briefing entra como um bloco **antes** do log (estilo Author's Note) ou
  como "nota" de sistema no próprio log.

**Benchmark de schemas proposto** (os mesmos contextos do maichat, os 4 atores permitidos, as métricas D do relatório 13,
a coincidência de movimento e o banco do Jev):

| schema | exemplo |
|---|---|
| livre (baseline) | persona + histórico como mensagens user/assistant |
| WhatsApp export | `[30/09/2026, 18:31:04] Cesar: texto` |
| WhatsApp só com hora | `[18:31:04] Cesar: texto` |
| Messenger/Facebook JSON | `{"sender_name": "Cesar", "timestamp_ms": …, "content": "…"}` |
| Snapchat JSON | `{"From": "cesar", "Media Type": "TEXT", "Created": "…", "Content": "…"}` |
| IRC/Discord log | `<cesar> texto` / `cesar — hoje às 18:31` |
| SMS/iMessage | um par remetente/texto por linha |

Hipótese: os formatos de linha (WhatsApp) vencem os JSON (mais "máquina"), e os timestamps ajudam no ritmo. Em PT-BR, o
formato WhatsApp é o mais natural, porque o próprio corpus brasileiro a coletar virá nesse formato.

### D. Cabeçalho de roleplay no system prompt
Deixar explícito que é **atuação**, no formato consagrado pelos front-ends de RP (SillyTavern, Character Cards):

```
SYSTEM: You are roleplaying as {{char}} in a private text chat with {{user}}. Stay consistent with {{char}}'s identity,
voice and history. You are not an assistant.
CHARACTER: Name / personality (as behaviours, not adjectives) / background / "{{char}} never…" / how {{char}} texts
RELATIONSHIP: (the live state, section E)
MEMORY: (OptMem wake)
DIRECTOR NOTE (this turn): (briefing do Jev, curto e imperativo; relatórios 09, 12 e 13)
```

**Respaldo:**
- CoSER mostra que pensamentos internos e motivações melhoram a atuação.
- Personalidade descrita por **comportamentos situacionais** ("quando desconfortável, desvia com humor seco e responde
  curto") supera adjetivos ("tsundere").
- A nota do diretor fica **no fim do contexto**, onde tem mais influência (Author's Note do SillyTavern).
- O RoleCDE mostra que alguns modelos abandonam o papel quando ele conflita com o alinhamento (claude-haiku-4.5 no pior
  lugar). É mais um motivo para testar o ator.

### E. Estado da relação vivo, atualizado pelo Jev a cada mensagem
O traço estático ("tímida, sarcástica, gosta de gatos") não evolui. O que evolui é a **relação**:

```
relationship_toward_user:
  trust 0.72 · comfort 0.84 · romantic_interest 0.51 · resentment 0.19 · protectiveness 0.62 · respect … · jealousy …
unresolved:   - user cancelled dinner last Friday · - suspects user was avoiding her
shared_history: - met at bookstore · - joke about terrible cappuccino · - he stayed up talking after her bad day
```

**Cascata do Jev (o "neurônio" que o usuário descreveu), em tempo real:**
1. **Jev 1, detecção:** "quais dimensões esta mensagem do usuário move?", com **um Noul por dimensão** (ex.: "a mensagem
   aumenta o ressentimento de Alice?", "…aumenta a confiança?"), não um Choice. O relatório 10 mostrou que "qual o
   principal?" falha e um Noul por item funciona.
   - Nouls de evento, compostos com AND no código como o detector "defensivo" do relatório 11: "ele se desculpou?", "fez
     piada de mau gosto?", "cumpriu algo que prometeu?", "mencionou outra pessoa de forma que gera ciúme?", "revelou algo
     vulnerável?"
   - **State:** a mensagem, os 6–8 turnos anteriores, o `relationship` atual e as `unresolved` (o contexto muda a leitura:
     a mesma piada ofende no dia 2 e não no dia 200).
2. **Jev 2, magnitude:** só para as dimensões ativadas, um **Score com níveis descritivos**, **não números**. A
   documentação do Jev (*jaggedness*) desaconselha números: "quanto isso mexe com o ressentimento de Alice, dado o
   contexto?" com os níveis {nada, leve, moderado, forte, marcante}.
3. **Código, a física do estado:**
   - mapeia o nível para um delta (ex.: leve +0,03, moderado +0,07, forte +0,15, marcante +0,25), escalado por
     `(1 − valor)` ao subir e por `valor` ao descer, para não saturar;
   - aplica **decaimento** por tempo (o ressentimento esfria em dias, a confiança muda devagar) e **inércia** (a confiança
     cai rápido e sobe devagar);
   - aplica **histerese** para mudar de "modo" (ex.: `resentment > 0,5` → Alice fica seca), sem oscilar a cada turno. O
     relatório 06 mostrou que a relação lida turno a turno é instável (65–70%): aqui ela é um **acumulador**, não uma
     leitura;
   - aplica **eventos compostos**: um pedido de desculpas **só** reduz o ressentimento se houver `unresolved` ligada. Uma
     desculpa repetida sem mudança rende menos a cada vez.
4. **Eventos viram memória:** o Jev decide se o evento entra em `unresolved` ou `shared_history` (Noul "é algo que Alice
   lembraria daqui a um mês?"), e a LLM redige uma nota curta **fora do caminho crítico** (OptMem, relatório 15). Pendências
   resolvidas saem da lista.
5. **O estado alimenta o diretor, não só o prompt:** `resentment` alto + `unresolved` → o movimento sorteado favorece
   "responder seco", "cobrar indiretamente" ou "ignorar o carinho", com **subtexto**: "olha só quem resolveu aparecer", e
   não "estou chateada porque…". `trust` e `comfort` altos → mais autorrevelação e provocação carinhosa. É exatamente o
   "mesma frase no dia 2 × dia 200".

**Cuidados:**
- **Anti-bajulação:** a relação precisa poder **piorar** de verdade e o personagem precisa querer coisas próprias, senão
  vira espelho.
- **Ética** (relatório 15): nunca usar ciúme, carência ou culpa para prender o usuário (37% das despedidas em apps de
  companhia fazem isso). O ressentimento se expressa, mas não vira chantagem.
- **Custo:** Jev 1 cabe na chamada de leitura já existente (+10–20 Nouls). Jev 2 é uma chamada condicional pequena (~0,5 s),
  paralela à geração.
- **Teste proposto:** conversas sintéticas roteirizadas (desculpa, piada ruim, sumir por 3 dias, cumprir promessa), com
  checagem de que as trajetórias do estado são plausíveis. E um A/B com humanos: bot com estado vivo × estático, medindo
  consistência e "a relação evolui?".

### Onde isso entra no fluxo (seção 4.2)
```
mensagem → [Jev leitura + Jev 1 da relação] → código atualiza o relationship_state (e decaimento)
         → diretor (movimento condicionado ao estado) → [Jev 2 da magnitude, em paralelo, se necessário]
         → PROMPT = cabeçalho de roleplay (D) + ficha com "nunca…" (B) + relação (E) + OptMem + nota do diretor
         → LLM completa o LOG no schema de chat (C), gerando N variações com probabilidade (A)
         → filtros de código + banco atômico do Jev → sorteio → entrega (bolhas e tempos sugeridos pelo log, validados em código)
```

---

## 0. Resumo em 13 pontos

0. **O teste direto da tese (A/B, 119 pontos reais) confirma o essencial.** Uma LLM **barata + briefing do Jev** ficou mais perto
   do humano do que a mesma LLM sozinha **e** do que uma LLM ~3× mais cara sem briefing (claude-haiku-4.5), em todas as métricas
   de forma e na coincidência do "movimento" (o que fazer na resposta: 34,5% × 23,5% × 26%). Os juízes automáticos, que separavam
   com facilidade a LLM pura do humano, ficaram **no acaso** com o briefing. **Mas, honestamente, ~80% do ganho veio das regras de
   código calibradas às taxas humanas** (tamanho, "sem pergunta reflexa", "sem !", estilo da persona). Um controle com o mesmo
   briefing **sem** Jev chegou quase lá. O Jev acrescentou o que só ele faz: fazer o **tamanho acompanhar o momento** (ρ 0,47 ×
   0,28) e acertar um pouco mais o **movimento**. E, quando lê errado, a LLM obedece o erro ("love u dad" → "delete your
   account"). Conclusão: o Jev é o sistema nervoso, mas o **esqueleto** são as distribuições humanas codificadas. E portões de
   confiança são obrigatórios.
1. **A tese se sustenta, com uma correção importante.** O Jev funciona muito bem como **leitor** do momento: emoção por
   família (≈75–85% de acerto com confiança alta), seriedade, se é hora de brincar (AUC 0,78), risco de fim de conversa
   (AUC 0,75–0,91), gancho (o melhor preditor de a conversa continuar) e movimento de flerte. Ele funciona **mal** como
   juiz de "isso soa humano?": prefere a resposta caricata da LLM à resposta humana real em 71–88% dos pares (a mais longa
   em 63%), e o gpt-4o-mini faz o mesmo. Então o desenho é **"o Jev lê e classifica, o código decide, sorteia e conta, a LLM
   escreve"**, e não "o Jev julga a qualidade final".
2. **Cabem centenas de perguntas por mensagem.** 132 perguntas numa chamada levam ≈0,55 s e custam ≈US$ 0,00023; a latência
   é quase a mesma de 1 pergunta, e as perguntas não interferem entre si. Um usuário com 200 mensagens/dia custa entre
   ≈US$ 0,50 e 3 por mês em Jev. **O Jev não é o gargalo; a LLM é.**
3. **O maior defeito das LLMs não é o vocabulário, é o formato da resposta.** No mesmo ponto da conversa, a LLM escreve
   **2,2–2,6× mais** que a pessoa, usa "!" em 40–92% das respostas (humano: 7%), termina em pergunta em 44–58% (humano:
   20%) e segue o molde "reação → comentário ou paráfrase → pergunta" (20–25% contra 1,6%). As palavras "de LLM" existem
   ("totally", "sorry to hear", "that's so sweet", o nome do usuário), mas vêm depois disso em impacto.
4. **Os dois vícios que você descreveu existem, e cada modelo puxa para um lado.** O GPT-4o-mini puxa para o formal de
   assistente ("Absolutely!", "I totally get that"); o Gemini Flash, para a gíria forçada (💀, "lmaooo", "bestie", memórias
   inventadas). E **proibir não resolve**: o prompt "seja casual e curto" zerou a lista negra, mas criou vícios novos ("wait"
   no início em 12%, "tbh" em 10%, resposta curta demais, nenhuma pergunta). O que resolve é dar **alvos numéricos por
   momento** (tamanho, pergunta sim/não, riso sim/não) e controlar a frequência de cada vício em código.
5. **Humanos são banais e não respondem a tudo.** Metade das respostas é "true", "fair", "haha", "same". Em 1 de cada 5
   vezes a pessoa nem responde à última mensagem e segue o próprio fio. Diante de uma notícia, ela comenta **o fato** ("u
   did scream") em vez de nomear a emoção ("That's amazing!"). "Congrats" e "sorry to hear" praticamente não existem em
   chat real entre conhecidos (0–1 a cada 1.000 mensagens).
6. **Flerte e carinho se respondem abaixo, não acima.** A resposta mais comum a um carinho é **não comentá-lo** e seguir
   o assunto (32%) ou mandar só um "hehe" (13%). As pessoas respondem em média meio nível **abaixo** da intensidade do
   outro, e só 3% passam de +1. O que mantém o flerte vivo é **encabular-se** ("para kkk", "não me expõe"), **devolver
   a provocação** ou subir **meio** passo. A LLM faz o contrário: "Aww you're making me blush 🥺 can't wait to see u!!",
   com pergunta.
7. **Sua hipótese da ansiedade se confirma em parte.** Quem "picota" mais é a **agitação** (arousal): +6–7 pp de chance
   de várias bolhas por nível, com bolhas 32–36% mais curtas. A **ansiedade** gera mais bolhas (2,2 × 1,6 no WhatsApp), mas
   com **mais texto** e não mais rápido. O **momento sério** alonga as bolhas, sem reduzir o número. O que mais determina o
   número de bolhas é simplesmente **quanto a pessoa tem a dizer** (R² ≈ 0,25–0,29) e o **estilo pessoal** (7% a 63% de
   turnos com várias bolhas).
8. **Fragmentar é a regra, não a exceção.** Têm 2 ou mais bolhas 31–47% dos turnos, que concentram 53–71% de todas as
   mensagens. A gramática é quase fixa: **reação curta primeiro → conteúdo → continuação → pergunta por último**; o riso
   abre, o emoji fecha.
9. **A demora dramática não é humana.** Depois de algo sério a pessoa escreve ≈15–20% a mais, tira emoji e brincadeira,
   mas **não demora mais** (até responde um pouco antes). O que determina a latência é o **ritmo do outro** (ρ ≈ 0,4): cada
   um responde no tempo em que é respondido. A pausa longa aparece **antes de responder a um flerte** (11 s × 4 s até
   começar a digitar).
10. **Conversas íntimas não terminam, pausam.** 94% das sessões de WhatsApp acabam sem despedida e 61% numa frase comum. Quem vai
    sair **para de perguntar** 1–2 turnos antes. Entre íntimos quase não se diz "oi": 54% das sessões abrem direto com uma pergunta,
    e a saudação só cresce depois de dias de silêncio, sempre com um gancho concreto e **nunca** com "quanto tempo!".
11. **O que mata uma conversa é o turno sem gancho.** Depois de "ok"/emoji/"haha" sozinho, a chance de o outro responder cai
    para 76–79%; depois de uma pergunta, é 94%. O `hook` do Jev é o melhor preditor de continuidade (OR 11,9). Mas o que **sobe
    o engajamento** do outro não é perguntar, é **contar algo de si** (autorrevelação +0,15, história +0,18). E devolver
    "e você?" é raro em humanos (2–8%).
12. **Para decidir o número de bolhas, código é melhor que Jev.** O `p_n_msgs` do Jev respondeu "1 bolha" em ≈100% dos casos.
    Previsões de eventos raros (vai rir? vai usar emoji?) **ordenam** bem, mas vêm descalibradas (p_laugh médio de 0,34 contra 4,7%
    real). Regra: **o Jev fornece o estado; o código sorteia a partir de distribuições empíricas calibradas.**

---

## 1. Dados e método (resumo)

| corpus | o que é | tamanho | para que serviu |
|---|---|---|---|
| **MaiChat** (Edinburgh, 2026) | chat 1:1 ao vivo em inglês, entre pessoas que se conhecem, **com logs de digitação** (cada estado do texto, em ms) | 42 conversas, 4.177 msgs | timing, digitação, hesitação, texto apagado; comparação humano × LLM no mesmo ponto |
| **WhatsApp Berntzen** (DANS) | WhatsApp real doado, holandês, 2012–14 | 57 chats 1:1, 60.469 msgs, 3.744 sessões | bursts, latências reais, aberturas, reaberturas, fins de sessão |
| **NPS Chat** | salas públicas de 2006, atos de diálogo anotados à mão | 10.567 posts | aberturas entre desconhecidos, flerte explícito |
| **NUS SMS** | SMS reais (Singapura) | 55.835 msgs | registro, abreviações |
| **EmpatheticDialogues** | chat induzido, com **emoção-ouro** (32 classes) | 24.850 conversas | validação do detector de emoção; como o ouvinte responde |

- **Camada base do Jev** (`scripts/annotate_base.py`), sobre 5.968 turnos:
  - **passe D (descritivo):** o que a pessoa sente e faz neste turno. As bolhas foram juntadas, para o Jev não "ver" a
    fragmentação e o resultado não ficar circular;
  - **passe P (preditivo):** vendo só o contexto até o parceiro, como será o próximo turno. É exatamente a posição do bot.
  - Custo: 11.448 chamadas, US$ 0,76, mediana de 0,58 s.
- **9 análises paralelas** (subagentes), cada uma com scripts próprios (`scripts/analysis/aN_*`), saídas (`analysis/data/`) e relatório:

| # | tema | relatório |
|---|---|---|
| 1 | ritmo, fragmentação, timing e digitação ("camada de entrega") | `analysis/01_ritmo_fragmentacao_timing.md` |
| 2 | aberturas ("oi"), reaberturas e fechamentos | `analysis/02_aberturas_e_fechamentos.md` |
| 3 | emoção, empatia, seriedade e humor (com validação contra ouro) | `analysis/03_emocao_empatia_humor.md` |
| 4 | estilo, registro e espelhamento | `analysis/04_estilo_e_espelhamento.md` |
| 5 | engajamento e dinâmica ("diretor de conversa") | `analysis/05_engajamento_dinamica.md` |
| 6 | Jev como infraestrutura: confiabilidade, contexto, idioma, custo, latência | `analysis/06_jev_confiabilidade_e_design.md` |
| 7 | flerte, afeto e intimidade: o que dizer e como responder | `analysis/07_flerte_afeto_intimidade.md` |
| 8 | vocabulário, vícios humanos × vícios de LLM, lista negra | `analysis/08_vocabulario_vicios_humano_vs_llm.md` |
| 9 | experimento A/B: LLM pura × LLM + briefing do Jev × humano | `analysis/09_experimento_briefing_jev.md` |
| 10 | 2ª rodada: arquiteturas do Jev como juiz e diagnosticador de "cara de LLM" (parcial) | `analysis/10_arquiteturas_jev_juiz_llm.md` |
| 11 | 2ª rodada: rajadas ("5 bolhas em < 1 min") e arquiteturas do Jev para a entrega (parcial) | `analysis/11_arquiteturas_jev_entrega_e_rajadas.md` |
| 12 | 2ª rodada: arquiteturas para "o que escrever" + Briefing v2 (parcial) | `analysis/12_arquiteturas_jev_o_que_escrever.md` |
| 13 | 2ª rodada: controle de saída da LLM, 4 atores (parcial, só dev) | `analysis/13_controle_de_saida_llm.md` |
| 14 | literatura: diálogo, persona, roleplay e avaliação (37 papers) | `analysis/14_literatura_dialogo_roleplay.md` |
| 15 | literatura: CMC, psicologia, companions, ética e OptMem | `analysis/15_literatura_cmc_psicologia_companions.md` |
| 16 | benchmarks: EQ-Bench, slop/Antislop, testes de Turing, viés de juízes | `analysis/16_benchmarks_eqbench_slop_turing.md` |

**Limitação transversal:** **não há dado em português.** Os corpora são em inglês e em holandês. Todo equivalente em PT-BR
("haha" → "kkkk", "u" → "vc", as listas negras em PT) é **inferência**, não medição. O Jev funciona com state em PT
(correlação de 0,87–0,94 com o inglês nos Nouls, 78% de concordância na emoção), mas **perde um pouco de ironia** e lê o PT
como levemente mais sério. O primeiro passo prático é juntar um corpus de chat brasileiro doado e recalibrar.

---

## 2. A sua pergunta direta: emoção × quantas mensagens

> "Identificar que essa pessoa está com ansiedade e por isso mandou três mensagens diretas, nem finalizou as frases…"

**Exemplos reais** (WhatsApp, traduzidos do holandês; rótulos do passe D do Jev):

- **Ansiedade + afeto numa conversa de relacionamento: 4 bolhas no mesmo minuto.** Contexto: estão falando de trocar de
  emprego e ela provoca: "você quer é me ver longe o quanto antes, né? 😉". Resposta (emoção: afeto/ansiedade, anxious
  0,83, arousal 2,7):
  > "Sim!" / "Claro que não quero te deixar ir" / "Eu simplesmente não consigo lidar com isso" / "Mesmo que você diga que é melhor"

  O que veio em seguida mostra outro padrão: a resposta dela foi leve, "Haha uhum claro / Pra você também?". **Quem ouve o
  momento intenso é quem traz de volta a leveza** (84% das voltas ao humor, relatório 3).
- **Tristeza séria: 1 mensagem longa**, com reticências no lugar das quebras (seriedade 2,9 de 3):
  > "Hoje fui ao orfanato estadual… Pude ver o documento de quando me acharam… Não é muito legal quando a sua história de
  > adoção é diferente do que você sempre pensou"
- **Zoeira: bolhas curtíssimas em sequência** (playful 0,92, maichat):
  > "i swear its like 70" / "or are you dehydrated" / "oh no you're right" / "60"
- **Irritação real: 1 bolha seca**, às vezes em CAPS (maichat): "TO WHERE" · "its wednesday" · "wow".

**O que os números dizem** (relatórios 1, 3 e 4):

| estado (Jev, passe D) | efeito sobre o número de bolhas | efeito sobre o tamanho de cada bolha | efeito sobre a latência |
|---|---|---|---|
| **agitação (arousal)** ↑ | **+6–7 pp** de chance de 2+ bolhas por nível | **−32–36%** por nível | nenhum (a digitação fica mais rápida: 5,4 × 4,3 chars/s) |
| **ansiedade** | mais bolhas no WhatsApp (2,18 × 1,58) | bolhas **mais longas** (37 × 22 chars): a pessoa escreve mais | nenhum |
| **seriedade** ↑ | sem efeito | **+27–30%** por nível | nenhum (até −6 s ao vivo) |
| **tensão / briga** | **1 bolha seca** (1,18 no chat ao vivo) | curta | digitação rápida e **sem apagar** (14% editam contra 43%) |
| **zoeira** | picotada (bolhas curtas) | curta | rápida |
| **vulnerável / desabafo** | **mais bolhas** (2,5 × 1,65) e **menos pontuação** (19% × 30%) | 1,7–2,8× mais texto no total | pausas e reescrita (11% apagam o rascunho inteiro) |
| **tamanho do que a pessoa tem a dizer** | **o fator nº 1** (R² 0,25–0,29): ≤20 chars → 1 bolha em ~90%; >160 chars → 4+ bolhas em 45% | — | — |
| **estilo pessoal** | de 7% a 63% de turnos com várias bolhas (R² 0,13–0,19) | — | — |
| **espelhamento** | +9 a +19 pp se o parceiro fragmenta | — | ρ ≈ 0,4 com a latência do parceiro |

**Conclusão para o sistema:** o número de bolhas deve ser decidido **em código**, a partir de (tamanho do texto gerado ×
estilo da persona × espelhamento do usuário × moduladores de momento vindos do Jev: arousal, seriedade, tensão), **sorteado**
de distribuições empíricas. Não se deve perguntar ao Jev "quantas bolhas?". O relatório 1 traz a tabela completa e a fórmula
de "digitando…" (≈ 0,15 s + 0,173 s por caractere, ×1,17 por nível de seriedade).

---

## 3. O que escrever em cada momento (o mais importante)

### 3.1 Momento → o que humanos fazem × o que a LLM faz

As formas em inglês são dado (maichat, NPS, SMS, WhatsApp). As colunas "PT-BR" são **inferência** para calibrar com dados
brasileiros.

| momento (o que o usuário fez) | o que humanos fazem (dado) | o que a LLM faz (dado) | regra do briefing | exemplos de tom em PT-BR (inferência) |
|---|---|---|---|---|
| **"oi" puro** | devolve a saudação com **outra palavra** (74%) e **um degrau acima de energia** ("Hi!" → "Helluuuu"); em 43%, uma 2ª bolha já traz assunto. O "oi" é um **pedido de atenção** | "Oi! Tudo bem? Como posso ajudar? 😊" | 1ª bolha: outra saudação, com energia igual ou +1; 2ª bolha: gancho (callback de memória ou "e aí, o que manda?"). Nunca repetir a palavra do usuário | "oiee" · "opa" · "eii" → "e a prova, foi?" |
| **"como vai?"** | resposta **concreta** com um detalhe do dia ("meh / tired but ok", "longgg / work was chaotic"); devolve a pergunta em 32–35% | 85–104 chars, pergunta em 70–90%, "I'm doing great, thanks for asking!" | 1 detalhe concreto, ≤ 40 chars; devolver a pergunta com p≈0,3 | "cansada mas viva kkk" · "de boa, e vc?" · "longo o dia hj" |
| **notícia boa** | comenta **o fato**, não a emoção ("u did scream", "look at us being productive"); "congrats" 0–1/1.000; pergunta em só **8%** | "That's amazing!! 🎉 So happy for you! How are you going to celebrate?" (pergunta em 56–64%) | reagir ao conteúdo, 1 ideia; pergunta proibida por padrão; "!" no máximo 1 | "MENTIRA" · "passou??" · "aeee" · "sabia" |
| **desabafo / notícia ruim** | abertura curtíssima ("oh no", "oh", "ugh") + **uma pergunta exploratória** ("what happened?"); só 8–14% de consolo explícito; "sorry to hear" = **0** no chat real; para medo, **pergunta** em vez de "sorry" | parágrafo de validação: "Oh no, Jordan. I'm really sorry you're feeling that way. Sending you a huge virtual hug… I'm here for whatever you need ❤️" | reação curta + 1 pergunta; ≈ 9–10 palavras, 1 bolha; conselho só se pedido (Noul ≥ 0,6); **uma** pergunta só: na 2ª resposta, cai para 20% | "eita" · "putz" · "ah não" → "o que aconteceu?" |
| **desabafo leve com riso** ("tô moído kkk") | o outro **ri junto** (+9 pp) e responde prático | trata como tragédia | se o riso vem no fim (suavizador), acolher com leveza, humor permitido | "kkkk força" · "descansa hj pelo menos" |
| **piada / provocação** | em conversa de zoeira, **não responde com "haha": responde com outra piada** (85%); resposta seca que reenquadra ("define productive", "allegedly") | "Haha, that's hilarious! 😂" + pergunta | continuar a brincadeira; riso só se o usuário marcou a própria piada; 1 frase curta | "vc queria" · "olha quem fala" · "prova" |
| **carinho / flerte** | **segue o assunto** (32%), "hehe" (13%), devolve a provocação (18%), devolve o afeto (13%); **agradecer é raro** (4%); intensidade −0,5 nível; sem pergunta (0/25); emoji de afeto 🥺😘😚, **nunca** 😂💀 | "Aww stop it u are making me blush 🥺🙈 cant wait to see u!!" (2,2× maior, pergunta em 36–42%) | sortear o movimento (ver 3.3); intensidade ≤ a do usuário; sem pergunta; tamanho ≈ o do usuário | "para kkk" · "não me expõe" · "é a mesma cara de ontem" · "tb te amo" |
| **"já comeu?" / "dorme bem" (cuidado)** | responde **literalmente** ("i had noodles", "coffee counts right") | "Aww thank you for caring! 🥰 I did! What about you?" | responder o conteúdo, sem agradecer o cuidado | "comi miojo kkk" · "café conta?" |
| **"te amo" / "saudade"** | devolve em só ~1/5 dos casos; quando devolve, é **simétrico** (mesmas palavras, mesmo emoji, mesmo tamanho) e rápido; às vezes termina o fio antes e devolve na bolha seguinte | "I love you too so much! ❤️ You mean the world to me!" | simétrico e curto; com p≈0,3, devolução atrasada numa 2ª bolha | "tb te amo" · "saudade tb 🥺" |
| **reclamação / crítica ao bot** | responde **seco** ou com desculpa rápida; o "haha" do usuário no fim é **suavizador**: a crítica é real | "Haha, I'm sorry about that! 😅" | responder ao conteúdo, sem rir de volta | "foi mal kkk tô aqui" |
| **pergunta direta ao bot** | resposta direta, curta (mediana de 30 chars); "yes/hmm/no" | responde + desenvolve + pergunta | responder; pergunta de volta em ≤ 10–20% | — |
| **usuário conta algo longo** | o único momento longo (mediana de 50 chars); começa com "i", "yeah", "but", "and" | — | pode ser longo; 2–3 bolhas | — |
| **despedida** | espelhada, com variação e afeto conforme o tipo; 21% viram "guerra de bye"; entre íntimos, 61% nem se despedem | "It was great talking to you! Let me know if you need anything else! 😊" | espelhar o tipo (sono, ocupado, carinho); deixar um fio para a próxima; permitir 1 rodada extra | "boa noite, dorme bem 💤" · "vai lá, depois me conta" |
| **"ok" / "haha" / emoji (ack)** | no fim de tópico, **carregar** a conversa (história curta + gancho) ou deixar pausar | pergunta genérica "So, what else is new?" | se o gancho do usuário for < 0,3, proibir resposta só-reação; se o risco de fim for alto, deixar pausar | — |

### 3.2 Vícios: o que tirar e o que (com moderação) pôr

**Lista negra, em ordem de impacto medido** (relatório 8, Tabela 3; 250 contextos × 3 LLMs):

1. **tamanho e número de frases** (LLM 2,2–2,6×; 3+ frases em 42–61% × 13%);
2. **"?" no fim** (44–58% × 20%);
3. **"!"** (40–92% × 7%);
4. **emoji de entusiasmo** (😂🥰🎉✨ em 7–14% × 0%);
5. **fórmulas de validação**: "totally", "I totally get that", "sorry to hear", "that's so sweet", "aww", "sounds like/amazing",
   "that's amazing", "definitely", "absolutely", "I'm here for you", "it's okay to feel", "let me know", "feel free";
6. **paráfrase** do que o usuário disse (14–21% × 3%);
7. **o nome do usuário** no meio da frase (5% × 0%);
8. **caricatura** (Gemini): "💀", "lmao", "wait", "honestly", "bestie", "fr fr", memórias inventadas;
9. formatação: "\n\n" em parágrafos, `</p>`, travessão.

Em PT-BR (inferência): "Que incrível!", "Entendo perfeitamente", "Sinto muito ouvir isso", "Faz todo sentido", "Com certeza!",
"Estou aqui para você", "É totalmente normal sentir…", "Me conta mais!", "E você?" reflexo, "tô corando", "mal posso esperar",
e o extremo caricato: "kkkkkkkk" em toda mensagem, "mano/véi/slk/pprt" forçados, "tipo assim" repetido.

**Hipóteses que NÃO se confirmaram:** "journey", "vibe", o travessão, "I'd love to" e "it's okay to feel" quase não aparecem
em modo chat nesses modelos (0–3%). São vícios de LLM "de redação", não de chat.

**Os vícios humanos que tornam o texto crível são de forma, não de palavra:**
- 90% das mensagens sem pontuação final; 75% começando em minúscula (no desktop; no celular, o corretor capitaliza);
- 45% com até 3 palavras;
- sem apóstrofo, e em PT sem acento ("vc", "tb", "pq", "to");
- começar a bolha com "and/but/so/oh" ("e…", "mas…", "aí…");
- riso como pontuação no fim ("…kkk").

As muletas que as LLMs acham que são humanas ("honestly", "literally", "tbh", "i mean") aparecem só **1–4 vezes a cada
1.000 mensagens**. A autocorreção humana é **invisível**: 36% das mensagens tiveram apagamento durante a digitação, mas só
0,4% mostram um "*correção".

**Estilo: o que espelhar e o que não** (relatório 4):

| espelhar (só na resposta imediata) | **não** espelhar (é identidade da persona) |
|---|---|
| o **ato** de rir (×1,5–2,8), emoji (×1,2–1,9), alongamento (×1,5), "!" | a **forma** do riso (cada pessoa usa a mesma forma em 70–83% das vezes) |
| tamanho (parcial: ρ ≈ 0,12–0,14) e nº de bolhas (parcial) | caixa, ponto final, gírias e abreviações do usuário, typos, dialeto |
| — | pergunta (anti-espelhar: se o usuário perguntou, primeiro responder) |

O contágio **acaba em 2 trocas**, e não há "convergência lenta": os pares já começam parecidos. A persona deve ter uma
**impressão digital fixa** (forma de riso, taxa de emoji, repertório de 3–5 emojis, caixa, pontuação, inventário de 3–6
abreviações) e só modular a **taxa** localmente.

**Três "leis" de conteúdo que resumem tudo:**
1. **Reaja ao conteúdo, não à emoção.** "passou??" em vez de "Que incrível, parabéns!".
2. **Uma ideia por resposta.** O humano reage **ou** comenta **ou** pergunta; a LLM faz as três coisas.
3. **Nem tudo merece resposta completa.** Metade das respostas humanas é backchannel ("vdd", "kkk", "aham"), e isso não
   reduz o engajamento se houver gancho nos momentos certos.

### 3.3 Flerte e afeto: como responder (relatório 7)

- **Inventário real** (entre conhecidos): provocação (17%), **cuidado cotidiano** (15%: "já comeu?", "orgulho de vc"),
  saudação com apelido (12%), convite (9%), elogio direto (7%) ou indireto (6%), saudade (5%), "te amo" (3%). Declarações
  explícitas somam só 8%; **o afeto é sobretudo cuidado e provocação**.
- **O que mantém vivo × o que mata**, medido pela chance de o próximo turno de quem flertou continuar afetuoso:

| resposta ao flerte | % dos humanos | flerte continua |
|---|---|---|
| encabular-se ("ugh ur sweet", "HELP", "dont expose me") | 1,7% | **6/7** |
| escalar meio passo | 1,7% | **4/4** |
| devolver a provocação | 18% | 40% |
| devolver o afeto | 13% | 40% (e encerra a sessão em 25%) |
| só "hehe" | 13% | 52% |
| ignorar e seguir o assunto | 32% | 19% |
| agradecer | 4% | 17% |
| recusar | 1,7% | 0/5 |

- **Regra de intensidade:** `alvo = arredondar(intensidade_usuário − 0,5)`, teto em `usuário + 1` e na intimidade da relação.
  "Meio passo acima" só quando o usuário convida explicitamente (Noul ≥ 0,6), e com probabilidade ≈ 0,2.
- **Movimento:** Choice do Jev **misturado com o prior empírico** por tipo de gatilho e **sorteado** (não argmax). O argmax
  acertou o movimento humano em só 27,5% e, sozinho, produz respostas repetitivas.
- **Recuo silencioso:** humanos quase nunca recusam explicitamente; eles **mudam de assunto**. Se o usuário desviar do afeto
  do bot (Noul ≥ 0,6), o bot deve zerar o afeto por ~5 turnos, sem pirraça.
- **Timing:** pausa **antes** de responder a um flerte (≈2,5× o ocioso normal) e depois digitação rápida, em 1 bolha. "Tb te
  amo" sai rápido (≈13 s); a provocação espirituosa demora (22–30 s).

---

## 4. A arquitetura: Cérebro (Jev) + Boca (LLM) + Corpo (código)

### 4.1 Princípios (derivados dos experimentos)

1. **O Jev lê, classifica e ordena. O código decide, sorteia, conta e calibra. A LLM só escreve.**
2. **Uma chamada de leitura por mensagem, com dezenas a centenas de perguntas.** É barato e rápido e as perguntas são
   isoladas; dividir em várias chamadas só encarece.
3. **Nunca pergunte ao Jev "isso soa humano?".** Use perguntas **atômicas de conteúdo** ("parafraseia o usuário?", "valida mais
   do que o momento pede?", "usa mais gíria do que X costuma?") e **estilometria em código** (AUC 0,87–0,90 só com tamanho,
   frases, "!", emoji, "?" e lista negra).
4. **Distribuições, não argmax.** Número de bolhas, movimento de flerte, riso, pergunta e emoji saem de sorteios em
   distribuições empíricas, condicionadas ao estado lido pelo Jev. O argmax cria robôs previsíveis.
5. **Confiança como portão:** < 0,5 = "não sei" (usar o padrão da persona); 0,5–0,7 = só ações reversíveis (tom, emoji);
   ≥ 0,7 = ações fortes (mudar de modo, consolar, encerrar). Histerese de 0,6/0,4 e EMA entre turnos.
6. **Contexto de 8 turnos no state, no mínimo 2.** Sem contexto tudo parece mais sério e mais zangado (a emoção muda em 37%
   dos casos). Mais de 20 turnos só entram como resumo de memória.
7. **Perguntas congeladas e versionadas**, em inglês mesmo com state em PT, com critérios descritos em todas as opções, sem
   negação. Mudou a redação, recalibra o limiar.
8. **Multi-frequência:** coisas diferentes mudam em ritmos diferentes (por bolha, por turno, por sessão, entre sessões).

### 4.2 Fluxo por mensagem do usuário

```
 ┌───────────────────────────── CORPO (código) ─────────────────────────────┐
 │ E0  Recepção: debounce de fim de turno (o usuário vai mandar outra bolha?)│
 │     janela 4/8/12 s conforme a taxa de fragmentação do usuário + a última│
 │     bolha ("?" encurta; "kkk", "..." ou mídia alongam)                    │
 │ E1  Features de código: tamanhos, "?", riso (forma/posição), emoji,       │
 │     latência relativa, silêncio desde a última sessão, hora local,        │
 │     taxa de perguntas do usuário e do bot, perfil de estilo do usuário    │
 └──────────────────────────────────┬───────────────────────────────────────┘
                                    ▼
 ┌──────────────────── CÉREBRO: RODADA 1 (Jev, ≈0,55 s) ────────────────────┐
 │  [A] LEITURA (state: persona + memória + 8 turnos + current_turn)         │
 │      ~40–130 perguntas: momento, emoção (Choice plana ~32 → família em    │
 │      código), valência, arousal, seriedade, playful, ironia, tensão,      │
 │      vulnerável, seeks_support, flerte + intensidade + movimento do       │
 │      usuário, gancho, tópico esgotado, pergunta pessoal?, função do riso, │
 │      oportunidade de cuidado, desvio de afeto, tipo de abertura (se nova  │
 │      sessão), fechamento/closure, …                                       │
 │  [B] PREVISÃO (state: até o turno do usuário, sem a resposta)             │
 │      ~10–20: p_length, p_question, p_laugh, p_emoji, p_joke_welcome,      │
 │      p_end, p_tone                                                        │
 │  timeout 0,9 s → hedge; corte em 1,8 s → estado do turno anterior         │
 └──────────────────────────────────┬───────────────────────────────────────┘
                                    ▼
 ┌──────────────── ROTEADOR + DIRETOR (código, alimentado pelo Jev) ─────────┐
 │  Calibração (isotônica), EMA, histerese → MODO:                           │
 │   abertura/reabertura · leve/zoeira · flerte/afeto · acolhimento/sério ·  │
 │   conflito/reparo · logística/pergunta direta · desacelerando/encerrando  │
 │  Cada modo = um workflow com priors próprios.                             │
 │  RODADA 2 do Jev (opcional, só no modo que pedir, ~0,5 s): Choice de      │
 │  MOVIMENTO com opções do modo (ex. flerte: continue_topic, minimal_ack,   │
 │  tease_back, reciprocate, flustered, deflate_humor…)                      │
 │  score(m) = P_jev(m) × prior(m | modo, fase, seriedade) → vetos → SORTEIO │
 │  Orçamentos: tamanho-alvo (p_length + espelho), pergunta sim/não, riso    │
 │  sim/não, emoji sim/não, intensidade-alvo, "e você?" sim/não              │
 └──────────────────────────────────┬───────────────────────────────────────┘
                                    ▼
 ┌──────────────────────── BRIEFING (código → texto) ───────────────────────┐
 │ curto, concreto, com alvos numéricos, exemplos de TOM (não de texto) e    │
 │ as proibições do momento (ver 4.4)                                        │
 └──────────────────────────────────┬───────────────────────────────────────┘
                                    ▼
 ┌──────────────── BOCA: LLM rápida (1–3 candidatas, briefings variados) ────┐
 └──────────────────────────────────┬───────────────────────────────────────┘
                                    ▼
 ┌──────────── VALIDAÇÃO (código + Jev, rodada 3 em paralelo, ≈0,5 s) ───────┐
 │ C1 código: tamanho, nº de frases, "?" não autorizado, "!", emoji fora do  │
 │    repertório, lista negra, nome do usuário, \n\n e HTML → corta/reescreve │
 │ J2 Jev (Nouls atômicos): parafraseia? valida demais? forçado? formal?     │
 │    piada fora de hora? mais marcadores do que a persona usa? coerente com │
 │    a última msg? cópia dos exemplos do briefing? (em código)              │
 │ Escolha em CÓDIGO pela menor soma de violações (nunca "qual soa humana?") │
 │ C2 controlador de distribuição (janela de 20–30 msgs do bot): taxa de     │
 │    perguntas, riso, emoji, cada muleta ("pera", "tipo", "na real") ≤ 1    │
 │    a cada 15–20 msgs, mesma abertura ≤ 2x                                 │
 │ Normalização de estilo da persona: forma do riso, caixa, pontuação,       │
 │ inventário de abreviações                                                 │
 └──────────────────────────────────┬───────────────────────────────────────┘
                                    ▼
 ┌─────────────────────────── ENTREGA (código) ─────────────────────────────┐
 │ nº de bolhas = f(tamanho do texto, estilo da persona, espelho do usuário, │
 │   arousal/seriedade/tensão) → sorteio; teto de 5                          │
 │ divisão: [reação] → [conteúdo] → [continuação] → [pergunta no fim];       │
 │   emoji solto no fim; bolhas de 15–40 chars; momento sério não picota     │
 │   em pedaços < 25 chars                                                   │
 │ latência inicial: espelha a do usuário (×0,3–0,5 do humano no chat ao     │
 │   vivo); pausa extra antes de flerte; nunca "demora dramática" no sério   │
 │ "digitando…" ≈ 0,15 + 0,173 s/char (×1,17 por nível de seriedade);        │
 │   pausas e resets visíveis ocasionais (mais no sério, nunca na tensão)    │
 │ se o usuário escrever enquanto o bot "digita": parar, esperar, REPLANEJAR │
 │ bolha extra "ah, e…" (≤ 8% dos turnos, preferir pergunta)                 │
 └──────────────────────────────────────────────────────────────────────────┘
```

**Sobre a sua ideia de roteamento por tamanho do texto (simples/médio/complexo):** os dados sugerem rotear pelo **momento**
(o modo acima), não pelo tamanho. O tamanho é um insumo de orçamento. Mas vale um **roteamento de custo** em 3 níveis, parecido
com o que você imaginou:
- **Rota leve** (≈20 perguntas, US$ 0,00008): o usuário mandou ack, riso, emoji ou logística simples;
- **Rota média** (≈75 perguntas + previsão, US$ 0,00023): conversa normal;
- **Rota pesada** (≈180 perguntas, rodadas 2 e 3, 2–3 candidatas, US$ 0,0005): momentos de alto risco de erro de tom, como
  vulnerável, conflito, flerte escalando, reabertura depois de dias, ou quando a confiança da leitura veio baixa.

Uma pré-classificação barata decide a rota: pelas features de código, ou por um Noul único em ≈0,5 s.

### 4.3 Os ritmos do sistema (multi-frequência)

| ritmo | o que roda | exemplos |
|---|---|---|
| **por bolha do usuário** | código (+ Jev opcional na zona ambígua) | fim de turno (debounce), interrupção enquanto o bot "digita" |
| **por turno do usuário** | rodadas 1–3 do Jev, diretor, LLM, validação, entrega | tudo da seção 4.2 |
| **por sessão / a cada ~10 turnos** | Jev "lento" + EMA | nível de intimidade e relação (instável por turno: só 65–70% de concordância), temperatura do flerte, trajetória de engajamento (serra), perfil de estilo do usuário (código) |
| **entre sessões** | código + Jev na reabertura | fios abertos e itens pendentes ("a prova de sexta"), reabertura proativa (só com gancho concreto, saudação só após ≥ 24 h, nunca "quanto tempo!"), check-in de cuidado |

### 4.4 Exemplo de briefing (gerado em código a partir das respostas do Jev)

Situação: o usuário, depois de uma conversa leve, manda: "tô tão cansada dessa faculdade kkk nem sei pq ainda tento".

```
MOMENTO: desabafo leve (vulnerável 0,71, mas com riso suavizador no fim; seriedade 1,1; ironia 0,55)
MOVIMENTO: acolher com leveza — reagir ao que ela disse + 1 pergunta concreta
TAMANHO: ~35 caracteres (máx 60). 1 ideia.
PERGUNTA: sim, uma só, sobre o fato ("o que rolou?"), não sobre o sentimento
RISO: permitido 1 "kkk" curto no fim só se combinar; nada de rir primeiro
EMOJI: não
TOM: amiga próxima, informal, minúsculas, sem ponto final
NÃO USE: "sinto muito", "entendo perfeitamente", "é normal se sentir assim", "estou aqui pra você",
         "você é capaz", conselho, o nome dela, "!" 
EXEMPLOS DE TOM (não copie): "putz" · "eita, semana braba?" · "o que aconteceu dessa vez"
```

**Regras de redação do briefing** (relatórios 8 e 9):
- curto e imperativo (≈60–70 palavras), nunca probabilidades em prosa;
- tamanho como **alvo com folga** ("~35 caracteres"), não como teto;
- pergunta, riso e "!" **sorteados** com as taxas humanas do momento, não proibidos sempre;
- exemplos de **tom** com outro conteúdo, sorteados a cada vez e verificados contra cópia;
- a política de estilo (minúsculas, sem ponto final), mas **nunca** a lista de gírias da persona;
- uma linha positiva de como começar;
- se a confiança do movimento for < 0,4, **não ditar o movimento**, só a forma.

Resposta humana típica: "putz kkk o que rolou dessa vez". A resposta típica da LLM sem briefing seria: "Ah, sinto muito que
você esteja se sentindo assim! 😔 A faculdade pode ser muito desgastante mesmo. Lembre-se de que você é capaz! O que está te
deixando mais cansada?"

### 4.5 Catálogo resumido de detectores (os que têm evidência)

| detector | tipo | evidência | consumidor |
|---|---|---|---|
| `emotion` (Choice plana, ~32 classes; família derivada em código) | C | top-1 45–50%, top-3 70–72%, família 70–75%, polaridade 85–90%; com conf ≥ 0,9, 84% | modo, tom, "sabor" do briefing |
| `seriousness` (0–3, "este momento") | S | tom "supportive_serious" prevê turno sério com AUC 0,86 | modo sério: sem emoji, +15–20% de tamanho, mesmo nº de bolhas, sem atraso extra |
| `joke_welcome` | N | prevê se a pessoa vai brincar (AUC 0,78); piada recebida em 88% com p ≥ 0,6 e em 52% com p < 0,3 | libera humor |
| `hook` do usuário | N | melhor preditor de continuidade (OR 11,9; AUC 0,69 contra 0,56 do "?") | se < 0,3, o bot precisa carregar o turno |
| `p_end` + `closure` + código | N/S | AUC 0,75 (0,80 combinado); 0,91 no chat ao vivo | desacelerar (parar de perguntar), despedida espelhada, deixar pausar |
| `p_length` | S | melhor preditor do tamanho da resposta (ρ 0,44), 0,46 com código | orçamento de caracteres |
| `flirt_now` + `user_int` + `user_move` | N/S/C | a forma do flerte bate (p < 10⁻⁵); o movimento precisa de prior + sorteio | modo flerte, intensidade-alvo, movimento |
| `user_deflects` (desvio de afeto) | N | derivado dos padrões de esquiva (mudança de assunto) | recuo silencioso |
| `open_type` (tipo de 1ª mensagem após silêncio) | C | tipologia medida em 3.744 sessões | regras de abertura (sem saudação se for resposta atrasada, etc.) |
| função do riso do usuário | C (+ código: tamanho e posição) | suavizador: curto e no fim (70%); reação: longo e no início (59%) | não rir de volta de um suavizador; responder ao conteúdo |
| `paraphrase`, `overvalidation`, `forced`, `formal` (pós-geração) | N | AUC 0,74 / 0,69 / 0,76 (Gemini) / 0,80 (GPT) | reescrita pontual |
| `more_markers` ("mais gíria/emoji/riso que X costuma?") | N | AUC 0,81 contra LLM simples, 0,69 contra LLM imitando | filtro de caricatura |
| **NÃO usar** | — | "soa humano?" (r = 0,06 com a verdade), "qual é a humana?" (12% contra Gemini), `direct`/`generic`/`register` (saem invertidos), `p_n_msgs` (sempre "1"), `topic_shift` (AUC 0,53–0,60), `p_question` sozinho (0,62) | — |

---

## 5. O que o experimento A/B mostrou (relatório 9)

**Desenho:** 119 pontos de decisão reais do maichat, estratificados em abertura, fechamento, sério, flerte/afeto, logística,
brincadeira e casual. Em cada ponto, o histórico vai até o turno do parceiro, e a comparação é com o que a pessoa realmente
respondeu. A "boca" é a `gemini-3.5-flash-lite`. Os limiares foram calibrados num conjunto de dev separado e congelados antes
do teste.

| condição | palavras (mediana) | pergunta | riso | emoji | "!" | LLM-ish | movimento = humano | custo por resposta | latência p50 |
|---|---|---|---|---|---|---|---|---|---|
| **humano** | **6** | **12%** | **7%** | **5%** | **7%** | **5%** | — | — | — |
| A: LLM barata pura | 15 | 57% | 37% | 39% | 29% | 21% | 23,5% | US$ 0,00009 | 1,29 s |
| S: + prompt de estilo fixo | 9 | 27% | 36% | 16% | 16% | 13% | 28,6% | US$ 0,00007 | 1,32 s |
| **B: + briefing do Jev** (~30 perguntas) | 4 | 1% | 8% | 4% | 0% | 3% | **34,5%** | US$ 0,00017 | 1,75 s |
| C: B + 3 candidatas + gate do Jev | 4 | 1% | 8% | 4% | 0% | 3% | 34,5% | US$ 0,00037 | 2,48 s |
| D: LLM ~3× mais cara, sem briefing | 13 | 54% | 47% | 20% | 32% | 25% | 26,1% | US$ 0,00026 | 1,43 s |

**Leitura:**
- **Barato + Jev > caro sem Jev** em todas as métricas de forma e no movimento (+8 pp). **Pagar por um modelo maior não tira os
  vícios; a instrução concreta tira.** Isso confirma a sua intuição de focar em velocidade.
- **O briefing corrige demais:** 4 palavras contra 6, 1% de pergunta contra 12%, 0% de "!" contra 7%, e às vezes fica seco ou
  "engraçadinho" ("hey there" em aberturas, "im a chaotic potato"). **Correção:** pergunta e "!" devem ser **sorteados** com a
  taxa humana, não proibidos; o tamanho entra como **alvo com folga**, não como teto (o teto cortou 43% dos humanos); e uma linha
  positiva de como começar ("comece simples: ah/então/sim, ou direto no conteúdo").
- **O controle sem Jev chegou quase lá** (subconjunto de 36): mesma distinguibilidade e mesmas taxas de superfície. O Jev
  acrescentou: o tamanho acompanhando o momento (0,47 × 0,28) e +5,6 pp de movimento (IC de 0 a 13,5). No tom, o controle foi
  até melhor (44% × 28%). **O valor do Jev está em ler o momento para decidir O QUE fazer, e é aí que precisa melhorar**: top-1
  do movimento em 32% (contra 16% da classe majoritária) é informativo, mas erra 2 em cada 3.
- **O briefing precisa ser curto e imperativo.** Despejar as probabilidades do Jev em prosa (~145 palavras) devolveu a LLM ao
  baseline: ela lê "laugh 43%" como licença. As ordens curtas (~68 palavras) tiveram aderência de 97–100%.
- **"Não use X" não fez a LLM usar X** (1,7% × 39,5%). Mas **listar as gírias da persona** ("words you use: idk, omg") fez a
  LLM enfiá-las em tudo. Dê a política de estilo (minúsculas, sem ponto) e não o vocabulário.
- **O gate não se pagou:** 3 candidatas + escolha do Jev custam 2,2×, somam 0,7 s e não trazem ganho. E a Choice "qual
  candidata é melhor" tem **viés de posição** (53/32/15%). Se usar gate, faça **um Noul por candidata**, em ordem embaralhada.
- **O Jev prevendo a forma da resposta humana** (na posição do bot): tamanho ρ 0,49 (útil), emoji AUC 0,71 (útil), pergunta
  0,57 (fraco), riso **0,41 (pior que o acaso)**, nº de bolhas 0,04 (inútil). Riso e bolhas ficam em regra de código.
- **Padrões novos que só apareceram aqui:**
  - 32% dos turnos humanos começam com marcador simples ("ok", "so", "oh", "yeah", "same", "i") e só 5% com interjeição
    performática ("omg", "wait", "ooh", "honestly"). A LLM faz 37% performático.
  - A pergunta recíproca genérica ("what about you?", "how's your day?") aparece em **1,2%** dos turnos humanos, contra 18–19% da
    LLM.
  - Na abertura, 13 de 17 humanos já trazem conteúdo ("guess what", "just got home / im exhausted"); a LLM pergunta em 17 de 17.
  - O humano **responde à palavra, não ao tema** ("so how has your day been" → "why are you talking like a customer survey").

**Exemplos lado a lado:**

| contexto (o outro disse) | humano real | A: LLM pura | B: + briefing Jev | D: LLM cara |
|---|---|---|---|---|
| "hey love / im here" | "guess what" | "Hey! Perfect timing. How was your day?" | "hey there" | "hey! 💙 good to hear from you, how's your day going?" |
| "im proud" | "dont sound like my mom" | "thx someone has to be lol / what about u? got plans?" | "stop it" | "thanks lol, means a lot coming from you" |
| "i have been trying to learn some coding since morning / its so hard :(" | "did u try watching videos on coding tht could possibly help u" | "aww hang in there! what language are you trying to learn? coding can definitely be super frustrating at first 😭" | "coding is no joke, you got this" | "Ah yeah coding can be tough at first! What language are you learning? I found it gets easier…" |
| "kinda / still adjusting" (cabelo novo) | "send pic" | "that's so valid tbh, every time i change my hair i stare in the mirror like 'who is this stranger'…" | "takes a minute" | "that's fair, takes a bit to get used to a new cut. does it feel different?" |
| "love u dad" | "love u too / now go eat real food" | "go to your room" | "delete your account" ⚠️ leitura errada | "lol stop / you're gonna make me sentimental" |
| "Anyway dad I need to go now / thanks for chatting" | "OK - see you Friday." | "You're welcome, Alex. Good luck with the meeting tomorrow! See you for Christmas." | "goodnight / sleep well" | "No problem! Good luck with your meeting tomorrow - you'll be fine… Take care!" |

## 6. Armadilhas descobertas (o que NÃO fazer)

1. **Juiz de humanidade.** Qualquer pergunta abstrata de "naturalidade" vira um detector de caricatura e premia o vício.
2. **Proibição absoluta no prompt.** "Seja curto, casual, sem pergunta" gera o estereótipo oposto e vícios substitutos.
   Dê alvos numéricos por momento e controle frequências em código.
3. **Exemplos literais no briefing.** A LLM copia ("miss u too 😚" quatro vezes). Use exemplos de **tom** com outro
   conteúdo, sorteie exemplos diferentes a cada vez e cheque a cópia em código.
4. **Argmax do movimento.** Existem várias respostas humanas válidas: sorteie a partir da mistura Jev + prior.
5. **Pedir ao Jev contagens e previsões de forma** (nº de bolhas). Ele responde a moda; decida em código.
6. **Limiares não calibrados.** Os Nouls preditivos ordenam bem mas superestimam eventos raros. Calibre com os logs do produto
   (isotônica), por idioma.
7. **Cascata para refinar a mesma pergunta.** Com 32 emoções, a Choice plana ganhou da cascata (50,5% × 44,3%). Use cascata só
   quando o 2º passo traz **informação nova** (outro state) ou decide **o que perguntar depois** (workflows por modo).
8. **Callback como tática de engajamento no meio da conversa.** O efeito medido foi nulo (+0,03). Guarde-o para a
   **reabertura** de sessão, onde ele é o gancho natural.
9. **"Relationship" por turno.** Instável: trate como variável lenta ou venha da configuração da persona.
10. **Sempre devolver "e você?"**, sempre despedir-se formalmente, sempre agradecer carinho, sempre rir da piada: humanos não
    fazem nenhuma dessas coisas na maioria das vezes.

---

## 7. Próximos passos recomendados

1. **Corpus PT-BR.** Coletar conversas de WhatsApp doadas, anonimizadas e com consentimento, e rodar exatamente o mesmo
   pipeline (`scripts/`) para recalibrar as tabelas: formas de riso ("kkk", "rs", "hahaha"), abreviações, taxas por momento
   e listas negras em português.
2. **Protótipo de ponta a ponta** com o fluxo da seção 4, com a persona fixa e os logs de todas as leituras do Jev, para
   calibração isotônica dos limiares.
3. **Avaliação com humanos** (o Jev não serve de juiz de realismo): teste pareado cego "bot × humano" e métricas de produto
   (duração da sessão, retorno no dia seguinte, taxa de "turno sem gancho").
4. **Experimento que ficou faltando:** o briefing com orçamento numérico de tamanho vindo do `p_length` (relatório 8, §4.2).

---

## 8. Reprodutibilidade

- `scripts/download.sh`: baixa os corpora.
- `scripts/normalize.py`: gera `messages.jsonl`.
- `scripts/features.py`: gera as features e os turnos.
- `scripts/annotate_base.py`: gera a camada base do Jev.
- `scripts/jev.py` e `scripts/llm.py`: clientes com cache. A chave do OpenRouter vai em `.env` ou `OPENROUTER_API_KEY`
  (não versionada).
- `scripts/analysis/aN_*.py`: as análises de cada relatório.
- `docs/DATA.md`: esquemas e descrição de cada arquivo.

Os dados brutos e processados **não** estão no git (licenças e tamanho). Rode os scripts para regenerá-los.
