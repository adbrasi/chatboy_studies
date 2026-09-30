# 18 · Estado da relação vivo: cascata Jev 1 → Jev 2 → física em código

> **RASCUNHO EM ANDAMENTO** (retomada após reinício do container). As seções 1–4b estão com resultados finais; 4c, 5, 6
> e 7 estão sendo preenchidas.

Prefixo `c2`. Scripts em `scripts/analysis/c2_*.py`; saídas pequenas em `analysis/data/c2_*`; grandes em
`data/processed/c2_*` (fora do git). Modelos de LLM usados: só os 4 permitidos (`openai/gpt-6-luna` como 2º rotulador;
`~deepseek/deepseek-flash-latest` para redigir as falas dos cenários; os 4 como atores nos experimentos 5 e 7).

## 1. Desenho

### 1.1 Taxonomia (`c2_common.py`)

**8 dimensões** do personagem C em relação ao usuário U, cada uma em [0, 1] com linha de base própria:
`trust` (confiança), `comfort` (conforto/proximidade), `affection` (afeto/interesse romântico), `resentment`
(ressentimento/irritação), `jealousy` (ciúme), `respect` (respeito/admiração), `protectiveness` (proteção/preocupação),
`playfulness` (cumplicidade brincalhona).

**18 eventos** (Noul "olhando só a `user_message`", mais um de contexto):
pedido de desculpa, piada de mau gosto, insulto/crítica séria, faz promessa, cumpre promessa, cancela/quebra promessa,
explica o sumiço, elogio, vulnerabilidade, menciona outra pessoa (ciúme), cuidado prático ("comeu?", "chegou bem?" —
o afeto como cuidado cotidiano do relatório 07), defensividade (relatório 11), mentira percebida, desdém, expressa afeto,
provocação amigável, gratidão, interesse pelo personagem; e `left_waiting` ("a última mensagem de C ficou sem resposta?",
usado com o intervalo de tempo para detectar sumiço).

Mais os **Nouls por item** (fan-out, princípio "Noul por item em vez de Choice"): para cada pendência `unresolved_issues[i]`,
"a mensagem pede desculpa/repara *esta* pendência?" e "repete/agrava *esta* pendência?"; para cada promessa
`open_promises[i]`, "cumpriu?" e "quebrou?". É isso que permite o AND em código: *a desculpa só reduz o ressentimento se
houver pendência e se ela se referir à pendência*.

### 1.2 Arquiteturas do Jev 1 comparadas

| código | o que é |
|---|---|
| DIM | 16 Nouls, um por dimensão × direção ("esta mensagem aumenta o ressentimento de C?" / "…diminui?") |
| BIP | 8 Scores bipolares ("como a mensagem muda X?" {baixa claramente … sobe claramente}) |
| EVT | 18 Nouls de evento → mapa evento→dimensões em código (noisy-OR ponderado, `EVT_MAP`) |
| COMB | DIM + EVT: média, máximo, ou regressão logística agrupada ajustada no dev |

Cada uma **sem** o estado (state = 8 turnos + mensagem + intervalo) e **com** o estado (mais o bloco
`relationship_of_C_toward_U` em níveis descritivos, `unresolved_issues`, `open_promises`, `shared_history`).

### 1.3 Jev 2 (magnitude), só para as dimensões ativadas

- **Score descritivo** {nada, leve, moderado, forte, marcante} (cada nível com uma frase: "leve: um efeito pequeno e
  passageiro", "forte: um efeito real que duraria dias"…);
- **Choice de números** 0, 10, 20…100 ("em quantos pontos, numa escala 0–100…") — a ideia original do usuário;
- **só Jev 1**: magnitude inferida da probabilidade do Noul de direção.

### 1.4 Física em código (`RelState`, `physics_step`)

nível → delta (0 / 0,03 / 0,07 / 0,15 / 0,25) × taxa da dimensão (inércia: a confiança cai 1,5× e sobe 0,5×) ×
saturação suave (`min(1, 2(1−v))` ao subir, `min(1, 2v)` ao descer); decaimento exponencial para a linha de base com
meia-vida por dimensão (ressentimento 72 h, ciúme 24 h, cumplicidade 12 h, confiança/respeito não decaem); piso do
ressentimento enquanto houver pendência; desculpa repetida pela mesma pendência vale 0,5^(n−1); sumiço = intervalo
(código) AND C ficou esperando (Jev) AND não explicou (Jev); modos com histerese (frio, ciumento, na defensiva,
preocupado, caloroso, brincalhão). Detalhe completo na §8 (especificação final).

## 2. Jev 1 nos dados reais (validação 4a)

**Dados.** 30 janelas ricas em eventos de relação (14 do maichat, 16 do whatsapp_nl), escolhidas pelo `jev_base` +
regex (desculpa, ciúme, promessa, intervalos ≥ 20 h): **1.895 turnos**, cada turno tratado como "mensagem do usuário" e o
outro falante como o personagem. Split por conversa: **dev 847 turnos (13 conversas) / teste 1.048 (17 conversas)**.
**Ouro:** `openai/gpt-6-luna` (reasoning low) como 2º rotulador independente, vendo até 24 turnos anteriores com
intervalos, direção e nível por dimensão + eventos (US$ 0,39). **Checagem manual:** 60 turnos rotulados por mim
(30 com sinal forte de evento, 30 aleatórios, estratificados sem olhar o ouro).

**Cuidado com a distribuição:** conversa real entre amigos/casais é pouco dramática. No ouro, 53% dos turnos movem alguma
dimensão, quase tudo para cima (cumplicidade 747, conforto 274, afeto 175) e quase tudo "leve" (1.359 de 1.364 movimentos
com nível 1). Ressentimento sobe em só 19 turnos (8 no teste), ciúme em 2. Os números de detecção abaixo são dominados
pelas dimensões positivas; as negativas são avaliadas melhor pelos pares contrastivos (§4b) e cenários (§4c).

### 2.1 Arquiteturas (teste, 1.048 turnos × 16 células; limiar escolhido no dev; IC 95% bootstrap por conversa)

| arquitetura | estado | AUC agrupada | AUC só nível ≥ 2 | AUC macro (por célula) | limiar (dev) | F1 teste | prec. | rec. |
|---|---|---|---|---|---|---|---|---|
| DIM | sem | 0,951 [0,942–0,959] | 0,949 | 0,880 | 0,67 | 0,573 [0,507–0,625] | 0,56 | 0,59 |
| DIM | **com** | 0,958 [0,947–0,968] | 0,913 | 0,885 | 0,74 | 0,598 [0,518–0,658] | 0,56 | 0,64 |
| BIP | sem | 0,858 [0,836–0,885] | 0,787 | 0,844 | 0,45 | 0,429 | 0,38 | 0,50 |
| BIP | com | 0,896 [0,874–0,913] | 0,708 | 0,833 | 0,42 | 0,536 | 0,56 | 0,51 |
| EVT→mapa | sem | 0,917 [0,910–0,926] | 0,972 | 0,848 | 0,57 | 0,452 | 0,42 | 0,49 |
| EVT→mapa | com | 0,924 [0,915–0,935] | 0,955 | 0,849 | 0,54 | 0,476 | 0,44 | 0,53 |
| COMB média | com | 0,963 [0,956–0,971] | 0,941 | 0,890 | 0,60 | 0,587 | 0,57 | 0,60 |
| COMB máx. | com | 0,958 | 0,908 | 0,878 | 0,74 | 0,598 | 0,53 | 0,69 |
| COMB reg. log. | sem | 0,969 [0,960–0,979] | 0,923 | 0,891 | 0,27 | 0,651 [0,562–0,714] | 0,62 | 0,69 |
| **COMB reg. log.** | **com** | **0,968 [0,958–0,979]** | 0,929 | **0,894** | 0,28 | **0,657 [0,565–0,726]** | 0,62 | 0,69 |

**Com − sem estado (pareado, mesmos turnos):** DIM +0,007 [+0,004; +0,011]; BIP +0,038 [+0,021; +0,055]; EVT +0,008
[0,000; +0,014]; COMB-LR −0,001 [−0,003; +0,001]. Nos dados reais o estado ajuda pouco (é conversa sem pendências
graves: o estado quase sempre diz "nenhuma pendência"); onde ele importa é nos pares contrastivos (§4b).

Leitura: **o Noul por dimensão × direção (DIM) é a espinha dorsal**; os Nouls de evento mapeados em código são piores
sozinhos (0,92), mas somam informação (COMB-LR 0,97; F1 0,66 vs 0,60) e — mais importante — são eles que alimentam os
portões AND (desculpa × pendência, promessa cumprida, sumiço). O **Score bipolar é o pior** (0,86–0,90): um só Score de 5
níveis por dimensão comprime "sobe" e "desce" e fica perto do meio. Direção separada (2 Nouls) > Score bipolar.

**Por célula (AUC teste, DIM com estado):** confiança↑ 0,91 · conforto↑ 0,80 · afeto↑ 0,89 · ressentimento↑ 0,96 (8 pos.)
· ressentimento↓ 1,00 (3 pos.) · respeito↑ 0,84 · proteção↑ 0,81 · cumplicidade↑ 0,93. O mais difícil é o conforto↑
(o luna marca "conforto leve" em muito small talk caloroso; o Jev e eu discordamos em parte deles).

**Eventos (AUC teste, com estado):** desculpa 1,00 (F1@0,5 0,88) · promessa feita 0,97 · promessa cumprida 0,96 ·
cancela 1,00 · explica sumiço 0,99 · elogio 0,99 · vulnerabilidade 0,99 · outra pessoa 0,99 · cuidado prático 0,97 ·
defensivo 0,94 · desdém 0,96 · afeto 0,98 · provocação amigável 0,88 · gratidão 1,00 · interesse 0,97. A ordenação é
quase perfeita, mas **no limiar 0,5 o Jev dispara demais nos eventos raros** (F1 de defensivo 0,07, desdém 0,09,
outra pessoa 0,42): o limiar tem de ser calibrado por evento (ou composto com AND), nunca 0,5 fixo.

### 2.2 Checagem manual (60 turnos; κ de Cohen por célula turno × dimensão-direção)

| par | κ | Jaccard nas células movidas | taxa de "move" (a / b) |
|---|---|---|---|
| eu × luna (ouro) | **0,68** | 0,53 | 3,9% / 6,0% |
| eu × Jev com estado, limiar 0,5 | 0,35 | 0,24 | 3,9% / 15,9% |
| eu × Jev sem estado, limiar 0,5 | 0,34 | 0,23 | 3,9% / 14,3% |
| **eu × Jev com estado, limiar 0,74 (escolhido no dev)** | **0,65** | 0,50 | 3,9% / 5,2% |
| luna × Jev com estado, limiar 0,74 | 0,55 | 0,40 | 6,0% / 5,2% |

O ouro do luna concorda comigo tanto quanto eu esperava de um 2º anotador (κ 0,68). **No limiar calibrado (0,74) o Jev
concorda comigo quase tanto quanto o luna (κ 0,65 vs 0,68)**; no limiar ingênuo de 0,5 ele marca movimento em 4× mais
células do que eu (16% vs 4%) — a principal fonte de deriva do estado (§4d). Na metade "com sinal" o Jev a 0,74 chega a
κ 0,74 comigo.

## 3. Jev 2: magnitude (pares contrastivos de gravidade)

12 escadas (insulto, piada, cancelamento, desculpa, elogio, vulnerabilidade, ciúme, promessa cumprida, cuidado, desdém,
mentira, provocação) × 3 gravidades × 2 paráfrases × 2 contextos (amigos / ficando) = 144 mensagens, 288 pares de
gravidade diferente (`c2_pairs.py magnitude`).

| método | acerto na ordenação dos pares | grave > leve | Spearman médio (gravidade) | |dif.| entre paráfrases (escala normalizada) |
|---|---|---|---|---|
| Jev 1, prob. do Noul | 0,917 | 0,979 | 0,83 | 0,079 |
| Jev 1, Score bipolar | 0,936 | 0,990 | 0,86 | 0,100 |
| **Jev 2, Score descritivo** | **0,969** | **1,000** | **0,91** | **0,044** |
| Jev 2, Choice de números (0–100) | 0,665 | 0,740 | 0,63 | 0,031* |

\* a "consistência" dos números é enganosa: eles quase sempre caem em 10 ou 20 (colapso), por isso as paráfrases
concordam e a ordenação falha.

Níveis inteiros do Score descritivo por gravidade (contagem): leve → {leve 48}; moderado → {leve 19, moderado 28, forte 1};
grave → {leve 3, moderado 34, forte 11}. **O Jev nunca usa "marcante" e raramente "forte"**: a escala é comprimida para
baixo. A física tem de ser calibrada para isso (§8: a tabela nível → delta foi desenhada assumindo que "forte" é raro).
Exemplos (contexto amigos): "you're kinda slow sometimes lol" → Score 1,03 / número 10; "honestly you're so
self-absorbed…" → 1,74 / 20; "you're pathetic and i regret ever wasting my time on you" → 2,76 / 70. Desculpa: "oops
sorry" 0,95 / 0; "sorry for cancelling, that wasn't cool" 1,56 / 10; a desculpa longa com reparação 2,19 / **0** (o
Choice de números devolve 0 para a desculpa mais forte).

**Nos dados reais** o ouro é quase todo "leve", então a magnitude só pode ser avaliada como "move ou não": Spearman
(nível do ouro × magnitude, 4.622 pares turno × dimensão ativada no teste) Score descritivo 0,49 [0,42–0,54], prob. do
Jev 1 0,48, bipolar 0,40, **números 0,25 [0,22–0,28]**. Entre os movimentos positivos do ouro não há variância para medir
(ρ ≈ 0).

**Decisão:** Jev 2 = Score descritivo. O Choice de números (a ideia original) é o pior método em todas as medidas —
confirma a recomendação da documentação (*jaggedness*: níveis descritivos, não números).

## 4. Validação

### 4b. Pares contrastivos de contexto (o Jev usa o estado e as pendências?)

15 itens com a MESMA mensagem em contexto neutro (N) × carregado, com a carga apresentada só nos turnos anteriores (T),
só no bloco de estado/pendências (S) ou nos dois (TS); 28 perguntas-alvo com sinal esperado pré-registrado. "Acerto" =
a pergunta-alvo se move ≥ 0,05 no sentido esperado em relação a N.

| carga no state | acerto | diferença média no sentido esperado |
|---|---|---|
| só turnos (T), com bloco de estado neutro | 0,71 | +0,23 |
| só turnos, sem bloco de estado | 0,75 | +0,23 |
| só estado + pendências (S) | 0,68 | +0,18 |
| **turnos + estado (TS)** | **0,86** | **+0,32** |

Exemplos (probabilidade do Noul, N → TS):
- "lol you're such a nerd, no wonder you're single" — dia 2 (acabaram de se conhecer) × dia 200 (melhores amigos que se
  zoam): ressentimento↑ 0,60 → **0,07**; piada de mau gosto 0,64 → 0,20; cumplicidade↑ 0,36 → **0,95**. Aqui o estado
  sozinho (S) já faz quase todo o efeito (0,09 / 0,94): a história ("há 200 dias", "zoam um ao outro sobre estar
  solteiro") não está nos turnos.
- "sorry about earlier" sem briga × depois de briga real: ressentimento↓ 0,44 → 0,57–0,65.
- "can't make it tonight, something came up" 1ª vez × 3ª vez: confiança↓ 0,24 → **0,91**; ressentimento↑ 0,15 → 0,89.
- "i promise i'll be there tomorrow" depois de duas promessas quebradas: confiança↑ 0,60 → **0,18**.
- "sorry, i'm really sorry" 1ª × desculpa repetida sem mudança: ressentimento↓ 0,70 → **0,27**.
- "haha whatever" depois de "me magoou quando você me zoou na frente dos seus amigos": ressentimento↑ 0,07 → 0,93.
- "hey" depois de 3 dias de silêncio sobre a cirurgia da mãe: ressentimento↑ 0,02 → 0,47 (a carga está nos turnos; o
  estado sozinho não basta: 0,08).
- Falhas: "you're the best, you know that?" como bajulação depois de esquecer o aniversário *aumenta* o afeto (0,56 →
  0,63; esperado: diminuir); "going out with Ana tonight 😊" depois de um beijo: ciúme↑ só 0,06 → 0,25 (o evento "outra
  pessoa" sobe para 0,86, mas o Noul de dimensão não); "miss you" depois de sumir 2 semanas: afeto↑ fica em 0,63.

Leitura: **o Jev lê o contexto** (a mesma mensagem muda de sinal), e o melhor é pôr as duas coisas no state: os turnos
(o que aconteceu) e o bloco de estado/pendências (o que não está mais na janela). O ciúme é a dimensão menos sensível
no Noul de dimensão — por isso na física ele entra também pelo evento "outra pessoa" (via mapa, escalado pelo afeto).

### 4d. Replay da física nos dados reais (Jev × ouro, mesma física)

Para cada janela real, a física foi executada duas vezes: com as detecções do Jev (Jev 1 com estado + Jev 2 Score) e com
as do ouro (luna convertido para o formato do Jev). Métricas: correlação de Spearman, entre personagens, do estado final
por dimensão; correlação dos deltas por turno; concordância do modo; saturação. Variantes escolhidas no **dev**,
reportadas no **teste** (`c2_replay.py`).

| variante da física | ρ estado final (média) | r dos deltas | modo igual | saturação Jev / ouro | movimentos/turno Jev / ouro |
|---|---|---|---|---|---|
| v0 (limiar 0,5) | 0,605 | 0,357 | 0,748 [0,64–0,84] | **0,25** / 0,02 | 2,15 / 0,76 |
| v0 + habituação | 0,580 | 0,278 | 0,795 | 0 / 0 | |
| v0 + habituação + humor volta 10%/msg | 0,581 | 0,280 | 0,696 | 0 / 0 | |
| v1 (+ portão de rotina) | 0,615 | 0,252 | 0,699 | 0 / 0 | 1,37 / 0,51 |
| v1 + limiar de rotina 0,7 | 0,609 | 0,268 | 0,711 | 0 / 0 | |
| v1 + limiares 0,8/0,6 | 0,633 | 0,270 | 0,742 | 0 / 0 | |
| v0 + limiar 0,74 | 0,699 | 0,403 | 0,821 [0,72–0,91] | **0,21** / 0,02 | 0,87 / 0,76 |
| v1 + limiar 0,74 | 0,672 | 0,267 | 0,721 | 0 / 0 | |
| v0 + limiar 0,74 + habituação | 0,681 | 0,305 | 0,818 | 0 / 0 | |
| **v2 = v0 + limiar 0,74 + humor volta 10%/msg** (escolhida) | **0,697** | **0,469** | **0,809 [0,72–0,90]** | **0 / 0** | 0,87 / 0,76 |

O critério original (ρ + modo − excesso de saturação) escolhia "v0 + limiar 0,74", que satura a cumplicidade em 18% dos
turnos do dev (nas conversas de zoeira constante ela vai a > 0,95). Passei a exigir saturação ≤ a do ouro + 0,02
(declarado depois de ver isso; as duas escolhas são reportadas). A v2 resolve a saturação com o retorno do "humor"
(cumplicidade, preocupação, ciúme) à base a cada mensagem e mantém o melhor ρ e a melhor correlação de deltas.
**O que mais importa não é a física, é o limiar de detecção**: subir de 0,5 para 0,74 leva o Jev de 2,15 para 0,87
movimentos por turno (ouro: 0,76) e o ρ de 0,61 para 0,70.

### 4c. Cenários roteirizados

(em execução: 16 cenários, 161 mensagens do usuário por configuração, critérios pré-registrados em `c2_scen_run.py`)

## 5. Efeito do estado na resposta

(pendente)

## 6. Postura do personagem (pedido extra)

(pendente)

## 7. Custo e latência

(pendente)

## 8. Especificação final

(pendente)
