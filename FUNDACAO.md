# A Fundação

> O desenho mínimo de um companion que conversa como gente, tirado de 19 relatórios de evidência.
> Regra de ouro: **o Jev lê e decide, o código governa, a LLM atua.** Três papéis, nenhum repetido.

---

## 1. O mapa

```
                         mensagem(ns) do usuário
                                   │
                        (código) junta as bolhas
                                   │
      ┌────────────────────────────▼────────────────────────────┐
      │  ① LER     1 chamada ao Jev, ~80 perguntas, 0,5 s        │
      │            o momento · a postura · a relação · a memória │
      └────────────────────────────┬────────────────────────────┘
                                   │  respostas tipadas
      ┌────────────────────────────▼────────────────────────────┐
      │  ② DIRIGIR   código puro, < 10 ms                        │
      │   atualiza a Relação ─► escolhe o modo ─► sorteia o      │
      │   movimento ─► fixa os orçamentos ─► escreve a NOTA      │
      │   DO DIRETOR (≤ 70 palavras) e o PLANO DE ENTREGA        │
      └────────────────────────────┬────────────────────────────┘
                                   │         "digitando…" liga aqui
      ┌────────────────────────────▼────────────────────────────┐
      │  ③ ATUAR    1 chamada à LLM                              │
      │   [cabeçalho de roleplay · ficha · relação em frases ·   │
      │    memória (OptMem wake) · últimos 8 turnos · NOTA]      │
      └────────────────────────────┬────────────────────────────┘
                                   │
      ┌────────────────────────────▼────────────────────────────┐
      │  ④ CONFERIR  normalizador (código) + 1 chamada ao Jev    │
      │   ~25 perguntas atômicas; violou? ─► regenera 1× (raro)  │
      └────────────────────────────┬────────────────────────────┘
                                   │
      ┌────────────────────────────▼────────────────────────────┐
      │  ⑤ ENTREGAR  código: bolhas, ordem, "digitando…", ritmo  │
      └────────────────────────────┬────────────────────────────┘
                                   │
                              usuário vê
                                   ·
                 ·····  fora do caminho crítico (assíncrono)  ·····
                   OptMem: grava a nota do que valeu lembrar
                   Agenda: reabertura/check-in quando há pendência
```

**Quatro objetos de estado.** São tudo o que o sistema sabe:

| objeto | o que é | muda quando | quem escreve |
|---|---|---|---|
| **Ficha** | quem o personagem é: comportamentos situacionais, os "nunca…", a voz (forma do riso, emojis, caixa, pontuação, abreviações), 10–15 mensagens-exemplo | nunca (é o autor que escreve) | humano |
| **Relação** | 8 dimensões (confiança, conforto, afeto, ressentimento, ciúme, respeito, proteção, cumplicidade) + modo + pendências + história compartilhada | a cada mensagem | código, com os sinais do Jev |
| **Memória** | OptMem: log de notas de uma linha + árvore de resumos; o `wake` devolve ~8k tokens | depois da resposta, assíncrono | Jev decide o quê, LLM redige a linha |
| **Conversa** | últimos 8 turnos + perfil de estilo do usuário (taxas contadas em código) | a cada mensagem | código |

Não há banco vetorial, nem "world state" separado, nem LLM planejadora, nem crítico LLM, nem várias candidatas por padrão.
Cada uma dessas ausências tem motivo (seção 5).

---

## 2. As cinco peças

### ① LER: uma chamada ao Jev
As perguntas do Jev são isoladas e a latência não cresce com o número delas (132 perguntas em 0,55 s, relatório 06). Então
tudo o que precisa ser **lido** vai numa chamada só. O padrão oficial é o *speculative fan-out*: perguntar até o que talvez
não se use.

**State:** `{ficha resumida, relação em frases, memória (wake), últimos 8 turnos, turno atual do usuário}`. As perguntas são
em inglês, literais, com critérios descritos em todas as opções.

| bloco | perguntas (tipo) | por quê (evidência) |
|---|---|---|
| **momento** | emoção (Choice plana de ~32 classes; a família sai em código), seriedade (S 0–3), brincadeira, ironia, tensão, vulnerável, pede apoio (N), flerte + intensidade (N+S), função do riso do usuário (C), o usuário deu gancho? (N), tópico esgotado? (N), pergunta pessoal? (N), a conversa está desacelerando? (S) | família da emoção 75–85% com confiança alta; hora de brincar AUC 0,78; gancho é o melhor preditor de continuidade (rel. 03, 05) |
| **postura** | como **o personagem** reage (C: revidar, desdenhar, ignorar, magoar-se, rir, provocar de volta, aceitar, recuar, desculpar-se) + intensidade (S) | o soldado revida e a tímida se magoa; 0% de "pedir desculpa" diante de insulto; 86% fiel à ficha (rel. 18) |
| **relação** | uma N por dimensão × direção + uma S descritiva de magnitude por dimensão (nada → marcante) + N de eventos (desculpa, promessa cumprida ou quebrada, sumiço, ciúme, cuidado…) | detecção AUC 0,96–0,97; magnitude em palavras ordena 97% (os números ordenam 66%); com contexto, a mesma frase muda no dia 2 × dia 200 em 86% dos pares (rel. 18) |
| **decisão da resposta** | movimento (C, 16 opções), tamanho (S), cabe piada? (N), **a que palavra reagir** (C sobre as palavras numeradas), encerrar? (N) | movimento 38%, e **64% com confiança ≥ 0,7**; tamanho ρ 0,49; palavra certa 62% contra 43% da regra (rel. 09, 12) |
| **memória** | vale lembrar? (N) + tipo (C: fato, pendência, piada interna, limite) | alimenta o OptMem (rel. 15) |

A magnitude da relação vai **na mesma chamada** que a detecção, em vez de numa segunda. Com o fan-out, perguntar a
magnitude das 8 dimensões custa o mesmo que perguntar só a das que dispararem, e a cascata de 2 chamadas vira 1.

### ② DIRIGIR: só código
É o coração, e é determinístico, testável e barato. Tudo o que é conta, limiar, sorteio ou memória de curto prazo mora
aqui (o Jev não conta bem, rel. 06):

1. **Relação:** aplica a física. A direção só vale acima do limiar **por dimensão**, calibrado (o Jev é generoso com o
   positivo de rotina). O nível vira delta (0 / 0,04 / 0,10 / 0,18 / 0,28), com saturação, meia-vida, inércia (a confiança
   cai rápido e sobe devagar), histerese de modo e eventos com AND (a desculpa só reduz o ressentimento se houver pendência
   ligada; rel. 18).
2. **Modo:** abertura · leve · flerte · acolhimento · conflito · logística · encerrando. A seriedade, a tensão e a
   vulnerabilidade decidem, com histerese 0,6/0,4 e EMA.
3. **Postura e movimento:** a postura do Jev filtrada pela ficha e pela relação. O movimento é a distribuição do Jev ×
   prior do modo, **sorteado**; só é ditado se a confiança for ≥ 0,7, senão a nota dita só a forma. Vetos: nada de humor
   com seriedade ≥ 1; "e você?" em ≤ 25% das perguntas pessoais; carregar a conversa quando o usuário não deu gancho;
   nada de mudança abrupta de tópico (rel. 03, 05, 12).
4. **Orçamentos:** o tamanho é um **alvo com folga**, nunca um teto (o `p_length` do Jev + um leve espelho do usuário).
   Pergunta, riso, emoji e "!" são **sorteados** com as taxas humanas do momento, não proibidos (proibir corrige demais,
   rel. 09 e 12). A intensidade fica ≤ a do usuário + 0 em afeto, e um degrau acima só quando ele convida (rel. 07).
5. **Nota do diretor:** ≤ 70 palavras, imperativa, concreta, em ordens **variadas** (ordens fixas viram vícios: "hey
   drinking coffee" 5×, rel. 12). Exemplo:
   > *Ele te insultou. Você não deixa barato: devolve seco, com desprezo, 1 frase. Pegue "chato". Sem pergunta, sem
   > emoji, sem pedir desculpa.*
6. **Plano de entrega:** o nº de bolhas é **sorteado** da distribuição prevista (features do Jev + regressão ordinal,
   AUC 0,92, rel. 11). Também saem o atraso inicial e a duração do "digitando…".

### ③ ATUAR: uma chamada à LLM
O prompt tem uma ordem fixa, e cada bloco tem motivo:
```
[SYSTEM] You are roleplaying as {{char}} in a private chat with {{user}}. Stay in character. You are not an assistant.
[FICHA]  comportamentos situacionais ("quando desconfortável, desvia com humor seco e responde curto"),
         os "nunca…", como {{char}} escreve (forma de riso, emojis, caixa, pontuação), 3–5 mensagens-exemplo
[RELAÇÃO] em frases, não números ("ela ainda está chateada com o jantar cancelado")
[MEMÓRIA] OptMem wake (o que importa desta relação)
[CONVERSA] últimos 8 turnos
[NOTA DO DIRETOR] — por último, onde pesa mais
```
**Evidência:**
- com o cabeçalho + a ficha comportamental + a nota no fim + o normalizador, os 4 atores caem de D 1,6–3,8 para **≈1,2**
  (teste, rel. 17);
- a mesma ficha com adjetivos fica em 2,9–3,3, igual a não ter ficha;
- a relação em frases dá +65 pp de frieza no estado ressentido, contra +25 pp com números (rel. 18);
- LLM barata + nota supera uma LLM ~3× mais cara sem nota (rel. 09);
- **o modelo-ator é trocável**: a fundação não depende dele.

### ④ CONFERIR: normalizador + uma chamada ao Jev
- **Normalizador (código, grátis):** impõe a voz da ficha (a forma do riso, a caixa, a pontuação, o repertório de emoji);
  corta "!", emoji e marcadores acima do orçamento; limpa HTML, "\n\n" e o nome do usuário; aplica a lista negra de chat
  ("sorry to hear", "totally", "that's so sweet"…). Sozinho, corta 25–47% da distância ao humano (rel. 13).
- **Jev, ~25 perguntas atômicas** (nunca "isso soa humano?", que dá AUC ≈ 0,5 e prefere a caricatura, rel. 08, 10, 19):
  - parafraseia o usuário;
  - valida demais;
  - pediu desculpa ou acalmou como um assistente;
  - termina em pergunta genérica;
  - faz reação → comentário → pergunta;
  - entusiasmo acima do usuário;
  - fora da postura pedida;
  - viola algum "nunca…" (um Noul por restrição, 80 de 80 contra a checagem manual, rel. 17);
  - incoerente com a última mensagem;
  - guarda de segurança.

  Combinadas com as features de código, essas perguntas separam humano de LLM com AUC ≈ 0,9, e cada uma diz **o que**
  corrigir.
- **Regra:** violou algo grave → regenera 1× com a correção cirúrgica na nota. Senão, segue. Várias candidatas por padrão
  não se pagam (+0–0,08 de D por 2–4× o custo, rel. 13 e 17).

### ⑤ ENTREGAR: só código
- **Bolhas:** o texto é dividido em fronteiras de frase na ordem humana: reação curta → conteúdo → continuação →
  **pergunta por último**. Emoji solto vai no fim. Momento sério não é picotado em pedaços pequenos (rel. 01).
- **"Digitando…":** ≈ 0,15 s + 0,17 s por caractere, × 1,17 por nível de seriedade. Há pausas visíveis ocasionais no sério e
  nunca na briga.
- **Ritmo:** o atraso espelha o do usuário (ρ ≈ 0,4). Há uma pausa extra antes de responder a um flerte. **Não** existe
  demora dramática no momento sério (os humanos não demoram, rel. 03). Se chegar bolha nova do usuário enquanto o bot
  "digita", replaneja (volta a ①).
- **Despedida:** espelha e deixa ir. Nunca culpa nem "não vai embora" (37% dos apps fazem isso; rel. 15).

---

## 3. Tempo e custo de uma mensagem

```
t=0      bolhas do usuário agrupadas (código)
t≈0,5 s  ① LER pronto                       ─┐
t≈0,5 s  ② DIRIGIR pronto; "digitando…" liga │ caminho crítico:
t≈1,7 s  ③ texto pronto (LLM ~1,2 s)          │ 2 chamadas ao Jev + 1 à LLM
t≈2,2 s  ④ conferido (Jev 0,5 s)             ─┘ ≈ 2,2 s
t≈2,2 s+ ⑤ entrega no ritmo humano (o "digitando…" absorve tudo acima)
```
- Um humano no chat ao vivo leva ~14 s para responder (mediana). O bot pode responder em ~5 s e ainda parecer humano, então
  os ~2,2 s de computação ficam **escondidos** atrás do "digitando…".
- **Custo do Jev** ≈ US$ 0,0003 por mensagem (≈ US$ 1–2 por mês para um usuário intenso). O custo dominante é o da LLM-ator.
- Comparação: o pipeline "Interpreter → State Manager → Retrieval → Planner → Writer → Critic" com LLMs em cada etapa teria
  4–6 chamadas de LLM em série (≈ 5–10 s), e o planejador LLM **decide pior** que o Jev (32,5% × 37,5% no movimento, rel. 12).

---

## 4. Por que é assim (cada escolha tem um experimento)

| princípio | a evidência |
|---|---|
| **O Jev decide pelo personagem, nunca adivinha o usuário** | o papel do Jev é ler a mensagem e escolher a reação fiel à ficha; o reflexo de assistente ("ohh, me desculpe") é o vício nº 1 em RP (RoleCDE; rel. 18) |
| **Perguntas atômicas, nunca holísticas** | "soa humano?" AUC 0,51 e prefere a caricatura; o banco atômico + código AUC 0,91 (rel. 10, 19) |
| **O Jev como features, o código como decisor** | bolhas: Jev + regressão AUC 0,92 contra "sempre 1" com uma pergunta direta (rel. 11) |
| **Sortear, não usar argmax** | o argmax cria robôs; o sorteio aproxima a variedade humana (rel. 12) |
| **Portão de confiança** | movimento 64% com confiança ≥ 0,7 e 25–30% abaixo de 0,55 (rel. 12) |
| **Palavras, não números, para o Jev e para a LLM** | magnitude em palavras 97% contra números 66% (rel. 18); relação em frases +65 pp contra +25 pp; o Jev não conta (rel. 06) |
| **Alvos, não proibições** | proibir pergunta e "!" corrige demais (1% contra 12% humano); a pergunta e o "!" sorteados acertam a taxa (rel. 09, 12, 13) |
| **Ficha por comportamentos, não adjetivos** | D 1,2 contra 3,0 (rel. 17) |
| **Nota curta, imperativa, no fim** | prosa com probabilidades devolve a LLM ao baseline (rel. 09); a nota no fim pesa mais (Author's Note) |
| **Timing e bolhas em código** | a LLM propondo horários perde para "copiar a latência anterior" (rel. 17); a fórmula do "digitando…" vem de logs reais (rel. 01) |
| **Planejador externo, não autorreflexão** | literatura: F1 13,5 → 21,1 com planejador externo, e CoT da própria LLM piora (rel. 14) |

---

## 5. O que foi cortado, e por quê

| cortado | por quê |
|---|---|
| LLM planejadora / "Social Planner" | decide pior que o Jev e custa ~1 s a mais por etapa |
| crítico LLM / reranker LLM | juízes LLM preferem a resposta mais longa (63–70%); as perguntas atômicas do Jev + código fazem o papel a 1/100 do custo |
| várias candidatas por padrão | +0–0,08 de D por 2–4× o custo. Fica só como regeneração pontual |
| cascata de Jevs em série | a magnitude entra no mesmo fan-out; para o movimento, a cascata **piorou** (33% × 37,5%, rel. 12) |
| banco vetorial / RAG de memória | o OptMem cobre memória de longo prazo com o log e a árvore; o Jev decide o que entra |
| retrieval de casos humanos no Jev | ganha +1,3 pp (não significativo) com uma etapa a mais. Fica para depois |
| "world state" separado | é desnecessário em companion; em roleplay com cena, a cena vira uma linha da memória e da nota |
| gerar dentro de um log de chat em produção | funciona sem diretor, mas a LLM inventa a fala do usuário (até 48%) e abandona o formato com a nota (rel. 17) |
| Verbalized Sampling por padrão | só vence quando não há diretor (rel. 17) |
| previsões sobre o usuário | não é papel do sistema (o Jev decide pelo personagem) |

---

## 6. Estado de validação (honesto)

| peça | status |
|---|---|
| ① leitura do momento, emoção, seriedade, hora de brincar, gancho | ✅ validado no teste (rel. 03, 05, 06, 12) |
| ① postura do personagem (a decisão) | ✅ decisão validada (86% fiel, checagem manual); ⏳ o efeito na resposta renderizada não foi medido (créditos) |
| ① + ② relação (detecção, magnitude, física) | ✅ detecção AUC 0,97, magnitude 97%, física 73% dos critérios (escolhida no dev), sem saturar |
| ② movimento, palavra-âncora, portão de confiança | ✅ teste (rel. 12) |
| ② orçamentos sorteados (em vez de proibidos) | ⚠️ dev e teste pequeno (rel. 12, 13); falta a validação grande |
| ③ cabeçalho + ficha comportamental + nota no fim | ✅ teste, 4 atores (rel. 17) |
| ④ normalizador + banco atômico | ✅ teste (rel. 10, 13, 19); ⏳ o banco sobre o pipeline final no teste (~3,5 mil chamadas, já preparado) |
| ⑤ bolhas, "digitando…", ritmo | ✅ modelo de bolhas no teste (rel. 11); a fórmula de digitação vem de logs reais (rel. 01) |
| **ponta a ponta** | ⏳ **falta o que nenhum juiz automático substitui: humanos.** Teste cego em 3 vias à la Jones & Bergen (bot × humano × bot sem Jev) + métricas de produto (retorno no dia seguinte, turnos sem gancho) + um corpus PT-BR para recalibrar as taxas |

---

## 7. O tamanho do código

É uma fundação, não um framework:

| arquivo | faz | tamanho estimado |
|---|---|---|
| `questions.yaml` | as ~80 perguntas de LER e as ~25 de CONFERIR, congeladas e versionadas | dados |
| `card.yaml` | a ficha (comportamentos, "nunca…", voz, exemplos) | dados |
| `rates.yaml` | as taxas humanas por momento (tamanho, pergunta, riso, emoji, bolhas, tempos), recalibráveis com logs | dados |
| `relation.py` | a física da relação (deltas, meias-vidas, histerese, eventos com AND) | ~150 linhas |
| `director.py` | modo, postura, movimento, orçamentos, nota | ~200 linhas |
| `normalize.py` | voz da ficha, lista negra, cortes | ~100 linhas |
| `deliver.py` | bolhas, ordem, "digitando…", ritmo | ~100 linhas |
| `loop.py` | ① → ⑤ + assíncronos (OptMem, agenda) | ~100 linhas |

**Cerca de 650 linhas de lógica.** Todo o resto são dados versionados: perguntas, ficha e taxas.
