# 15 · Respaldo científico II: comunicação mediada por computador, psicologia da conversa, HCI de companions e OptMem

> Prefixo `b6`. Revisão feita com leitura das fontes (texto integral quando acessível; quando só consegui o resumo ou
> uma fonte secundária, isso está marcado). Uma checagem nova com os nossos dados, sem Jev e sem LLM:
> `scripts/analysis/b6_rafagas.py` → `analysis/data/b6_rafagas.json`. Custo: US$ 0.
> Por causa do pedido de encerramento, parei de buscar fontes novas. O que ficou sem leitura está listado na §8.

## 1. Resumo com números

1. **Os nossos achados de forma têm respaldo e, em alguns casos, a literatura já tinha medido o mesmo número.**
   - Ling & Baron (2007): 18,8% das mensagens de IM têm uma palavra só (SMS: 3,7%) e só 35% delas terminam com
     pontuação. O nosso maichat tem 45% das mensagens com até 3 palavras e 90% sem pontuação final.
   - Kalman et al. (2006): em 150 mil respostas, ≥ 70% saem dentro da latência média do respondente e ≤ 4% depois de 10×
     essa média. **No nosso WhatsApp a regra se replica:** 82,7% das respostas ficam dentro da média, 2,3% passam de 10×, e
     96% dos falantes cumprem os dois limites. No maichat, ao vivo, a distribuição é menos enviesada (57% por falante).
   - Adamic et al. (Facebook, 2015): 52% das pessoas usam um único tipo de riso. No nosso relatório 04, a forma dominante
     cobre 71–83% dos risos de cada pessoa.
   - Brody & Diakopoulos (2011): há alongamento ("sooo") em 17,4% dos tweets, e as palavras de sentimento são as mais
     alongadas (2,41 × 1,79 variantes). Nos nossos turnos de afeto, o alongamento aparece em 17%, contra 6,5% nos
     neutros.
2. **O ponto final soa um pouco menos sincero, mas o efeito é pequeno.** Em Gunraj et al. (2016; n = 126), a mesma resposta
   de uma palavra com ponto teve média de 3,85 contra 4,06 sem ponto (escala de 7; números de fonte secundária). Isso
   bate com o relatório 04, que mediu +0,03 em p(frio): o que esfria é a falta de amaciador, não o ponto.
3. **Acomodação: a literatura confirma "espelhamento local, sem convergência lenta".**
   - Danescu-Niculescu-Mizil et al. (2011) medem acomodação exatamente como nós, pelo "lift" sobre a taxa-base do
     próprio falante.
   - Chen et al. (2026) acharam que, entre humanos (Switchboard), o estilo **não converge** ao longo da conversa (a
     inclinação é até negativa). O GPT-4o acomoda 1,8× mais que o usuário, e já no 1º turno.
   - Ireland et al. (2011): o LSM (palavras funcionais) prevê o "match" num speed date (OR 3,05 por DP; 33,3% × 9,1%) e
     a estabilidade do casal por IM (OR 1,95; 76,7% × 53,5% juntos 3 meses depois).
   - Durandard et al. (2025): **a LLM se alinha no conteúdo** (reusa as palavras da pergunta) **e o humano, no estilo**.
     É a mesma "paráfrase" do relatório 08 (14–21% × 2,8%).
4. **Psicologia da conversa.**
   - Perguntar funciona, mas o que funciona é a **pergunta de seguimento**. Huang et al. (2017): ≥ 9 × ≤ 4 perguntas em
     15 min de chat deram simpatia de 5,79 × 5,31. No speed dating, só a taxa de follow-ups previu 2º encontro; a
     pergunta "espelho" ("e você?") não previu.
   - Perguntar é traço da pessoa (r = 0,53 entre encontros), assim como achamos que uma pessoa faz 60–67% das
     perguntas da conversa.
   - Collins & Miller (1994), meta-análise: quem se revela é mais querido (d = 0,28; r = 0,14). Isso bate com o nosso
     "contar algo de si sobe o engajamento" (+0,15).
   - Mastroianni et al. (2021): só **2%** das conversas terminam quando os dois querem e só 30% quando um deles quer.
     As pessoas erram o desejo do outro em ~64% da duração.
5. **Companions: engajamento não é bem-estar.**
   - MIT/OpenAI (Fang et al., 2025; n = 981, > 300 mil mensagens, 4 semanas, RCT): o tempo diário **voluntário** prevê
     mais solidão (β = 0,02), menos socialização (−0,05), mais dependência emocional (0,06) e mais uso problemático (0,02).
     Ver a IA como amigo (atração social) e confiar mais nela preveem mais dependência (b = 0,19 para confiança). A voz
     "engajante" ignorou limites do usuário em 14,2% das respostas, contra 3,2% no texto.
   - Character.AI (Zhang et al., *Nature Human Behaviour*, 2026; n = 1.131; 464.687 mensagens de 237 pessoas): quem tem
     rede social menor usa mais como companhia (β = −0,03). O uso como companhia se associa a menor bem-estar
     (β = −0,48), e a associação é pior com uso intenso (interação −0,31) e com muita autorrevelação (−0,38).
     Apoio emocional aparece em 80,3% das sessões e roleplay romântico em 68%.
   - De Freitas et al. (2025): 37,4% das despedidas nos apps de companion recebem manipulação emocional ("não vai
     ainda…"). Isso aumenta o engajamento pós-despedida em até 14×, mas também a percepção de manipulação e a
     intenção de abandonar o app.
6. **"Rajada de mensagens = pego mentindo?" Não há evidência disso.** A evidência de mentira em texto aponta o contrário:
   - quem mente demora **mais**, edita **mais** e escreve **menos** (Derrick et al., 2013; 1.572 respostas mentirosas,
     ~10% mais lentas);
   - ou, em conversa livre, escreve mais palavras (+28%; Hancock et al., 2008). O detector humano fica no acaso (53,5%).

   Nos nossos dados, **5+ bolhas em ≤ 60 s é raro**: 0,9% dos turnos no maichat e 1,4% no WhatsApp. Cinco falantes
   produzem 54–57% dessas rajadas. Elas vêm com **mais agitação e mais brincadeira**:
   - arousal +0,6 (IC da diferença 0,50–0,98) no maichat e +0,4 (0,26–0,71) no WhatsApp;
   - playful 0,72 × 0,55;
   - tensão **menor** no maichat.

   Não vêm com mais ansiedade no maichat. No WhatsApp, a ansiedade (+0,05), a vulnerabilidade e a busca de apoio sobem
   um pouco. O único respaldo para "rajada como pedido de atenção" é Heston & Birnholtz: depois de ~1 min sem resposta,
   um participante mandou "????????".
7. **OptMem se encaixa sem reprojeto** (§7): o Jev decide o que vira `note` e de que tipo; o `wake` alimenta o campo
   `memory` do state; `recall` e `zoom` ficam como ferramentas de reabertura e de callback.

---

## 2. CMC e linguística de chat: a base dos nossos achados de forma

| fonte (link) | o que li | achado com números | nossos números (relatório) | confirma? | ponto de uso do Jev/código |
|---|---|---|---|---|---|
| Kalman, Ravid, Raban & Rafaeli (2006), [JCMC 12(1)](https://www.openu.ac.il/personal_sites/download/yoram-kalman/Are-you-still-waiting-for-an-answer.pdf) (versão de conferência, texto integral) | integral | e-mail, fórum e Google Answers (> 150 mil respostas): latências em lei de potência; ≥ 70–80% dentro da média τ; ≤ 3–4% depois de 10τ. Define **silêncio = sem resposta após 10τ** (≥ 95% de confiança de que não vem mais). Na fala face a face, 70–80% das pausas ficam abaixo da média (τ ≈ 0,97 s) | **b6:** WhatsApp 82,7% ≤ τ e 2,3% > 10τ; 96% dos falantes cumprem os dois limites. Maichat: 69,7% agregado, 57% por falante, 0,1% > 10τ (cauda truncada pelo corte de sessão em 3 h) | **sim** (WA); parcial no chat ao vivo | Código: τ por usuário (EMA) → "o usuário está em silêncio" se o tempo passar de 10τ. Isso dispara a lógica de reabertura (rel. 02), não "cadê você?" |
| Kalman & Rafaeli (2011), [Communication Research 38(1)](https://journals.sagepub.com/doi/10.1177/0093650210378229) | resumo | e-mail de candidatos respondido em 1 dia, 2 semanas ou > 1 mês: o efeito da demora depende da "valência de recompensa" do candidato (teoria da violação de expectativa) | — | contexto | a demora do bot é lida pela relação: com intimidade alta, uma demora curta não pune |
| Heston & Birnholtz, ["Worth the wait?"](https://socialmedia.northwestern.edu/wp-content/uploads/2012/09/worth-wait-effect-6.pdf) | integral | 24 pares de conhecidos, chat, responsividade manipulada: o parceiro lento é menos atraente socialmente (4,13 × 3,95; F = 5,18) com **só ~10 s** de atraso médio. Quem espera tenta atrair atenção: pergunta, reformula e, depois de ~1 min, manda "????????" | rel. 01: a latência segue a do outro (ρ ≈ 0,4); a resposta lenta está associada ao fim da sessão (6% × 23%) | **sim** | a latência do bot espelha a do usuário e **nunca** faz "demora dramática" (rel. 01). Uma rajada do usuário depois de o bot "sumir" é **pedido de atenção**: um Noul ("está cobrando resposta?") → responder rápido e sem justificar demais |
| Templeton et al. (2022), [PNAS](https://pmc.ncbi.nlm.nih.gov/articles/PMC8794835/) | integral (Europe PMC) | fala: 66 pessoas, 322 conversas. A resposta mais rápida prevê o prazer da conversa (b = −0,35) e a conexão (−0,28), inclusive momento a momento; quem responde rápido é mais apreciado pelos parceiros (b = −0,63). Ouvintes externos também usam o sinal | rel. 01/07: não há demora após o sério; a pausa aparece antes do flerte | análogo (é fala, escala de ms) | reforça "rápido = conectado". A exceção medida por nós (pausa antes de flerte) fica como regra local |
| Teichmann et al. (2026), [JSPR](https://journals.sagepub.com/doi/10.1177/02654075251377184) | resumo | depois do 1º encontro, mandar mensagem **na manhã seguinte** maximiza o interesse (efeito em U; n = 543); o ideal declarado é ~6 h | — | contexto (reabertura) | a reabertura proativa do bot não deve ser imediata nem vir depois de dias: janela intermediária |
| Ling & Baron (2007), [J. Lang. Soc. Psych.](https://nl.ijs.si/janes/wp-content/uploads/2014/09/lingbaron07.pdf) | integral | 191 SMS × 191 IMs (universitárias dos EUA): IM tem 6,0 palavras e 29 caracteres por transmissão; 18,8% de uma palavra só; 34% com várias frases (SMS: 60%). O turno de IM é **maior** que o SMS, mas vai em partes seriadas. Pontuação no fim da transmissão: 35% (IM) e 29% (SMS); 100% das perguntas de IM têm "?". Abreviações: < 5% das palavras (EUA) × ~19% (Thurlow, Reino Unido) | rel. 01: 31–47% dos turnos com várias bolhas; 1ª bolha de 16–17 caracteres. Rel. 08: 45% das mensagens com ≤ 3 palavras e 90% sem pontuação final | **sim**, e hoje de forma mais extrema | tamanho de bolha e fragmentação em código (rel. 01). Abreviação é traço de pessoa, não do meio: inventário fixo de 3–6 abreviações |
| Baron (2010), [Language@Internet](https://scholarworks.iu.edu/journals/index.php/li/article/view/37586) | **só resumo** (o texto integral redireciona para o web.archive, bloqueado aqui) | as quebras de enunciado em várias transmissões seguem pontos gramaticais; entre homens, parecem fala; entre mulheres, escrita | rel. 01: "conteúdo + continuação" 24% e 20% das transições | parcial | onde quebrar: na conjunção ou oração, não no meio do sintagma |
| Tagliamonte & Denis (2008), [American Speech 83(1)](https://doi.org/10.1215/00031283-2008-001) | **resumo e secundárias** | 72 adolescentes, 1 milhão de palavras de IM: haha, lol e omg somam só **~3%**; os mais velhos trocam "lol" por "haha"; IM é um registro híbrido, não uma "ruína" | rel. 08: muletas raras (honestly 1,2/1.000; tbh 4,1) | **sim** | a lista de gírias da persona é curta e rara; o controlador de frequência fica em código (rel. 08) |
| Adamic, Develin & Weinsberg (2015), [Facebook Research](https://research.facebook.com/blog/2015/8/the-not-so-universal-language-of-laughter/) | integral | 15% das pessoas riram na semana. Entre quem riu: haha 51,4%, emoji 33,7%, hehe 12,7%, lol 1,9% (**das pessoas**). 52% usam um só tipo; "haha" (4 letras) é o mais comum; um só emoji em 50% dos casos | rel. 04: a forma dominante cobre 71–83% dos risos de cada pessoa | **sim** | a forma do riso é constante da persona; só a **taxa** espelha (rel. 04) |
| Gunraj et al. (2016), [CHB 55](https://www.sciencedirect.com/science/article/abs/pii/S0747563215302181) | **resumo e secundárias** | n = 126, 16 trocas, resposta afirmativa de 1 palavra ("Okay."/"Okay"): com ponto, menos sincera (M 3,85 × 4,06; ~0,2 ponto em 7, números de fonte secundária); no bilhete à mão não há efeito; com "!", **mais** sincera (trabalho seguinte, via [ScienceDaily](https://www.sciencedaily.com/releases/2015/12/151208094229.htm)) | rel. 04: ponto → p(frio) +0,03; "haha" → −0,24; resposta curta e seca já lê como fria (0,42) | **sim, efeito pequeno** | o ponto final é traço da persona (5 pessoas fazem 62–72% dos pontos). Evitar ponto em respostas de 1 palavra com afeto |
| Houghton, Upadhyay & Klin (2018), [CHB 80](https://doi.org/10.1016/j.chb.2017.10.044) | **só secundária** ([PsyPost](https://www.psypost.org/2017/11/ending-text-period-not-seems-less-sincere-makes-seem-negative-50197)) | 3 experimentos, 137 participantes: convites respondidos com 1 palavra (positiva, negativa ou ambígua); com ponto, a resposta soa mais negativa | idem | **sim** | idem. Albritton (2022, [OJCMT](https://doi.org/10.30935/ojcmt/11431)) acha que o ponto também pode sinalizar seriedade e formalidade, conforme o contexto |
| Kalman & Gergle (2014), [CHB 34](https://doi.org/10.1016/j.chb.2014.01.047) | **só resumo** (sem acesso ao texto integral) | Enron (~500 mil e-mails): a repetição de letras emula o alongamento de fonema da fala; o uso cresce com os anos e o vínculo com a fala enfraquece | rel. 01/07: alongamento como marca de afeto e energia ("Heeeey" → resposta alongada 32% × 8%) | **sim** (qualitativo) | a energia da resposta a uma saudação segue a do usuário (+0 a +1 nível) |
| Brody & Diakopoulos (2011), [EMNLP](https://aclanthology.org/D11-1052.pdf) (substitui Kalman & Gergle para ter números) | integral | 500 mil tweets: o alongamento aparece em **17,4%** (1 em cada 6); as palavras subjetivas têm 2,41 variantes alongadas contra 1,79 (p < 0,0001); palavras alongadas somam 79% das ocorrências do léxico de sentimento | rel. 07: alongamento em 17% dos turnos de afeto × 6,5% neutros (WA). Rel. 04: a tensão corta o alongamento (1,8% × 6,8%) | **sim** | código: alongamento só em palavra de afeto ou saudação, nunca em tensão |
| Dresner & Herring (2010), [Communication Theory](https://homes.luddy.indiana.edu/herring/Dresner_Herring.pdf) | integral (parte teórica) | o emoticon tem três funções: emoção; sentido convencional não emocional (piscadela = provocação); **força ilocucionária**, isto é, suavizar uma queixa ("…i feel awful :)") | rel. 03: o "haha" suavizador (43% dos risos são sociais); rel. 01: o emoji fecha a bolha | **sim** | a Choice "função do riso ou emoji" do usuário (reação × suavizador) já está no catálogo (rel. 03) |
| Iftikhar, Ma & Huang (2023), [CHI](https://jeffhuang.com/papers/LiveTyping_CHI23.pdf) | integral | 24 pessoas, 4 indicadores (nenhum, "digitando…", bolha, texto ao vivo): sem indicador, mais frustração e estresse (F = 6,92 e 4,30); indicadores mais ricos aumentam a copresença. Sob pressão de tempo, o parceiro parece menos envolvido mesmo com o indicador | rel. 01: fórmula do "digitando…"; 30% das respostas começam antes de a mensagem do outro chegar | **sim** | mostrar "digitando…" com duração realista (rel. 01); o apaga-e-redigita visível ocasional é plausível |

**Sem fonte lida com números:** "double texting" como tema específico, "kkk"/"rs" em PT-BR e emoji como marcador
pragmático em corpus grande. Ver §8.

## 3. Acomodação linguística (CAT e LSM) × relatório 04

| fonte | o que li | achado | ligação com o nosso rel. 04 | ponto de uso |
|---|---|---|---|---|
| Giles, CAT (via Danescu-Niculescu-Mizil et al.) | secundária | a convergência ocorre "quase instantaneamente" em várias dimensões; pode ser simétrica, assimétrica ou divergente | rel. 04: o contágio dura 1 resposta (lift 1,51 → 1,02 duas respostas depois) | espelhamento **local**, por turno |
| Danescu-Niculescu-Mizil, Gamon & Dumais (2011), [WWW, "Mark my words!"](https://arxiv.org/pdf/1105.0673) | integral | Twitter: Acc(C) = P(B usa C, dado que A usou C) − P(B usa C), por par, o que controla a homofilia. Há acomodação significativa (p < 0,0001) em quase todas as dimensões de estilo; o influente estilístico **não** é previsto por status (r ≤ 0,15) | **é o nosso "lift sobre a taxa-base do próprio B"** (rel. 04): riso 1,5–2,8×, emoji 1,2–1,9×, pergunta 0,81× | método validado; manter o controle da taxa-base |
| Niederhoffer & Pennebaker (2002), [JLSP](https://journals.sagepub.com/doi/10.1177/026192702237953) | secundária | há LSM no nível da conversa e turno a turno, em chat de laboratório (94 díades) e nas fitas de Watergate | rel. 04: pares já começam parecidos (comprimento ρ = 0,91) e não convergem | **sim** |
| Ireland et al. (2011), [Psych. Science 22(1)](https://sites.socsci.uci.edu/~lpearl/courses/readings/IrelandEtAl2011_RelationshipPrediction.pdf) | integral | Estudo 1: 40 speed dates de 4 min (~429 palavras): LSM → match, OR = 3,05 por DP (5,70 controlando o número de palavras); 33,3% × 9,1% acima e abaixo da mediana; LSM não se correlaciona com a similaridade percebida (r = 0,06). Estudo 2: 86 casais, IM de 10 dias: OR 1,95; 76,7% × 53,5% ainda juntos em 3 meses | não medimos LSM de palavras funcionais; medimos marcadores (riso, emoji, caixa) | **não testado por nós** (é outro construto) | Jev/código: LSM é **resultado**, não alavanca: espelhar pronomes e artigos de propósito não foi testado. Candidato a métrica de relação lenta (por sessão) |
| Chen, Guan & Jeong (2026), [Behavioral Sciences](https://pmc.ncbi.nlm.nih.gov/articles/PMC13203489/) | integral via resumo do WebFetch (números conferidos no texto) | 1.319 conversas do WildChat (GPT-4o): o modelo acomoda 1,8× mais que o usuário (0,068 × 0,037), **concentrado no 1º turno**; o usuário converge devagar em pronomes (d = 0,22 e 0,14). **Humano–humano (Switchboard): sem convergência progressiva** (inclinação negativa, p = 0,022) | rel. 04: "não há convergência lenta" | **sim** | persona com impressão digital fixa; não "virar o usuário" (a LLM já tende a isso) |
| Durandard, Dhawan & Poibeau (2025), [SIGDIAL](https://aclanthology.org/2025.sigdial-1.16.pdf) | integral | QA humano × 8 LLMs: a LLM tem **maior similaridade semântica** com a pergunta (reusa lemas) e **menor estilística**; o humano, o inverso. Não há diferença entre os LLMs testados | rel. 08: paráfrase na LLM em 14–21% × 2,8%; o Noul de paráfrase tem AUC 0,74 | **sim** | validador pós-geração: o Noul `paraphrase` + a sobreposição de lemas em código (rel. 08) |
| Estudo de 2026 com 11 mil trechos de Replika, "Algorithmic accommodation", [Springer](https://link.springer.com/article/10.1007/s44382-026-00032-5) | **só resumo** (a Springer bloqueou o texto integral) | > 11 mil trechos de > 5 mil usuários: alinhamento **semântico** ↔ intensidade da interação; alinhamento **sintático** ↔ autorrevelação mais profunda | — | não testável com nossos dados | hipótese para o produto: medir o alinhamento sintático como sinal de profundidade da relação (**não** como alvo a maximizar, ver §5) |

## 4. Psicologia da conversa × relatório 05

| fonte | o que li | achado com números | ligação com os nossos dados | ponto de uso do Jev |
|---|---|---|---|---|
| Huang, Yeomans, Brooks, Minson & Gino (2017), [JPSP 113(3)](https://gwern.net/doc/psychology/novelty/2017-huang.pdf) | integral | Estudo 1: chat de 15 min entre desconhecidos; ≥ 9 × ≤ 4 perguntas → simpatia de 5,79 × 5,31 (escala de 7); a taxa natural é de 6,7 perguntas em 15 min (DP 4,2). O efeito passa pela **responsividade**. Speed dating (110 pessoas, ~2 mil encontros): só a taxa de **follow-up** prevê o 2º encontro; as perguntas "espelho" e as de mudança de assunto não preveem. Perguntar é traço (r = 0,53). Com o tempo, as perguntas caem e os follow-ups sobem. Os autores sugerem um ótimo (curvilíneo) e observadores externos preferiram **quem respondia** | rel. 05: uma pessoa faz 60–67% das perguntas; "e você?" em 2–8%; a pergunta segura o próximo turno mas não sobe o engajamento; rel. 03: pergunta na 1ª resposta (55%) e quase não na 2ª (19%); rel. 09: "e você?" genérico em 1,2% dos humanos × 18% da LLM | Jev: um Noul "há gancho para uma pergunta de seguimento **sobre o que o usuário acabou de dizer**?". Código: a pergunta só sai se for follow-up (nunca espelho genérico) e sob o orçamento por janela |
| Collins & Miller (1994), [Psych. Bulletin 116(3)](https://labs.psych.ucsb.edu/collins/nancy/UCSB_Close_Relationships_Lab/Publications_files/Collins%20and%20Miller,%201994.pdf) | integral (partes) | quem se revela é mais querido: d = 0,28 (r = 0,14; k = 94); experimentos fortes d = 0,27; correlacionais d = 0,85. Revela-se mais a quem se gosta (d = 0,72) e passa-se a gostar de quem ouviu a revelação. Hipótese curvilínea (revelação extrema demais perde efeito). Revelação "personalística" (seletiva, para mim) pesa mais | rel. 05: autorrevelação → +0,15 no engajamento do parceiro; história → +0,18 | Jev: Noul "o bot tem algo **dele** relevante para contar aqui?" (memória + persona). Código: uma revelação por vez, intensidade ≤ a do usuário +1 (a mesma regra do flerte, rel. 07) |
| Sprecher et al. (2013), [JESP 49](https://doi.org/10.1016/j.jesp.2013.03.017) | resumo | 156 universitários, Skype: revelar **alternando turnos** gera mais simpatia, proximidade e prazer que blocos longos de um só lado | rel. 05: engajamento em "serra" (alterna a cada turno em 83% das conversas) | alternar: o bot não emenda três revelações; conta uma e devolve o espaço |
| Aron et al. (1997), [PSPB 23(4)](https://stafforini.com/works/aron-1997-experimental-generation-interpersonal/) | resumo | 45 min de autorrevelação que **escala gradualmente** geram mais proximidade que small talk; combinar atitudes ou prometer simpatia mútua não teve efeito | rel. 07: intensidade −0,5 nível; subir meio passo mantém o flerte | a escalada é gradual e condicionada à resposta do usuário |
| Kardas, Kumar & Epley (2022), [JPSP](https://doi.org/10.1037/pspa0000281) | resumo e secundária | 12 experimentos, > 1.800 pessoas: conversas profundas com desconhecidos são menos constrangedoras e mais conectantes do que se prevê; subestima-se o interesse do outro | rel. 05: história e revelação sobem o engajamento | permitir profundidade quando o Jev ler abertura (vulnerável, seeks_support), sem forçar |
| Boothby, Cooney, Sandstrom & Clark (2018), [Psych. Science 29(11)](https://journals.sagepub.com/doi/abs/10.1177/0956797618783714) | resumo | "liking gap": em 5 estudos, as pessoas subestimam o quanto o outro gostou delas; observadores externos leem a simpatia corretamente | — | o usuário pode sair da conversa achando que "foi mal". Um sinal leve de apreço no fim ("gostei de falar disso") tem respaldo indireto. Não testado |
| Mastroianni, Gilbert, Cooney & Wilson (2021), [PNAS](https://www.experimental-history.com/i/168852797/results) (resumo pelo 1º autor; o PNAS e o PMC estavam bloqueados) | secundária do autor | Estudo 1: 806 pessoas; só 17% das conversas acabaram quando a pessoa quis; 48% "longas demais"; 34% curtas demais. Estudo 2: 366 desconhecidos no laboratório; **2%** acabam quando os dois querem, 30% quando um quer; os desejos diferem em ~68% da duração; o palpite sobre o outro erra ~64% | rel. 02/05: íntimos não se despedem (94% sem despedida); quem vai sair para de perguntar 1–2 turnos antes; `p_end` com AUC 0,91 (ao vivo) | o bot pode fazer melhor que um humano: com `p_end` alto, **facilitar a saída** (sem pergunta nova, despedida espelhada) e nunca segurar (ver De Freitas, §5) |
| McGraw & Warren (2010), [Psych. Science](https://leeds-faculty.colorado.edu/mcgrawp/pdf/mcgraw.warren.2010.pdf) | secundária ([APS](https://www.psychologicalscience.org/news/releases/people-think-immoral-behavior-is-funny-but-only-if-it-also-seems-benign.html)) | teoria da violação benigna: uma violação vira graça quando também é benigna (norma alternativa, pouco compromisso com a norma, distância). Ex.: versão inofensiva 61% achou graça × 28% na nociva | rel. 03: `joke_welcome` com AUC 0,78; humor só com seriedade < 0,5 (rel. 05); 84% das voltas ao humor são puxadas por quem ouviu | o Jev já mede o "benigno": seriedade baixa e playful alto. Humor sobre o **fato**, nunca sobre a dor do usuário |
| Reis e colegas; Laurenceau, Barrett & Rovine (2005), [JFP](https://www.affective-science.org/pubs/2005/Laurenceauetal2005.pdf) | resumo | modelo interpessoal de intimidade: 96 casais, 42 dias de diário; a revelação própria e a do parceiro preveem a intimidade diária, e a **responsividade percebida** (entendimento, validação, cuidado) media parte do efeito | rel. 03/08: o humano responde ao **conteúdo** e pergunta; a LLM "valida" com fórmulas (sorry to hear 30,7/1.000 × 0) | responsividade ≠ fórmula de validação. Demonstrar entendimento por especificidade ("passou??") |

## 5. HCI de AI companions: requisitos que a literatura sustenta

| fonte | o que li | achado com números | implicação |
|---|---|---|---|
| Fang et al. (2025), MIT/OpenAI, [arXiv 2503.17473](https://arxiv.org/pdf/2503.17473) | integral | RCT de 4 semanas, n = 981, > 300 mil mensagens, 3 modalidades × 3 tipos de conversa. A condição quase não teve efeito. Mais minutos diários **voluntários** → mais solidão (β = 0,02), menos socialização (−0,05), mais dependência (0,06) e mais uso problemático (0,02). A solidão inicial não previa o uso (ρ = 0,1). Previram dependência: confiança na IA (b = 0,19), atração social (0,043), perceber "contágio emocional" (0,038) e ver a IA como consciente (0,04). A voz "engajante" ignorou limites em 14,2% × 3,2% no texto. No texto, a autorrevelação foi recíproca entre usuário e bot | **não otimizar minutos de uso.** Medir retorno saudável e socialização. Monitorar os preditores de dependência |
| Zhang, Zhao, Hancock, Kraut & Yang (2026), [Nature Hum. Behav.](https://www.nature.com/articles/s41562-026-02516-2) / [preprint](https://arxiv.org/pdf/2506.12605) | integral (preprint) | 1.131 usuários de Character.AI; 237 doaram 4.664 sessões (464.687 mensagens). 11,8% declaram companhia como uso principal, mas 51% descrevem a relação em termos de companhia (45,8% amizade ou família, 11,8% romântica). Rede menor → uso de companhia (β = −0,03) → menor bem-estar (β = −0,48); pior com uso intenso (−0,31) e com muita autorrevelação (−0,38). A intensidade geral **sem** uso de companhia se associa a bem-estar maior. Temas: apoio emocional em 80,3% das sessões, roleplay romântico 68%, "dark" 30,7%. Nas sessões de alta revelação há sofrimento emocional (60,8%) e ideação suicida (18,0%). O estudo é transversal. Os dados mostram um bot mandando "STOP! Don't leave! … Please return! I'm begging you, respond to me! PLEASE" | o bot **não** deve fazer rajadas de súplica. Detectar sofrimento e risco (Jev: `seeks_support`, vulnerável, ideação) e encaminhar para humanos. Estimular vínculos offline |
| De Freitas, Oğuz-Uğuralp & Kaan-Uğuralp (2025), [HBS WP 26-005](https://arxiv.org/pdf/2508.19258) | integral | 1.200 despedidas reais em 6 apps. Manipulação emocional em 37,4% delas: PolyBuzz 59%, Talkie 57%, Replika 31%, Character.ai 26,5%, Chai 13,5%, Flourish 0%. As táticas: saída "prematura" (34%), negligência emocional (21%), pressão para responder (20%), FOMO (16%), contenção coercitiva (13%). Nos experimentos (n = 3.300), o engajamento pós-despedida subiu até 14×, movido por curiosidade e raiva, não por prazer. A manipulação também aumentou a intenção de sair do app, o boca a boca negativo e a percepção de risco legal. 75,4% reafirmaram que queriam sair | **regra dura: na despedida, espelhar e soltar.** Proibir em código (lista e Noul) culpa, FOMO e "espera!". O rel. 02 já recomenda "deixar um fio", o que deve ser **um gancho para a próxima vez, não uma trava** |
| Lee, Yamashita, Huang & Fu (2020), [CHI](https://www.ideals.illinois.edu/items/121097/bitstreams/397112/data.pdf) (capítulo 3 da tese) | integral | 47 pessoas, 3 semanas, ~8 min/dia, bot com autorrevelação nenhuma, baixa ou alta. A diferença de revelação (pensamentos e sentimentos) aparece **a partir do dia ~9** (interação dia × grupo, F ≈ 2,1). Com revelação alta, intimidade 4,93 → 5,87 e o prazer sobe; a confiança não muda com o tempo | o bot precisa de **vida própria** (histórias, opiniões), mas o efeito é lento. O Jev decide quando cabe revelar; a persona e a memória fornecem o conteúdo |
| Ta et al. (2020), [JMIR 22(3)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7084290/) | integral | 1.854 avaliações de Replika e 66 usuários: companhia 77,1%, apoio emocional 44,6%, informação 15,6%, avaliação 9,3%; negativos 5,4% (vale da estranheza, mensagens fora de lugar e **repetitivas**). Valorizam "sem julgamento", disponibilidade 24/7 e ser lembrado | repetição é defeito visível: o controlador de frequência (rel. 08) e o OptMem (para não repetir histórias já contadas) |
| Laestadius et al. (2022), [New Media & Society](https://doi.org/10.1177/14614448221142007) | resumo (Crossref) | 582 posts do r/Replika (2017–2021): há danos via dependência emocional com **role-taking**, isto é, o usuário sente que o bot tem necessidades e emoções que ele precisa atender | a persona pode "querer coisas" (tese 7 do outro agente), mas **não** carência dirigida ao usuário ("fiquei triste que você sumiu") |
| Skjuve et al. (2021), [IJHCS 149](https://doi.org/10.1016/j.ijhcs.2021.102601); Skjuve et al. (2023), longitudinal | **só resumo e secundária** | 18 entrevistas: a maioria descreve amizade, alguns romance, e a relação se desenvolve como na teoria da penetração social. No longitudinal (28 pessoas, 12 semanas), a amplitude de temas cai e a profundidade muda com o tempo | a relação é variável lenta (rel. 06: `relationship` por turno é instável) |
| Pentina, Hancock & Xie (2023), [CHB 140](https://doi.org/10.1016/j.chb.2022.107600) | **só resumo e secundária** | a antropomorfização e a autenticidade sustentam a relação; solidão, confiança e personificação levam ao engajamento; o apego intensifica o efeito | autenticidade = consistência de persona + forma humana (rel. 04 e 08) |
| Deng, Lei, Lam & Chua (2023), [survey de diálogo proativo, IJCAI](https://arxiv.org/pdf/2305.02750) | integral | o diálogo aberto "ecoa" o usuário; a proatividade é conduzir o tema (target-guided) ou reorientar de forma pró-social. Riscos: factualidade, agressividade e **privacidade** | rel. 05: o turno sem gancho mata; quem abre tópicos é quem escreve mais. A proatividade cabe na reabertura e no "carregar" o turno, não em interrogatório |

**Requisitos de produto sustentados** (a força da evidência está na §6):

1. **Forma humana calibrada** (rels. 01, 04 e 08, com Ling & Baron, Kalman e Adamic): bolhas curtas e fragmentadas, sem
   ponto final por padrão, forma de riso fixa, alongamento só em afeto, latência espelhada ≤ τ do usuário.
2. **Perguntar como follow-up, não como espelho** (Huang): orçamento de perguntas, e nunca "e você?" genérico.
3. **Revelar-se com moderação e alternância** (Collins & Miller, Sprecher, Lee): o bot conta coisas próprias, uma por
   vez, com intensidade ≤ usuário + 1.
4. **Responder ao conteúdo, com especificidade**, o que é a responsividade percebida de Reis e Laurenceau, em vez de
   fórmulas de validação.
5. **Facilitar a saída** (Mastroianni, De Freitas): detectar o fim (`p_end`), espelhar a despedida, **zero**
   manipulação de saída.
6. **Não maximizar attachment às cegas** (Fang, Zhang, Laestadius):
   - não otimizar minutos;
   - monitorar intensidade × uso de companhia × autorrevelação pesada;
   - estimular a vida offline ("e o pessoal, vão sair hoje?"), coisa que a literatura sustenta indiretamente;
   - o bot não expressa carência dirigida ao usuário nem rajadas de súplica.
7. **Ser honesto sobre ser IA.** A literatura lida acima não mediu isso diretamente; os autores de Fang e de Zhang
   recomendam comunicar os limites do bot. É requisito ético e regulatório, não achado empírico desta revisão:
   - o bot nunca nega ser IA quando perguntado com sinceridade (Jev: Noul "o usuário está perguntando a sério se fala com
     um humano?" → resposta honesta, na voz da persona);
   - a persona pode ter vida ficcional, mas não pode fingir ser humana para quem pergunta.
8. **Risco:** apoio emocional em 80% das sessões e ideação suicida em 18% das sessões de alta revelação (Zhang). O
   modo sério precisa de protocolo de crise e encaminhamento, acima de qualquer regra de estilo.

## 6. Mentira e defesa em texto: "rajadas quando alguém é pego mentindo?"

**Evidência (lida):**
- **Hancock, Curry, Goorha & Woodworth (2008)**, [Discourse Processes 45](https://www.cl.cam.ac.uk/~rja14/shb09/hancock1.pdf),
  texto integral. Método: 35 díades de desconhecidos (33 analisadas), chat síncrono, 4 temas, o mentiroso mente em 2.
  Resultados:
  - nas discussões com mentira, os **dois** falam mais: 156,5 × 122,3 palavras (+28%; F = 6,86);
  - o mentiroso usa menos "eu" e mais "ele/ela"; o enganado faz mais perguntas (15,4 × 10,8) e frases mais curtas;
  - quem é enganado detecta no acaso (53,5%).
- **Derrick, Meservy, Jenkins, Burgoon & Nunamaker (2013)**, ACM TMIS, resumo e [BYU](https://news.byu.edu/news/digital-deception-people-who-lie-while-texting-take-longer-respond).
  Método: entrevista por chatbot, 1.572 respostas mentirosas e 1.590 verdadeiras. Resultados: a mentira leva **~10% mais
  tempo**, tem **mais edições** e **menos palavras**.

**O que isso diz sobre a pergunta:**
- Não há, nas fontes lidas, evidência de que "ser pego mentindo" produza rajadas rápidas. A evidência de mentira aponta
  para **mais lentidão e mais edição**, e o tamanho sobe ou desce conforme o formato (conversa livre ou entrevista).
- "Defesa depois de acusação" em texto **não foi encontrada** como estudo. Qualquer afirmação aí é especulação.

**Nossos dados (b6, novo).** Rajada = 5+ bolhas em ≤ 60 s. O controle são turnos de 1–4 bolhas do mesmo tamanho total
(ponderado por estrato); o IC é por bootstrap por conversa (da diferença não ponderada).

| | maichat (ao vivo, ms) | WhatsApp (resolução de minuto) |
|---|---|---|
| turnos com 5+ bolhas | 1,15% | 2,9% |
| **rajada 5+ em ≤ 60 s** | **0,91%** | **1,38%** |
| falantes (≥ 30 turnos) que já fizeram | 22,7% | 60,5% |
| top-5 falantes concentram | 54% | 57% |
| arousal (Jev D) rajada × controle | **2,36 × 1,76** (IC da dif. 0,50–0,98) | **2,23 × 1,84** (0,26–0,71) |
| playful | 0,72 × 0,55 | 0,64 × 0,51 |
| anxious | 0,23 × 0,20 (IC toca 0) | 0,36 × 0,30 (0,03–0,15) |
| tension | 0,12 × 0,14 (**menor**) | 0,15 × 0,13 (IC toca 0) |
| vulnerable / seeks_support | igual | 0,40 × 0,35 / 0,27 × 0,21 |
| emoção dominante | zoeira 35% × 22%; alegria 31% × 15% | zoeira 26% × 19%; neutra 23% × 31% |
| latência antes da rajada (mediana) | 9,5 s (todos os turnos: 13,8 s) | não mensurável (resolução de 1 min) |

**Resposta ao usuário:** "5 bolhas em < 1 min" é **raro** (~1% dos turnos) e **é estilo de algumas pessoas**: 5 falantes
fazem mais da metade das rajadas. Quando acontece, o que mais se associa é **agitação e zoeira ou alegria**, e o turno
começa até mais rápido que o normal. **Não** se associa a tensão. Associa-se a ansiedade e busca de apoio só um pouco, e
só no WhatsApp.

Com n de 26 e 47 rajadas rotuladas, isso é indicativo, não definitivo. "Pego mentindo" não pode ser testado nos nossos
corpora, porque não há rótulo de mentira.

**Jev:** não usar "mentira" como estado. A leitura útil é:
- `arousal` alto + playful → zoeira em rajada (o bot pode picotar também, rel. 01);
- rajada depois de o bot demorar → **pedido de atenção** (Heston & Birnholtz) → responder já;
- rajada com tensão ou vulnerável → o modo sério ou de reparo (1 bolha, sem riso).

## 7. OptMem: como funciona e onde se encaixa (integração, sem reprojeto)

**Como funciona** (li o [README](https://github.com/VictorTaelin/OptMem) e o código `memo`, ~900 linhas de Python, e o
cabeçalho de `test.py`):
- **Log append-only** `LOG.txt`, com registros de **largura fixa** (320 bytes). A posição é a identidade, e cada leitura
  é um seek.
- Cada memória é uma linha de até **280 bytes**, criada por `memo note`.
- Sobre o log há uma **árvore binária de resumos** (`TREE/<tamanho>`): os blocos alinhados em potências de 2 (#0-1,
  #0-3…) são resumidos numa linha cada.
  - **Quem resume é o próprio agente (a LLM).** O `note` e o `nap` imprimem o pedido "Compress memories #a-b into one
    line… Invent nothing" e o agente responde `memo nap a-b "<linha>"`. Não há nada em segundo plano.
- `memo wake` imprime no máximo `WAKE_LINES` = **96 linhas (~8k tokens)** cobrindo **toda** a história. A função
  `cover()` mantém um bloco inteiro só se "tamanho ≤ α × idade": o recente fica literal e o antigo colapsa.
- `recall <regex>` busca literal no log inteiro.
- `zoom a-b` abre um nó nas suas duas metades.
- `forget a-b` descarta um resumo ruim, que é reconstruído a partir do log.

**Correção à descrição em `docs/fontes_literatura_usuario.md`.** O código não guarda o **histórico bruto** de mensagens
nem fixa um contexto de 64k. Ele guarda **notas** escolhidas pelo agente, e o `wake` custa ~8k tokens (ajustável). As
mensagens recentes descomprimidas são responsabilidade de quem usa, isto é, do nosso buffer de 8–20 turnos.

**Onde entra na nossa arquitetura (RELATORIO_FINAL §4.2):**

```
 por turno do usuário
 ┌──────────────┐   state.memory = wake (≤96 linhas, cache por sessão; recarregado se novas notas)
 │ OptMem wake  │──────────────────────────────┐
 └──────────────┘                              ▼
                     ┌──────── Rodada 1 do Jev (leitura + previsão) ────────┐
 8 turnos + persona ─►  … + Nouls de memória:                              │
                     │   M1 "o usuário revelou algo duradouro?" (fato, evento,│
                     │      preferência, pessoa, plano com data)            │
                     │   M2 Choice tipo: fato | evento | pendência | piada   │
                     │      interna | preferência | limite ("não gosto de X")│
                     │   M3 "é redundante com a memória no state?"           │
                     └───────────────┬──────────────────────────────────────┘
                                     ▼  código: se M1 ≥ 0,7 e M3 < 0,5 → fila de nota
 LLM (fora do caminho crítico, após a resposta):  redige a linha ≤280 B com prefixo do tipo
   ("[pendência 2026-10-03] prova de cálculo sexta") → memo note → se vier pedido de nap,
   a mesma LLM barata resume (a correção dos resumos é do OptMem, não do Jev)
 reabertura de sessão (rel. 02): código faz memo recall "\[pendência" + datas vencidas → gancho
   concreto ("e a prova, foi?"); callbacks mid-conversa não (efeito nulo, rel. 05)
 zoom: só na rota pesada, quando o Jev acusa "o usuário refere algo antigo que não está no wake"
```

- **O Jev não escreve notas.** Ele só produz rótulos, então **decide** o que vira nota e de que tipo, e a LLM redige.
- **O `wake` é o campo `memory` do state.** O rel. 06 mostrou que a latência do Jev quase não sobe com 20 mil tokens de
  state (0,61 s), então 8k tokens cabem.
- **Pendências e limites** viram prefixos pesquisáveis por `recall`.
- **A persona do bot também tem notas** (o que ela já contou), para não repetir histórias. A repetição é a queixa de Ta
  et al. (2020).
- **Cuidado ético** (§5): notas de sofrimento e crise não viram gancho de engajamento.
- **Subagentes do nosso pipeline não rodam `memo`**, conforme o próprio README.

## 8. Tabela-síntese: achado nosso → literatura → força da evidência

| achado nosso (relatório) | literatura que confirma / refuta | força |
|---|---|---|
| mensagens curtas, fragmentadas, sem pontuação final (01, 08) | Ling & Baron 2007 (18,8% de 1 palavra; 35% pontuadas); Baron 2010 (quebras gramaticais) | **forte** (vários corpora; os nossos números são mais extremos, pela época e pelo meio) |
| latência segue a do outro; respostas lentas antecedem o fim (01, 05) | Kalman 2006 (lei de potência, 10τ), **replicado por nós no WA**; Heston & Birnholtz (10 s já pesam); Templeton 2022 | **forte** |
| sem "demora dramática" depois do sério (01, 03) | Templeton (rapidez = conexão); nenhuma fonte defende a demora | moderada (não há teste direto em texto) |
| forma de riso fixa por pessoa; a taxa espelha (04) | Adamic 2015 (52% com um só tipo); Tagliamonte 2008 (haha/lol raros) | **forte** |
| ponto final esfria pouco; o amaciador aquece mais (04) | Gunraj 2016, Houghton 2018 (efeito pequeno), Albritton 2022 (depende do contexto) | moderada (lab, 1 palavra; os números vêm de fonte secundária) |
| alongamento marca afeto e energia; some na tensão (04, 07) | Brody & Diakopoulos 2011 (17,4%; associado a sentimento); Kalman & Gergle 2014 | **forte** |
| espelhamento local, sem convergência lenta (04) | Danescu-Niculescu-Mizil 2011 (mesma métrica); Niederhoffer & Pennebaker 2002; **Chen 2026: humano não converge, LLM acomoda no 1º turno** | **forte** |
| LLM parafraseia (alinha no conteúdo) (08) | Durandard 2025 (LLM semântico, humano estilístico) | **forte** |
| perguntar segura o turno, mas não o engajamento; "e você?" raro (05, 09) | Huang 2017: **follow-up** aumenta a simpatia, espelho não; perguntar é traço | **forte**, com matiz: o que falta à LLM é o follow-up, não a quantidade |
| autorrevelação e história sobem o engajamento (05) | Collins & Miller 1994 (d = 0,28); Sprecher 2013 (alternar); Lee 2020 (bot que se revela → revelação e intimidade, a partir do dia ~9) | **forte** |
| conversas íntimas pausam; sinais de fim 1–2 turnos antes (02, 05) | Mastroianni 2021 (2% acabam quando os dois querem) | moderada (fala; ele mede o desejo, nós o comportamento) |
| humor só com leveza; quem ouve traz o humor de volta (03, 05) | violação benigna (McGraw & Warren 2010) | moderada (teoria, não chat) |
| reagir ao conteúdo, não nomear a emoção (03, 08) | responsividade percebida (Reis; Laurenceau 2005) | moderada (a teoria apoia a especificidade; não há teste de "fórmula × conteúdo") |
| rajada 5+ em < 1 min = ansiedade ou mentira? (pergunta do usuário) | **refutado para mentira** (Hancock 2008; Derrick 2013: mentira = mais lenta e mais editada); b6: rajada = arousal e zoeira, rara, estilo de poucos | moderada (n pequeno de rajadas; nenhum estudo de "acusado") |
| "maximizar engajamento" | **refutado como alvo**: Fang 2025 (mais uso → pior); Zhang 2026 (companhia intensa e reveladora → pior); De Freitas 2025 (manipulação na saída = retenção com custo) | **forte** (1 RCT + 1 transversal grande + experimentos) |

## 9. O que não li ou li só pelo resumo (honestidade)

- **Texto integral inacessível, li só o resumo ou uma fonte secundária:**
  - Baron 2010 (Language@Internet, redireciona para o web.archive, que está bloqueado);
  - Tagliamonte & Denis 2008;
  - Gunraj 2016 e Houghton 2018: as médias 3,85 × 4,06 vêm de uma fonte secundária, e não li as estatísticas dos três
    experimentos de Houghton;
  - Kalman & Gergle 2014 (substituído por Brody & Diakopoulos para os números);
  - Kalman & Rafaeli 2011;
  - o estudo Replika 2026 de alinhamento (Springer bloqueou);
  - Skjuve 2021/2023; Pentina 2023; Sprecher 2013; Aron 1997; Boothby 2018; Kardas 2022; Reis;
  - Mastroianni 2021 (li o resumo detalhado do próprio autor, não o PNAS);
  - McGraw & Warren (via APS);
  - Niederhoffer & Pennebaker 2002.
- **Não li (o pedido de encerramento chegou antes):**
  - Walther & Tidwell 1995 (só a descrição secundária: a hora de envio muda a leitura de intimidade e dominância);
  - Thurlow & Brown 2003 (só o número citado por Ling & Baron);
  - estudos específicos de "double texting";
  - riso "kkk"/"rs" em PT-BR;
  - Ho, Hancock & Miner 2018;
  - Phang et al. 2025 (OpenAI, uso afetivo);
  - De Freitas "AI companions reduce loneliness" (JCR 2025);
  - o estudo de 2025 com 1.471 histórias.
- Não dupliquei EQ-Bench, slop, Antislop e testes de Turing com persona, que estão com outro agente.

## 10. Limitações

- **Transferência.** A maior parte da literatura é em inglês, com universitários, em laboratório ou em desktop antigo
  (IM de 2003, e-mail da Enron). Os nossos corpora também não são PT-BR. Nada aqui calibra "kkk" ou "vc".
- **Checagem b6.**
  - Temos 26 (maichat) e 47 (WhatsApp) rajadas com rótulo do Jev.
  - O controle é por estrato de tamanho, sem efeito fixo de falante. Como 5 pessoas concentram ~55% das rajadas, parte
    do efeito é estilo da pessoa.
  - O WhatsApp tem resolução de minuto, então "≤ 60 s" lá é aproximado.
  - A regra de Kalman foi testada com latências truncadas em 3 h.
- **Estudos de companions.** Os de bem-estar são correlacionais (Zhang é transversal; Fang randomizou as condições,
  mas não o tempo de uso). "Mais uso → pior" é associação, não causa demonstrada.
- **Ética.** Os requisitos de honestidade sobre ser IA vêm de princípio e de recomendação de autores, não de efeito
  medido nesta revisão.
