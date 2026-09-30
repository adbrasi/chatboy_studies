# Dados e pipeline

Todos os dados brutos ficam em `data/raw/` e os processados em `data/processed/`, ambos fora do git.
Para regenerar: rode `scripts/normalize.py` → `scripts/features.py` → `scripts/annotate_base.py`.

## Corpora

| corpus (campo `corpus`) | Origem | Tipo | Tamanho | O que tem de especial |
|---|---|---|---|---|
| `maichat` | MaiChat (Univ. Edinburgh, LREC 2026, CC BY-SA 4.0) | chat 1:1 em inglês, ao vivo, entre pessoas que já se conhecem | 42 conversas, 4.177 mensagens, 2.871 turnos | **Logs de digitação**: cada estado intermediário do texto, com timestamps em ms (pausas, apagamentos, texto abandonado) |
| `whatsapp_nl` | WhatsApp Corpus Berntzen (DANS, doi:10.17026/DANS-XZZ-UGTW, CC BY 4.0) | WhatsApp real doado, **holandês**, 2012–2014 | 60 chats (57 diádicos), 60.469 mensagens, 33.896 turnos, 3.744 sessões | Conversas reais do dia a dia (família, amigos, casais), timestamps com resolução de minuto, marcadores de mídia (`<afbeelding weggelaten>` = imagem) |
| `nps_chatroom` | NPS Chat Corpus (NLTK) | salas de chat públicas, 2006, em grupo | 10.567 posts (15 sessões) | **Atos de diálogo anotados à mão** (Greet, Emotion, ynQuestion, whQuestion, Bye, Accept, Reject…); muito flerte e muitas aberturas entre desconhecidos |
| `nus_sms` | NUS SMS Corpus 2015 | SMS reais, em inglês, sobretudo de Singapura | 55.835 mensagens | Não são conversas, só mensagens avulsas por remetente; úteis para estilo (abreviações, informalidade) |
| `empathetic` | EmpatheticDialogues (Meta, CC BY-NC 4.0) | chat humano-humano induzido por crowdworkers | 24.850 conversas, 107.220 falas | **Rótulo-ouro de emoção** (32 classes) por conversa e a situação descrita; A = quem conta a situação, B = quem escuta |

## Arquivos processados

- `messages.jsonl`: uma linha por mensagem: `corpus, conv_id, idx, speaker, text, ts, lang` + campos específicos
  (`typing` no maichat, `dialogue_act` no nps, `gold_emotion`/`situation` no empathetic, `media` no whatsapp).
  - `typing` (maichat): `compose_s` (1º estado digitado → envio), `n_states`, `n_deletion_events`, `chars_deleted`,
    `peak_len` (maior comprimento atingido; se > len(final) houve texto abandonado), `max_pause_s`,
    `n_pauses_over_2s`, `idle_before_typing_s` (mensagem anterior no chat → começar a digitar).
- `messages_feat.jsonl`: o mesmo, mais `f` = features determinísticas (regex/código): `n_chars, n_words, has_q,
  n_excl, laugh, laugh_text, n_emoji, emoticon, emoji_only, elongation, caps_word, n_slang, ellipsis, ends_punct,
  starts_lower, self_correction, greeting, farewell, media, url`.
- `turns.jsonl`: um **turno** é a sequência máxima de mensagens consecutivas do mesmo falante (o "burst").
  Campos: `n_msgs` (quantas bolhas), `total_chars`, `mean_chars`, `burst_span_s` (1ª→última bolha do burst),
  `response_latency_s` (última msg do outro → 1ª msg deste turno; só na mesma sessão), `session`
  (nova sessão após mais de 3 h de silêncio), `turn_in_session`, flags agregadas (laugh, n_emoji, has_q, media,
  greeting, farewell, self_correction) e, no maichat, `typing` agregado (`compose_s_total, chars_per_s,
  chars_deleted, deletion_ratio, n_deletion_events, max_pause_s, abandoned_text`).
- `jev_base.jsonl`: camada base de anotação Jev (ver abaixo), um registro por turno anotado (todos os turnos
  do maichat, mais janelas contíguas de até 60 turnos de cada chat do whatsapp_nl).
- `jev_cache.jsonl`: cache de todas as chamadas ao Jev (hash de estado+perguntas → respostas).

## Camada base Jev (`scripts/annotate_base.py`)

Para cada turno T do falante S:

- **D (descritivo)**: o estado tem os 8 turnos anteriores da mesma sessão e o T (com as mensagens de T **juntadas
  em um texto só**, para o Jev não ver a fragmentação; assim, emoção × nº de bolhas não fica circular).
  Perguntas: `emotion` (choice, 12 classes), `valence` (score 0–4), `arousal` (0–4), `anxious`, `playful`, `flirting`,
  `vulnerable`, `seeks_support`, `tension`, `hook`, `topic_shift`, `mirrors` (nouls), `seriousness` (0–3),
  `intent` (choice, 13), `phase` (choice, 7), `relationship` (choice, 6), `engagement` (0–4).
- **P (preditivo)**: o estado vai só até o turno anterior do parceiro; o T **não aparece**. Pergunta como será o
  próximo turno de S: `p_n_msgs` (1/2/3/4+), `p_length` (0–4), `p_laugh`, `p_question`, `p_emoji`,
  `p_topic_shift`, `p_end`, `p_joke_welcome` (nouls), `p_tone` (choice), `p_emotion` (choice).
  Depois comparamos com o que S realmente fez. É exatamente a posição do chatbot na hora de responder.

Formato das respostas: noul → `{"type":"noul","noul":p}`; score → `{"score":valor_esperado,"probabilities":{…},
"confidence":c}`; choice → `{"choice":rótulo,"probabilities":{…},"confidence":c}`.

## Cliente Jev (`scripts/jev.py`)

`POST https://openrouter.ai/api/v1/systemone`, modelo `typesafe/jev-1.13` (a chave vem de `OPENROUTER_API_KEY` ou
de `.env`). `ask(state, questions)` faz uma chamada com cache; `ask_many([(state, questions), ...], workers=N)`
roda em paralelo. Os helpers `noul()`, `choice()` e `score()` montam as perguntas, e `top()` extrai o valor escalar
de uma resposta.
Custo medido: ~US$0,00004 por 1k tokens de entrada, com latência de ~0,6–0,9 s por chamada via OpenRouter.
