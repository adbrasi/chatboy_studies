# chatboy_studies

Estudo sobre o que torna uma conversa de chat realista e engajante, a partir de conversas humanas reais, e sobre como
usar o **Jev** (TypeSafe, modelo "System One") como sistema nervoso de uma LLM rápida: o Jev lê o momento, o código decide
e a LLM só escreve.

**Comece por [`FUNDACAO.md`](FUNDACAO.md)** (o desenho final do sistema, uma página) e depois [`RELATORIO_FINAL.md`](RELATORIO_FINAL.md), que traz a síntese, as respostas às perguntas, a tabela "o que
escrever em cada momento", a arquitetura proposta e o resultado do experimento A/B.

## Estrutura

| caminho | conteúdo |
|---|---|
| `FUNDACAO.md` | **o mapa final**: 5 peças, 4 objetos de estado, cada escolha ligada à evidência |
| `RELATORIO_FINAL.md` | síntese de todas as rodadas e detalhes da proposta |
| `analysis/01…09_*.md` | 1ª rodada: ritmo, aberturas, emoção, estilo, engajamento, confiabilidade do Jev, flerte, vocabulário × LLM, A/B |
| `analysis/10…16_*.md` | 2ª rodada: arquiteturas de chamadas do Jev (juiz, entrega/rajadas, o que escrever), controle de saída, literatura e benchmarks |
| `analysis/17…19_*.md` | 3ª rodada: laboratório de geração, estado da relação + postura do personagem, variantes do juiz |
| `analysis/data/` | saídas pequenas das análises (json/csv/jsonl) |
| `docs/DATA.md` | corpora, esquemas e camada base de anotação do Jev |
| `scripts/` | pipeline: `download.sh` → `normalize.py` → `features.py` → `annotate_base.py`; clientes `jev.py` e `llm.py` |
| `scripts/analysis/` | scripts de cada relatório (prefixos `a1_`–`a9_`, `b1_`–`b7_`, `c1_`–`c2_`, `d1_`) |

Os dados brutos e processados ficam em `data/` e não são versionados. Para usar os clientes, ponha a chave do OpenRouter
em `.env` como `OPENROUTER_API_KEY=...`.

## Corpora

MaiChat (Univ. of Edinburgh, CC BY-SA 4.0) · WhatsApp Corpus Berntzen (DANS, CC BY 4.0) · NPS Chat Corpus (NLTK) ·
NUS SMS Corpus · EmpatheticDialogues (Meta, CC BY-NC 4.0).
