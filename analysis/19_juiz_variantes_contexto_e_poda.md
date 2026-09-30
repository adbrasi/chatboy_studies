# 19 · Juiz de "cara de LLM": as variantes que faltavam (rubrica, referência do falante, exemplos, frase culpada)

> Prefixo `d1`. Script em `scripts/analysis/d1_judge_variants.py`; resultados em `analysis/data/d1_results.json`.
> O relatório 10 (b1) fechou sem rodar essas quatro arquiteturas. Aqui elas foram medidas com as mesmas unidades, a mesma
> divisão dev/teste por conversa, as mesmas features de código e o mesmo banco enxuto de 9 perguntas.
> Amostra: todas as respostas humanas mais até 2 de LLM por contexto (1 instruída, 1 não instruída). Dev: 612; teste: 579.
> Gasto: ≈4.300 chamadas ao Jev (US$ 0,13), sem LLM. IC 95% por bootstrap por conversa.

## Resumo

- **Pôr contexto extra no state não ajuda o juiz.** A rubrica com as nossas taxas medidas, 6 mensagens reais do próprio
  falante e 6 exemplos rotulados "pessoa × chatbot" não melhoram o que já temos com código + o banco de 9 perguntas
  atômicas:

| combinação (regressão logística treinada no dev) | AUC no teste (geral) | só saídas instruídas | só saídas não instruídas |
|---|---|---|---|
| só código | 0,846 [0,80–0,89] | 0,792 | 0,900 |
| código + banco enxuto (9 perguntas) | **0,887** [0,85–0,92] | 0,830 | 0,945 |
| + V1: rubrica no state | 0,887 | 0,829 | 0,945 |
| + V2: referência ao estilo do próprio falante | 0,890 | 0,835 | 0,946 |
| + V3: exemplos rotulados no state | 0,889 | 0,833 | 0,944 |
| + V1 + V2 + V3 | 0,891 [0,85–0,92] | 0,837 | 0,946 |

- **A pergunta holística continua falhando mesmo com a rubrica ao lado:** "parece coisa de amigo, segundo a descrição?"
  0,48; "tem os traços de chatbot descritos?" 0,53; a escala amigo ↔ chatbot 0,51. A rubrica não resgata a pergunta
  holística. Isso confirma o relatório 10: o Jev funciona com perguntas atômicas e concretas.
- **As perguntas relativas ao próprio falante dão sinal fraco, mas real:** "mais entusiasmada que ele costuma escrever?"
  0,63; "mais marcadores (emoji, riso, gíria)?" 0,63; "mais longa?" 0,59. "Parece a mesma pessoa?" dá 0,40, invertida. O
  Jev acha "a mesma pessoa" a versão mais caprichada, o mesmo viés visto antes.
- **Os exemplos rotulados** ("parece mais com os exemplos de chatbot?") dão 0,57–0,58 sozinhos: pouco.
- **Poda frase a frase (V4, 177 respostas de LLM com 2–6 frases).** Medido pelo escore de "cara de LLM" do modelo de código,
  onde o original está em 0,80 e o humano em 0,385:

| variante | escore de LLM (↓ melhor) | faz sentido como resposta (Jev) | ainda responde ao que foi dito (Jev) |
|---|---|---|---|
| original | 0,799 | — | — |
| Jev: manter só a frase essencial | **0,466** | 0,859 | 0,678 |
| regra: manter só a 1ª frase | 0,482 | 0,871 | 0,724 |
| Jev: apagar a frase "culpada" | 0,663 | 0,876 | 0,760 |
| regra: apagar a última frase | 0,591 | 0,898 | 0,799 |
| regra: apagar as frases com "?" | 0,623 | 0,894 | 0,821 |
| Jev: apagar todas as frases marcadas como enchimento | 0,766 | — | — |

  **O Jev escolhendo a frase não bate as regras simples.** "Manter a 1ª frase" empata com "o Jev escolhe a essencial" e
  preserva um pouco mais o sentido. "Apagar a última" é melhor que "o Jev apaga a culpada". O relatório 13 viu o mesmo.
  Conclusão prática: **podar é tarefa de código**. O Jev fica com o diagnóstico: um Noul por vício diz O QUE está errado e
  qual regra aplicar.

## Conclusão sobre o juiz

Com todas as arquiteturas testadas nos relatórios 10 e 19, o melhor juiz é **código + um banco de perguntas atômicas do Jev**
(9 a 85 perguntas numa chamada), combinados por regressão logística: AUC ≈ 0,89–0,91 no geral e 0,83 nas saídas instruídas.
Não ajudaram:
- enriquecer o state com rubrica, exemplos ou o estilo do falante;
- cascata e encadeamento (relatório 10);
- paráfrases em ensemble;
- a poda frase a frase.

O ganho real do Jev está em diagnosticar vícios atômicos (AUC 0,82–0,98 por vício) e em somar +0,04–0,05 ao código.

## Limitações

- As respostas instruídas são poucas e de poucos atores. O escore de "cara de LLM" usado para avaliar a poda é o próprio
  modelo de código, que favorece regras parecidas com as suas features. A coerência foi julgada pelo Jev e não por humanos.
