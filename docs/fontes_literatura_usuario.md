# Fontes indicadas pelo usuário (pesquisa feita por outro agente, colada na conversa)

Lista de papers e linhas de pesquisa a ler e cruzar com os nossos achados:

- **Meena**, "Towards a Human-like Open-Domain Chatbot" (2020): SSA (sensibleness + specificity).
- **Recipes for Building an Open-Domain Chatbot / BlenderBot** (2021): a boa conversa mistura conhecimento, empatia,
  personalidade, consistência e interesse.
- **LaMDA** (2022): Sensibleness, Specificity, Interestingness.
- **PersonaChat**, "Personalizing Dialogue Agents…" (2018): persona consistente.
- **EmpatheticDialogues** (2019) + **ESConv** (2021): estratégias de suporte emocional (pergunta, reflexão de sentimentos,
  reafirmação, informação, sugestão, self-disclosure).
- **LIGHT**, "Learning to Speak and Act in a Fantasy Text Adventure Game" (2019): diálogo condicionado pelo mundo.
- **Generative Agents** (Park et al., 2023): memória → reflexão → planejamento → comportamento.
- **SOTOPIA** (ICLR 2024): inteligência social em interações.
- **LoCoMo** (ACL 2024) e **LongMemEval** (ICLR 2025): memória em conversas longas.
- **RoleLLM / RoleBench** (ACL 2024): benchmark de roleplay.
- **InCharacter** (ACL 2024): personalidade consistente medida por entrevistas psicológicas.
- **CharacterEval** (ACL 2024): avaliação multidimensional de RP; reward models especializados > GPT-4 como juiz.
- **CharacterBench** (AAAI 2025): 3.956 personagens, 11 dimensões.
- **CoSER** (ICML 2025): personagens de livros, com pensamentos internos (atuação dramática).
- **RMTBench** (2025): roleplay multi-turno centrado na intenção do usuário.
- **PersonaEval** (2025): LLMs erram quem é quem em diálogos de RP (humanos 90,8%, melhor LLM ~69%); põe em dúvida o
  LLM-as-a-judge.
- **Survey de avaliação de Role-Playing Agents** (ACL 2025): 1.676 trabalhos.
- **RoleCDE** (ACL 2026): conflito entre fidelidade ao personagem e outras instruções.
- **The Curious Case of Neural Text Degeneration** (Holtzman et al.): sampling e texto insípido.
- **WritingBench**: escrita como capacidade separada.
- Estudo de 2025 com 1.471 histórias: críticos, estudantes e leitores valorizam coisas diferentes.
- **Replika**: estudos de HCI (disponibilidade, não julgamento, self-disclosure, evolução da relação); self-disclosure do
  chatbot aumenta a confiança e a reciprocidade.
- Estudo de 2026 com 11 mil trechos de Replika: alinhamento semântico ↔ intensidade; alinhamento sintático ↔
  self-disclosure mais profundo.
- **MIT/OpenAI**, estudo longitudinal de 4 semanas (981 participantes, 300 mil mensagens): uso, solidão e dependência.
- **Nature Human Behaviour** (ago/2026), Character.AI: 1.131 usuários, 464.687 mensagens; uso de companhia × bem-estar.
- **Proactive conversational AI** (survey): passividade dos agentes.
- **Linguistic accommodation / Communication Accommodation Theory.**
- **SillyTavern**: Character Cards, World Info/Lorebook, Author's Note (quanto mais perto do fim, mais influência), Data
  Bank/Vector Storage, Prompt Manager.

## Teses do texto do outro agente (para validar, refutar ou complementar com os nossos dados)

1. LLM bom ≠ chatbot bom ≠ personagem bom ≠ relacionamento convincente.
2. O pipeline atual é "persona + histórico → próxima mensagem"; o desejável é "persona + memórias + estado psicológico
   + relação + mundo + objetivos + modelo do usuário + cena → **intenção comunicativa** → texto".
3. **Relationship state** (trust, comfort, romantic_interest, resentment, pendências, piadas internas) é a variável
   subestimada: a mesma frase no dia 2 e no dia 200 deve gerar respostas diferentes.
4. **Assistant prior**: reconhecer → explicar → validar → oferecer ajuda → perguntar. É ótimo para assistente e péssimo
   para roleplay.
5. **Subtexto**: humanos mostram, não descrevem ("ah, então você foi com ela").
6. Exemplos comportamentais ("quando desconfortável: desvia com humor seco, responde curto") valem mais que adjetivos
   ("tsundere").
7. O personagem precisa **querer** algo; se só quer agradar, vira espelho bajulador.
8. Iniciativa e agência; reparo de mal-entendidos; consistência imperfeita (estáveis são os mecanismos, não a superfície).
9. Avaliação longitudinal em 20 dimensões (sensibleness, specificity, interestingness, fidelidade, memória, iniciativa,
   subtexto, assistant-ness…); LLM-as-a-judge é perigoso em RP (PersonaEval).

## Memória (fora de escopo: já resolvida pelo usuário)

**OptMem** (github.com/VictorTaelin/OptMem): o histórico inteiro vira um log append-only comprimido hierarquicamente numa
árvore binária de resumos. Com ~64k tokens de contexto fixo, as mensagens recentes ficam descomprimidas (~8k tokens) e há
`zoom()` para navegar pelo passado e `recall <regex>`. O usuário considera a memória resolvida. Não gastar esforço aí;
apenas mostrar onde ela se encaixa na arquitetura (ex.: o Jev decide o que vira `note`, e o state do Jev recebe o `wake`
resumido).
