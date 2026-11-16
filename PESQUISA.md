# Fundamentacao Teorica: De Redes Neurais Recorrentes (RNNs) aos LLMs Modernos

**Instituicao:** Centro Universitario Presidente Antonio Carlos (UNIPAC Barbacena)
**Curso:** Ciencia da Computacao
**Disciplina:** Inteligencia Artificial
**Professor:** Felipe Roncalli de Paula Carneiro
**Aluno:** Lucas Cimino Alves

---

## 1. O Gargalo Sequencial (RNNs)

### 1.1. Processamento Serial
As Redes Neurais Recorrentes (RNNs) processam sequencias de dados de forma estritamente linear ao longo do eixo temporal ($t = 1, 2, \dots, T$). Em cada instante de tempo $t$, o estado oculto $h_t$ e calculado por meio de uma funcao de transicao recorrente aplicada sobre a entrada atual $x_t$ e o estado oculto imediatamente anterior $h_{t-1}$:

$$h_t = f(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$$

Essa relacao cria uma dependencia de dados direta e inquebravel: o vetor $h_t$ nao pode ser computado enquanto $h_{t-1}$ nao tiver sido totalmente avaliado, o que exige o calculo previo de $h_{t-2}$, e assim sucessivamente ate $h_0$. Arquiteturalmente, mesmo variantes com portas de controle de fluxo de gradiente, como LSTM (*Long Short-Term Memory*) e GRU (*Gated Recurrent Unit*), preservam essa cadeia causal de estados ocultos. Como consequencia, torna-se impossivel dividir uma mesma sequencia de entrada em segmentos independentes para processamento simultaneo ao longo da dimensao temporal; a complexidade sequencial de operacoes cresce linearmente em $O(T)$, impedindo a paralelizacao intra-sequencia.

### 1.2. Ociosidade de Hardware
As Unidades de Processamento Grafico (GPUs) modernas — como arquiteturas baseadas em *Tensor Cores* — sao projetadas sob o paradigma SIMD/SIMT (*Single Instruction, Multiple Threads*), alcancando maximo desempenho quando executam milhares de operacoes matriciais densas (GEMM - *General Matrix Multiply*) de maneira simultanea e massivamente paralela.

Quando uma RNN e executada, a dependencia sequencial obriga a GPU a realizar multiplicacoes matriz-vetor pequenas, passo a passo, sincronizando a memoria global e os registradores a cada token da sequencia. Enquanto a unidade logica calcula o passo $t$, a vasta maioria dos milhares de nucleos CUDA e *Tensor Cores* permanece ociosa, aguardando a conclusao da etapa atual para iniciar a proxima. Alem disso, o custo de transferencia de dados entre a memoria de alta largura de banda (HBM) e a cache SRAM da GPU torna-se o fator limitante (*memory-bandwidth bound*), subutilizando severamente a capacidade de computacao em FLOPs do acelerador e inviabilizando o escalonamento do treinamento para corpora na ordem de trilhoes de tokens.

---

## 2. A Quebra de Paradigma (Transformers)

### 2.1. Revolucao de 2017
A publicacao do artigo *"Attention Is All You Need"* por Vaswani et al. (2017) redefiniu a arquitetura de processamento de linguagem natural e, posteriormente, de toda a inteligencia artificial generativa. Ate entao, o estado da arte em traducao automatica e modelagem de sequencias dependia de arquiteturas *Encoder-Decoder* baseadas em RNNs ou redes convolucionais (CNNs) acopladas a mecanismos de atencao auxiliares. A grande ruptura do trabalho foi eliminar completamente as camadas recorrentes e convolucionais, demonstrando que uma arquitetura baseada exclusivamente em mecanismos de autoatencao (*Self-Attention*) e redes *feed-forward* pontuais atingia qualidade superior com uma fracao do custo computacional de treinamento. Essa mudanca destravou as leis de escala (*Scaling Laws*), permitindo o treinamento paralelo de modelos com bilhoes e trilhoes de parametros.

### 2.2. Abandono da Recorrencia
O mecanismo de *Self-Attention* substitui a propagacao sequencial de estados ocultos por uma operacao matricial global aplicada simultaneamente a todos os tokens da sequencia de entrada $X \in \mathbb{R}^{T \times d_{model}}$. Em vez de transmitir informacoes passo a passo atraves de uma cadeia de estados (onde sinais distantes sofrem atenuacao ou explosao de gradiente), a autoatencao conecta qualquer par de posicoes $(i, j)$ na sequencia com um caminho de comprimento constante $O(1)$.

Como nao ha dependencia temporal entre o calculo das representacoes dos diferentes tokens dentro de uma mesma camada durante o treinamento, toda a matriz de entrada e projetada e processada em paralelo em uma unica passagem matricial. Para compensar a ausencia de ordem inerente provocada pela remocao da recorrencia, a arquitetura injeta explicitamente a nocao de ordenacao por meio de Codificacoes Posicionais (*Positional Encodings*, como funcoes senoidais ou *RoPE - Rotary Position Embeddings*) somadas ou aplicadas as representacoes dos tokens.

---

## 3. A Mecanica do Sistema de Busca (Q, K, V)

### 3.1. Arquitetura de Vetores
O mecanismo de *Scaled Dot-Product Attention* opera como um sistema de recuperacao de informacao diferenciavel e continuo. Dada uma matriz de embeddings de entrada $X$, o modelo aplica tres matrizes de pesos aprendiveis ($W^Q, W^K, W^V$) para projetar cada token em tres subespacos vetoriais distintos:

$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

A equacao central da atencao e definida por:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

* **Query (Consulta - $Q$):** Representa o vetor de intencao de busca emitido pelo token atual. Funcionalmente, codifica a pergunta ou o criterio sintatico-semantico que aquela posicao precisa resolver no contexto (por exemplo, um verbo transitivo buscando seu objeto direto, ou um pronome buscando seu referente nominal).
* **Key (Chave - $K$):** Representa o vetor de indexacao ou assinatura semantica que cada token da sequencia expoe para ser consultado. Matematicamente, o produto escalar $QK^T$ mede a afinidade ou similaridade geometrica entre a *Query* de um token e as *Keys* de todos os outros tokens. O fator de escala $\frac{1}{\sqrt{d_k}}$ (onde $d_k$ e a dimensao dos vetores *Key*) evita que o produto escalar atinja magnitudes elevadas, o que empurraria a funcao $\text{softmax}$ para regioes de gradiente proximo a zero.
* **Value (Valor - $V$):** Contem a representacao semantica efetiva e o conteudo informacional de cada token que sera extraido. Apos a normalizacao dos escores de afinidade pela funcao $\text{softmax}$, obtem-se uma matriz de pesos de atencao cuja soma por linha e igual a $1$. A multiplicacao dessa matriz por $V$ produz uma soma ponderada dos vetores de valor, agregando ao token atual apenas as informacoes mais relevantes do contexto global.

---

## 4. A Linha de Montagem de um LLM

### 4.1. Processo Industrial

1. **Tokenizacao e Embeddings:**
   O texto bruto nao pode ser processado diretamente por operacoes algebricas; por isso, passa inicialmente por um tokenizador sub-palavra, sendo o *Byte-Pair Encoding* (BPE) o algoritmo predominante (junto a variantes como *WordPiece* e *Unigram*). O BPE segmenta o texto iterativamente fundindo os pares de bytes ou caracteres mais frequentes no corpus, equilibrando o tamanho do vocabulario com a capacidade de representar palavras raras, morfologia complexa e codigo-fonte sem gerar tokens desconhecidos (`<UNK>`). Cada token recebe um identificador inteiro unico que indexa uma matriz de *Embeddings*, convertendo indices discretos em vetores densos continuos em um espaco de alta dimensao ($d_{model}$), onde a proximidade geometrica passa a refletir relacoes semanticas.

2. **Pre-treinamento:**
   Nesta etapa de maior custo computacional, o modelo (geralmente uma arquitetura *Decoder-only* causal) e exposto a trilhoes de tokens extraidos de livros, artigos cientificos, repositorios de codigo e paginas da web. O objetivo matematico e a Modelagem de Linguagem Causal (*Next-Token Prediction*), minimizando a entropia cruzada (*Cross-Entropy Loss*) sobre a distribuicao de probabilidade condicional $P(x_t \mid x_1, x_2, \dots, x_{t-1})$. Embora seja frequentemente referenciado na literatura critica como o "paradoxo do papagaio estocastico" — por otimizar estritamente a verossimilhanca estatistica da proxima palavra sem aterramento fisico —, a escala massiva forca o modelo a comprimir padroes sintaticos, fatos enciclopedicos, raciocinio logico e estruturas de programacao em seus pesos internos, formando um modelo base (*Base Model*) capaz de completar textos, mas ainda incapaz de seguir instrucoes conversacionais de forma confiavel.

3. **Ajuste Fino Supervisionado (SFT):**
   Para transformar um completador de texto generico em um assistente util, aplica-se o *Supervised Fine-Tuning* (SFT). O modelo pre-treinado continua seu treinamento com uma taxa de aprendizado reduzida sobre um conjunto de dados altamente curado, composto por dezenas ou centenas de milhares de pares de instrucao e resposta ideal formulados por especialistas humanos. O SFT ensina o modelo a reconhecer tokens especiais de controle de turno (como `<|user|>` e `<|assistant|>`), estruturar respostas passo a passo e adotar o comportamento esperado de um agente conversacional, mantendo a mesma funcao de perda de entropia cruzada, porem aplicada exclusivamente sobre os tokens da resposta esperada.

4. **Alinhamento (RLHF / DPO):**
   Mesmo apos o SFT, o modelo pode gerar respostas factualmente incorretas, toxicas ou desalinhadas com as preferencias humanas sutis. A etapa de alinhamento refina a politica do modelo ($\pi_\theta$) utilizando dados de comparacao humana (onde anotadores classificam duas ou mais respostas para um mesmo prompt como escolhida $y_w$ versus rejeitada $y_l$):
   * **RLHF (*Reinforcement Learning from Human Feedback*):** Opera em duas fases apos o SFT. Primeiro, treina-se um Modelo de Recompensa (*Reward Model* - RM) separado para atribuir um escore escalar de qualidade humana a qualquer resposta gerada. Em seguida, utiliza-se um algoritmo de aprendizado por reforco, tipicamente o PPO (*Proximal Policy Optimization*), para atualizar os pesos do LLM de modo a maximizar a recompensa estimada pelo RM, mantendo uma penalidade de divergencia KL (*Kullback-Leibler*) em relacao ao modelo SFT original para evitar o colapso de modo (*reward hacking*). E um processo computacionalmente caro e instavel, exigindo multiplos modelos carregados simultaneamente na memoria.
   * **DPO (*Direct Preference Optimization*):** Elimina a necessidade de treinar um Modelo de Recompensa explicito e de executar loops complexos de PPO. Por meio de uma reparametrizacao analitica da funcao objetivo do RLHF, o DPO demonstra que o proprio LLM pode atuar implicitamente como seu modelo de recompensa. O alinhamento e convertido diretamente em uma funcao de perda binaria de classificacao sobre os pares de preferencia $(y_w, y_l)$, aumentando a log-probabilidade relativa da resposta preferida em relacao a rejeitada ponderada pela politica de referencia ($\pi_{\text{ref}}$). Isso torna o alinhamento substancialmente mais estavel, rapido e economico em memoria de GPU.

---

## 5. Registro Comentado dos Experimentos Praticos (Hiperparametros)

Abaixo apresenta-se a analise comparativa da execucao local do modelo `qwen2.5-coder:7b` via Ollama (taxa media de ~39 tokens/s) sob tres configuracoes distintas de amostragem para o mesmo prompt tecnico (*"Implemente uma funcao em Python para verificar se uma arvore binaria e uma Arvore Binaria de Busca (BST) valida, explicando a complexidade de tempo e espaco"*), mantendo o teto `num_predict = 512`:

* **Cenario A (Temperature = 0.0 | Top-P = 0.9 | Num Predict = 512 | Velocidade: 38.59 tokens/s):**
  * **Mecanismo:** Decodificacao deterministica (*Greedy Decoding*), selecionando invariavelmente o token de maior probabilidade na distribuicao `softmax`.
  * **Analise:** Produziu codigo sintaticamente limpo e logicamente exato (`left < node.value < right`), utilizando a abordagem recursiva otima com limites inferior e superior (`-inf` a `+inf`). Observou-se tambem o efeito pratico do limite `num_predict = 512`, que interrompeu a geracao ao atingir exatamente 512 tokens antes da conclusao do ultimo topico explicativo.

* **Cenario B (Temperature = 0.85 | Top-P = 0.9 | Num Predict = 512 | Velocidade: 39.48 tokens/s):**
  * **Mecanismo:** Suavizacao moderada dos logits combinada com amostragem por nucleo (*Nucleus Sampling*), restringindo a escolha ao menor conjunto de tokens cuja soma de probabilidades atinge 90%.
  * **Analise:** Preservou integralmente a corretude algoritmica da validacao da BST, mas introduziu variacao lexica nos identificadores (`key` e `val` no lugar de `value`, `is_valid_bst` no lugar de `is_bst_util`), alternancia de idioma nos comentarios (`# Output: True`) e reorganizacao dos topicos explicativos, alcancando a secao de complexidade de tempo ($O(n)$) dentro do mesmo orcamento de 512 tokens.

* **Cenario C (Temperature = 1.5 | Top-P = 1.0 | Num Predict = 512 | Velocidade: 39.17 tokens/s):**
  * **Mecanismo:** Elevacao artificial da probabilidade de tokens raros pertencentes a cauda longa da distribuicao (*long tail*), com filtro `top_p` totalmente aberto (`1.0`).
  * **Analise:** Provocou degradacao sintatica, mistura de idiomas (explicacao em portugues com docstring em ingles e erros de concordancia) e **falha logica critica (alucinacao funcional)**: na verificacao do limite superior do no, o modelo gerou `node.val > max(l_limit, node.val)` em vez de `node.val >= r_limit`, invalidando a restricao superior da BST e fazendo a funcao aceitar arvores invalidas. Isso comprova como a amostragem de alta entropia compromete a precisao em tarefas formais e de engenharia de software.