# Inferencia Local de LLMs e Analise de Hiperparametros com Ollama

Este repositorio contem a implementacao pratica desenvolvida para a 3a Etapa da disciplina de Inteligencia Artificial do curso de Ciencia da Computacao (UNIPAC Barbacena). O projeto explora a execucao local de Modelos de Linguagem de Grande Escala (LLMs) via API REST do Ollama, manipulacao dinamica de hiperparametros de amostragem e persistencia automatizada dos resultados.

## Arquitetura e Objetivo

O script `OllamaInference.py` estabelece uma comunicacao HTTP em fluxo continuo (`stream=True`) com o servidor local do Ollama (`http://localhost:11434/api/generate`), permitindo:

1. Selecao dinamica do modelo quantizado instalado localmente (padrao: `qwen2.5-coder:7b`).
2. Configuracao interativa em tempo de execucao dos parametros `temperature`, `top_p` e `num_predict` com validacao de limites.
3. Exibicao autorregressiva token por token no terminal em tempo real.
4. Calculo da taxa de geracao (tokens por segundo) a partir das metricas de telemetria retornadas pelo motor de inferencia.
5. Exportacao automatica do prompt, hiperparametros, metricas e resposta completa para arquivos `.txt` organizados no diretorio `resultados_inferencia/`.

## Pre-requisitos

- Python 3.10 ou superior
- Ollama instalado e em execucao local
- Modelo de linguagem baixado no ambiente Ollama (recomendado: `qwen2.5-coder:7b` ou `llama3:8b`)

## Instalacao e Configuracao

1. Clone ou baixe os arquivos deste repositorio e acesse o diretorio do projeto:

```bash
git clone https://github.com/LucasCimino-prog/llm-local-pipeline-python.git
cd llm-local-pipeline-python