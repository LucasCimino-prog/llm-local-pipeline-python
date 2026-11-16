import datetime
import json
import os
import time
import requests


def obter_float(mensagem: str, padrao: float, minimo: float, maximo: float) -> float:
    """Solicita um valor de ponto flutuante dentro de um intervalo valido."""
    while True:
        entrada = input(f"{mensagem} [Padrao: {padrao}]: ").strip()
        if not entrada:
            return padrao
        try:
            valor = float(entrada)
            if minimo <= valor <= maximo:
                return valor
            print(f"Erro: Informe um valor entre {minimo} e {maximo}.")
        except ValueError:
            print("Erro: Entrada invalida. Digite um numero decimal.")


def obter_int(mensagem: str, padrao: int, minimo: int, maximo: int) -> int:
    """Solicita um valor inteiro dentro de um intervalo valido."""
    while True:
        entrada = input(f"{mensagem} [Padrao: {padrao}]: ").strip()
        if not entrada:
            return padrao
        try:
            valor = int(entrada)
            if minimo <= valor <= maximo:
                return valor
            print(f"Erro: Informe um valor entre {minimo} e {maximo}.")
        except ValueError:
            print("Erro: Entrada invalida. Digite um numero inteiro.")


def salvar_resultado_txt(
    modelo: str,
    prompt: str,
    temperatura: float,
    top_p: float,
    num_predict: int,
    resposta_completa: str,
    tokens_gerados: int,
    tokens_por_segundo: float,
) -> str:
    """Salva o log completo da inferencia em um arquivo .txt estruturado."""
    os.makedirs("resultados_inferencia", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_arquivo = f"resultados_inferencia/inferencia_T{temperatura}_P{top_p}_{timestamp}.txt"

    cabecalho = (
        "====================================================================\n"
        "RELATORIO DE INFERENCIA LOCAL - OLLAMA\n"
        "====================================================================\n"
        f"Data/Hora       : {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n"
        f"Modelo          : {modelo}\n"
        f"Temperature     : {temperatura}\n"
        f"Top-P           : {top_p}\n"
        f"Num Predict     : {num_predict}\n"
        f"Tokens Gerados  : {tokens_gerados}\n"
        f"Velocidade      : {tokens_por_segundo:.2f} tokens/s\n"
        "--------------------------------------------------------------------\n"
        "PROMPT ENVIADO:\n"
        f"{prompt}\n"
        "--------------------------------------------------------------------\n"
        "RESPOSTA GERADA:\n\n"
    )

    with open(nome_arquivo, "w", encoding="utf-8") as arquivo:
        arquivo.write(cabecalho)
        arquivo.write(resposta_completa)
        arquivo.write("\n\n====================================================================\n")

    return nome_arquivo


def executar_inferencia():
    url = "http://localhost:11434/api/generate"

    print("=== Configuracao Dinamica de Inferencia Local (Ollama) ===\n")

    modelo = input("Modelo Ollama [Padrao: qwen2.5-coder:7b]: ").strip() or "qwen2.5-coder:7b"

    prompt_padrao = (
        "Implemente uma funcao em Python para verificar se uma arvore binaria "
        "e uma Arvore Binaria de Busca (BST) valida, explicando a complexidade "
        "de tempo e espaco."
    )
    prompt_usuario = input(f"\nDigite o prompt (ou Enter para usar o prompt tecnico padrao):\n> ").strip()
    prompt = prompt_usuario if prompt_usuario else prompt_padrao

    print("\n--- Definicao de Hiperparametros de Amostragem ---")
    temperatura = obter_float("Temperature (0.0 a 2.0)", padrao=0.2, minimo=0.0, maximo=2.0)
    top_p = obter_float("Top-P / Nucleus Sampling (0.01 a 1.0)", padrao=0.9, minimo=0.01, maximo=1.0)
    num_predict = obter_int("Num Predict / Limite maximo de tokens (32 a 4096)", padrao=512, minimo=32, maximo=4096)

    payload = {
        "model": modelo,
        "prompt": prompt,
        "options": {
            "temperature": temperatura,
            "top_p": top_p,
            "num_predict": num_predict,
        },
        "stream": True,
    }

    buffer_resposta = []
    tokens_gerados = 0
    tokens_por_segundo = 0.0

    try:
        inicio_requisicao = time.perf_counter()
        response = requests.post(url, json=payload, stream=True, timeout=120)
        response.raise_for_status()

        print("\n--- Iniciando Geracao do Modelo (Streaming) ---\n")

        for chunk in response.iter_lines():
            if chunk:
                data = json.loads(chunk.decode("utf-8"))
                token_text = data.get("response", "")

                print(token_text, end="", flush=True)
                buffer_resposta.append(token_text)

                if data.get("done", False):
                    tempo_total = time.perf_counter() - inicio_requisicao
                    tokens_gerados = data.get("eval_count", len(buffer_resposta))
                    eval_duration_ns = data.get("eval_duration", 0)

                    if eval_duration_ns > 0:
                        tokens_por_segundo = tokens_gerados / (eval_duration_ns / 1e9)
                    elif tempo_total > 0:
                        tokens_por_segundo = tokens_gerados / tempo_total

                    print("\n\n--- Geracao Concluida ---")
                    print(f"Total de tokens gerados : {tokens_gerados}")
                    print(f"Taxa de geracao         : {tokens_por_segundo:.2f} tokens/s")

        resposta_final = "".join(buffer_resposta)
        caminho_salvo = salvar_resultado_txt(
            modelo=modelo,
            prompt=prompt,
            temperatura=temperatura,
            top_p=top_p,
            num_predict=num_predict,
            resposta_completa=resposta_final,
            tokens_gerados=tokens_gerados,
            tokens_por_segundo=tokens_por_segundo,
        )
        print(f"Arquivo salvo com sucesso em: {caminho_salvo}")

    except requests.exceptions.ConnectionError:
        print(
            "\nErro: Nao foi possivel conectar ao Ollama. "
            "Verifique se o servico esta ativo localmente (ollama serve)."
        )
    except requests.exceptions.Timeout:
        print("\nErro: Tempo limite de conexao excedido ao aguardar resposta do modelo.")
    except Exception as e:
        print(f"\nOcorreu um erro inesperado: {e}")


if __name__ == "__main__":
    executar_inferencia()