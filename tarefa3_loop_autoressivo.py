"""
Laboratório 3 — Tarefa 3: Loop de Inferência Auto-Regressivo
=============================================================
Disciplina : Tópicos em Inteligência Artificial 2026.1
Professor  : Dimmy Magalhães — iCEV
Aluno      : Arthur

Fundamento
----------
O modelo de linguagem trabalha como um laço de repetição condicionado na
produção anterior. A cada passo, o vetor final de 512 dimensões emitido pelo
Decoder sofre duas operações:

    1. Projeção linear → vocab_size   (512 → 10.000)
    2. Softmax                        (distribuição de probabilidades)

O token com maior probabilidade é selecionado via argmax, adicionado à
sequência de contexto, e o processo se repete até gerar <EOS>.

Loop de inferência
------------------
    sequencia = ["<START>"]
    enquanto True:
        probs    = generate_next_token(sequencia, encoder_out)
        next_tok = vocab[argmax(probs)]
        sequencia.append(next_tok)
        se next_tok == "<EOS>": parar
"""

import numpy as np


# ---------------------------------------------------------------------------
# Configuração do vocabulário fictício
# ---------------------------------------------------------------------------

VOCAB_SIZE = 10_000

# Tokens especiais e algumas palavras do vocabulário
VOCAB = {
    "<PAD>"  : 0,
    "<START>": 1,
    "<EOS>"  : 2,
    "o"      : 3,
    "rato"   : 4,
    "roeu"   : 5,
    "a"      : 6,
    "roupa"  : 7,
    "do"     : 8,
    "rei"    : 9,
}
# Demais tokens recebem IDs 10..9999
for _i in range(10, VOCAB_SIZE):
    VOCAB[f"token_{_i}"] = _i

# Mapeamento inverso ID → token
ID_TO_TOKEN = {v: k for k, v in VOCAB.items()}

EOS_ID   = VOCAB["<EOS>"]
START_ID = VOCAB["<START>"]

# Máximo de passos para evitar loop infinito
MAX_LEN = 50


# ---------------------------------------------------------------------------
# Função auxiliar
# ---------------------------------------------------------------------------

def softmax(x):
    """Softmax numericamente estável para vetor 1-D."""
    x_shifted = x - np.max(x)
    e = np.exp(x_shifted)
    return e / e.sum()


# ---------------------------------------------------------------------------
# Mock do Decoder: generate_next_token
# ---------------------------------------------------------------------------

# Pesos da camada de projeção linear (512 → VOCAB_SIZE)
# Inicializados uma única vez para manter determinismo entre os passos
np.random.seed(0)
W_VOCAB = np.random.randn(512, VOCAB_SIZE) * 0.01


# Sequência de tokens que o mock vai forçar para tornar a demo legível:
# <START> → o → rato → roeu → a → roupa → do → rei → <EOS>
_FORCED_SEQUENCE = [
    VOCAB["o"],
    VOCAB["rato"],
    VOCAB["roeu"],
    VOCAB["a"],
    VOCAB["roupa"],
    VOCAB["do"],
    VOCAB["rei"],
    EOS_ID,
]


def generate_next_token(current_sequence, encoder_out):
    """
    Simula a passagem pelo bloco Decoder e retorna um vetor de probabilidades
    sobre o vocabulário fictício (V = 10.000).

    Na prática este bloco executaria:
        1. Masked Multi-Head Self-Attention  (sub-camada 1)
        2. Cross-Attention com encoder_out   (sub-camada 2)
        3. Feed-Forward Network              (sub-camada 3)
        4. Projeção linear 512 → V + Softmax

    Aqui usamos um mock determinístico: o decoder_vector é construído de forma
    que o argmax recaia sobre o token correto da sequência forçada,
    permitindo verificar o loop sem um Decoder treinado.

    Parâmetros
    ----------
    current_sequence : list[int]
        Lista de IDs de tokens gerados até agora.
    encoder_out : np.ndarray, shape (1, T_enc, 512)
        Saída simulada do Encoder.

    Retorna
    -------
    probs : np.ndarray, shape (VOCAB_SIZE,)
        Distribuição de probabilidades sobre o vocabulário.
    """
    # Índice do próximo token forçado (baseado no número de tokens gerados)
    step = len(current_sequence) - 1          # passo atual (0-indexado)
    step = min(step, len(_FORCED_SEQUENCE) - 1)
    target_id = _FORCED_SEQUENCE[step]

    # Vetor de saída simulado do Decoder (512 dimensões)
    np.random.seed(step * 31)
    decoder_vector = np.random.randn(512) * 0.1

    # Projeção linear 512 → VOCAB_SIZE
    logits = decoder_vector @ W_VOCAB         # shape (VOCAB_SIZE,)

    # Amplifica o logit do token-alvo para garantir que argmax o selecione
    logits[target_id] += 10.0

    # Softmax → distribuição de probabilidades
    probs = softmax(logits)
    return probs


# ---------------------------------------------------------------------------
# Loop de Inferência Auto-Regressivo
# ---------------------------------------------------------------------------

def loop_inferencia(encoder_out):
    """
    Executa o loop de geração auto-regressiva até <EOS> ou MAX_LEN.

    Parâmetros
    ----------
    encoder_out : np.ndarray, shape (1, T_enc, 512)
        Saída simulada do Encoder (frase de entrada codificada).

    Retorna
    -------
    sequencia : list[str]
        Lista de tokens gerados (incluindo <START> e <EOS>).
    """
    # Inicializar sequência com token de início
    sequencia_ids   = [START_ID]
    sequencia_tokens = [ID_TO_TOKEN[START_ID]]

    print("\n--- Iniciando loop de geração ---")
    print(f"  Sequência inicial: {sequencia_tokens}\n")

    passo = 0
    while True:
        passo += 1

        # 1. Gerar distribuição de probabilidades
        probs = generate_next_token(sequencia_ids, encoder_out)

        # 2. Selecionar token com maior probabilidade (argmax)
        next_id    = int(np.argmax(probs))
        next_token = ID_TO_TOKEN[next_id]
        next_prob  = probs[next_id]

        # 3. Adicionar à sequência de contexto
        sequencia_ids.append(next_id)
        sequencia_tokens.append(next_token)

        print(f"  Passo {passo:2d}: "
              f"argmax → ID={next_id:5d}  "
              f"token='{next_token}'  "
              f"prob={next_prob:.4f}")

        # 4. Verificar condição de parada
        if next_id == EOS_ID:
            print(f"\n  ✓ Token <EOS> gerado — encerrando loop.")
            break

        if passo >= MAX_LEN:
            print(f"\n  ⚠ Limite MAX_LEN={MAX_LEN} atingido — encerrando loop.")
            break

    return sequencia_tokens


# ---------------------------------------------------------------------------
# Demonstração principal
# ---------------------------------------------------------------------------

def demonstracao():
    print("=" * 60)
    print("TAREFA 3 — Loop de Inferência Auto-Regressivo")
    print("=" * 60)

    np.random.seed(42)

    # Encoder fictício: simula uma frase codificada (ex: "Le chat ronge...")
    T_enc       = 8
    d_model     = 512
    encoder_out = np.random.randn(1, T_enc, d_model)

    print(f"\nEncoder output shape: {encoder_out.shape}")
    print(f"Vocabulário fictício: V = {VOCAB_SIZE:,} tokens")
    print(f"Token de início : <START> (ID={START_ID})")
    print(f"Token de parada : <EOS>   (ID={EOS_ID})")
    print(f"Limite máximo   : {MAX_LEN} passos")

    # Executar loop
    frase_gerada = loop_inferencia(encoder_out)

    # Imprimir frase final
    print("\n" + "=" * 60)
    print("FRASE FINAL GERADA:")
    print("  " + " ".join(frase_gerada))
    print("=" * 60)

    # Estatísticas
    n_tokens = len(frase_gerada) - 2   # exclui <START> e <EOS>
    print(f"\nEstatísticas:")
    print(f"  Tokens gerados (sem marcadores): {n_tokens}")
    print(f"  Sequência completa             : {len(frase_gerada)} tokens")


if __name__ == "__main__":
    demonstracao()
