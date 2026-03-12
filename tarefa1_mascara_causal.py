"""
Laboratório 3 — Tarefa 1: Máscara Causal (Look-Ahead Mask)
===========================================================
Disciplina : Tópicos em Inteligência Artificial 2026.1
Professor  : Dimmy Magalhães — iCEV
Aluno      : Arthur

Fundamento
----------
No treinamento paralelizado em GPUs, a frase de destino completa entra no
Decoder de uma só vez (Teacher Forcing). Para impedir que o token na posição i
"veja" os tokens em posições i+1, i+2, ..., injetamos uma máscara matricial M
antes do cálculo do Softmax:

    Attention(Q, K, V) = softmax( QK^T / sqrt(d_k)  +  M ) * V

A matriz M é triangular superior preenchida com -inf:
- Posições permitidas (j <= i) recebem 0   → score inalterado
- Posições futuras   (j >  i) recebem -inf → exp(-inf) = 0 no Softmax
"""

import numpy as np


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def softmax(x, axis=-1):
    """Softmax numericamente estável (subtrai o máximo antes de exp)."""
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x_shifted)
    return e / e.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Tarefa 1 — create_causal_mask
# ---------------------------------------------------------------------------

def create_causal_mask(seq_len):
    """
    Cria a máscara causal (Look-Ahead Mask) para uma sequência de comprimento
    seq_len.

    Retorna
    -------
    mask : np.ndarray, shape (seq_len, seq_len)
        - Triangular inferior (incluindo diagonal): 0.0
        - Triangular superior: -inf  (representado por -1e9 para estabilidade)
    """
    # np.triu com k=1 preenche apenas ACIMA da diagonal principal
    mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)
    return mask


# ---------------------------------------------------------------------------
# Prova Real
# ---------------------------------------------------------------------------

def prova_real():
    """
    Multiplica matrizes fictícias Q e K, adiciona a máscara M e aplica Softmax.
    Demonstra que as probabilidades das palavras FUTURAS são estritamente 0.0.
    """
    print("=" * 60)
    print("TAREFA 1 — Máscara Causal (Look-Ahead Mask)")
    print("=" * 60)

    np.random.seed(42)

    seq_len = 5
    d_k     = 8   # dimensão das chaves/queries

    # Matrizes fictícias Q e K
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)

    # 1. Produto escalar escalonado
    scores = Q @ K.T / np.sqrt(d_k)          # shape (5, 5)

    # 2. Gerar máscara causal
    M = create_causal_mask(seq_len)

    # 3. Somar máscara aos scores
    masked_scores = scores + M

    # 4. Aplicar Softmax linha a linha
    weights = softmax(masked_scores)          # shape (5, 5)

    # -----------------------------------------------------------------------
    # Exibir resultados
    # -----------------------------------------------------------------------
    tokens = ["t1", "t2", "t3", "t4", "t5"]

    print(f"\nParâmetros: seq_len={seq_len}, d_k={d_k}")

    print("\n--- Máscara M (0 = permitido, -inf = bloqueado) ---")
    header = "       " + "  ".join(f"{t:>6}" for t in tokens)
    print(header)
    for i, row_tok in enumerate(tokens):
        row = "  ".join(
            f"{'   0  ':>6}" if M[i, j] == 0 else f"{' -inf ':>6}"
            for j in range(seq_len)
        )
        print(f"  {row_tok}: {row}")

    print("\n--- Pesos de Atenção após Softmax (com máscara) ---")
    print(header)
    for i, row_tok in enumerate(tokens):
        row = "  ".join(f"{weights[i, j]:6.4f}" for j in range(seq_len))
        print(f"  {row_tok}: {row}")

    print("\n--- Verificação: tokens futuros = 0.0 ---")
    all_zeros = True
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            val = weights[i, j]
            status = "✓" if val == 0.0 else "✗ FALHOU"
            print(f"  weights[{tokens[i]}, {tokens[j]}] = {val:.10f}  {status}")
            if val != 0.0:
                all_zeros = False

    print()
    if all_zeros:
        print("✓ PROVA REAL CONFIRMADA: todas as posições futuras têm")
        print("  probabilidade estritamente 0.0 após o Softmax.")
    else:
        print("✗ Erro: alguma posição futura possui probabilidade > 0.")

    print("=" * 60)


if __name__ == "__main__":
    prova_real()
