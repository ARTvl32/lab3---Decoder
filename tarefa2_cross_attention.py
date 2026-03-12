"""
Laboratório 3 — Tarefa 2: A Ponte Encoder-Decoder (Cross-Attention)
====================================================================
Disciplina : Tópicos em Inteligência Artificial 2026.1
Professor  : Dimmy Magalhães — iCEV
Aluno      : Arthur

Fundamento
----------
Diferente do Self-Attention, onde Q, K e V derivam da mesma sequência, o
Cross-Attention (ou Encoder-Decoder Attention) cruza fronteiras:

    Q  ←  decoder_state  (o que o Decoder está gerando agora)
    K  ←  encoder_out    (índice de busca da frase original)
    V  ←  encoder_out    (conteúdo semântico a ser extraído)

A equação é a mesma do Scaled Dot-Product Attention, porém SEM máscara causal:
o Decoder deve ter permissão de olhar para toda a frase do Encoder.

    CrossAttention(Q, K, V) = softmax( QK^T / sqrt(d_k) ) * V

Dimensões nesta tarefa
----------------------
    encoder_out   : (batch=1, seq_len_frances=10, d_model=512)
    decoder_state : (batch=1, seq_len_ingles=4,   d_model=512)
    scores        : (batch=1, seq_len_ingles=4,   seq_len_frances=10)
    output        : (batch=1, seq_len_ingles=4,   d_model=512)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Função auxiliar
# ---------------------------------------------------------------------------

def softmax(x, axis=-1):
    """Softmax numericamente estável."""
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x_shifted)
    return e / e.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Tarefa 2 — cross_attention
# ---------------------------------------------------------------------------

def cross_attention(encoder_out, decoder_state):
    """
    Calcula o Cross-Attention (Encoder-Decoder Attention).

    Parâmetros
    ----------
    encoder_out   : np.ndarray, shape (B, T_enc, d_model)
        Saída final do Encoder — representa a "memória" da frase de entrada.
    decoder_state : np.ndarray, shape (B, T_dec, d_model)
        Estado atual do Decoder — representa o que já foi gerado.

    Retorna
    -------
    output  : np.ndarray, shape (B, T_dec, d_model)
        Representações do Decoder enriquecidas com contexto do Encoder.
    weights : np.ndarray, shape (B, T_dec, T_enc)
        Pesos de atenção (alinhamento Decoder ↔ Encoder).
    """
    B, T_enc, d_model = encoder_out.shape
    _, T_dec, _       = decoder_state.shape
    d_k               = d_model  # projeção na mesma dimensão

    # -----------------------------------------------------------------------
    # Matrizes de projeção (arbitrárias — simulam pesos aprendidos)
    # -----------------------------------------------------------------------
    np.random.seed(7)
    Wq = np.random.randn(d_model, d_k) * 0.01   # projeção Query (Decoder)
    Wk = np.random.randn(d_model, d_k) * 0.01   # projeção Key   (Encoder)
    Wv = np.random.randn(d_model, d_k) * 0.01   # projeção Value (Encoder)

    # -----------------------------------------------------------------------
    # Projeções
    # Q vem do Decoder, K e V vêm do Encoder
    # -----------------------------------------------------------------------
    Q = decoder_state @ Wq   # (B, T_dec, d_k)
    K = encoder_out   @ Wk   # (B, T_enc, d_k)
    V = encoder_out   @ Wv   # (B, T_enc, d_k)

    # -----------------------------------------------------------------------
    # Scaled Dot-Product Attention  (SEM máscara causal)
    # -----------------------------------------------------------------------
    # scores[b, i, j] = similaridade entre token Decoder i e token Encoder j
    scores  = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)  # (B, T_dec, T_enc)
    weights = softmax(scores, axis=-1)                   # (B, T_dec, T_enc)
    output  = weights @ V                                # (B, T_dec, d_k)

    return output, weights


# ---------------------------------------------------------------------------
# Demonstração
# ---------------------------------------------------------------------------

def demonstracao():
    print("=" * 65)
    print("TAREFA 2 — Cross-Attention (Ponte Encoder-Decoder)")
    print("=" * 65)

    np.random.seed(42)

    # Tensores fictícios conforme especificação do laboratório
    batch_size       = 1
    seq_len_frances  = 10   # tokens da frase em francês (Encoder)
    seq_len_ingles   = 4    # tokens já gerados em inglês (Decoder)
    d_model          = 512

    encoder_out   = np.random.randn(batch_size, seq_len_frances, d_model)
    decoder_state = np.random.randn(batch_size, seq_len_ingles,  d_model)

    print(f"\nTensores de entrada:")
    print(f"  encoder_out   : {encoder_out.shape}   "
          f"(batch, seq_frances, d_model)")
    print(f"  decoder_state : {decoder_state.shape}    "
          f"(batch, seq_ingles, d_model)")

    output, weights = cross_attention(encoder_out, decoder_state)

    print(f"\nResultados do Cross-Attention:")
    print(f"  scores/weights shape : {weights.shape}  "
          f"(batch, T_dec, T_enc)")
    print(f"  output shape         : {output.shape}   "
          f"(batch, T_dec, d_model)")

    # Verificação: pesos somam 1 ao longo do eixo Encoder (axis=-1)
    soma_pesos = weights[0].sum(axis=-1)
    print(f"\n--- Verificação: pesos de atenção somam 1.0 por token ---")
    tokens_dec = ["<START>", "The", "cat", "sat"]
    for i, tok in enumerate(tokens_dec):
        ok = "✓" if abs(soma_pesos[i] - 1.0) < 1e-6 else "✗"
        print(f"  {tok:10s}: soma = {soma_pesos[i]:.8f}  {ok}")

    # Mostrar para qual token do Encoder cada token do Decoder "presta mais atenção"
    print(f"\n--- Foco principal de cada token do Decoder ---")
    fr_tokens = [f"fr{i+1}" for i in range(seq_len_frances)]
    for i, tok in enumerate(tokens_dec):
        foco_idx = np.argmax(weights[0, i])
        foco_val = weights[0, i, foco_idx]
        print(f"  '{tok:8s}' → {fr_tokens[foco_idx]} "
              f"(peso = {foco_val:.4f})")

    print(f"\n--- Confirmação das origens de Q, K, V ---")
    print("  Q  ←  decoder_state  (estado atual da geração em inglês)")
    print("  K  ←  encoder_out    (índice da frase em francês)")
    print("  V  ←  encoder_out    (conteúdo semântico da frase em francês)")
    print("  ✓  SEM máscara causal — Decoder vê toda a entrada do Encoder")

    print("=" * 65)


if __name__ == "__main__":
    demonstracao()
