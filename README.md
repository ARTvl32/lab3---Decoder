# Laboratório 3 — Implementando o Decoder

**Disciplina:** Tópicos em Inteligência Artificial 2026.1  
**Instituição:** iCEV — Instituto de Ensino Superior  
**Professor:** Dimmy Magalhães  

---

## Objetivos de Aprendizagem

1. Dominar a álgebra linear por trás do **mascaramento causal** (Look-Ahead Masking)
2. Implementar a **ponte Encoder-Decoder** via Cross-Attention
3. Simular um **loop de geração auto-regressiva**

---

## Estrutura do Repositório

```
lab3-decoder/
│
├── tarefa1_mascara_causal.py     # Look-Ahead Mask + prova com Softmax
├── tarefa2_cross_attention.py    # Cross-Attention Encoder-Decoder
├── tarefa3_loop_autoressivo.py   # Loop de inferência auto-regressivo
└── README.md
```

---

## Tarefas

### Tarefa 1 — Máscara Causal (Look-Ahead Mask)

Implementa `create_causal_mask(seq_len)` que retorna uma matriz quadrada onde:

- Triangular inferior + diagonal principal → `0` (posições permitidas)
- Triangular superior → `-∞` (posições futuras bloqueadas)

A máscara é injetada na equação de atenção antes do Softmax:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

**Prova real:** após aplicar o Softmax com a máscara, todos os tokens futuros
recebem probabilidade estritamente `0.0`.

---

### Tarefa 2 — Cross-Attention (Ponte Encoder-Decoder)

Implementa `cross_attention(encoder_out, decoder_state)` simulando a segunda
sub-camada do bloco Decoder:

| Matriz | Origem | Papel |
|--------|--------|-------|
| **Q** | `decoder_state` | "O que preciso buscar agora?" |
| **K** | `encoder_out`   | "Índice do conteúdo da entrada" |
| **V** | `encoder_out`   | "Conteúdo semântico a extrair" |

> Nota: **sem máscara causal** nesta etapa — o Decoder pode ver toda a frase
> original do Encoder.

---

### Tarefa 3 — Loop de Inferência Auto-Regressivo

Simula o loop completo de geração token a token:

1. `generate_next_token(current_sequence, encoder_out)` → vetor de probabilidades (V = 10.000)
2. `argmax` seleciona o token mais provável
3. Token é adicionado à sequência de contexto
4. Loop para ao gerar `<EOS>`

---

## Como Executar

> Requer apenas **Python 3** e **NumPy**.

```bash
# Tarefa 1 — Máscara Causal
python tarefa1_mascara_causal.py

# Tarefa 2 — Cross-Attention
python tarefa2_cross_attention.py

# Tarefa 3 — Loop Auto-Regressivo
python tarefa3_loop_autoressivo.py
```

---

## Fundamentos Teóricos

### Por que a máscara usa −∞?

A função Softmax aplica a exponencial em cada score:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

Como $e^{-\infty} = 0$, qualquer posição mascarada com $-\infty$ recebe
probabilidade **exatamente zero**, impedindo que o modelo "olhe para o futuro"
durante o treinamento paralelizado com Teacher Forcing.

### Por que Cross-Attention não usa máscara?

Na sub-camada 2 do Decoder, o modelo **deve** ter acesso irrestrito a toda a
frase de entrada. A restrição causal só se aplica à sub-camada 1 (Masked
Self-Attention entre os tokens já gerados).

### Geração Auto-Regressiva

$$y_i = \arg\max \; \text{softmax}(W_{\text{vocab}} \cdot h_i)$$

onde $h_i$ é o vetor de saída do Decoder no passo $i$, condicionado em
$y_1, \ldots, y_{i-1}$.

---

## Referências

- Vaswani et al. (2017). *Attention Is All You Need*. NeurIPS.
- Notas de aula — Prof. Dimmy Magalhães, iCEV 2026.1
