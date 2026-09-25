"""
Formação dos lotes de treino.

O modelo aprende a prever o PRÓXIMO token. Para um bloco de contexto de tamanho
block_size, pegamos block_size + 1 tokens seguidos:

    tokens:  [ 12, 40, 7, 33, 5 ]      (block_size = 4)
    x (entrada):  [ 12, 40,  7, 33 ]
    y (alvo):     [ 40,  7, 33,  5 ]   <- x deslocado uma posição

Cada posição t vira um exemplo de treino ao mesmo tempo: "dado x[0..t], preveja y[t]".
Um bloco de 64 tokens gera 64 exemplos de uma vez; um lote de 32 blocos, 2.048 exemplos.
É classificação multiclasse (1 entre vocab_size classes) em cada posição.

Este módulo usa só NumPy. Na semana 2, o train.py converte para PyTorch com
torch.from_numpy(x).
"""

from pathlib import Path

import numpy as np

PASTA = Path(__file__).resolve().parent / "processed"


def carregar_tokens(split, pasta=PASTA):
    """np.memmap lê do disco sob demanda: não carrega o arquivo inteiro na RAM."""
    return np.memmap(Path(pasta) / f"{split}.bin", dtype=np.uint16, mode="r")


def get_batch(dados, batch_size, block_size, rng=None):
    """
    Sorteia batch_size posições iniciais e devolve (x, y), ambos com forma
    (batch_size, block_size) e tipo int64 (o que as camadas de embedding esperam).
    """
    if len(dados) <= block_size:
        raise ValueError(f"Dados com {len(dados)} tokens, menos que block_size + 1.")
    rng = rng or np.random.default_rng()
    inicios = rng.integers(0, len(dados) - block_size, size=batch_size)
    x = np.stack([dados[i : i + block_size] for i in inicios]).astype(np.int64)
    y = np.stack([dados[i + 1 : i + 1 + block_size] for i in inicios]).astype(np.int64)
    return x, y
