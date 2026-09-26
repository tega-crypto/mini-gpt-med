"""
Testes de sanidade da Etapa 2 (arquitetura). Rodam na CPU em poucos segundos.

    python tests/test_etapa2.py

Se data/processed/ existir (rode antes python data/prepare.py), o último teste
confere a perda inicial com um lote de dados reais.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "data"))

from model import GPT, GPTConfig  # noqa: E402
from loader import PASTA, carregar_tokens, get_batch  # noqa: E402

PEQUENO = GPTConfig(vocab_size=50, block_size=16, n_embd=32, n_head=4, n_layer=2, dropout=0.0)


def test_formas():
    modelo = GPT(PEQUENO)
    ids = torch.randint(0, 50, (3, 16))
    logits, perda = modelo(ids)
    assert logits.shape == (3, 16, 50) and perda is None
    # Sequências mais curtas que block_size também funcionam (usado na geração).
    logits, _ = modelo(ids[:, :5])
    assert logits.shape == (3, 5, 50)
    try:
        modelo(torch.randint(0, 50, (1, 17)))
        assert False, "deveria falhar com sequência maior que block_size"
    except ValueError:
        pass


def test_causalidade():
    # Trocar um token no fim não pode mudar as previsões das posições anteriores.
    torch.manual_seed(0)
    modelo = GPT(PEQUENO).eval()
    ids = torch.randint(0, 50, (1, 16))
    alterado = ids.clone()
    alterado[0, 10] = (ids[0, 10] + 1) % 50
    with torch.no_grad():
        a, _ = modelo(ids)
        b, _ = modelo(alterado)
    assert torch.allclose(a[:, :10], b[:, :10], atol=1e-6), "o modelo está vendo o futuro"
    assert not torch.allclose(a[:, 10:], b[:, 10:]), "o token alterado deveria influenciar o que vem depois"


def test_perda_inicial():
    # Recém-criado, o modelo não sabe nada: a perda deve ficar perto de ln(vocab_size).
    torch.manual_seed(0)
    modelo = GPT(PEQUENO)
    ids = torch.randint(0, 50, (8, 16))
    _, perda = modelo(ids, ids)
    assert abs(perda.item() - math.log(50)) < 0.1, perda.item()


def test_decora_um_lote():
    # Um modelo correto consegue decorar um único lote pequeno. Se a perda não
    # cai, há erro no gradiente ou na ligação entre as camadas.
    torch.manual_seed(0)
    modelo = GPT(PEQUENO)
    x = torch.randint(0, 50, (4, 16))
    y = torch.randint(0, 50, (4, 16))
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=3e-3)
    for _ in range(200):
        _, perda = modelo(x, y)
        otimizador.zero_grad()
        perda.backward()
        otimizador.step()
    assert perda.item() < 0.1, perda.item()


def test_config_base_com_dados_reais():
    if not (PASTA / "meta.json").exists():
        print("    (pulado: rode python data/prepare.py para gerar data/processed/)")
        return
    meta = json.loads((PASTA / "meta.json").read_text(encoding="utf-8"))
    config = GPTConfig.from_json(RAIZ / "configs" / "base.json", meta["vocab_size"])
    torch.manual_seed(1337)
    modelo = GPT(config).eval()
    treino = carregar_tokens("train")
    x, y = get_batch(treino, 32, config.block_size, rng=np.random.default_rng(1337))
    with torch.no_grad():
        _, perda = modelo(torch.from_numpy(x), torch.from_numpy(y))
    print(f"    {modelo.num_parametros():,} parâmetros | perda inicial {perda.item():.4f}"
          f" | esperada {meta['perda_inicial_esperada']}")
    assert abs(perda.item() - meta["perda_inicial_esperada"]) < 0.1
    del treino  # no Windows, o memmap aberto trava o arquivo


if __name__ == "__main__":
    for nome, fn in list(globals().items()):
        if nome.startswith("test_"):
            fn()
            print(f"ok  {nome}")
    print("Todos os testes da Etapa 2 passaram.")
