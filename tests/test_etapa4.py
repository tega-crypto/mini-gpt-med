"""
Testes de sanidade da Etapa 4 (geração). Usam um modelo minúsculo sem treino,
então rodam em segundos e não precisam do checkpoint.

    python tests/test_etapa4.py
"""

import sys
from pathlib import Path

import torch

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from generate import gerar, taxa_palavras_reais  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402


def modelo_pequeno():
    torch.manual_seed(0)
    config = GPTConfig(vocab_size=20, block_size=8, n_embd=16, n_head=2, n_layer=1, dropout=0.0)
    return GPT(config).eval()


def test_tamanho_e_contexto():
    # Gera além de block_size: o contexto precisa ser cortado nos últimos 8 tokens.
    modelo = modelo_pequeno()
    inicio = torch.tensor([[1, 2, 3]])
    saida = gerar(modelo, inicio, 30)
    assert saida.shape == (1, 33)
    assert saida[0, :3].tolist() == [1, 2, 3], "o início precisa ser mantido"
    assert int(saida.max()) < 20


def test_top_k_1_escolhe_o_mais_provavel():
    # Com top-k = 1 só sobra um candidato: o sorteio vira escolha do maior logit.
    modelo = modelo_pequeno()
    ids = torch.tensor([[4, 5]])
    for _ in range(10):
        with torch.no_grad():
            logits, _ = modelo(ids[:, -8:])
        esperado = logits[0, -1].argmax().item()
        ids = gerar(modelo, ids, 1, top_k=1)
        assert ids[0, -1].item() == esperado


def test_temperatura():
    modelo = modelo_pequeno()
    ids = torch.tensor([[1]])
    try:
        gerar(modelo, ids, 5, temperatura=0)
        assert False, "temperatura 0 deveria falhar (divisão por zero)"
    except ValueError:
        pass
    # Mesma semente, mesmo texto: é o que permite comparar temperaturas de forma justa.
    a = gerar(modelo, ids, 20, 1.0, gerador=torch.Generator().manual_seed(3))
    b = gerar(modelo, ids, 20, 1.0, gerador=torch.Generator().manual_seed(3))
    assert torch.equal(a, b)


def test_palavras_reais():
    vocabulario = {"febre", "dias", "paciente"}
    # Palavras com 3+ letras: paciente, com, febre, dias. Só "com" não está no vocabulário.
    assert taxa_palavras_reais("Paciente com febre há 3 dias", vocabulario) == 0.75
    assert taxa_palavras_reais("febre xptoz", vocabulario) == 0.5
    assert taxa_palavras_reais("", vocabulario) == 0.0


if __name__ == "__main__":
    for nome, fn in list(globals().items()):
        if nome.startswith("test_"):
            fn()
            print(f"ok  {nome}")
    print("Todos os testes da Etapa 4 passaram.")
