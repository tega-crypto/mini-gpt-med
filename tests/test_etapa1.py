"""
Testes de sanidade da Etapa 1. Rodam sem internet, com um mini-dataset falso.

    python tests/test_etapa1.py
"""

import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "data"))

from tokenizer import CharTokenizer  # noqa: E402
from prepare import limpar, formatar  # noqa: E402
from loader import carregar_tokens, get_batch  # noqa: E402


def test_tokenizer_ida_e_volta():
    tok = CharTokenizer.from_text("Paciente com febre há 3 dias.")
    texto = "febre há 3 dias"
    assert tok.decode(tok.encode(texto)) == texto
    try:
        tok.encode("Z")
        assert False, "deveria falhar com caractere desconhecido"
    except ValueError:
        pass


def test_limpeza():
    sujo = "Mulher  com 72 anos — “anemia”\r\n\n\n\nFim"
    assert limpar(sujo) == 'Mulher com 72 anos - "anemia"\n\nFim'
    # "é" decomposto (e + acento combinante) vira um caractere só
    assert limpar("é") == "é"


def test_formatacao():
    linha = {"source": "Revalida", "year": 2013, "group": "Pediatria", "question": "Enunciado", "answer": "C"}
    assert formatar(linha) == "### Revalida 2013 | Pediatria\nEnunciado\nGabarito: C"
    linha["group"] = None
    assert formatar(linha).startswith("### Revalida 2013\n")


def _dataset_falso(n=60):
    rng = np.random.default_rng(0)
    areas = ["Pediatria", "Cirurgia", "Clínica Médica", None]
    frases = ["Homem com {} anos apresenta dor torácica.", "Criança de {} anos com febre e tosse.",
              "Gestante com {} semanas refere cefaleia. Qual a conduta?"]
    linhas = []
    for i in range(n):
        linhas.append({
            "id": f"{i:08x}", "source": "Revalida", "year": int(rng.integers(2011, 2025)),
            "group": areas[i % 4],
            "question": frases[i % 3].format(int(rng.integers(1, 90))) + "\n(A) x\n(B) y\n(C) z",
            "answer": "ABCDE"[i % 5],
        })
    return pd.DataFrame(linhas)


def test_prepare_e_lotes():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _dataset_falso().to_csv(tmp / "falso.csv", index=False)
        saida = tmp / "processed"
        r = subprocess.run([sys.executable, str(RAIZ / "data" / "prepare.py"), "--arquivo",
                            str(tmp / "falso.csv"), "--saida", str(saida), "--min-freq", "1"],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        print(r.stdout)

        tok = CharTokenizer.load(saida / "tokenizer.json")
        train, val = carregar_tokens("train", saida), carregar_tokens("val", saida)
        corpus = (saida / "corpus.txt").read_text(encoding="utf-8")
        # Nada se perdeu na codificação: decodificar treino + validação devolve o corpus.
        assert tok.decode(train) + tok.decode(val) == corpus
        assert "### Revalida" in corpus

        # x e y: y é x deslocado uma posição.
        x, y = get_batch(train, batch_size=8, block_size=16, rng=np.random.default_rng(1))
        assert x.shape == y.shape == (8, 16) and x.dtype == np.int64
        assert (x[:, 1:] == y[:, :-1]).all()

        # Perda inicial esperada: um modelo que ainda não aprendeu nada prevê
        # distribuição uniforme, e a entropia cruzada fica em -ln(1/V) = ln(V).
        V = tok.vocab_size
        probs_uniformes = np.full(V, 1 / V)
        perda = -np.mean(np.log(probs_uniformes[y]))
        assert abs(perda - math.log(V)) < 1e-9


if __name__ == "__main__":
    for nome, fn in list(globals().items()):
        if nome.startswith("test_"):
            fn()
            print(f"ok  {nome}")
    print("Todos os testes da Etapa 1 passaram.")
