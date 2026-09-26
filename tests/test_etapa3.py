"""
Teste de sanidade da Etapa 3 (treino). Roda sem internet, em poucos segundos,
com um corpus repetitivo que um modelo minúsculo aprende rápido.

    python tests/test_etapa3.py
"""

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from char_tokenizer import CharTokenizer  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402


def test_treino_curto():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        dados = tmp / "processed"
        dados.mkdir()
        texto = "### Revalida 2020 | Pediatria\nFebre há 3 dias.\n(A) sim\nGabarito: A\n\n" * 200
        tok = CharTokenizer.from_text(texto)
        corte = int(len(texto) * 0.9)
        np.array(tok.encode(texto[:corte]), dtype=np.uint16).tofile(dados / "train.bin")
        np.array(tok.encode(texto[corte:]), dtype=np.uint16).tofile(dados / "val.bin")
        tok.save(dados / "tokenizer.json")
        (dados / "meta.json").write_text(
            json.dumps({"perda_inicial_esperada": math.log(tok.vocab_size)}), encoding="utf-8"
        )
        config = {
            "block_size": 16, "batch_size": 8, "n_embd": 32, "n_head": 2, "n_layer": 2,
            "dropout": 0.0, "learning_rate": 3e-3, "max_iters": 150, "eval_interval": 50,
            "eval_iters": 5, "seed": 1,
        }
        (tmp / "mini.json").write_text(json.dumps(config), encoding="utf-8")

        saida = subprocess.run(
            [sys.executable, str(RAIZ / "train.py"), "--config", str(tmp / "mini.json"),
             "--dados", str(dados), "--saida", str(tmp), "--dispositivo", "cpu"],
            capture_output=True, text=True, encoding="utf-8",
        )
        assert saida.returncode == 0, saida.stderr

        resultado = json.loads((tmp / "results" / "mini" / "perdas.json").read_text(encoding="utf-8"))
        historico = resultado["historico"]
        assert [h["passo"] for h in historico] == [0, 50, 100, 150]
        # O texto se repete: a perda precisa cair bem abaixo de ln(vocab).
        assert historico[-1]["val"] < 0.5 * historico[0]["val"], historico
        assert (tmp / "results" / "mini" / "curva_perda.png").exists()

        # O checkpoint traz tudo para recriar o modelo e o tokenizador sem a config.
        ckpt = torch.load(tmp / "checkpoints" / "mini.pt", weights_only=True)
        modelo = GPT(GPTConfig(**ckpt["config_modelo"]))
        modelo.load_state_dict(ckpt["modelo"])
        assert CharTokenizer(ckpt["chars"]).vocab_size == tok.vocab_size
        assert abs(ckpt["perda_val"] - resultado["melhor_perda_val"]) < 1e-3


if __name__ == "__main__":
    for nome, fn in list(globals().items()):
        if nome.startswith("test_"):
            fn()
            print(f"ok  {nome}")
    print("Todos os testes da Etapa 3 passaram.")
