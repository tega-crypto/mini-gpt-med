"""
Treino do Mini-GPT.

    python train.py                                   # usa configs/base.json
    python train.py --config configs/base.json --max-iters 500   # teste rápido

Cada passo do treino:
  1. sorteia um lote (x, y) do conjunto de treino;
  2. o modelo calcula os logits e a perda (entropia cruzada) de prever y a partir de x;
  3. backward: calcula o gradiente da perda em relação a cada peso;
  4. o otimizador AdamW ajusta os pesos um pouco na direção que reduz a perda.

A cada eval_interval passos, mede a perda média em treino e em validação.
Se a validação melhorou, salva o modelo em checkpoints/<nome>.pt.
No fim, salva as perdas e a curva em results/<nome>/.

Treino caindo e validação subindo = overfitting: o modelo está decorando as
questões de treino em vez de aprender o padrão do texto.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "data"))

from char_tokenizer import CharTokenizer  # noqa: E402
from loader import PASTA, carregar_tokens, get_batch  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402


def escolher_dispositivo(pedido):
    if pedido != "auto":
        return pedido
    if torch.cuda.is_available():
        return "cuda"  # GPU NVIDIA (Kaggle, Colab)
    if torch.backends.mps.is_available():
        return "mps"  # GPU dos Macs com chip Apple
    return "cpu"


@torch.no_grad()
def estimar_perda(modelo, dados, cfg, rng, dispositivo):
    """
    A perda de um único lote oscila muito; a média de eval_iters lotes é estável.
    modelo.eval() desliga o dropout, para medir o modelo como ele será usado.
    """
    modelo.eval()
    medias = {}
    for split, tokens in dados.items():
        perdas = []
        for _ in range(cfg["eval_iters"]):
            x, y = lote(tokens, cfg, rng, dispositivo)
            _, perda = modelo(x, y)
            perdas.append(perda.item())
        medias[split] = float(np.mean(perdas))
    modelo.train()
    return medias


def lote(tokens, cfg, rng, dispositivo):
    x, y = get_batch(tokens, cfg["batch_size"], cfg["block_size"], rng)
    return torch.from_numpy(x).to(dispositivo), torch.from_numpy(y).to(dispositivo)


def salvar_curva(historico, caminho, perda_inicial):
    import matplotlib

    matplotlib.use("Agg")  # gera o PNG sem abrir janela
    import matplotlib.pyplot as plt

    passos = [h["passo"] for h in historico]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(passos, [h["train"] for h in historico], label="treino")
    ax.plot(passos, [h["val"] for h in historico], label="validação")
    ax.axhline(perda_inicial, color="gray", linestyle=":", label=f"ln(vocab) = {perda_inicial:.2f}")
    ax.set_xlabel("passo")
    ax.set_ylabel("perda (entropia cruzada)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        # line_buffering: cada linha aparece na hora, mesmo com a saída indo para um arquivo.
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(RAIZ / "configs" / "base.json"))
    p.add_argument("--dados", default=str(PASTA), help="pasta gerada pelo data/prepare.py")
    p.add_argument("--max-iters", type=int, help="substitui o max_iters da config")
    p.add_argument("--dispositivo", default="auto", help="auto, cpu, cuda ou mps")
    p.add_argument("--nome", help="nome do experimento (padrão: nome do arquivo de config)")
    p.add_argument("--saida", default=str(RAIZ), help="onde criar checkpoints/ e results/")
    args = p.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.max_iters is not None:
        cfg["max_iters"] = args.max_iters
    nome = args.nome or Path(args.config).stem
    dispositivo = escolher_dispositivo(args.dispositivo)

    # Semente fixa: rodar de novo dá o mesmo resultado (na mesma máquina).
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])

    pasta = Path(args.dados)
    if not (pasta / "meta.json").exists():
        sys.exit(f"{pasta / 'meta.json'} não existe. Rode antes: python data/prepare.py")
    meta = json.loads((pasta / "meta.json").read_text(encoding="utf-8"))
    tok = CharTokenizer.load(pasta / "tokenizer.json")
    dados = {"train": carregar_tokens("train", pasta), "val": carregar_tokens("val", pasta)}

    config_modelo = GPTConfig.from_json(args.config, tok.vocab_size)
    modelo = GPT(config_modelo).to(dispositivo)
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=cfg["learning_rate"])

    pasta_ckpt = Path(args.saida) / "checkpoints"
    pasta_res = Path(args.saida) / "results" / nome
    pasta_ckpt.mkdir(parents=True, exist_ok=True)
    pasta_res.mkdir(parents=True, exist_ok=True)

    print(f"Experimento '{nome}' | dispositivo {dispositivo} | {modelo.num_parametros():,} parâmetros")
    print(f"{cfg['max_iters']} passos | lote {cfg['batch_size']} x {cfg['block_size']} caracteres | "
          f"perda inicial esperada {meta['perda_inicial_esperada']}")

    historico = []
    melhor_val = float("inf")
    inicio = time.time()
    for passo in range(cfg["max_iters"] + 1):
        if passo % cfg["eval_interval"] == 0 or passo == cfg["max_iters"]:
            perdas = estimar_perda(modelo, dados, cfg, rng, dispositivo)
            minutos = (time.time() - inicio) / 60
            historico.append({"passo": passo, **perdas, "minutos": round(minutos, 2)})
            marca = ""
            if perdas["val"] < melhor_val:
                melhor_val = perdas["val"]
                # Guardamos tudo o que a geração vai precisar: pesos, arquitetura e vocabulário.
                torch.save(
                    {
                        "modelo": modelo.state_dict(),
                        "config_modelo": vars(config_modelo),
                        "config_treino": cfg,
                        "chars": tok.chars,
                        "passo": passo,
                        "perda_val": melhor_val,
                    },
                    pasta_ckpt / f"{nome}.pt",
                )
                marca = "  * salvo"
            print(f"passo {passo:5d} | treino {perdas['train']:.4f} | val {perdas['val']:.4f} | "
                  f"{minutos:5.1f} min{marca}")
        if passo == cfg["max_iters"]:
            break

        x, y = lote(dados["train"], cfg, rng, dispositivo)
        _, perda = modelo(x, y)
        otimizador.zero_grad(set_to_none=True)
        perda.backward()
        otimizador.step()

    resultado = {
        "experimento": nome,
        "config": cfg,
        "parametros": modelo.num_parametros(),
        "dispositivo": dispositivo,
        "melhor_perda_val": round(melhor_val, 4),
        "historico": historico,
    }
    (pasta_res / "perdas.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    salvar_curva(historico, pasta_res / "curva_perda.png", meta["perda_inicial_esperada"])
    print(f"Melhor perda de validação: {melhor_val:.4f}")
    print(f"Modelo em {pasta_ckpt / (nome + '.pt')} | curva em {pasta_res / 'curva_perda.png'}")

    del dados  # no Windows, o memmap aberto trava os arquivos


if __name__ == "__main__":
    main()
