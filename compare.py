"""
Compara os experimentos treinados (Etapa 5).

    python compare.py

Lê results/<experimento>/perdas.json de cada treino e gera:
- results/comparacao.png: curvas de perda de todos os experimentos juntas;
- results/comparacao.md: tabela com parâmetros, perdas, tempo e qualidade das amostras.

Se o checkpoint do experimento existir, também gera amostras com a mesma
semente e temperatura para todos e mede a taxa de palavras reais.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

from generate import carregar_modelo, gerar, palavras, taxa_palavras_reais  # noqa: E402

RESULTADOS = RAIZ / "results"


def carregar_experimentos(pasta):
    experimentos = []
    for arquivo in sorted(pasta.glob("*/perdas.json")):
        experimentos.append(json.loads(arquivo.read_text(encoding="utf-8")))
    # Do menor para o maior modelo: a tabela fica na ordem da escala.
    return sorted(experimentos, key=lambda e: e["parametros"])


def salvar_grafico(experimentos, caminho):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, exp in enumerate(experimentos):
        cor = f"C{i}"
        passos = [h["passo"] for h in exp["historico"]]
        rotulo = f"{exp['experimento']} ({exp['parametros'] / 1e6:.1f} M parâmetros)"
        ax.plot(passos, [h["val"] for h in exp["historico"]], color=cor, label=f"{rotulo}: validação")
        ax.plot(passos, [h["train"] for h in exp["historico"]], color=cor, linestyle="--", alpha=0.6,
                label=f"{exp['experimento']}: treino")
    ax.set_xlabel("passo")
    ax.set_ylabel("perda (entropia cruzada)")
    ax.set_ylim(top=2.5)  # o começo (4,9) achataria a parte interessante do gráfico
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)


def qualidade_amostras(nome, vocabulario, temperatura, n_amostras=5, n=400, inicio="### Revalida", seed=1337):
    ckpt = RAIZ / "checkpoints" / f"{nome}.pt"
    if vocabulario is None or not ckpt.exists():
        return None, None
    modelo, tok, _ = carregar_modelo(ckpt)
    ids = torch.tensor([tok.encode(inicio)] * n_amostras)
    saida = gerar(modelo, ids, n, temperatura, gerador=torch.Generator().manual_seed(seed))
    textos = [tok.decode(linha.tolist()) for linha in saida]
    taxa = sum(taxa_palavras_reais(t[len(inicio):], vocabulario) for t in textos) / n_amostras
    return taxa, textos[0]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--temperatura", type=float, default=0.8)
    args = p.parse_args()

    experimentos = carregar_experimentos(RESULTADOS)
    if not experimentos:
        sys.exit("Nenhum results/*/perdas.json encontrado. Treine antes: python train.py")

    corpus = RAIZ / "data" / "processed" / "corpus.txt"
    vocabulario = set(palavras(corpus.read_text(encoding="utf-8"))) if corpus.exists() else None

    linhas = [
        "# Comparação dos experimentos\n",
        "![Curvas de perda](comparacao.png)\n",
        f"Palavras reais: média de 5 amostras de 400 caracteres a partir de `### Revalida`, "
        f"temperatura {args.temperatura}, mesma semente para todos.\n",
        "| Experimento | Parâmetros | Perda val | Perplexidade | Treino − val (fim) | Tempo | Palavras reais |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    amostras = []
    for exp in experimentos:
        fim = exp["historico"][-1]
        taxa, amostra = qualidade_amostras(exp["experimento"], vocabulario, args.temperatura)
        linhas.append(
            f"| {exp['experimento']} | {exp['parametros']:,} | {exp['melhor_perda_val']:.4f} | "
            f"{math.exp(exp['melhor_perda_val']):.2f} | {fim['train'] - fim['val']:+.4f} | "
            f"{fim['minutos']:.0f} min ({exp['dispositivo']}) | {f'{taxa:.0%}' if taxa is not None else '–'} |"
        )
        if amostra:
            amostras.append(f"## {exp['experimento']}\n\n```\n{amostra}\n```\n")
        print(linhas[-1])

    linhas.append("\nTreino − val negativo: o modelo vai melhor no treino que na validação (começo de overfitting).\n")
    linhas += ["# Amostras (primeira de cada experimento)\n"] + amostras
    salvar_grafico(experimentos, RESULTADOS / "comparacao.png")
    (RESULTADOS / "comparacao.md").write_text("\n".join(linhas), encoding="utf-8")
    print(f"Tabela em {RESULTADOS / 'comparacao.md'} | gráfico em {RESULTADOS / 'comparacao.png'}")


if __name__ == "__main__":
    main()
