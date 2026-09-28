"""
Compara os experimentos treinados (Etapa 5).

    python compare.py

Lê results/<experimento>/perdas.json de cada treino e gera:
- results/comparacao.png: curvas de perda de todos os experimentos juntas;
- results/comparacao.md: tabela com parâmetros, perdas, tempo e qualidade das amostras.

Se o checkpoint do experimento existir, também:
- mede a perda no conjunto de validação inteiro, com janela deslizante;
- gera amostras com a mesma semente e temperatura para todos e mede a taxa de
  palavras reais e se as alternativas saem em ordem, (A) -> (B) -> (C).
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

from generate import carregar_modelo, gerar, palavras, taxa_palavras_reais  # noqa: E402

RESULTADOS = RAIZ / "results"
VAL = RAIZ / "data" / "processed" / "val.bin"


def carregar_experimentos(pasta):
    experimentos = []
    for arquivo in sorted(pasta.glob("*/perdas.json")):
        experimentos.append(json.loads(arquivo.read_text(encoding="utf-8")))
    # Do menor para o maior modelo; com o mesmo tamanho, do menor para o maior contexto.
    return sorted(experimentos, key=lambda e: (e["parametros"] // 100_000, e["config"]["block_size"]))


def salvar_grafico(experimentos, caminho):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, exp in enumerate(experimentos):
        cor = f"C{i}"
        passos = [h["passo"] for h in exp["historico"]]
        rotulo = (f"{exp['experimento']} ({exp['parametros'] / 1e6:.1f} M parâmetros, "
                  f"contexto {exp['config']['block_size']})")
        ax.plot(passos, [h["val"] for h in exp["historico"]], color=cor, label=f"{rotulo}: validação")
        ax.plot(passos, [h["train"] for h in exp["historico"]], color=cor, linestyle="--", alpha=0.6,
                label=f"{exp['experimento']}: treino")
    ax.set_xlabel("passo")
    ax.set_ylabel("perda (entropia cruzada)")
    ax.set_ylim(top=2.5)  # o começo (4,9) achataria a parte interessante do gráfico
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)


@torch.no_grad()
def perda_janela_deslizante(modelo, tokens, lote=64):
    """
    Perda média no conjunto inteiro, com cada caractere previsto a partir de
    pelo menos block_size/2 caracteres de contexto.

    Por que não usar só janelas lado a lado? Nelas, o primeiro caractere de cada
    janela é previsto sem contexto nenhum, o segundo com 1, e assim por diante.
    A média mistura previsões com pouco e com muito contexto, e a mistura muda
    com o block_size: comparar modelos com contextos diferentes ficaria injusto.
    Aqui as janelas andam block_size/2 por vez e só a segunda metade de cada uma
    entra na conta (na primeira janela, entra tudo).
    """
    T = modelo.config.block_size
    passo = T // 2
    tokens = torch.from_numpy(np.asarray(tokens, dtype=np.int64))
    inicios = list(range(0, len(tokens) - T, passo))
    soma, total = 0.0, 0
    for i in range(0, len(inicios), lote):
        grupo = inicios[i : i + lote]
        x = torch.stack([tokens[s : s + T] for s in grupo])
        y = torch.stack([tokens[s + 1 : s + T + 1] for s in grupo])
        logits, _ = modelo(x)
        perdas = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none").view(len(grupo), T)
        for j, s in enumerate(grupo):
            conta = perdas[j] if s == 0 else perdas[j, T - passo :]
            soma += conta.sum().item()
            total += conta.numel()
    return soma / total


def ordem_alternativas(texto):
    """
    Dentro de cada questão (separadas por "###"), conta os pares de alternativas
    seguidas e quantos estão em ordem: (A) depois (B), (B) depois (C)...
    Para acertar, o modelo precisa lembrar qual letra usou por último, que pode
    estar a mais de 64 caracteres de distância.
    """
    certos, pares = 0, 0
    for questao in texto.split("###"):
        letras = re.findall(r"^\(([A-E])\)", questao, flags=re.MULTILINE)
        for a, b in zip(letras, letras[1:]):
            pares += 1
            certos += ord(b) == ord(a) + 1
    return certos, pares


def qualidade_amostras(modelo, tok, vocabulario, temperatura, n_amostras=20, n=1000,
                       inicio="### Revalida", seed=1337):
    ids = torch.tensor([tok.encode(inicio)] * n_amostras)
    saida = gerar(modelo, ids, n, temperatura, gerador=torch.Generator().manual_seed(seed))
    textos = [tok.decode(linha.tolist()) for linha in saida]
    taxa = sum(taxa_palavras_reais(t[len(inicio):], vocabulario) for t in textos) / n_amostras
    certos = pares = 0
    for t in textos:
        c, p = ordem_alternativas(t)
        certos, pares = certos + c, pares + p
    return taxa, (certos, pares), textos[0]


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
    val = np.fromfile(VAL, dtype=np.uint16) if VAL.exists() else None

    linhas = [
        "# Comparação dos experimentos\n",
        "![Curvas de perda](comparacao.png)\n",
        "- **Perda val. (treino):** melhor média de 100 lotes sorteados, medida durante o treino.",
        "- **Perda val. (inteira):** validação inteira, cada caractere previsto com pelo menos "
        "metade do contexto do modelo. É a medida justa entre contextos diferentes.",
        f"- **Palavras reais** e **alternativas em ordem:** 20 amostras de 1000 caracteres a partir de "
        f"`### Revalida`, temperatura {args.temperatura}, mesma semente para todos.\n",
        "| Experimento | Parâmetros | Contexto | Perda val. (treino) | Perda val. (inteira) | Perplexidade | "
        "Treino − val. (fim) | Tempo | Palavras reais | Alternativas em ordem |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    amostras = []
    for exp in experimentos:
        nome, fim = exp["experimento"], exp["historico"][-1]
        ckpt = RAIZ / "checkpoints" / f"{nome}.pt"
        inteira = taxa = ordem = None
        if ckpt.exists():
            modelo, tok, _ = carregar_modelo(ckpt)
            if val is not None:
                inteira = perda_janela_deslizante(modelo, val)
            if vocabulario is not None:
                taxa, ordem, amostra = qualidade_amostras(modelo, tok, vocabulario, args.temperatura)
                amostras.append(f"## {nome}\n\n```\n{amostra}\n```\n")
        perda_ref = inteira if inteira is not None else exp["melhor_perda_val"]
        linhas.append(
            f"| {nome} | {exp['parametros']:,} | {exp['config']['block_size']} | {exp['melhor_perda_val']:.4f} | "
            f"{f'{inteira:.4f}' if inteira is not None else '–'} | {math.exp(perda_ref):.2f} | "
            f"{fim['train'] - fim['val']:+.4f} | {fim['minutos']:.0f} min ({exp['dispositivo']}) | "
            f"{f'{taxa:.0%}' if taxa is not None else '–'} | "
            f"{f'{ordem[0] / ordem[1]:.0%} ({ordem[0]}/{ordem[1]})' if ordem and ordem[1] else '–'} |"
        )
        print(linhas[-1])

    linhas.append("\nTreino − val. negativo: o modelo vai melhor no treino que na validação (começo de overfitting).\n")
    linhas += ["# Amostras (primeira de cada experimento)\n"] + amostras
    salvar_grafico(experimentos, RESULTADOS / "comparacao.png")
    (RESULTADOS / "comparacao.md").write_text("\n".join(linhas), encoding="utf-8")
    print(f"Tabela em {RESULTADOS / 'comparacao.md'} | gráfico em {RESULTADOS / 'comparacao.png'}")


if __name__ == "__main__":
    main()
