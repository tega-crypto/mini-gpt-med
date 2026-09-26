"""
Geração de texto com o modelo treinado.

    python generate.py                                    # começa com "### Revalida"
    python generate.py --inicio "### Enare" --temperatura 0.5
    python generate.py --comparar                         # tabela de temperaturas para o relatório

Geração auto-regressiva: o modelo prevê só o PRÓXIMO caractere. Para escrever um
texto, repetimos:
  1. passa o texto até agora pelo modelo (no máximo os últimos block_size caracteres);
  2. pega os logits da última posição: uma nota para cada caractere do vocabulário;
  3. transforma em probabilidades e sorteia um caractere;
  4. cola o caractere no fim do texto e volta ao passo 1.

Dois controles do sorteio:
- temperatura: divide os logits antes do softmax. Abaixo de 1 exagera as
  diferenças (texto mais seguro e repetitivo); acima de 1 achata (mais variado e
  mais erros). Perto de 0, sempre escolhe o caractere mais provável.
- top-k: só sorteia entre os k caracteres mais prováveis. Corta a "cauda" de
  caracteres improváveis, que é de onde vêm os erros mais estranhos.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

from char_tokenizer import CharTokenizer  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402


def carregar_modelo(caminho, dispositivo="cpu"):
    """O checkpoint do train.py traz pesos, arquitetura e vocabulário: não precisa da config."""
    ckpt = torch.load(caminho, map_location=dispositivo, weights_only=True)
    modelo = GPT(GPTConfig(**ckpt["config_modelo"]))
    modelo.load_state_dict(ckpt["modelo"])
    modelo.to(dispositivo).eval()  # eval(): desliga o dropout
    return modelo, CharTokenizer(ckpt["chars"]), ckpt


@torch.no_grad()
def gerar(modelo, ids, n_novos, temperatura=1.0, top_k=None, gerador=None):
    """ids: tensor (B, T) com o começo do texto. Devolve (B, T + n_novos)."""
    if temperatura <= 0:
        raise ValueError("A temperatura precisa ser maior que zero.")
    block_size = modelo.config.block_size
    for _ in range(n_novos):
        contexto = ids[:, -block_size:]  # o modelo só enxerga os últimos block_size caracteres
        logits, _ = modelo(contexto)
        logits = logits[:, -1, :] / temperatura  # só a última posição interessa
        if top_k is not None:
            # Tudo abaixo do k-ésimo maior logit vira -inf, ou seja, probabilidade 0.
            limite = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
            logits = logits.masked_fill(logits < limite, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        proximo = torch.multinomial(probs, num_samples=1, generator=gerador)
        ids = torch.cat([ids, proximo], dim=1)
    return ids


def palavras(texto):
    return re.findall(r"[a-záàâãéêíóôõúüç]{3,}", texto.lower())


def taxa_palavras_reais(texto, vocabulario):
    """Avaliação simples: fração das palavras geradas (3+ letras) que existem no corpus."""
    lista = palavras(texto)
    return sum(p in vocabulario for p in lista) / len(lista) if lista else 0.0


def comparar(modelo, tok, inicio, n, seed, top_k, saida, n_amostras=5):
    """
    Gera n_amostras textos por temperatura e salva uma tabela em Markdown para o relatório.
    A taxa de palavras reais é a média das amostras: uma amostra só oscila demais.
    """
    corpus = RAIZ / "data" / "processed" / "corpus.txt"
    vocabulario = set(palavras(corpus.read_text(encoding="utf-8"))) if corpus.exists() else None

    linhas = [f"# Amostras por temperatura\n\nInício: `{inicio}` | {n} caracteres | top-k: {top_k or 'não'}\n"]
    if vocabulario:
        linhas.append(f"Palavras reais: fração das palavras geradas (3+ letras) que aparecem no corpus, "
                      f"média de {n_amostras} amostras. Abaixo, a primeira amostra de cada temperatura.\n")
    for temperatura in [0.5, 0.8, 1.0, 1.2]:
        gerador = torch.Generator().manual_seed(seed)  # mesma semente: só a temperatura muda
        ids = torch.tensor([tok.encode(inicio)] * n_amostras)  # as amostras saem juntas, num lote
        textos = [tok.decode(linha.tolist()) for linha in gerar(modelo, ids, n, temperatura, top_k, gerador)]
        titulo = f"## Temperatura {temperatura}"
        if vocabulario:
            taxa = sum(taxa_palavras_reais(t[len(inicio):], vocabulario) for t in textos) / n_amostras
            titulo += f" | palavras reais: {taxa:.0%}"
        print(f"\n{titulo}\n{textos[0]}")
        linhas.append(f"{titulo}\n\n```\n{textos[0]}\n```\n")
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text("\n".join(linhas), encoding="utf-8")
    print(f"\nTabela salva em {saida}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # mensagens do sys.exit
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=str(RAIZ / "checkpoints" / "base.pt"))
    p.add_argument("--inicio", default="### Revalida", help="texto inicial (prompt)")
    p.add_argument("--n", type=int, default=500, help="quantos caracteres gerar")
    p.add_argument("--temperatura", type=float, default=0.8)
    p.add_argument("--top-k", type=int, help="sorteia só entre os k mais prováveis")
    p.add_argument("--amostras", type=int, default=1)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--comparar", action="store_true", help="gera com temperaturas 0.5, 0.8, 1.0 e 1.2")
    args = p.parse_args()

    if not Path(args.checkpoint).exists():
        sys.exit(f"{args.checkpoint} não existe. Treine antes: python train.py")
    modelo, tok, ckpt = carregar_modelo(args.checkpoint)
    nome = Path(args.checkpoint).stem
    print(f"Modelo '{nome}': passo {ckpt['passo']}, perda de validação {ckpt['perda_val']:.4f} "
          f"(perplexidade {math.exp(ckpt['perda_val']):.2f})")
    try:
        tok.encode(args.inicio)
    except ValueError as e:
        sys.exit(f"{e}. O início só pode usar caracteres do vocabulário.")

    if args.comparar:
        comparar(modelo, tok, args.inicio, args.n, args.seed, args.top_k, RAIZ / "results" / nome / "amostras.md")
        return

    gerador = torch.Generator().manual_seed(args.seed)
    for i in range(args.amostras):
        ids = torch.tensor([tok.encode(args.inicio)])
        texto = tok.decode(gerar(modelo, ids, args.n, args.temperatura, args.top_k, gerador)[0].tolist())
        print(f"\n--- amostra {i + 1} (temperatura {args.temperatura}, top-k {args.top_k or 'não'}) ---\n{texto}")


if __name__ == "__main__":
    main()
