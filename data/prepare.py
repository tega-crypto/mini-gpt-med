"""
Etapa 1 — Dados e tokenização.

Baixa o HealthQA-BR (questões de Revalida e Enare, licença CC-BY 4.0), limpa o texto,
separa treino/validação, cria o vocabulário de caracteres e salva tudo em formato binário
pronto para o treino.

Uso:
    python data/prepare.py                      # baixa do Hugging Face (precisa de internet)
    python data/prepare.py --arquivo meu.parquet  # usa um arquivo local (.parquet ou .csv)

Saídas em data/processed/:
    corpus.txt       texto completo, para ler e conferir
    train.bin        tokens de treino (uint16)
    val.bin          tokens de validação (uint16)
    tokenizer.json   vocabulário de caracteres
    meta.json        estatísticas (tamanho, vocabulário, perda inicial esperada...)
"""

import argparse
import json
import math
import random
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from tokenizer import CharTokenizer, char_frequencies  # noqa: E402

DATASET_HF = "Larxel/healthqa-br"
PARQUET_HF = f"https://huggingface.co/api/datasets/{DATASET_HF}/parquet/default/train/0.parquet"
SAIDA = RAIZ / "data" / "processed"

# Trocas que reduzem o vocabulário sem mudar o sentido do texto.
SUBSTITUICOES = {
    "“": '"', "”": '"', "‘": "'", "’": "'",  # aspas curvas
    "–": "-", "—": "-", "−": "-",                   # travessões e sinal de menos
    " ": " ", "\t": " ", " ": " ", "​": "",           # espaços especiais
    "…": "...",
    "µ": "μ", "ẞ": "β",                                   # micro e "beta" digitados com o símbolo errado
    "₁": "1", "₂": "2", "₃": "3", "⁺": "+",              # O₂ -> O2, igual ao resto do corpus
}

# Homóglifos: letras gregas e cirílicas idênticas às latinas, que vieram coladas dos PDFs
# originais ("Allium ceра" com "р" cirílico, alternativa "Ε" com épsilon). Para o modelo,
# "o" latino e "о" cirílico seriam tokens diferentes sem nenhum motivo.
HOMOGLIFOS = str.maketrans({
    "Α": "A", "Β": "B", "Ε": "E", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N",  # grego
    "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X", "ο": "o",
    "А": "A", "В": "B", "Е": "E", "М": "M", "Н": "H", "О": "O", "Р": "P",  # cirílico
    "С": "C", "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x",
})

# As alternativas vêm como dicionário Python:  'A': 'texto;'  ->  (A) texto;
# Algumas ocupam várias linhas (quebra de linha do PDF original); essas linhas são unidas.
ALTERNATIVA = re.compile(r"""^(['"])([A-E])\1: (['"])(.*?)\3$""", re.MULTILINE | re.DOTALL)


# ---------------------------------------------------------------- carregamento
def carregar(arquivo=None):
    """Devolve um DataFrame do pandas com as questões."""
    import pandas as pd

    if arquivo:
        arquivo = Path(arquivo)
        if arquivo.suffix == ".parquet":
            return pd.read_parquet(arquivo)
        return pd.read_csv(arquivo)

    print(f"Baixando {DATASET_HF} do Hugging Face...")
    try:
        from datasets import load_dataset
    except ImportError:
        # Sem a biblioteca 'datasets': o Hugging Face também serve o parquet direto por URL.
        return pd.read_parquet(PARQUET_HF)
    return load_dataset(DATASET_HF, split="train").to_pandas()


# ---------------------------------------------------------------- limpeza
def limpar(texto):
    texto = unicodedata.normalize("NFC", str(texto))  # "é" sempre como 1 caractere, nunca "e" + acento
    for velho, novo in SUBSTITUICOES.items():
        texto = texto.replace(velho, novo)
    texto = texto.translate(HOMOGLIFOS)
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    texto = re.sub(r" {2,}", " ", texto)                          # espaços repetidos
    texto = "\n".join(linha.strip() for linha in texto.split("\n"))
    texto = re.sub(r"\n{3,}", "\n\n", texto)                      # linhas em branco repetidas
    texto = ALTERNATIVA.sub(lambda m: f"({m[2]}) " + m[4].replace("\n", " "), texto)
    return texto.strip()


def formatar(linha, com_cabecalho=True, com_gabarito=True):
    """
    Cada questão vira um bloco de texto. O cabeçalho dá estrutura ao corpus:
    na demo, basta começar com "### Revalida" para o modelo escrever um enunciado.
    """
    partes = []
    if com_cabecalho:
        fonte = linha.get("source") or "Prova"
        ano = linha.get("year")
        area = linha.get("group")
        cab = f"### {fonte}"
        if ano is not None and not (isinstance(ano, float) and math.isnan(ano)):
            cab += f" {int(ano)}"
        if isinstance(area, str) and area.strip():
            cab += f" | {area.strip()}"
        partes.append(cab)
    partes.append(limpar(linha["question"]))
    if com_gabarito and isinstance(linha.get("answer"), str):
        partes.append(f"Gabarito: {linha['answer'].strip()}")
    return "\n".join(partes)


def remover_raros(textos, min_freq):
    """Remove caracteres que aparecem menos de min_freq vezes no corpus inteiro."""
    freq = char_frequencies("".join(textos))
    raros = {ch for ch, n in freq.items() if n < min_freq}
    if not raros:
        return textos, []
    tabela = str.maketrans({ch: "" for ch in raros})
    return [t.translate(tabela) for t in textos], sorted(raros)


# ---------------------------------------------------------------- principal
def main():
    # O console do Windows (cp1252) não imprime alguns símbolos, como "→" ou letras gregas.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arquivo", help="arquivo local .parquet ou .csv em vez de baixar")
    p.add_argument("--val", type=float, default=0.1, help="fração de validação (padrão 0.1)")
    p.add_argument("--min-freq", type=int, default=3, help="remove caracteres mais raros que isso")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--sem-cabecalho", action="store_true")
    p.add_argument("--sem-gabarito", action="store_true")
    p.add_argument("--saida", default=str(SAIDA))
    args = p.parse_args()

    df = carregar(args.arquivo)
    if "question" not in df.columns:
        sys.exit(f"Coluna 'question' não encontrada. Colunas: {list(df.columns)}")
    df = df.dropna(subset=["question"])
    df = df[df["question"].astype(str).str.strip().str.len() > 0]
    print(f"{len(df)} questões carregadas.")

    blocos = [formatar(l, not args.sem_cabecalho, not args.sem_gabarito) for l in df.to_dict("records")]

    # Embaralhar por QUESTÃO antes de separar: assim a validação tem todas as provas e anos,
    # e nenhuma questão aparece partida entre treino e validação (evita vazamento).
    random.Random(args.seed).shuffle(blocos)
    blocos, raros = remover_raros(blocos, args.min_freq)

    n_val = max(1, int(len(blocos) * args.val))
    val_txt = "\n\n".join(blocos[:n_val]) + "\n\n"
    train_txt = "\n\n".join(blocos[n_val:]) + "\n\n"

    # O vocabulário vem do corpus inteiro para a validação não ter caractere desconhecido.
    tok = CharTokenizer.from_text(train_txt + val_txt)
    if tok.vocab_size > 65535:
        sys.exit("Vocabulário grande demais para uint16.")

    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    np.array(tok.encode(train_txt), dtype=np.uint16).tofile(saida / "train.bin")
    np.array(tok.encode(val_txt), dtype=np.uint16).tofile(saida / "val.bin")
    tok.save(saida / "tokenizer.json")
    (saida / "corpus.txt").write_text(train_txt + val_txt, encoding="utf-8")

    meta = {
        "fonte": args.arquivo or DATASET_HF,
        "questoes": len(blocos),
        "questoes_val": n_val,
        "caracteres_treino": len(train_txt),
        "caracteres_val": len(val_txt),
        "tamanho_mb": round(len((train_txt + val_txt).encode("utf-8")) / 1e6, 2),
        "vocab_size": tok.vocab_size,
        "perda_inicial_esperada": round(math.log(tok.vocab_size), 4),
        "caracteres_raros_removidos": raros,
        "seed": args.seed,
    }
    (saida / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Corpus: {meta['tamanho_mb']} MB | treino {meta['caracteres_treino']:,} chars | "
          f"val {meta['caracteres_val']:,} chars")
    print(f"Vocabulário: {tok.vocab_size} caracteres | perda inicial esperada ln(V) = "
          f"{meta['perda_inicial_esperada']}")
    if raros:
        print(f"Removidos {len(raros)} caracteres raros: {''.join(raros)!r}")
    print(f"Arquivos salvos em {saida}")


if __name__ == "__main__":
    main()
