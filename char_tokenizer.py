"""
Tokenizador em nível de caractere.

Ideia central: cada caractere distinto do corpus vira um número inteiro (um "token").
O vocabulário é pequeno (tipicamente 100 a 200 símbolos em português), o código é
simples e fácil de explicar na apresentação.

Trade-off (bom para o relatório):
- Caractere: vocabulário mínimo, mas sequências longas (cada letra é um passo).
- Palavra: sequências curtas, mas vocabulário enorme e palavras nunca vistas.
- BPE (sub-palavras): meio-termo, usado por GPT-2/3/4. Experimento opcional da etapa 5.
"""

import json
from collections import Counter


class CharTokenizer:
    def __init__(self, chars):
        # Ordenar garante que o mesmo corpus gere sempre o mesmo vocabulário.
        self.chars = sorted(set(chars))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}  # string -> inteiro
        self.itos = {i: ch for i, ch in enumerate(self.chars)}  # inteiro -> string

    @classmethod
    def from_text(cls, text):
        return cls(text)

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        """Texto -> lista de inteiros. Falha com mensagem clara se aparecer caractere desconhecido."""
        try:
            return [self.stoi[ch] for ch in text]
        except KeyError as e:
            raise ValueError(f"Caractere fora do vocabulário: {e.args[0]!r}") from None

    def decode(self, ids):
        """Lista de inteiros -> texto."""
        return "".join(self.itos[int(i)] for i in ids)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"chars": self.chars}, f, ensure_ascii=False, indent=0)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f)["chars"])


def char_frequencies(text):
    """Contagem de cada caractere. Útil para achar símbolos raros que inflam o vocabulário."""
    return Counter(text)
