"""
Arquitetura do Transformer (só o decodificador, como no GPT).

Caminho de um lote pelo modelo, com B = batch_size, T = block_size, C = n_embd:

    ids (B, T)
      -> embedding do token + embedding da posição        (B, T, C)
      -> n_layer blocos: atenção causal + MLP             (B, T, C)
      -> LayerNorm final e camada linear                  (B, T, vocab_size)
      -> logits: uma nota para cada caractere possível, em cada posição

Baseado no nanoGPT de Karpathy, escrito para ser lido de cima para baixo.
"""

import json
import math
from dataclasses import dataclass, fields

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 125
    block_size: int = 64  # quantos caracteres o modelo enxerga para trás
    n_embd: int = 128  # tamanho do vetor que representa cada posição
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1

    @classmethod
    def from_json(cls, path, vocab_size):
        """Lê configs/*.json e ignora as chaves de treino (learning_rate, max_iters...)."""
        with open(path, encoding="utf-8") as f:
            dados = json.load(f)
        nomes = {campo.name for campo in fields(cls)} - {"vocab_size"}
        return cls(vocab_size=vocab_size, **{k: v for k, v in dados.items() if k in nomes})


class CausalSelfAttention(nn.Module):
    """
    Cada posição olha para as anteriores e decide quais importam para ela.

    Para cada posição, uma camada linear gera três vetores:
    - query (q): "o que estou procurando?"
    - key (k):   "o que eu tenho a oferecer?"
    - value (v): "o que eu passo adiante se for escolhido?"

    O peso que a posição t dá à posição s é softmax(q_t · k_s / sqrt(d)).
    A máscara causal zera os pesos de s > t: o modelo não pode ver o futuro,
    senão prever o próximo caractere seria trivial (bastaria copiá-lo).

    Multi-cabeça: n_head atenções menores em paralelo, cada uma com d = C / n_head.
    Uma cabeça pode aprender a olhar para a letra anterior, outra para o
    início da palavra, e assim por diante.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd precisa ser múltiplo de n_head"
        self.n_head = config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)  # q, k e v de uma vez
        self.proj = nn.Linear(config.n_embd, config.n_embd)  # junta as cabeças
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # Matriz triangular inferior: 1 onde a posição pode olhar, 0 no futuro.
        # register_buffer: vai junto para a GPU, mas não é treinada.
        mascara = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mascara", mascara.view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, d): cada cabeça trabalha de forma independente.
        d = C // self.n_head
        q = q.view(B, T, self.n_head, d).transpose(1, 2)
        k = k.view(B, T, self.n_head, d).transpose(1, 2)
        v = v.view(B, T, self.n_head, d).transpose(1, 2)

        # Dividir por sqrt(d) mantém os valores numa escala em que o softmax não satura.
        pesos = (q @ k.transpose(-2, -1)) / math.sqrt(d)  # (B, n_head, T, T)
        pesos = pesos.masked_fill(self.mascara[:, :, :T, :T] == 0, float("-inf"))
        pesos = F.softmax(pesos, dim=-1)  # exp(-inf) = 0: o futuro recebe peso zero
        pesos = self.attn_dropout(pesos)

        y = pesos @ v  # média ponderada dos values: (B, n_head, T, d)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # junta as cabeças de volta
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    """
    Depois de trocar informação entre posições (atenção), cada posição
    processa o que recebeu sozinha: camada 4x mais larga, ativação não linear
    e volta ao tamanho C.
    """

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    Um bloco do Transformer: comunicação (atenção) seguida de computação (MLP).

    Conexões residuais (x + ...): cada subcamada só soma um ajuste ao que já existe.
    O gradiente passa direto pelas somas até as primeiras camadas, o que torna
    redes profundas treináveis.
    LayerNorm antes de cada subcamada ("pre-norm", como no GPT-2) estabiliza o treino.
    """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)  # qual caractere
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)  # em que posição
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size)
        self.apply(self._init_pesos)

    @staticmethod
    def _init_pesos(modulo):
        # Pesos pequenos (desvio 0,02, como no GPT-2): no início os logits ficam
        # perto de zero, o softmax fica quase uniforme e a perda começa em ln(vocab_size).
        if isinstance(modulo, (nn.Linear, nn.Embedding)):
            nn.init.normal_(modulo.weight, mean=0.0, std=0.02)
        if isinstance(modulo, nn.Linear) and modulo.bias is not None:
            nn.init.zeros_(modulo.bias)

    def num_parametros(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, ids, alvos=None):
        """
        ids: (B, T) com T <= block_size. alvos: (B, T), os ids deslocados uma posição.
        Devolve (logits, perda); a perda é None quando não há alvos (na geração).
        """
        B, T = ids.shape
        if T > self.config.block_size:
            raise ValueError(f"Sequência com {T} tokens, mas block_size = {self.config.block_size}.")
        posicoes = torch.arange(T, device=ids.device)
        x = self.drop(self.tok_emb(ids) + self.pos_emb(posicoes))  # (B, T, C)
        for bloco in self.blocks:
            x = bloco(x)
        logits = self.head(self.ln_f(x))  # (B, T, vocab_size)

        perda = None
        if alvos is not None:
            # A entropia cruzada espera (N, classes): junta lote e posições numa dimensão só.
            perda = F.cross_entropy(logits.view(B * T, -1), alvos.view(B * T))
        return logits, perda
