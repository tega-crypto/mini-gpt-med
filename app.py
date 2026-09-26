"""
Interface no navegador para testar o modelo, com barra de prompt.

    python app.py                  # abre http://localhost:8000 no navegador
    python app.py --porta 8080

Usa só a biblioteca padrão do Python (http.server), sem instalar nada.
O texto aparece caractere por caractere, conforme o modelo gera:
cada caractere é enviado ao navegador assim que é sorteado.
"""

import argparse
import json
import math
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from torch.nn import functional as F

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

from generate import carregar_modelo  # noqa: E402

PAGINA = RAIZ / "interface" / "index.html"
CHECKPOINTS = RAIZ / "checkpoints"
MAX_CARACTERES = 2000

_modelos = {}
_trava = threading.Lock()  # carregar o mesmo modelo duas vezes ao mesmo tempo seria desperdício


def modelo(nome):
    with _trava:
        if nome not in _modelos:
            _modelos[nome] = carregar_modelo(CHECKPOINTS / f"{nome}.pt")
        return _modelos[nome]


def listar_modelos():
    lista = []
    for caminho in sorted(CHECKPOINTS.glob("*.pt")):
        ckpt = torch.load(caminho, map_location="cpu", weights_only=True)
        # A máscara causal está no state_dict (register_buffer), mas não é parâmetro.
        n_param = sum(t.numel() for k, t in ckpt["modelo"].items() if not k.endswith("mascara"))
        lista.append({
            "nome": caminho.stem,
            "parametros": n_param,
            "perda_val": round(ckpt["perda_val"], 4),
            "perplexidade": round(math.exp(ckpt["perda_val"]), 2),
            "vocabulario": "".join(ckpt["chars"]),
        })
    # Menor perda primeiro: o melhor modelo aparece selecionado.
    return sorted(lista, key=lambda m: m["perda_val"])


@torch.no_grad()
def gerar_caracteres(gpt, tok, inicio, n, temperatura, top_k, seed):
    """Mesmo laço do generate.gerar, mas devolve cada caractere assim que ele é sorteado."""
    gerador = torch.Generator().manual_seed(seed)
    ids = torch.tensor([tok.encode(inicio)])
    block_size = gpt.config.block_size
    for _ in range(n):
        logits, _ = gpt(ids[:, -block_size:])
        logits = logits[:, -1, :] / temperatura
        if top_k:
            limite = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
            logits = logits.masked_fill(logits < limite, float("-inf"))
        proximo = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=gerador)
        ids = torch.cat([ids, proximo], dim=1)
        yield tok.decode([proximo.item()])


class Servidor(BaseHTTPRequestHandler):
    def log_message(self, formato, *args):
        pass  # sem uma linha no terminal para cada requisição

    def responder(self, status, corpo, tipo="application/json; charset=utf-8"):
        dados = corpo if isinstance(corpo, bytes) else json.dumps(corpo, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(dados)))
        self.end_headers()
        self.wfile.write(dados)

    def do_GET(self):
        if self.path == "/":
            self.responder(200, PAGINA.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/modelos":
            self.responder(200, listar_modelos())
        else:
            self.responder(404, {"erro": "não encontrado"})

    def do_POST(self):
        if self.path != "/gerar":
            return self.responder(404, {"erro": "não encontrado"})
        try:
            pedido = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            nome = str(pedido["modelo"])
            if not (CHECKPOINTS / f"{nome}.pt").exists() or "/" in nome or "\\" in nome:
                raise ValueError(f"Modelo '{nome}' não encontrado em checkpoints/.")
            inicio = str(pedido["inicio"])
            if not inicio:
                raise ValueError("Escreva um começo de texto.")
            n = max(1, min(int(pedido.get("n", 400)), MAX_CARACTERES))
            temperatura = float(pedido.get("temperatura", 0.8))
            if not 0.05 <= temperatura <= 3:
                raise ValueError("A temperatura precisa ficar entre 0,05 e 3.")
            top_k = int(pedido.get("top_k") or 0)
            seed = int(pedido.get("seed", 1337))
            gpt, tok, _ = modelo(nome)
            desconhecidos = sorted({ch for ch in inicio if ch not in tok.stoi})
            if desconhecidos:
                raise ValueError("Caracteres fora do vocabulário do modelo: " + " ".join(repr(c) for c in desconhecidos))
        except (KeyError, TypeError, ValueError) as e:
            return self.responder(400, {"erro": str(e)})

        # Sem Content-Length: o navegador lê o texto aos pedaços até a conexão fechar.
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            for ch in gerar_caracteres(gpt, tok, inicio, n, temperatura, top_k, seed):
                self.wfile.write(ch.encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # o usuário apertou "parar": o navegador fechou a conexão


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--porta", type=int, default=8000)
    p.add_argument("--sem-navegador", action="store_true", help="não abre o navegador sozinho")
    args = p.parse_args()

    if not list(CHECKPOINTS.glob("*.pt")):
        sys.exit("Nenhum modelo em checkpoints/. Treine antes: python train.py")
    endereco = f"http://localhost:{args.porta}"
    servidor = ThreadingHTTPServer(("127.0.0.1", args.porta), Servidor)
    print(f"Mini-GPT Médico rodando em {endereco}  (Ctrl+C para encerrar)")
    if not args.sem_navegador:
        webbrowser.open(endereco)
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\nEncerrado.")


if __name__ == "__main__":
    main()
