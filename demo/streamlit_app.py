"""
Versão em Streamlit da interface, para publicar com um link no Streamlit Community Cloud.

    pip install -r demo/requirements.txt
    streamlit run demo/streamlit_app.py

Usa os mesmos checkpoints e o mesmo laço de geração do app.py. Na nuvem, os modelos
precisam estar commitados em checkpoints/ (o .gitignore ignora *.pt: use git add -f).
"""

import math
import sys
from pathlib import Path

import streamlit as st

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from app import CHECKPOINTS, MAX_CARACTERES, gerar_caracteres, montar_lista_modelos  # noqa: E402
from generate import carregar_modelo  # noqa: E402

SUGESTOES = [
    "### Revalida 2022 | Pediatria",
    "### Enare Residência Médica 2024 | Clínica Médica",
    "### Revalida 2017\nUm homem com 45 anos",
]


@st.cache_resource
def modelos_disponiveis():
    # Ler cada checkpoint é lento: o Streamlit guarda a lista entre as visitas.
    return montar_lista_modelos() if CHECKPOINTS.exists() else []


@st.cache_resource
def modelo(nome):
    return carregar_modelo(CHECKPOINTS / f"{nome}.pt")


st.set_page_config(page_title="Medical Mini-GPT", page_icon="🩺")
st.title("Medical Mini-GPT")
st.write(
    "A small GPT built from scratch in PyTorch and trained on Brazilian medical exam questions "
    "(Revalida and Enare). Write the beginning of a text in Portuguese and the model continues it, "
    "one character at a time. It imitates the style of the questions; it does **not** give medical advice."
)

lista = modelos_disponiveis()
if not lista:
    st.error("No models found in checkpoints/. Train one first: python train.py")
    st.stop()

with st.sidebar:
    st.header("Settings")
    nomes = [m["nome"] for m in lista]  # menor perda primeiro: o melhor modelo vem selecionado
    escolhido = st.selectbox("Model", nomes)
    info = next(m for m in lista if m["nome"] == escolhido)
    st.caption(f"{info['parametros']:,} parameters · perplexity {info['perplexidade']}")
    temperatura = st.slider("Temperature", 0.1, 1.5, 0.8, 0.05,
                            help="Low: safer and repetitive. High: more varied, more mistakes.")
    top_k = st.slider("Top-k (0 = off)", 0, 50, 0,
                      help="Sample only among the k most likely characters.")
    n = st.slider("Characters to generate", 100, MAX_CARACTERES, 500, 50)
    seed = st.number_input("Seed", value=1337, step=1)

sugestao = st.radio("Start from", SUGESTOES + ["Your own text"], index=0)
inicio = st.text_area("Beginning of the text", value="" if sugestao == "Your own text" else sugestao)

if st.button("Generate", type="primary"):
    gpt, tok, _ = modelo(escolhido)
    desconhecidos = sorted({ch for ch in inicio if ch not in tok.stoi})
    if not inicio:
        st.warning("Write the beginning of a text.")
    elif desconhecidos:
        st.error("Characters outside the model's vocabulary: " + " ".join(repr(c) for c in desconhecidos))
    else:
        with st.container(border=True):
            st.text(inicio)
            # O texto aparece caractere por caractere, como no app.py.
            st.write_stream(gerar_caracteres(gpt, tok, inicio, n, temperatura, top_k, int(seed)))

st.caption("Code and write-up: [github.com/tega-crypto/mini-gpt-med](https://github.com/tega-crypto/mini-gpt-med)")
