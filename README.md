# Mini-GPT Médico

Um GPT pequeno, implementado do zero em PyTorch, treinado com questões de provas médicas brasileiras (Revalida e Enare). Trabalho de faculdade em grupo, baseado no nanoGPT de Andrej Karpathy.

> O modelo prevê o próximo caractere. Ele imita o estilo das questões, mas **não** dá informação médica confiável.

## Status

| Etapa | Conteúdo | Status |
| --- | --- | --- |
| 1. Dados e tokenização | `data/prepare.py`, `tokenizer.py`, `data/loader.py` | Pronta |
| 2. Arquitetura | `model.py` | Semana 2 |
| 3. Treino | `train.py` | Semana 3 |
| 4. Geração e avaliação | `generate.py` | Semana 3 |
| 5. Experimentos | `configs/`, `results/` | Semana 4 |

## Estrutura

```
data/prepare.py        baixa, limpa, separa e tokeniza o corpus
data/loader.py         forma os lotes (x, y) para o treino
tokenizer.py           tokenizador em nível de caractere
configs/base.json      hiperparâmetros iniciais
notebooks/             exploração (rodar no Kaggle)
tests/test_etapa1.py   testes de sanidade, rodam sem internet
results/               curvas de perda e amostras geradas
```

## Como rodar a Etapa 1

```bash
pip install -r requirements.txt
python tests/test_etapa1.py   # confere se está tudo certo (não precisa de internet)
python data/prepare.py        # baixa o HealthQA-BR e gera data/processed/
```

No Kaggle: crie um notebook, ative **Settings → Internet: On**, clone o repositório e rode `notebooks/01_exploracao_dados.ipynb`.

## Dados

[HealthQA-BR](https://huggingface.co/datasets/Larxel/healthqa-br): 5.632 questões de múltipla escolha do Revalida, do Enare Residência Médica e do Enare Multiprofissional, em português, licença CC-BY 4.0.

Cada questão vira um bloco de texto:

```
### Revalida 2013 | Clínica Médica
Homem com 49 anos de idade apresenta...
(A) ...
Gabarito: C
```

O cabeçalho dá estrutura ao corpus: na demo, basta começar com `### Revalida` para o modelo escrever um enunciado.

## Decisões da Etapa 1 (para o relatório)

- **Tokenização por caractere:** vocabulário pequeno (~100–200 símbolos) e código fácil de explicar. BPE fica como experimento da etapa 5.
- **Limpeza:** normalização Unicode (NFC), aspas e travessões padronizados, espaços repetidos removidos, caracteres com menos de 3 ocorrências descartados. Tudo isso reduz o vocabulário sem mudar o sentido.
- **Separação treino/validação (90/10) por questão:** as questões são embaralhadas antes da divisão. Assim a validação cobre todas as provas e anos, e nenhuma questão fica partida entre os dois conjuntos.
- **Lotes:** `y` é `x` deslocado uma posição; cada posição do bloco é um exemplo de "preveja o próximo caractere".
- **Teste de sanidade:** a perda inicial esperada é ln(tamanho do vocabulário), gravada em `data/processed/meta.json`. Na semana 2, a primeira perda do modelo precisa bater com esse valor.

## Referências

- Karpathy, *Let's build GPT: from scratch, in code, spelled out* (vídeo) e repositório nanoGPT.
- Vaswani et al., *Attention Is All You Need* (2017).
