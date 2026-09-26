# Mini-GPT Médico

Um GPT pequeno, implementado do zero em PyTorch, treinado com questões de provas médicas brasileiras (Revalida e Enare). Trabalho de faculdade em grupo, baseado no nanoGPT de Andrej Karpathy.

> O modelo prevê o próximo caractere. Ele imita o estilo das questões, mas **não** dá informação médica confiável.

## Status

| Etapa | Conteúdo | Status |
| --- | --- | --- |
| 1. Dados e tokenização | `data/prepare.py`, `char_tokenizer.py`, `data/loader.py` | Pronta |
| 2. Arquitetura | `model.py` | Pronta |
| 3. Treino | `train.py` | Semana 3 |
| 4. Geração e avaliação | `generate.py` | Semana 3 |
| 5. Experimentos | `configs/`, `results/` | Semana 4 |

## Estrutura

```
data/prepare.py        baixa, limpa, separa e tokeniza o corpus
data/loader.py         forma os lotes (x, y) para o treino
char_tokenizer.py      tokenizador em nível de caractere
configs/base.json      hiperparâmetros iniciais
notebooks/             exploração (rodar no Kaggle)
model.py               o Transformer: atenção causal, blocos e cabeça de saída
tests/test_etapa1.py   testes de sanidade, rodam sem internet
tests/test_etapa2.py   testes do modelo: formas, causalidade, perda inicial
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

- **Tokenização por caractere:** vocabulário pequeno e código fácil de explicar. BPE fica como experimento da etapa 5.
- **Limpeza:** normalização Unicode (NFC), aspas e travessões padronizados, espaços repetidos removidos, caracteres com menos de 3 ocorrências descartados. Tudo isso reduz o vocabulário sem mudar o sentido.
- **Homóglifos:** o texto original, copiado de PDFs, tem letras gregas e cirílicas iguais às latinas (`Allium ceра` com "р" cirílico, alternativa `Ε` com épsilon grego). Elas são trocadas pela letra latina; sem isso, "o" e "о" seriam tokens diferentes.
- **Alternativas:** no dataset elas vêm como dicionário Python (`'A': 'texto'`, às vezes em várias linhas). Viram `(A) texto` numa linha só.

### Corpus após o `prepare.py`

| Medida | Valor |
| --- | --- |
| Questões (treino / validação) | 5.069 / 563 |
| Tamanho | 5,25 MB |
| Caracteres de treino / validação | 4.574.978 / 508.218 |
| Vocabulário | 125 caracteres (144 antes de tratar homóglifos e símbolos equivalentes) |
| Perda inicial esperada ln(125) | 4,83 |
- **Separação treino/validação (90/10) por questão:** as questões são embaralhadas antes da divisão. Assim a validação cobre todas as provas e anos, e nenhuma questão fica partida entre os dois conjuntos.
- **Lotes:** `y` é `x` deslocado uma posição; cada posição do bloco é um exemplo de "preveja o próximo caractere".
- **Teste de sanidade:** a perda inicial esperada é ln(tamanho do vocabulário), gravada em `data/processed/meta.json`. Na semana 2, a primeira perda do modelo precisa bater com esse valor.

## Decisões da Etapa 2 (para o relatório)

O `model.py` é um Transformer só com decodificador, como o GPT:

```
ids -> embedding do caractere + embedding da posição
    -> 4 blocos [LayerNorm -> atenção causal (4 cabeças) -> soma residual
                 LayerNorm -> MLP (4x mais largo, GELU)  -> soma residual]
    -> LayerNorm -> camada linear -> logits (125 notas, uma por caractere)
```

- **Atenção causal:** uma máscara triangular impede cada posição de ver as seguintes. Sem ela, prever o próximo caractere seria só copiá-lo.
- **Escala 1/sqrt(d):** mantém os produtos q·k pequenos, para o softmax não saturar logo no início.
- **Pre-norm e conexões residuais:** como no GPT-2; deixam o gradiente chegar às primeiras camadas e estabilizam o treino.
- **Inicialização com desvio 0,02:** os logits começam perto de zero e a perda inicial fica perto de ln(125).
- **Atenção escrita à mão** (em vez de `F.scaled_dot_product_attention`): mais lenta, mas cada passo da fórmula aparece no código.

Com `configs/base.json`, o modelo tem **833.661 parâmetros**. A perda inicial medida num lote real foi **4,90**, perto da esperada (4,83).

Os testes em `tests/test_etapa2.py` conferem que:
- as formas de entrada e saída estão certas;
- mudar um caractere não altera as previsões das posições anteriores (causalidade);
- a perda inicial é próxima de ln(vocabulário);
- o modelo consegue decorar um lote pequeno, sinal de que o gradiente chega a todas as camadas.

## Referências

- Karpathy, *Let's build GPT: from scratch, in code, spelled out* (vídeo) e repositório nanoGPT.
- Vaswani et al., *Attention Is All You Need* (2017).
