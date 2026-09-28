# Comparação dos experimentos

![Curvas de perda](comparacao.png)

- **Perda val. (treino):** melhor média de 100 lotes sorteados, medida durante o treino.
- **Perda val. (inteira):** validação inteira, cada caractere previsto com pelo menos metade do contexto do modelo. É a medida justa entre contextos diferentes.
- **Palavras reais** e **alternativas em ordem:** 20 amostras de 1000 caracteres a partir de `### Revalida`, temperatura 0.8, mesma semente para todos.

| Experimento | Parâmetros | Contexto | Perda val. (treino) | Perda val. (inteira) | Perplexidade | Treino − val. (fim) | Tempo | Palavras reais | Alternativas em ordem |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 833,661 | 64 | 1.2542 | 1.1925 | 3.30 | -0.0287 | 13 min (cpu) | 80% | 24% (8/33) |
| grande | 4,819,581 | 64 | 1.1270 | 1.0547 | 2.87 | -0.0363 | 48 min (cpu) | 87% | 40% (17/42) |
| contexto | 4,835,965 | 128 | 1.0569 | 1.0149 | 2.76 | -0.0314 | 120 min (cpu) | 88% | 73% (16/22) |

Treino − val. negativo: o modelo vai melhor no treino que na validação (começo de overfitting).

# Amostras (primeira de cada experimento)

## base

```
### Revalida 2017
Um mulher com 44 anos, seguindo a nasmáticas desconfirmações as aleitas de diâmetro da determinação clínica. Ao exame físico de base aumento de preenche propedem diagnóstica ou morformalidade, solução não indica a sobre emergência ao exame físico, com o diagnóstico de orientação do resecimento vesicular à vista que a diagnóstica sintomas com gastroplasia, visialmente que na indicação alimentar indicada em pacientes assinale a opção que a opção que apresenta as mumentos assintomas presentes e que associada, em familiares contrainde os músculos seguinte do concentrole abertura do três o resultado da família.
Gabarito: D

### Enare Residência Médica 2024 | Medicina de Família ausência
A paciente recebeu agente e está neurológica em exercícios recidos ou atividades. Assintomática após observação de extremidades na cunha sorolaridade de doença consulta uso de pacientes de idade. No tem exame físico, humante no primeiro direto de fibre, no cálculo de membros superiores indivíduos. Entr
```

## grande

```
### Revalida 2017
Um médico com 14 anos, sem leite meneral, depressiva de linfonoda em visualidade (Talando O2 expenso e por investigação de novos achados;
(E) contexto capilar em uma monoculação do deno-Consideral, menor e pesquisa de alergia pulmonar. A paciente ocorrida foi estabelecida miocárdica de piora objetiva.
assinale a opção que indica a medicação física a transfusão devido ao município do calor e enquanto de referência renal.
A respeito tumor do paciente é em regular, mas não são sentir o contexto de saúde.
Gabarito: B

### Enare Residência Médica 2024 | Pediatria
A doença de Saúde complementar a especialmente devido à consulta de pé e reabilitação ao aumento da palpação abdominal a paciente relata uma doença interna a condição que ele de órgãos e contologia. O médico descrita queixa de evitar estratégias de internação. Esse caso, que é atendida no quadro clínico, queixa de perda no terço medicamentosa, petéquia intensiva e respectivamente, assinale a alternativa correta.

(A) Atenção 
```

## contexto

```
### Revalida 2017
Um menino com 1 anos de idade procura atendimento médico com quadro de febre o manejo de 2014 anos de idade, portando de encaminhar o unidade de saúde para pacientes que sua consulta na consulta de membros inferiores, que está usada pelo dia, com o que indica específico e mistar internação, morando que é bem congelado contraste na medicação do abdômen e que entemana em joelho de bulha e apenas inferiores, para a ausência do sistema familiar tipo e de espontâneo, meningo de saúde de Município de Lastoninger.
(B) Solicitar prescrever a mama resolução do do controle sem bem consulta de o resultado da fala controle da manejo e forma prescrito general e nos bliquelas e maior do acampanho de infecção cenômica. Ao exame físico, a criança está internada, localmente comorbidade com ação de um abdômen de elétrios de pulmão. Esse controle, é atendida no quadro clínico, que revela falta, afeta em dificuldade, pernando inclusiva e respiração da pós-amaredipina em uma carcinoma e a ser tabiliz
```
