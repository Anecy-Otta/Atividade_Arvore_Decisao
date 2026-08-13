# Relatório de Análise: Árvores de Decisão e Comportamento de Probabilidades em Dados Clínicos

**Disciplina:** Aprendizado de Máquina\
**Dataset utilizado:** Heart Disease --- UCI Machine Learning Repository
(ID 45)\
**Ferramentas:** Python 3, Pandas, Scikit-Learn, Matplotlib, Google
Colab e Git

------------------------------------------------------------------------

## 1. Introdução

Este relatório documenta o estudo prático sobre o comportamento do
algoritmo de classificação **Árvore de Decisão (CART)** aplicado ao
conjunto de dados clínicos *Heart Disease*, disponibilizado pelo UCI
Machine Learning Repository.

O objetivo da atividade foi:

1.  identificar a existência de **colisões de registros**, isto é,
    pacientes com atributos preditores idênticos, mas com diagnósticos
    de saída diferentes;
2.  treinar uma árvore de decisão preliminar **sem limitação de
    profundidade** (`max_depth=None`) e analisar o comportamento das
    folhas puras;
3.  aplicar uma limitação de profundidade máxima (`max_depth=3`) e
    comparar o comportamento das probabilidades estimadas nas folhas com
    o modelo sem limitação.

------------------------------------------------------------------------

## 2. Extração e Pré-processamento dos Dados

O conjunto de dados foi obtido diretamente da API do UCI Machine
Learning Repository por meio da biblioteca `ucimlrepo`, utilizando o
identificador **45**.

O dataset original possui **303 registros** e **13 variáveis
preditoras**. Entre as variáveis estão informações clínicas como idade,
sexo, tipo de dor no peito, pressão arterial em repouso e colesterol.

### 2.1 Variável alvo

A variável original `num`, que possui valores de 0 a 4, foi transformada
em uma variável binária:

-   `0` → Sem doença cardíaca;
-   `1` → Com doença cardíaca, correspondendo aos valores originais de
    `num` maiores que zero.

### 2.2 Tratamento de valores ausentes

Foram identificados valores ausentes nas variáveis:

-   `ca`: 4 registros;
-   `thal`: 2 registros.

Esses registros foram removidos antes do treinamento dos modelos.

Assim:

-   **Registros originais:** 303;
-   **Registros removidos:** 6;
-   **Registros utilizados na análise e treinamento:** 297.

------------------------------------------------------------------------

## 3. Análise de Colisões de Registros

Uma colisão ocorre quando dois ou mais registros apresentam **exatamente
os mesmos valores em todas as variáveis preditoras**, mas possuem
valores diferentes para a variável alvo.

A análise foi realizada agrupando os registros pelas variáveis
preditoras e verificando se algum grupo apresentava mais de uma classe
de diagnóstico.

### Resultado

**Não foram encontradas colisões perfeitas nos atributos utilizados na
análise.**

O resultado obtido foi:

> Total de registros em colisão encontrados: **0**

Portanto, não houve, neste conjunto de dados, casos em que pacientes com
o mesmo vetor completo de atributos preditores apresentassem
diagnósticos de saída diferentes.

Esse resultado é relevante para a análise das árvores, pois não foi
identificada, neste experimento, uma restrição decorrente de registros
totalmente idênticos com classes conflitantes.

------------------------------------------------------------------------

## 4. Árvore de Decisão sem Limitação de Profundidade

Inicialmente foi treinado um modelo `DecisionTreeClassifier` sem
limitação explícita de profundidade:

``` python
DecisionTreeClassifier(max_depth=None, random_state=42)
```

O resultado obtido foi:

-   **Profundidade da árvore:** 11;
-   **Número total de folhas:** 57;
-   **Folhas puras:** 57;
-   **Folhas impuras:** 0;
-   **Percentual de folhas puras:** 100%.

Todas as folhas apresentaram **Índice Gini igual a 0**, indicando que,
nos dados utilizados para treinamento, cada folha continha registros de
uma única classe.

### 4.1 Comportamento das probabilidades

As probabilidades estimadas pelo modelo sem limitação de profundidade
apresentaram apenas dois valores:

-   `P(y=1) = 0.000`;
-   `P(y=1) = 1.000`.

Esse comportamento mostra que o modelo produziu previsões extremamente
polarizadas nos dados de treinamento.

A combinação de **profundidade 11**, **57 folhas**, **100% de folhas
puras** e probabilidades exclusivamente iguais a 0 ou 1 é compatível com
um comportamento de forte ajuste aos dados de treinamento.

Entretanto, como o experimento não utilizou um conjunto separado de
teste, não é possível medir diretamente, neste relatório, a capacidade
de generalização do modelo para novos pacientes.

------------------------------------------------------------------------

## 5. Árvore de Decisão com `max_depth=3`

Na segunda etapa foi treinado um novo modelo utilizando uma limitação de
profundidade máxima:

``` python
DecisionTreeClassifier(max_depth=3, random_state=42)
```

A restrição reduziu significativamente a complexidade da árvore.

### Resultados

-   **Profundidade máxima:** 3;
-   **Número total de folhas:** 8.

Como uma árvore binária com profundidade máxima 3 pode possuir no máximo
`2³ = 8` folhas, o modelo atingiu exatamente esse limite.

### 5.1 Comportamento das probabilidades

Diferentemente do modelo sem limitação, a árvore com `max_depth=3`
apresentou diferentes probabilidades estimadas nas folhas:

``` text
0.083
0.241
0.296
0.619
0.667
0.765
0.850
0.971
```

Esses valores mostram que as folhas passaram a representar grupos
maiores de pacientes, nos quais a proporção entre as classes influencia
a probabilidade estimada.

Assim, as probabilidades deixaram de ficar restritas aos extremos 0 e 1
e passaram a apresentar valores intermediários, refletindo a composição
das classes em cada folha.

------------------------------------------------------------------------

## 6. Comparação entre os Modelos

  -----------------------------------------------------------------------
  Métrica / Aspecto           Árvore sem limite           Árvore limitada
                             (`max_depth=None`)           (`max_depth=3`)
  ------------------- ------------------------- -------------------------
  Profundidade                               11                         3

  Número de folhas                           57                         8

  Folhas puras                        57 (100%)   Não avaliadas como foco
                                                   principal da atividade

  Probabilidades                  0.000 e 1.000      0.083, 0.241, 0.296,
  únicas `P(y=1)`                                    0.619, 0.667, 0.765,
                                                             0.850, 0.971

  Complexidade da                          Alta                  Reduzida
  árvore                                        

  Comportamento das     Extremamente polarizado Mais variado, com valores
  probabilidades                                           intermediários
  -----------------------------------------------------------------------

A comparação evidencia que a limitação de profundidade modifica
significativamente o comportamento da árvore.

No modelo sem limite, a árvore cresceu até profundidade 11 e criou 57
folhas, todas puras nos dados de treinamento. Consequentemente, as
probabilidades estimadas ficaram restritas a 0 ou 1.

Com `max_depth=3`, a árvore passou a possuir apenas 8 folhas. Como cada
folha reúne um número maior de registros, as probabilidades estimadas
passaram a assumir valores intermediários, de acordo com a proporção das
classes presentes em cada região da árvore.

------------------------------------------------------------------------

## 7. Análise do Sobreajuste

O modelo sem limitação de profundidade apresentou características
compatíveis com **sobreajuste aos dados de treinamento**:

-   árvore com profundidade elevada (11);
-   grande quantidade de folhas (57);
-   100% das folhas puras;
-   probabilidades de classificação somente 0 ou 1.

Esse comportamento sugere que a árvore conseguiu separar muito
detalhadamente os registros utilizados no treinamento.

Já a limitação `max_depth=3` reduziu a complexidade estrutural da árvore
e produziu probabilidades menos extremas.

É importante destacar que **não foi realizada uma avaliação com conjunto
de teste ou validação cruzada** neste experimento. Portanto, a conclusão
sobre sobreajuste é baseada no comportamento estrutural e nas
probabilidades observadas no treinamento, e não em uma comparação de
desempenho entre treino e teste.

------------------------------------------------------------------------

## 8. Conclusão

A atividade permitiu observar, de forma prática, o efeito da
profundidade de uma árvore de decisão sobre sua estrutura e sobre as
probabilidades estimadas.

Primeiramente, a análise dos registros não encontrou **colisões
perfeitas**, ou seja, não foram identificados casos com todos os
atributos preditores idênticos e diagnósticos diferentes.

Na árvore sem limitação de profundidade, o modelo atingiu **profundidade
11 e 57 folhas**, sendo todas as folhas puras. As probabilidades
estimadas ficaram restritas a **0.000 ou 1.000**, demonstrando um
comportamento extremamente polarizado nos dados utilizados para
treinamento.

Ao limitar a profundidade para **3**, a árvore passou a possuir apenas
**8 folhas** e apresentou probabilidades variadas:

**0.083, 0.241, 0.296, 0.619, 0.667, 0.765, 0.850 e 0.971.**

Dessa forma, o experimento demonstra que a limitação da profundidade
reduz a complexidade da árvore e faz com que as folhas representem
grupos maiores de registros, produzindo probabilidades menos extremas.

Por fim, embora o comportamento da árvore sem limite seja compatível com
sobreajuste, uma avaliação definitiva da capacidade de generalização
exigiria uma etapa adicional com dados de teste ou validação cruzada,
que não fez parte deste experimento.

------------------------------------------------------------------------

## 9. Resumo dos Resultados

  -----------------------------------------------------------------------
  Resultado                                                         Valor
  ------------------------------ ----------------------------------------
  Registros originais                                                 303

  Registros após tratamento                                           297

  Colisões perfeitas                                                    0

  Profundidade                                                         11
  (`max_depth=None`)             

  Folhas (`max_depth=None`)                                            57

  Folhas puras                                                  57 (100%)
  (`max_depth=None`)             

  Probabilidades                                            0.000 e 1.000
  (`max_depth=None`)             

  Profundidade (`max_depth=3`)                                          3

  Folhas (`max_depth=3`)                                                8

  Probabilidades (`max_depth=3`)       0.083, 0.241, 0.296, 0.619, 0.667,
                                                      0.765, 0.850, 0.971
  -----------------------------------------------------------------------
