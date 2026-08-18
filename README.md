#  Análise de Árvores de Decisão em Dados Clínicos

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Jupyter-yellow.svg)

Trabalho prático desenvolvido para a disciplina de **Aprendizado de Máquina**, com foco na investigação de **colisões de registros**, **sobreajuste (overfitting)** e no **comportamento das probabilidades estimadas nas folhas** de Árvores de Decisão (CART).

---

##  Objetivos da Atividade

1. **Importação e Preparação:** Extrair o *Heart Disease Dataset* por meio da biblioteca `ucimlrepo`, utilizando o conjunto de dados disponibilizado pela Universidade da Califórnia em Irvine (o UCI Machine Learning Repository).

2. **Análise de Colisões:** Identificar se existem colisões de registros, ou seja, pacientes com atributos preditores idênticos, mas com diagnósticos de saída diferentes.

3. **Árvore sem Limitação de Profundidade (`max_depth=None`):** Treinar uma árvore de decisão sem limite de profundidade e analisar o comportamento e a proporção das folhas puras.

4. **Árvore com Profundidade Limitada (`max_depth=3`):** Limitar a profundidade da árvore e comparar o comportamento das probabilidades estimadas nas folhas com o modelo sem limitação.

5. **Documentação Técnica:** Registrar os resultados, análises e conclusões em relatório Markdown no repositório do projeto.

---

##  Principais Resultados

- **Registros originais:** 303
- **Registros utilizados após tratamento:** 297
- **Colisões perfeitas encontradas:** 0
- **Profundidade da árvore sem limite:** 11
- **Folhas da árvore sem limite:** 57
- **Folhas puras:** 57 (100%)
- **Folhas com `max_depth=3`:** 8
- **Probabilidades no modelo sem limite:** 0.000 e 1.000
- **Probabilidades no modelo com `max_depth=3`:** 0.083, 0.241, 0.296, 0.619, 0.667, 0.765, 0.850 e 0.971

Os resultados detalhados e a análise dos modelos estão disponíveis no relatório técnico.

---

##  Estrutura do Repositório

```text
.
├── Atividade_Arvore_Decisao.ipynb   # Notebook principal executado no Google Colab
├── Relatorio.md                     # Relatório técnico da atividade
└── README.md                        # Documentação do projeto
