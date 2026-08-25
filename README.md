# Classificação de Dados Cardíacos com Árvore de Decisão

Projeto desenvolvido em Python utilizando **Google Colab**, com o objetivo de trabalhar com um conjunto de dados clínicos e aplicar um modelo de **Árvore de Decisão (Decision Tree)** para classificação.

## 📌 Sobre o projeto

O projeto utiliza o arquivo `processed.cleveland.data`, carregado a partir do Google Drive, e realiza etapas de preparação dos dados, criação das variáveis de entrada e treinamento de um modelo de classificação.

O conjunto de dados utilizado possui **302 registros e 14 colunas**, incluindo características clínicas relacionadas a pacientes.

Entre as variáveis utilizadas estão:

* `age` — idade
* `sex` — sexo
* `cp` — tipo de dor no peito
* `trestbps` — pressão arterial em repouso
* `chol` — colesterol sérico
* `fbs` — glicemia em jejum
* `restecg` — eletrocardiograma de repouso
* `thalach` — frequência cardíaca máxima atingida
* `exang` — angina induzida pelo exercício
* `oldpeak` — depressão do segmento ST
* `slope` — inclinação do segmento ST
* `ca` — número de grandes vasos evidenciados por fluoroscopia
* `thal` — resultado relacionado ao exame de tálio
* `target` — variável de classificação

## 🧠 Tecnologias utilizadas

* Python
* Google Colab
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

## 🔄 Etapas realizadas

### 1. Conexão com o Google Drive

O notebook utiliza o Google Drive para acessar o arquivo de dados:

`processed.cleveland.data`

A conexão é realizada utilizando o recurso `drive.mount()` do Google Colab.

### 2. Carregamento e preparação dos dados

Os dados são carregados utilizando o Pandas e as colunas originalmente identificadas por valores numéricos são renomeadas para nomes mais descritivos.

Depois, são definidas:

* `X`: variáveis utilizadas como características de entrada;
* `y`: variável `target`, utilizada como variável de classificação.

### 3. Tratamento dos valores ausentes

O conjunto apresenta valores representados por `?`.

O projeto realiza o seguinte tratamento:

1. substituição de `?` por `NaN`;
2. conversão das variáveis para valores numéricos;
3. preenchimento dos valores ausentes utilizando a média da respectiva coluna.

Esse tratamento é realizado antes do treinamento da árvore de decisão.

### 4. Criação da Árvore de Decisão

O modelo utilizado é:

`DecisionTreeClassifier`

com `random_state=42`.

Inicialmente, o modelo é treinado sem estabelecer um limite de profundidade.

Posteriormente, é criada uma segunda versão utilizando:

`max_depth=3`

Essa configuração limita a profundidade máxima da árvore e permite visualizar uma estrutura mais simples do modelo.

### 5. Visualização da árvore

A árvore de decisão é visualizada utilizando `matplotlib` e a função `tree.plot_tree()` do Scikit-learn.

A visualização utiliza os nomes das características e as classes do `target`, permitindo observar a estrutura de decisões criada pelo modelo.

### 6. Predição

O notebook também cria um conjunto de valores de entrada utilizando as mesmas características utilizadas no treinamento e realiza uma previsão com:

`model.predict()`

No exemplo apresentado no notebook, o modelo retorna a classe `2`.

## 📊 Modelo utilizado

O algoritmo principal deste projeto é uma **Árvore de Decisão para classificação**.

A árvore realiza divisões sucessivas dos dados com base nas características disponíveis, formando uma estrutura de decisões que resulta em uma classe de saída.

O projeto permite observar a diferença entre uma árvore sem limitação de profundidade e uma árvore limitada a `max_depth=3`.

## ▶️ Como executar

1. Abra o arquivo `.ipynb` no Google Colab.
2. Disponibilize o arquivo `processed.cleveland.data` no Google Drive.
3. Execute as células do notebook na ordem.
4. Verifique o carregamento e a preparação dos dados.
5. Execute o treinamento da Árvore de Decisão.
6. Visualize a árvore gerada.
7. Execute a etapa de predição.

## 📁 Arquivo principal

* `Cardiaco_colab_3_nó.ipynb` — notebook contendo o código de carregamento, preparação dos dados, treinamento, visualização e predição.

## ⚠️ Observação

Este projeto tem finalidade **acadêmica e educacional**. Os resultados do modelo não devem ser utilizados como diagnóstico ou decisão médica.

## 👩‍💻 Projeto acadêmico

Projeto desenvolvido como atividade prática de **Ciência de Dados / Aprendizado de Máquina**, utilizando dados clínicos e o algoritmo de Árvore de Decisão.
