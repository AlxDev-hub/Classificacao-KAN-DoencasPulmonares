# Classificações de doenças pulmonares

Neste repositório, a Rede de Kolmogorov-Arnold (KAN) será utilizada para desenvolver um classificador de doenças pulmonares capaz de distinguir entre o estado natural, pneumonia e tuberculose. Também foi utilizado o algoritmo CNN por motivos de comparação.

## Etapa: Defesa da Qualificação
Nesta etapa inicial, resultados parciais são esperados para validar as técnicas utilizadas. Portanto, um banco de dados contendo 300 imagens foi criado e publicado no [Kaggle](https://www.kaggle.com/datasets/alexsanderlindolfo/raio-x-de-doenas-pulmonares-qualificao). Utilizando este banco de dados, quatro algoritmos foram desenvolvidos, sendo eles: **CNN** (teste1_cnn.ipynb), [**KAN**](https://github.com/KindXiaoming/pykan) (teste1_kan.ipynb), [**Efficient KAN**](https://github.com/Blealtan/efficient-kan) (teste1_efficientKAN.ipynb) e [**Fast KAN**](https://github.com/ZiyaoLi/fast-kan) (teste1_fastKAN.ipynb).

## Etapa: Defesa Final
Nesta etapa, três tipos de experimentos foram desenvolvidos, sendo o primeiro referente aos testes iniciais utilizando um conjunto amostral contendo 1000 imagens, onde o método *Holdout* foi utilizado para dividir os dados entre subconjuntos de treino (70%, ou seja, 700 imagens) e teste (30%, ou seja, 300 imagens). Os arquivos que fazem parte destes experimentos começam com "teste2_".

No segundo, foi utilizando o banco de dados completo contendo 13411 imagens, onde o método *Holdout* foi utilizado para dividir os dados em subconjuntos de treino (70%, ou seja, 9387 imagens) e teste (30%, ou seja, 4024 imagens). Os arquivos que fazem parte destes experimentos começam com "teste4_".

No terceiro, foi utilizado o banco de dados completo contendo 13411 imagens, onde o método *KFold* foi utilizado para dividir os dados com k = 3. Os arquivos que fazem parte destes experimentos são os arquivos no formato ".py".

O banco de dados completo pode ser encontrado no [Kaggle](https://www.kaggle.com/datasets/alexsanderlindolfo/raio-x-de-doenas-pulmonares-completo).
