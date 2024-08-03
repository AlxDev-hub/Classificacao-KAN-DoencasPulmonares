# Classificações de doenças pulmonares

Neste repositório, diversos classificadores de doenças pulmonares serão desenvolvidos utilizando as técnicas de Redes Neurais Artificiais, CNN e KAN.

## Primeiro versionamento

Teste realizado utilizando a técnica CNN (Otimizador=Adam, Função de perda=CEL, LR=0,001, Épocas=20) com um conjunto amostral contendo 100 imagens, sendo 50 para o treinamento (25-NORMAL e 25-PNEUMONIA) e 50 para teste (25-NORMAL e 25-PNEUMONIA).

## Segundo versionamento 

No arquivo teste1_cnn.ipynb, foi feita uma modificação no parâmetro Épocas, indo de 20 para 500. Foi desenvolvido dois gráficos de performance, sendo o primeiro referente ao comportamento do Train loss e o segundo Test loss. Também foram adicionadas as Métricas de Avaliação, Confusion Matrix, Accuracy, Precision, Recall, F1 e AUC-ROC. 
