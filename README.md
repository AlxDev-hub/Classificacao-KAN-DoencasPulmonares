# Classificações de doenças pulmonares

Neste repositório, diversos classificadores de doenças pulmonares serão desenvolvidos utilizando as técnicas de Redes Neurais Artificiais, CNN e KAN.

## Primeiro versionamento

Teste realizado utilizando a técnica CNN (Otimizador=Adam, Função de perda=CEL, LR=0,001, Épocas=20) com um conjunto amostral contendo 100 imagens, sendo 50 para o treinamento (25-NORMAL e 25-PNEUMONIA) e 50 para teste (25-NORMAL e 25-PNEUMONIA).

## Segundo versionamento 

No arquivo teste1_cnn.ipynb, foi feita uma modificação no parâmetro Épocas, indo de 20 para 500. Foi desenvolvido dois gráficos de performance, sendo o primeiro referente ao comportamento do Train loss e o segundo Test loss. Também foram adicionadas as Métricas de Avaliação, Confusion Matrix, Accuracy, Precision, Recall, F1 e AUC-ROC. 

## Terceiro versionamento

No arquivo teste1_cnn.ipynb, grande parte da estrutura do código foi refatorada. Também foi realizado alterações no batch_size do trainloader (50 para 10) e do shuffle do testloader (True para False), isso na parte "Pré-processamento dos dados". Na "Implementação da arquitetura CNN" foi o bloco de maior mudança, sendo, alteração da construção de uma classe para a utilização do nn.Sequential(), ficando mais simples o entendimento da arquitetura desenvolvida. Também foram alteradas as quantidades das camadas bem como seus parâmetros e disposição na estrutura. A função de custo, "CrossEntropyLoss" foi alterada para "BCELoss" para classificação binária. No versionamento alterior foi utilizado 500 épocas, mas para este, foi utilizado 20 épocas.
