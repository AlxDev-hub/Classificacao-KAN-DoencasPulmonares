# Classificações de doenças pulmonares

Neste repositório, diversos classificadores de doenças pulmonares serão desenvolvidos utilizando as técnicas de Redes Neurais Artificiais, CNN e KAN.

## Primeiro versionamento

Teste realizado utilizando a técnica CNN (Otimizador=Adam, Função de perda=CEL, LR=0,001, Épocas=20) com um conjunto amostral contendo 100 imagens, sendo 50 para o treinamento (25-NORMAL e 25-PNEUMONIA) e 50 para teste (25-NORMAL e 25-PNEUMONIA).

## Segundo versionamento

No arquivo teste1_cnn.ipynb, foi feita uma modificação no parâmetro Épocas, indo de 20 para 500. Foi desenvolvido dois gráficos de performance, sendo o primeiro referente ao comportamento do Train loss e o segundo Test loss. Também foram adicionadas as Métricas de Avaliação, Confusion Matrix, Accuracy, Precision, Recall, F1 e AUC-ROC.

## Terceiro versionamento

No arquivo teste1_cnn.ipynb, grande parte da estrutura do código foi refatorada. Também foi realizado alterações no batch_size do trainloader (50 para 10) e do shuffle do testloader (True para False), isso na parte "Pré-processamento dos dados". Na parte de "Implementação da arquitetura CNN", foi o bloco de maior mudança, sendo, alteração da construção de uma classe para a utilização do nn.Sequential(), ficando mais simples o entendimento da arquitetura desenvolvida. Também foram alteradas as quantidades das camadas bem como seus parâmetros e disposição na estrutura. A função de custo, "CrossEntropyLoss" foi alterada para "BCELoss" para classificação binária. No versionamento alterior foi utilizado 500 épocas, mas para este, foi utilizado 20 épocas.

## Quarto versionamento

Foi adicionado o arquivo teste1_kan.ipynb para testes utilizando a técnica KAN com o mesmo conjunto amostral descrito no primeiro versionamento. No final do código foi constatado um erro de atualização do grid. Uma "possível" solução para este problema seria alterar a maneira que foi feito o pré-processamento das imagens, como por exemplo, aplicando flatten() nas imagens ou convertendo RGB para 256 níveis de cinza e depois aplicando o flatten().

## Quinto versionamento

Foram feitas modificação nos arquivos teste1_cnn.ipynb e teste1_kan.ipynb. No teste1_cnn.ipynb, o redimensionamento das imagens foi alterado para (512, 512), o batch_size do trainloader foi alterado para 50 e o shuffle do testloader foi alterado para False. Na construção do modelo CNN, foi alterado a quantidade de camadas convolucionais e ocultas, veja o arquivo teste1_cnn.ipynb para mais detalhes. Foi alterado o num_epoch para 50. No arquivo teste1_kan.ipynb, foi aplicado flatten() nas imagens durante o pré-processamento, ou seja, visto que imagens são dados bidimensionais (matrizes), cada imagem foi transformada em um vetor. Com está modificação, o erro mencionado no versionamento anterior foi resolvido.

## Sexto versionamento

No arquivo teste1_kan.ipynb, foi utilizado o train_test_split para, de maneira aleatória, modificar a quantidade e disposição dos dados de treinamento e teste.

## Setimo versionamento

A partir daqui foi adicionado mais uma classe no banco de dados e mais imagens, finalizando com 300 imagens, sendo 210 para o treinamento (70-NORMAL, 70-PNEUMONIA e 70-TUBERCULOSE) e 90 para teste (30-NORMAL, 30-PNEUMONIA e 30-TUBERCULOSE). Devido a isso, foram feitas algumas modificações no arquivo teste1_cnn.ipynb. O batch_size do treinamento foi alterado para 30 e do teste para 90. Na implementação da arquitetura CNN, foram adicionadas camadas de batch normalization a cada camada de convolução. Por fim, foi utilizado 30 épocas e função de perda Cross Entropy Loss para realizar o treinamento.

## Oitavo versionamento

Utilizando o banco de dados descrito no versionamento anterior, testes foram feitos no arquivo teste1_kan.ipynb. Quando o treinamento é feito em uma arquitetura com camadas ocultas (independente da quantidade de camadas e neurônios), o kernel do jupyter lab reinicia por volta da quarta ou quinta época. Até o momento deste commit ainda não foi descoberto o motivo deste problema. Também vale ser ressaltado que o valor de acurácia do treinamento já começa no valor máximo e não se altera no decorrer das épocas.

## Nono versionamento

Foi feito uma aplicação do algoritmo *efficient_kan*, que pode ser encontrado neste [link][https://github.com/Blealtan/efficient-kan]. Algo que vale ser mencionado é que a topologia da perda e acurácia do treinamento apresenta uma melhora significativa em relação ao teste1_kan.ipynb.
