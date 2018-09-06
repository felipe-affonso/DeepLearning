# Usando uma CNN para detectar gatos e cachorros.

Este sera um simples projeto onde seguiremos os passos da ANN, preparando o dataset, criando a rede, analisando a rede e no final avaliando e melhorando a performance da mesma. Utilizaremos um dataset padrão para esse tipo de aplicação, porém o mesmo pode ser substituído para a tarefa do seu interesse.

## Getting Started

Cada pasta do dataset contem os dados de treino e os dados de teste. Por isso poderemos "pular" algumas partes.

### Pre-requisitos

Você vai precisar instalar as bibliotecas: keras, theano e tensorflow.

## Parte 1 
### Convolution layer

Uma Rede Neural Convolucional, nada mais é do que uma Rede Neural Artificial com algun processos a mais. A maioria deles envolve tranformar a imagem em uma matriz de informações. Essas informações representam os pixeis da imagem. Ou seja, cada elemento da matriz é um pixel de uma determinada imagem.

Isso é o que fazemos nessa primeira etapa. 

### Max Pooling

Realizamos a operação de Pooling para detectar as features da imagem. Ao final desse processo, o resultado é uma matriz reduzida, porém que mantém as cracterísticas da matriz original.

### Flattening

Nesse momento possuímos uma matriz com as características da imagem original, porém reduzida. A parte de Flattening é o processo em que transformamos essa matriz reduzida na "input layer" de uma ANN. Ou seja, realizamos os três processos (Convolution, Pooling e Flattening) para transformar os dados em entradas para uma ANN comum.

### Criando uma ANN

Agora ja possuímos a primeira camada de uma ANN classica. Iremos criar as hidden layers.
Inicialmente utilizaremos apenas uma hidden layer e uma camada de saída. Como visto no repositório na parte de [ANN](https://github.com/felipe-affonso/DeepLearning/tree/master/ANN).

### Compilando a CNN

Da mesma forma com que foi realizado na ANN, devemos compilar a CNN. Isso consiste em adicionar um otimizador, o modo como vamos calcular a função de perda, e, também as métricas que serão utilizadas.
Escolhemos utilizar o otimizador: adam, ele representa o algoritmo Stochastic Gradient Descent. A perda será o algoritmo binary cross entropy, e, por fim, utilizaremos a precisão da rede como métrica.

## Parte 2

Da mesma forma com que devemos pre-processar dados antes de os utilizarmos em uma ANN, também devemos realizar esse mesmo processo em imagens, dessa forma evitaremos overfitting.

Nesse caso, em específico, utilizaremos uma função que aplica diversas transformações nas imagens, dessa forma aumentamos o nosso dataset. 8000 imagens, como é o caso desse dataset, não representa um número suficiente para a CNN aprender corretamente. Por isso, utilizaremos essas transformações, como por exemplo rotacionar, aplicar um zoom e escalar as imagens para que o dataset possua um numero maior de dados.

## Conclusão

Após executarmos o código, obtivemos uma precisão de 85% nos dados de treino e 81% nos dados de teste.

É possível melhorar o modelo adicionando mais camas e alterando alguns parametros, porém, no momento conseguimos realizar o objetivo que era treinar uma CNN para diferenciar cachorros e gatos.

