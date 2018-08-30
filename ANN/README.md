# Utilizando uma ANN para saber se os clientes sairão do banco ou não

Um banco da Europa deseja saber se os clientes vão fechar as suas contas ou não. Utilizaremos o dataset disponibilizado para tentar realizar essa predição com as técnicas de DeepLearning

## Getting Started

O importante é saber se a pessoa saiu ou não do banco após um determinado tempo (Coluna "Exited");

Esse conhecimento pode ser usado para detecção de fraude, pagamento, diversas situações.
Basicamente quando se tem uma serie de características e uma saída binaria.

Usaremos algumas bibliotecas: Theano, Tensorflow e Keras.

Theano: Biblioteca Opensource para calculos matematicos, roda com base em numpy. Pode ser usada para rodas os calculos na GPU tambem. 

Tensorflow: Biblioteca opensource para calculos numéricos. Originalmente feita pelo grupo de pesquisas do google.   

Keras: Feita para deep learning. Permite criar ANN poderosas com poucas linhas. Criada com base em Theano e Tensorflow.

Problema: Classificação

## Parte 1
## Processamento de dados

Primeiro importamos as bibliotecas
Depois importamos o dataset.
Nesse momento devemos analisar quais colunas poderão influenciar a decisão do cliente sair do banco, ou não.

As colunas: RowNumber (Numero da linha), CostumerID(Identificação do Cliente)e Surname (Sobrenome) não influenciam a permanencia do cliente no banco.

Já, as colunas: CreditScore (nota de credito), Geography (Geografia, país), Gender (Sexo), Age (idade), Tenure (Consultar), Balance (Valor na conta), NumOfProducts(Numero de produtos), hasCreditCard(Possui cartao de credito), IsActiveMember (é um cliente ativo) e  EstimatedSalry (Salario estimado) possuem influencia na permanencia do cliente no banco ou não.

O indice das colunas importantes é utilizado para criação da variavél X, que tambem chamaremos de variavies independentes

Ja o nosso resultado será descrito como Y, que corresponde a 13a coluna do dataset.

O proximo passo seria dividir o dataset entre treino e teste, porém, possuimos algumas variaveis categoricas. Por isso precisamos transforma-las antes de realizarmos a divisao.

Lideremos com os dados: País (Geography) e Sexo (Gender).

Fazemos isso através do labelencoder.

Porém, não existe ligação direta entre a numeração e a importancia do dado, o 2 não é mais importante que o 0, por exemplo. Para isso, nós iremos utilizar o OneHotEncoder. 

Agora estamos prontos pra dividir o dataset em treino e teste.

Posteriormente escalamos todas as variaveis, dessa forma elas ficarão todas com "tamanhos proximos", facilitando as operações que serão realizadas.

## Parte 2
## Rede Neural Artificial (ANN)

Agora começamos a montar a nossa Rede Neural Artificial
Primeiramente importamos as bibliotecas e módulos que iremos utilizar

Inicializando a rede neural:
- Devemos criar um classificador, ja que a tarefa é dizer se um conjunto de entradas vai ser classificado como 0 ou 1 (se o cliente sairá do banco ou não).
- Passamos para a parte das camadas da nossa rede neural. A primeira camada é de entrada de dados. Devemos ter o numero correspondente as variáveis independentes, que, nesse caso, são 11.
- Adicionaremos mais uma camada escondida.
- Agora criaremos a camada de saída, que deverá utilizar a ativação do tipo Sigmoid, dessa forma poderemos ter acesso a probabilidade do cliente sair ou não do banco.


A nossa rede ja esta pronta. Precisamos compila-la, ou seja, selecionar os parametros para que os pesos sejam calculados.

O proximo passo é fazer o fit, ou seja, aplicar a matriz ao algoritmo que criamos. Nesse momento a rede irá aprender com os dados e parametros que informamos.
Foram inserido valores para o batch_size e numero de epochs aleatórios, uma vez que queremos mostrar apenas o funcionamento da rede. Esses valores podem ser melhor adaptados para cada situação especifica.

## Parte 3
## Utilizando a ANN para fazer predições

Utilizando o modelo para fazer predições.

Podemos realizar a predição da mesma forma com que realizamos em qualquer algoritmo de machine learning.
Nesse momento vemos as probabilidades do cliente sair do banco.

Ao transformar os dados para binários, usando um valor de 50%, podemos calcular a acuracidade do modelo com os dados que ele não conhecia.
Após esse teste o valor alcançado foi de 84% de acerto.



