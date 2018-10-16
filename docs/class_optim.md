<span style="text-decoration:underline;">**CLASSIFICATION OPTIMIZATION**<span>

GILA provides arguments to select several optimizations options, some are basically the most commonly used in Keras.

### General Variables

Number of epochs

~~~shell
python3 gila.py -epochs 50
~~~

default=100

and batch size

~~~shell
python3 gila.py -batch 128
~~~

default=100

### Data Balance

In some cases the dataset is unbalanced. There are different strategies to balanced the dataset to escape from prior probability accuracy. To this end we can provide a generator that balance the batches.

~~~shell
python3 gilap.py -balance yes
~~~

default=no

### Optimizers

We can select different optimizers {sgd,adam,rmsprop}

~~~shell
python3 gilap.py -optim sgd
~~~

default=sgd

### Learning Rate

Learnig rate can be defined

~~~shell
python3 gilap.py -lr 0.01
~~~

default=0.1 (large lr assuming BatchNormalization is used)


### Learning Rate Annealing

We can perform a learning rate annealing.

~~~shell
python3 gilap.py -lra yes
~~~

default=no

In this case this is the LR scheduler:


|   Epoch	|  LR 	|
|---	|---	|
| 1-50%  	|  lr  	|   
| 50%-75%  	| lr/10  	|
| 75%-100%  	| lr/100  	|
| ---  	| ---  	|
