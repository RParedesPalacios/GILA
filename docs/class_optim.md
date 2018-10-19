<span style="text-decoration:underline;">**CLASSIFICATION OPTIMIZATION**<span>

GILA provides arguments to select several optimizations options, some are basically the most commonly used in Keras.

### General Variables

Number of epochs

~~~shell
-epochs 50
~~~

default=100

and batch size

~~~shell
-batch 128
~~~

default=100

### Data Balance

In some cases the dataset is unbalanced. There are different strategies to balanced the dataset to escape from prior probability accuracy. To this end we can provide a generator that balance the batches.

~~~shell
-balance
~~~

default=no

### Optimizers

We can select different optimizers {sgd,adam,rmsprop}

~~~shell
-optim sgd
~~~

default=sgd

### Learning Rate

Learnig rate can be defined

~~~shell
-lr 0.01
~~~

default=0.1 (large lr assuming BatchNormalization is used)


### Learning Rate Annealing

We can perform a learning rate annealing.

~~~shell
-lra
~~~

default=no

This is the LR scheduler:

|   Epoch	|  LR 	|
|---	|---	|
| 1-50%  	|  lr  	|   
| 50%-75%  	| lr/lra_scale  	|
| 75%-100%  	| lr/lra_scale^2  	|
| ---  	| ---  	|


We can define the annealing scale factor:

~~~shell
-lra_scale 5.0
~~~

default=2.0


In this case this is the LR scheduler:

|   Epoch	|  LR 	|
|---	|---	|
| 1-50%  	|  lr  	|   
| 50%-75%  	| lr/5  	|
| 75%-100%  	| lr/25  	|
| ---  	| ---  	|
