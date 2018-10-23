<span style="text-decoration:underline;">**CLASSIFICATION MODELS**<span>

GILA can create two different models:

### Auto models:

Automatically generates a convolutional network that fits the image sizes and the number of classes

~~~shell
-models auto
~~~

> Note: by default automodel uses 3x3 convolutions with stride=1 and padding. After each convolution, automodel stacks a batch normalization layer and a ReLu activation layer. Convolutional layers are stacked as many as defined by the -autonconv argument to form a convolutional block. Finally, after each convolutional block a 2x2 max-pooling with stride 2 is applied.

Some modifications can be done to the auto model.

To define the number of fully connected layers after the last convolution and before the softmax:

~~~shell
-autodlayers 2
~~~

And the size of this fully connected layers:

~~~shell
-autodsize 1024
~~~

Also we can define the number of filters, by default GILA doubles the number of filters after each maxpool. GILA starts with KINI number of filters up to KEND number of maximum filters:

~~~shell
-autokini 32 -autokend 1024
~~~

default values KINI=16 KEND=512

We can define as well the number of consecutive convolutions before each maxpooling:

~~~shell
-autonconv 1
~~~

default=2


We can add **residual** connections:
~~~shell
-autores
~~~

default=no

> Note: the first block of convolutions has not residual connections and it ends with a (2x2) maxpooling. After that all the convolutional blocks will have residual connections and the last convolution will have stride=2 to reduce the maps sizes. Moreover the last ReLu of each convolutional block is applied after the Add layer.


***


### Pre-trained models:

Load pretrained models and use GlobalAvgPooling to adapt the pretrained models to the size of the FC layers and numbers of classes

~~~shell
-models {vgg16,vgg19,resnet50,inceptionv3,inceptionresnetv2,densenet121,densenet169,densenet201,mobilenet,mobilenetv2}
~~~

Pretrained models can be frozen a number of epochs using the parameter:

~~~shell
-fepochs 5
~~~~

And we can define the last layer of the pretrained model that is goig to be frozen, for instance

~~~shell
-model resnet50 -flayer add_10
~~~

Moreover, similar to the auto model, we can define the number of fully connected layers after the last convolution and before the softmax:

~~~shell
-predlayers 2
~~~

And the size of this fully connected layers:

~~~shell
-predsize 1024
~~~

***

### For all the models

Model summary can be print out:

~~~shell
-summary
~~~

default=no

also a graphical version of the model is obtained in "model.png"


***

And models can be loaded or saved using json and h5 formats

~~~shell
-save_model my_model
~~~

will save the trained model in two files:
~~~shell
- my_model.json
- my_model.h5
~~~

Therefore, for loading
~~~shell
-load_model my_model
~~~

expect to find both files, json and h5
