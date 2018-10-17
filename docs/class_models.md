<span style="text-decoration:underline;">**CLASSIFICATION MODELS**<span>

GILA can create two different models:

### Auto models:

Automatically generates a convolutional network that fits the image sizes and the number of classes

~~~shell
-models auto
~~~

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

Model summary can be print out:

~~~shell
-summary
~~~

default=no

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
