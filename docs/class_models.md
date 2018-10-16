<span style="text-decoration:underline;">**CLASSIFICATION MODELS**<span>

GILA create two different models:

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
