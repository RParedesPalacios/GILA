<span style="text-decoration:underline;">**CLASSIFICATION EXAMPLES**<span>

## MNIST

Download MNIST images from the datasets right menu.

In this case images are grayscale and we use the channel argument to indicate it

~~~shell
-chan gray
~~~

try an auto model:

~~~shell
python3 gila.py -trdir training/ -tsdir testing/ -width 28 -height 28 -chan gray -mode class -da_width 10 -da_height 10 -da_rescale 255.0 -epochs 25 -plot -model auto -summary -lra -autodsize 128 -autonconv 1 -autokini 32
~~~

obtaining a 0,59% of test error rate.

![MNIST](https://raw.githubusercontent.com/RParedesPalacios/GILA/master/imgs/mnist.png)


***

## CIFAR 10

Download the images from the datasets right menu.

you will find two directories (train, test) and two files (tr.txt, ts.txt)

We can specify the data sources in both ways:

### Directories

~~~shell
python3 gila.py -trdir train -tsdir test -width 32 -height 32 -mode class -model auto
~~~

### Files

~~~shell
python3 gila.py -trfile tr.txt -tsfile ts.txt -width 32 -height 32 -mode class -model auto
~~~

In both cases GILA will use the training data to train the deep model and the test data as unseen data for test error estimation.

### Automodel with some data augmentation:

~~~shell
python3 ~/GITHUB/GILA/src/gila.py -trdir train -tsdir test -width 32 -height 32 -mode class -model auto -da_width 10 -da_height 10 -da_flip_h -lra -autokini 64 -plot -da_gauss 0.3 -summary -epochs 200 -autonconv 2
~~~

obtaining a 8,92% of test error rate.

![CIFAR10](https://raw.githubusercontent.com/RParedesPalacios/GILA/master/imgs/cifar10.png)



### Use a pre-trained model:

A **resnet50**:

~~~shell
python3 gila.py -trfile tr.txt -tsfile ts.txt -width 32 -height 32 -mode class -model resnet50 -da_width 20 -da_height 20 -da_flip_h
~~~

Or **inceptionv3**, in this case the images are too small so we increase the working size up to 128x128:

~~~shell
python3 gila.py -trfile tr.txt -tsfile ts.txt -width 128 -height 128 -mode class -model inceptionv3 -da_width 20 -da_height 20 -da_flip_h
~~~
