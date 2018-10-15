<span style="text-decoration:underline;">**CLASSIFICATION EXAMPLES**<span>


<span style="color:blue">### CIFAR 10<span>

Download the images from: [CIFAR10](https://www.dropbox.com/s/nn2bfybbfj3ly9b/CIFAR10.tgz)

you will find two directories (train, test) and two files (tr.txt, ts.txt)

We can specify the data sources in both ways:

* Directories

~~~shell
python3 ~/GITHUB/GILA/src/gila.py -trdir train -tsdir test -width 32 -height 32 -mode class -model auto
~~~

* Files

~~~shell
python3 gila.py -trfile tr.txt -tsfile ts.txt -width 32 -height 32 -mode class -model auto
~~~

In both cases GILA will use the training data to train the deep model and the test data as unseen data for error estimation.

* Add some data augmentation:

~~~shell
python3 gila.py -trfile tr.txt -tsfile ts.txt -width 32 -height 32 -mode class -model auto -da_width 20 -da_height 20 -da_flip_h yes
~~~

* Use a pre-trained model:

A resnet50:

~~~shell
python3 gila.py -trfile tr.txt -tsfile ts.txt -width 32 -height 32 -mode class -model resnet50 -da_width 20 -da_height 20 -da_flip_h yes
~~~

Or **inceptionv3**, in this case the images are too small so we increase the working size up to 128x128:

~~~shell
python3 ~/GITHUB/GILA/src/gila.py -trfile tr.txt -tsfile ts.txt -width 128 -height 128 -mode class -model inceptionv3 -da_width 20 -da_height 20 -da_flip_h yes
~~~
