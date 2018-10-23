
## GILA

GILA is a general image toolkit that uses deep learning techniques to solve different problems. GILA uses Keras and TensorFlow as DL framework.

Initially GILA is developed to deal with the following problems:

* Image Classification
* Object Detection
* Image Segmentation

### Install GILA

From git:

~~~shell
git clone https://github.com/RParedesPalacios/GILA.git
~~~

### Pre-requisites

GILA requires the following python libs:
  * Numpy
    ~~~shell
    pip3 install mumpy
    ~~~

  * Keras
    ~~~shell
    pip3 install keras
    ~~~

  * Pillow
    ~~~shell
    pip3 install pillow
    ~~~

  * MatPlolib (only if you use the -plot option)
    ~~~shell
    pip3 install matplotlib
    ~~~

  * Pydot (only if you use the -summary option)
    ~~~shell
    pip3 install pydot
    ~~~


### Using GILA

GILA is a command line tool:

~~~shell
python3 gila.py
~~~

some help:

~~~shell
python3 gila.py -h
~~~
