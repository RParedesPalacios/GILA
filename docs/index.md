
GILA is a general image toolkit that use deep learning techniques to solve different problems. GILA uses Keras and TensorFlow as DL framework.

Initially GILA is developed to deal with the following problems:

* Image Classification
* Object Detection
* Image Segmentation

#### Install GILA

From git:

~~~shell
git clone https://github.com/RParedesPalacios/GILA.git
~~~

#### Using GILA

As mentioned GILA is a command line tool:

~~~shell
python3 gila.py
~~~

some help:

~~~shell
python3 gila.py -h
~~~

~~~
General Image LAbelling using Depp Learning

optional arguments:
  -h, --help            show this help message and exit
  -trfile TRFILE        File with list of training images
  -tsfile TSFILE        File with list of test images
  -trdir TRDIR          Path of training images, class folders
  -tsdir TSDIR          Path of test images, class folders
  -chan {rgb,gray}      working image depth (rgb)
  -width WIDTH          working image width (240)
  -height HEIGHT        working image height (240)
  -resize {resize,crop}
                        resize mode (resize)
  -mode {class,detect,segment}
                        type of problem (class)
  -model {auto,vgg16,vgg19,resnet50,inceptionv3,inceptionresnetv2,densenet121,densenet169,densenet201,mobilenet,mobilenetv2}
                        neural network model (auto)
  -summary {yes,no}     print model summary (no)
  -predsize PREDSIZE    size of the dense layers attached to pre-trained
                        models (512)
  -predlayers PREDLAYERS
                        number of dense layers attached to pre-trained models
                        (1)
  -flayer FLAYER        freeze pre-trained model up to this layer
  -autodsize AUTODSIZE  size of the dense layers attached to the auto model
                        (512)
  -autodlayers AUTODLAYERS
                        number of dense layers attached to auto model (1)
  -autokini AUTOKINI    inital number of kernels for the auto model (16)
  -autokend AUTOKEND    Final number of kernels for the auto model (512)
  -optim {sgd,adam,rmsprop}
                        Optimizer (sgd)
  -lra {yes,no}         Learning Rate Annealing (yes)
  -lr LR                Learning rate (0.1)
  -epochs EPOCHS        Epochs (100)
  -fepochs FEPOCHS      Freeze pretrained epochs (10)
  -batch BATCH          Batch size (100)
  -balance {yes,no}     Class balance (no)
  -da_width DA_WIDTH    DA width shift % (0)
  -da_height DA_HEIGHT  DA height shift % (0)
  -da_rotation DA_ROTATION
                        DA rotation angle (0)
  -da_zoom DA_ZOOM      DA zoom rang [1-zoom,1+zoom] (0.0)
  -da_gauss DA_GAUSS    DA gaussian noise (0.0)
  -da_rescale DA_RESCALE
                        DA scale of values input map (255.0)
  -da_flip_v {yes,no}   DA vertical flip (no)
  -da_flip_h {yes,no}   DA horizontal flip (no)
  -load_model LOAD_MODEL
                        Load a model from file
  -save_model SAVE_MODEL
                        Save model to file
  -plot {yes,no}        Plot training (no)
risen:src rparedes$
~~~
