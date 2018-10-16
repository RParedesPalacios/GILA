<span style="text-decoration:underline;">**CLASSIFICATION DATA AUGMENTATION**<span>


Data augmentation is a crucial strategy when dealing with image classification. GILA uses the most commonly used data augmentation parameters that Keras provides:

### Width shift

Shift range in the horizontal axis as a percentage of the image width:

~~~shell
python3 gila.py -da_width 10
~~~

default value=0

### Height shift

Shift range in the vertical axis as a percentage of the image height:

~~~shell
python3 gila.py -da_height 20
~~~

default value=0

### Rotation

Maximum rotation angle

~~~shell
python3 gila.py -da_rotaion 10
~~~

default value=0

### Zoom

Zoom multiplier range applied to the image [1-zoom,1+zoom]

~~~shell
python3 gila.py -da_zoom 0.1
~~~

default value=0.0

### Horizontal Flip

Flipt the image horizontally

~~~shell
python3 gila.py -da_flip_h yes
~~~

default value=no

### Vertical Flip

Flip the image vertically

~~~shell
python3 gila.py -da_flip_v yes
~~~

default value=no

***

Not only for images:

### Gaussian noise

Additive gaussian noise to all the activation layers

~~~shell
python3 gila.py -da_gauss 0.3
~~~

default value=0.0

### Rescale

A fixed scale of the input values (1/value). Typically value=255 for images

~~~shell
python3 gila.py -da_rescale 255
~~~

***

All these data augmentation can be combined:

~~~shell
python3 ~/GITHUB/GILA/src/gila.py -trdir train -tsdir test -width 32 -height 32 -mode class -model resnet50 -da_width 20 -da_height 20 -da_flip_h yes -da_zoom 0.2 -da_rescale 255.0
~~~
