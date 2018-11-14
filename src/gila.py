
import sys
import argparse


###########################################
#########    ARGUMENTS
###########################################
ptrmodels=['auto','vgg16','vgg19','resnet50','inceptionv3','inceptionresnetv2','densenet121','densenet169','densenet201','mobilenet','mobilenetv2']
parser = argparse.ArgumentParser(description='General Image LAbelling using Depp Learning')

### SOURCES
parser.add_argument('-trfile', help='File with list of training images')
parser.add_argument('-tsfile',help='File with list of test images')
parser.add_argument('-trdir', help='Path of training images, class folders')
parser.add_argument('-tsdir', help='Path of test images, class folders')

# ### IMAGES
parser.add_argument('-chan', choices=['rgb', 'gray'], default='rgb',help='working image depth (rgb)')
parser.add_argument('-width', type=int, help='working image width (240)',default=240)
parser.add_argument('-height',type=int,help='working image height (240)',default=240)
parser.add_argument('-resize',choices=['resize', 'crop'],help='resize mode (resize)',default='resize')

# ### MODELS
parser.add_argument('-mode', choices=['class', 'detect', 'segment'], default='class',help='type of problem (class)')
parser.add_argument('-model',choices=ptrmodels,help='neural network model (auto)',default='auto')
parser.add_argument('-summary',action='store_true',help='print model summary (no)',default='no')
parser.add_argument('-predsize',type=int,help='size of the dense layers attached to pre-trained models (512)',default=512)
parser.add_argument('-predlayers',type=int,help='number of dense layers attached to pre-trained models (1)',default=1)
parser.add_argument('-flayer',help='freeze pre-trained model up to this layer')
parser.add_argument('-autodsize',type=int,help='size of the dense layers attached to the auto model (512)',default=512)
parser.add_argument('-autodlayers',type=int,help='number of dense layers attached to auto  model (1)',default=1)
parser.add_argument('-autokini',type=int,help='inital number of kernels for the auto model (16)',default=16)
parser.add_argument('-autokend',type=int,help='Final number of kernels for the auto model (512)',default=512)
parser.add_argument('-autonconv',type=int,help='Number of consecutive convolutions (2)',default=2)
parser.add_argument('-autocdwise',action='store_true',help='use depthwise convolutions',default='no')
parser.add_argument('-autores',action='store_true',help='auto model with residual connections',default='no')

# ## OPTIM
parser.add_argument('-optim', choices=['sgd', 'adam','rmsprop'], default='sgd',help='Optimizer (sgd)')
parser.add_argument('-lra_scale',type=float,help='Learning rate annealing scale factor (2.0)',default=2.0)
parser.add_argument('-lra', action='store_true', help='Learning Rate Annealing')
parser.add_argument('-lr',type=float,help='Learning rate (0.1)',default=0.1)
parser.add_argument('-flr',type=float,help='Learning rate for a pretrained model int the frozen phase (0.001)',default=0.001)
parser.add_argument('-epochs',type=int,help='Epochs (100)',default=100)
parser.add_argument('-fepochs',type=int,help='Freeze pretrained epochs (10)',default=10)
parser.add_argument('-batch',type=int,help='Batch size (100)',default=100)
parser.add_argument('-balance', action='store_true',help='Class balance')

# ## DA
parser.add_argument('-da_width', type=int, help='DA width shift %% (0)',default=0)
parser.add_argument('-da_height', type=int, help='DA height shift %% (0)',default=0)
parser.add_argument('-da_rotation', type=int, help='DA rotation angle (0)',default=0)
parser.add_argument('-da_zoom', type=float, help='DA zoom rang [1-zoom,1+zoom] (0.0)',default=0.0)
parser.add_argument('-da_gauss', type=float, help='DA gaussian noise (0.0)',default=0.0)
parser.add_argument('-da_rescale', type=float, help='DA scale of values input map (255.0)',default=255.0)
parser.add_argument('-da_flip_v', action='store_true',help='DA vertical flip')
parser.add_argument('-da_flip_h', action='store_true',help='DA horizontal flip')

# ## IO MODELS
parser.add_argument('-load_model',  help='Load a model from file')
parser.add_argument('-save_model',  help='Save  model to file')
parser.add_argument('-save_epochs', action='store_true',help='save model after all epochs')

# ## other
parser.add_argument('-plot', action='store_true',help='plot the accuracy and create a gila.png')
parser.add_argument('-history', action='store_true',help='create a gila.txt with accuracy evolution')
parser.add_argument('-log', action='store_true',help='create a gila_log.txt')

########## DETECTION ###########################
## Mode json annot file
parser.add_argument('-trannot', help='File with list of training images and annotations')
parser.add_argument('-tsannot', help='File with list of test images and annotations')
parser.add_argument('-fprefix', help='File name prefix, useful to match json image names with file names, e.g COCO json annot (0000000)',default="0000000")
parser.add_argument('-nmaps',  help='Numer of maps connecting to loss in detection mode (3)',default="3")
## Mode directory with backgrounds and objects
parser.add_argument('-trbackgounds', help='Directory with backgourd images')
parser.add_argument('-trobjects', help='Directory with object images, one sub-directory per object category')
## anchors
parser.add_argument('-anchors',type=float,nargs="+",help='Define the anchors geometry (0.75 0.75 1 1 1.5 1 1 1.5)',default=[0.75,0.75,1.5,1.5,2,2,1.5,1,1,1.5,1,2,2,1])
## maps
parser.add_argument('-minmap', type=int, help='detection minimum map size (2)',default=2)
parser.add_argument('-maxmap', type=int, help='detection maximum map size (2)',default=16)

args = parser.parse_args()

import numpy as np
from PIL import Image
from loaders import *
from deepmodels import *


print ("==================================")
if (args.chan=="rgb"):
    print ("setting depth to RGB")
    CHAN=3
else:
    print ("setting depth to Gray")
    CHAN=1

COLS=args.width
ROWS=args.height

print ("Setting size to ",ROWS,"x",COLS)

MODE=args.mode
if (MODE=="class"):
    print ("Setting mode to CLASIFICATION")
elif (MODE=="detect"):
    print ("Setting mode to DETECTION")
else:
    print ("Setting mode to SEGMENTATION")

RES=args.resize
if (RES=="resize"):
    print ("Setting resize mode")
else:
    print ("Setting crop mode")
print ("==================================")



############################
#### CLASSIFICATION MODE
############################
if (MODE=="class"):
    ### TRAIN
    if (args.trfile!=None)|(args.trdir!=None):
        train_class_model(args)
    ## EVAL
    elif (args.tsfile!=None)|(args.tsdir!=None):
        eval_class_model(args)
    else:
        print("Nothing to do, bye!")

############################
#### DETECTION MODE
############################
elif (MODE=="detect"):
    if (args.trannot!=None):
        from detect_deepmodels import *
        train_det_model(args)
    elif (args.tsannot!=None)and(args.tsdir!=None):
        from detect_eval import *
        eval_detect_model(args)
    else:
        print("Nothing to do, bye!")

############################
#### SEGMENTATION MODE
############################
elif (MODE=="segment"):
    print ("Segment mode not yet implemented")
    sys.exit()

















###########
