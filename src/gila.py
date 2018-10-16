
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
parser.add_argument('-summary',choices=['yes','no'],help='print model summary (no)',default='no')
parser.add_argument('-predsize',type=int,help='size of the dense layers attached to pre-trained models (512)',default=512)
parser.add_argument('-predlayers',type=int,help='number of dense layers attached to pre-trained models (1)',default=1)
parser.add_argument('-flayer',help='freeze pre-trained model up to this layer')
parser.add_argument('-autodsize',type=int,help='size of the dense layers attached to the auto model (512)',default=512)
parser.add_argument('-autodlayers',type=int,help='number of dense layers attached to auto  model (1)',default=1)
parser.add_argument('-autokini',type=int,help='inital number of kernels for the auto model (16)',default=16)
parser.add_argument('-autokend',type=int,help='Final number of kernels for the auto model (512)',default=512)

# ## OPTIM
parser.add_argument('-optim', choices=['sgd', 'adam','rmsprop'], default='sgd',help='Optimizer (sgd)')
parser.add_argument('-lra', choices=['yes', 'no'], default='yes',help='Learning Rate Annealing (yes)')
parser.add_argument('-lr',type=float,help='Learning rate (0.1)',default=0.1)
parser.add_argument('-epochs',type=int,help='Epochs (100)',default=100)
parser.add_argument('-fepochs',type=int,help='Freeze pretrained epochs (10)',default=10)
parser.add_argument('-batch',type=int,help='Batch size (100)',default=100)
parser.add_argument('-balance', choices=['yes', 'no'], default='no',help='Class balance (no)')

# ## DA
parser.add_argument('-da_width', type=int, help='DA width shift %% (0)',default=0)
parser.add_argument('-da_height', type=int, help='DA height shift %% (0)',default=0)
parser.add_argument('-da_rotation', type=int, help='DA rotation angle (0)',default=0)
parser.add_argument('-da_zoom', type=float, help='DA zoom rang [1-zoom,1+zoom] (0.0)',default=0.0)
parser.add_argument('-da_gauss', type=float, help='DA gaussian noise (0.0)',default=0.0)
parser.add_argument('-da_rescale', type=float, help='DA scale of values input map (255.0)',default=255.0)
parser.add_argument('-da_flip_v', choices=['yes', 'no'], default='no',help='DA vertical flip (no)')
parser.add_argument('-da_flip_h', choices=['yes', 'no'], default='no',help='DA horizontal flip (no)')

# ## IO MODELS
parser.add_argument('-load_model',  help='Load a model from file')
parser.add_argument('-save_model',  help='Save  model to file')

# ## other
parser.add_argument('-plot', choices=['yes', 'no'], default='no',help='Plot training (no)')

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
    if (args.trfile!=None):
        TEST=0
        print("Loading training file:",args.trfile)
        [X,L]=load_list_file_class_to_numpy(args.trfile,ROWS,COLS,CHAN,RES)
        Xt=0
        Lt=0
        if (args.tsfile!=None):
            TEST=1
            print("Loading test file:",args.tsfile)
            [Xt,Lt]=load_list_file_class_to_numpy(args.tsfile,ROWS,COLS,CHAN,RES)

        train_class_model(X,L,TEST,Xt,Lt,args)
    elif (args.trdir!=None):
        print("Setting training dir to",args.trdir)
        X=0
        L=0
        TEST=0
        Xt=0
        Lt=0
        if (args.tsdir!=None):
            print("Setting test dir to",args.tsdir)
            TEST=1
        train_class_model(X,L,TEST,Xt,Lt,args)
    ## EVAL
    elif (args.load_model!=None):
        TEST=0
        Xt=0
        Lt=0
        if (args.tsfile!=None):
            if (MODE=="class"):
                print("Loading test file:",args.tsfile)
                [Xt,Lt]=load_list_file_class_to_numpy(args.tsfile,ROWS,COLS,CHAN,RES)
                eval_class_model(Xt,Lt,args)
        elif (args.tsdir!=None):
            print("Setting test dir to",args.tsdir)
            if (args.numts==None):
                print("-numts must be specified in Tets Dir mode")
                sys.exit()
            TEST=1
            eval_class_model(Xt,Lt,args)
        else:
            print("No test file to eval model")
            sys.exit()

############################
#### DETECTION MODE
############################
elif (MODE=="detect"):
    print ("Detect mode not yet implemented")
    sys.exit()

############################
#### SEGMENTATION MODE
############################
elif (MODE=="segment"):
    print ("Segment mode not yet implemented")
    sys.exit()
