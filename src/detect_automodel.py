import math
import keras
from keras import models
from keras import layers
from automodel import FCN
from keras.backend import slice


def auto_det_model(args):

    [input,x]=FCN(args)

    model = models.Model(inputs=[input], outputs=[x])

    return input,x,model


## FCN target
def add_detect_target(input,args,maps,cat,anchors):

    depth=anchors*cat

    outs=[]
    for m in maps:
        x=layers.Conv2D(depth, kernel_size=(3, 3), strides=(1,1),
        padding='same',activation='sigmoid')(m.output)
        outs.append(x)

    model = models.Model(inputs=[input], outputs=outs)

    return outs,model
