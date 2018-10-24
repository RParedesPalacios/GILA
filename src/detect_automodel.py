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

def add_detect_target(input,args,maps):

    outs=[]
    for m in maps:
        size=int(m.output.shape[1])*int(m.output.shape[2])
        for mx in range(m.output.shape[1]):
            for my in range(m.
            output.shape[1]):
                s= layers.Lambda( lambda x: slice(x,(0,mx,my,0),(-1,1,1,-1)))(m.output)
                print(s.shape)
                x=layers.Dense(size,activation='linear')(s)
                outs.append(x)

    model = models.Model(inputs=[input], outputs=outs)

    return model
