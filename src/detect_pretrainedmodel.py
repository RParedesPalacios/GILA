import keras
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import layers
from pretrainedmodel import load_pretrained_model
import numpy as np
from automodel import *


def detect_pretrained_model(args,anchors,catlen):

    load_model=load_pretrained_model(args)

    for l in load_model.layers:
        print(l.name)
        if (args.olayer[0] in l.name):
            h=l.output.get_shape().as_list()[1]
            w=l.output.get_shape().as_list()[2]
            print(l.name,":",h,"x",w)
            x=l.output
            break


    s=min(h,w)
    DEPTH=int(math.log2(s))
    KINI=args.autokini
    KEND=args.autokend

    print("Additional Depth=",DEPTH)

    numf=int(KINI/2)
    tlist=[]
    for i in range(DEPTH):
        numf=numf*2
        if (numf>KEND):
            numf=KEND

        x=basic_block(x,numf,args,residual=False,tlist=tlist)


    depth=anchors*catlen

    outs=[]
    outm=[]
    for m in tlist:
        x=layers.Conv2D(depth, kernel_size=(3, 3), strides=(1,1),padding='same',activation='sigmoid')(m)
        outm.append(x)
        x=layers.Reshape((-1,catlen))(x)
        x=layers.Softmax(axis=2)(x)
        outs.append(x)

    output = layers.concatenate(outs,axis=1)
    model=Model(inputs=load_model.input, outputs=output)


    return load_model,model,outm
