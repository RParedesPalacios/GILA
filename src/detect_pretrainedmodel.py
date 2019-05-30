import keras
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import layers
from pretrainedmodel import load_pretrained_model
import numpy as np
from automodel import *
import sys

def detect_pretrained_model(args,anchors,catlen):

    load_model=load_pretrained_model(args)

    if (args.summary==True):
        load_model.summary()

    if (args.olayer==""):
        print("When using a pretrained network you have to specify the output layer to attach the detection network: -olayer")
        sys.exit(0)


    maps=[]
    print(args.olayer)
    model_layers=[]
    for l in load_model.layers:
        model_layers.append(l.name)

    print(model_layers)

    for l in args.olayer:
        if (l in model_layers):
            m=load_model.get_layer(l)
            h=m.output.get_shape().as_list()[1]
            w=m.output.get_shape().as_list()[2]
            print("Connecting",m.name,":",h,"x",w)
            maps.append(m.output)
            x=m.output
        else:
            print(l," not found in pretrained model")
            sys.exit(0)

            #break

    print(maps)


    s=min(h,w)
    DEPTH=int(math.log2(s))
    DEPTH=DEPTH+1
    KINI=args.autokini
    KEND=args.autokend

    print("Additional Depth=",DEPTH)

    numf=int(KINI/2)
    tlist=[]
    for i in range(DEPTH):
        numf=numf*2
        if (numf>KEND):
            numf=KEND

        if (i==0):
            x=layers.MaxPooling2D(pool_size=(2, 2))(x)
        else:
            x=basic_block(x,numf,args,residual=False,tlist=tlist)
            maps.append(tlist[-1])


    print(maps)


    depth=anchors*catlen

    outs=[]
    outm=[]
    for m in maps:
        x=layers.Conv2D(depth, kernel_size=(3, 3), strides=(1,1),padding='same',activation='sigmoid')(m)
        outm.append(x)
        x=layers.Reshape((-1,catlen))(x)
        x=layers.Softmax(axis=2)(x)
        outs.append(x)

    output = layers.concatenate(outs,axis=1)
    model=Model(inputs=load_model.input, outputs=output)


    return load_model,model,outm
