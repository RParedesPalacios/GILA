import keras
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from pretrainedmodel import load_pretrained_model
import numpy as np


def detect_pretrained_model(args,anchors,catlen):

    load_model=load_pretrained_model(args)

    print("Maps connected to target, from %d to %d" %(args.minmap,args.maxmap))

    maps=[]
    for l in load_model:
        if (min(l.shape[1],l.shape[2])>=args.minmap)and(max(l.shape[1],l.shape[2])<=args.maxmap):
            print(l.name,l.shape[1],"x",l.shape[2])
            maps.append(l)

    depth=anchors*catlen

    outs=[]
    for m in maps:
        x=layers.Conv2D(depth, kernel_size=(3, 3), strides=(1,1),padding='same',activation='sigmoid')(m)
        outs.append(x)

    model=Model(inputs=load_model.input, outputs=outs)


    return load_model,model
