import keras
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import layers
from pretrainedmodel import load_pretrained_model
import numpy as np


def detect_pretrained_model(args,anchors,catlen):

    load_model=load_pretrained_model(args)

    for l in load_model.layers:
        print(l.name)

    print(args.olayer)
    maps=[]
    for l in load_model.layers:
        if (l.name in args.olayer):
            print("layer connected to output:",l.name)
            maps.append(l.output)

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
