import math
import keras
from keras import models
from keras import layers
from automodel import FCN
from keras.backend import slice


def auto_det_model(args,anchors,catlen):

    [input,x,tlist]=FCN(args)

    list=[]
    for layer in tlist:
        if "re_lu" in layer.name:
            list.append(layer)

    #print(list)

    print("Maps connected to target, from %d to %d" %(args.minmap,args.maxmap))
    maps=[]
    for i in range(args.autonconv-1,len(list),args.autonconv):
        if (min(list[i].shape[1],list[i].shape[2])>=args.minmap)and(max(list[i].shape[1],list[i].shape[2])<=args.maxmap):
            print(list[i].name,list[i].shape[1],"x",list[i].shape[2])
            maps.append(list[i])


    depth=anchors*catlen

    ks=3
    outs=[]
    for m in maps:
        x=layers.Conv2D(depth, kernel_size=(ks, ks), strides=(1,1),padding='same',activation='sigmoid')(m)
        outs.append(x)


    model = models.Model(inputs=[input], outputs=outs)

    return model
