import math
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization as BN
from keras.layers import GaussianNoise as GN
from automodel import basic_block,FCN



def auto_det_model(args,cat):

    model=FCN(args)

    model.add(Flatten())
    for i in range(args.autodlayers):
        model.add(Dense(args.autodsize))
        model.add(BN())
        model.add(Activation('relu'))

    ## Detection output layer
    model.add(Dense(cat))
    model.add(Activation('softmax'))

    return model
