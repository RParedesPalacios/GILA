import math
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization as BN
from keras.layers import GaussianNoise as GN


def basic_block(model,K,args,ishape=0):
    if (ishape!=0):
        model.add(Conv2D(K, (3, 3), padding='same',input_shape=ishape))
    else:
        model.add(Conv2D(K, (3, 3), padding='same'))

    model.add(BN())
    if (args.da_gauss!=0.0):
        model.add(GN(args.da_gauss))
    model.add(Activation('relu'))

    for i range(args.autonconv-1):
        model.add(Conv2D(K, (3, 3), padding='same'))
        model.add(BN())
        if (args.da_gauss!=0.0):
            model.add(GN(args.da_gauss))
        model.add(Activation('relu'))


    model.add(MaxPooling2D(pool_size=(2, 2)))
    return model


def auto_model(X,L,args,num_classes):

    if (args.trdir==None):
        h=X.shape[1]
        w=X.shape[2]
        num_classes=len(set(L))
        shape=X.shape[1:]
    else:
        h=args.height
        w=args.width
        if (args.chan=="rgb"):
            shape=(h,w,3)
        else:
            shape=(h,w,1)

    s=min(h,w)

    DEPTH=int(math.log2(s))
    KINI=args.autokini
    KEND=args.autokend

    print("Depth=",DEPTH)

    model = Sequential()
    numf=int(KINI/2)
    for i in range(DEPTH):
        numf=numf*2
        if (numf>KEND):
            numf=KEND

        if (i==0):
            model=basic_block(model,numf,args,shape)
        else:
            model=basic_block(model,numf,args)


    model.add(Flatten())
    for i in range(args.autodlayers):
        model.add(Dense(args.autodsize))
        model.add(BN())
        model.add(Activation('relu'))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model
