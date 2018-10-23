import numpy as np
import keras
import json
from pprint import pprint

from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.callbacks import LearningRateScheduler as LRS

from detect_automodel import auto_det_model
from detect_generators import *


##################################
### TRAIN DETECTION MODELS
##################################
def train_det_model(args):


    ######### ANNOT FILE
    print("Loading JSON annotations file",args.trannot)
    with open(args.trannot) as f:
        data = json.load(f)

    cat=data['categories']
    pprint(data['categories'])
    cat=len(cat)
    print ("Categories in annotation file:",cat)
    f.close()

    dataimg=data['images']
    trsize=len(dataimg)
    print ("Images in annotation file:",trsize)

    ######### ANCHORS
    print("Anchors:")
    for i in range(len(args.anchors)//2):
        print("[%f,%f]" %(args.anchors[2*i],args.anchors[2*i+1]))

    ######### MODELS
    model=auto_det_model(args,cat)


    list=[]
    for layer in model.layers:
        if "conv2d" in layer.name:
            list.append(layer.name)


    print(list)

    for i in range(args.autonconv-1,len(list),args.autonconv):
        print(list[i])

    ######### Connectig to output map
    print("Connecting to output map:")
    if (args.summary==True):
        model.summary()


    #### GENERAL
    batch_size=args.batch
    print("Batch size",batch_size)
    epochs=args.epochs
    print("Epochs",epochs)

    #### OPTIMIZER
    if (args.optim=="sgd"):
        print ("Using SGD")
        opt = SGD(lr=args.lr, decay=1e-6,momentum=0.9)
    elif (args.optim=="adam"):
        print ("Using Adam")
        opt=Adam(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    elif (args.optim=="rmsprop"):
        print ("Using RMSprop")
        opt=RMSprop(lr=args.lr, rho=0.9, epsilon=None, decay=0.0)



    tr_steps=trsize/batch_size

    ## REGULAR TRAINING
    if (args.lra==True):
        e1=int(epochs*0.5)
        e2=int(epochs*0.75)
        print ("Learning rate annealing epochs: 0-->",e1,"-->",e2,"-->",epochs)
        print ("Learning rate annealing LR:",args.lr,"-->",args.lr/10,"-->",args.lr/100)

        def scheduler(epoch):
            if epoch < e1:
                return args.lr
            elif epoch < e2:
                if (epoch==e1):
                    print ("===============================")
                    print ("New learning rate:",args.lr/10)
                    print ("===============================")
                return args.lr/10
            else:
                if (epoch==e2):
                    print ("===============================")
                    print ("New learning rate:",args.lr/100)
                    print ("===============================")
                return args.lr/100

        set_lr = LRS(scheduler)
        callbacks=[set_lr]
    else:
        print ("Learning rate=",args.lr,"no annealing")
        callbacks=[]

    ### detection loss
    # detloss=
    #model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])


    #history = model.fit_generator(detect_train_generator(args),
                            #steps_per_epoch=tr_steps,
                            #epochs=epochs,
                            #callbacks=callbacks,
                            #verbose=1)
