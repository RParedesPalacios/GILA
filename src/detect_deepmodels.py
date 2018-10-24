import numpy as np
import keras
import json
from pprint import pprint

from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.callbacks import LearningRateScheduler as LRS

from detect_automodel import *
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
    anchors=len(args.anchors)//2
    for i in range(anchors):
        print("[%f,%f]" %(args.anchors[2*i],args.anchors[2*i+1]))

    ######### FCN MODEL
    input,x,model=auto_det_model(args)
    if (args.summary==True):
        model.summary()
    ######### Obtaining output maps and target size
    list=[]
    for layer in model.layers:
        if "re_lu" in layer.name:
            list.append(layer)

    print("Maps connected to target, from %d to %d" %(args.minmap,args.maxmap))
    maps=[]
    target_size=0
    for i in range(args.autonconv-1,len(list),args.autonconv):
        if (min(list[i].output.shape[1],list[i].output.shape[2])>=args.minmap)and(max(list[i].output.shape[1],list[i].output.shape[2])<=args.maxmap):
            print(list[i].name,list[i].output.shape[1],"x",list[i].output.shape[2])
            maps.append(list[i])
            target_size=target_size+(int(list[i].output.shape[1])*int(list[i].output.shape[2]))*(4+cat+1)

    print("Target size=",target_size)
    ######### Connect FCN model to target
    maps,model=add_detect_target(input,args,maps,cat,anchors)
    print(maps)
    # maps are target maps equal size that feat maps

    if (args.summary==True):
        from keras.utils import plot_model
        model.summary()
        plot_model(model, to_file='model.png')

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
    ### metrics

    #model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])


    #history = model.fit_generator(detect_train_generator(args),
                            #steps_per_epoch=tr_steps,
                            #epochs=epochs,
                            #callbacks=callbacks,
                            #verbose=1)
