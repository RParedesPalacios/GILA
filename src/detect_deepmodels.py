import numpy as np
import keras

from pprint import pprint

from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.callbacks import LearningRateScheduler as LRS
from keras.callbacks import ModelCheckpoint

from detect_automodel import *
from detect_generators import *
from detect_loss import *
from detect_tools import *
from detect_eval import *
from detect_pretrainedmodel import *
from files import *

##################################
### TRAIN DETECTION MODELS
##################################
def train_det_model(args):


    ######### ANNOT FILE
    [images,imglen,boxes,boxlen,catdict,catlen]=load_annot_json(args.trannot)

    ######### ANCHORS
    print("Anchors:")
    anchors=len(args.anchors)//2
    for i in range(anchors):
        print("[%f,%f]" %(args.anchors[2*i],args.anchors[2*i+1]))

    ######### MODEL
    PRETR=False
    if (args.load_model!=None):
        model=load_from_disk(args.load_model,hnm_loss,dif_pos_neg,score_pos,score_neg)
        outm=model.outputs
    elif (args.model=="auto"):
        print("Automodel")
        model,maps=auto_det_model(args,anchors,catlen)
        outm=model.outputs
    else:
        PRETR=True
        [base,model]=detect_pretrained_model(args,anchors,catlen)
        outm=model.outputs


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

    if (args.trsteps!=-1):
        tr_steps=args.trsteps
    else:
        tr_steps=imglen/batch_size



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



    #eval_detect_model(args,model)
    if (args.save_epochs):
        filepath=args.save_model+"_{epoch:03d}.h5"
        checkpoint = ModelCheckpoint(filepath,verbose=1, save_best_only=False)
        callbacks.append(checkpoint)


    fepochs=args.fepochs
    if (PRETR)and(fepochs>0):
        print("Freezing  the pre-trained model %d epochs" %(fepochs))
        for layer in base.layers:
            layer.trainable = False

        model.compile(loss=[hnm_loss],optimizer=opt,metrics=[dif_pos_neg,score_pos,score_neg,"accuracy"])

        history = model.fit_generator(detect_train_generator(args,maps,outm),
                             max_queue_size=10, workers=0,use_multiprocessing=False,
                             steps_per_epoch=tr_steps,
                             epochs=fepochs,
                             callbacks=callbacks,
                             verbose=1)
        for layer in base.layers:
            layer.trainable = True


    model.compile(loss=[hnm_loss],optimizer=opt,metrics=[dif_pos_neg,score_pos,score_neg,"accuracy"])

    history = model.fit_generator(detect_train_generator(args,maps,outm),
                            max_queue_size=10, workers=0,use_multiprocessing=False,
                            steps_per_epoch=tr_steps,
                            epochs=epochs,
                            callbacks=callbacks,
                            verbose=1)


    ## SAVE MODEL
    if (args.save_model!=None):
        save_to_disk(model,args.save_model)












    ###############
