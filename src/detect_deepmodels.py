import numpy as np
import keras

from pprint import pprint

from detect_generators import detect_train_generator
from detect_loss import *
from detect_automodel import *
from detect_tools import *
from detect_pretrainedmodel import *
from files import *
from setparams import set_tr_params

##################################
### TRAIN DETECTION MODELS
##################################
def train_det_model(args):

    ######### ANNOT FILE
    [images,imglen,boxes,boxlen,catdict,catlen,_]=load_annot_json(args.trannot)

    ######### ANCHORS
    if (len(args.anchors)==1):
        a=int(args.anchors[0])
        args.anchors.remove(a)

        args.anchors.append(0.75)
        args.anchors.append(0.75)

        args.anchors.append(1.5)
        args.anchors.append(1.5)
        for i in range(a-1):
          args.anchors.append(1)
          args.anchors.append(i+2)
          args.anchors.append(i+2)
          args.anchors.append(1)


    anchors=len(args.anchors)//2
    print(args.anchors)


    ######### MODEL
    PRETR=False
    if (args.load_model!=None):
        model=load_from_disk(args.load_model,hnm_loss,acc_pos,acc_neg)
        maps=[]
        for l in model.layers:
            print (l.name)
            if ("re_lu" in l.name):
                maps=[]
            elif ("reshape" in l.name):
                break
            else:
                maps.append(l.output)
    elif (args.model=="auto"):
        print("Automodel")
        model,maps=auto_det_model(args,anchors,catlen)
    else:
        PRETR=True
        [base,model,maps]=detect_pretrained_model(args,anchors,catlen)


    if (args.summary==True):
        from keras.utils import plot_model
        model.summary()
        plot_model(model, to_file='model.png')



    #### GENERAL
    batch_size=args.batch
    print("Batch size",batch_size)
    epochs=args.epochs
    print("Epochs",epochs)
    if (args.trsteps!=-1):
        tr_steps=args.trsteps
    else:
        tr_steps=imglen/batch_size

    [opt,callbacks]=set_tr_params(args)


    ## Pretraining epochs
    fepochs=args.fepochs
    if (PRETR)and(fepochs>0):
        print("Freezing the pre-trained model %d epochs" %(fepochs))
        for layer in base.layers:
            layer.trainable = False

        model.compile(loss=[hnm_loss],optimizer=opt,metrics=[acc_pos,acc_neg])
        #model.compile(loss='mean_squared_error',optimizer=opt,metrics=[acc_pos,acc_neg])

        history = model.fit_generator(detect_train_generator(args,maps),
                             max_queue_size=10, workers=0,use_multiprocessing=False,
                             steps_per_epoch=tr_steps,
                             epochs=fepochs,
                             callbacks=callbacks,
                             verbose=1)
        for layer in base.layers:
            layer.trainable = True


    ## Training epochs
    model.compile(loss=[hnm_loss],optimizer=opt,metrics=[acc_pos,acc_neg])
    #model.compile(loss='mean_squared_error',optimizer=opt,metrics=[acc_pos,acc_neg])

    history = model.fit_generator(detect_train_generator(args,maps),
                            max_queue_size=10, workers=0,use_multiprocessing=False,
                            steps_per_epoch=tr_steps,
                            epochs=epochs,
                            callbacks=callbacks,
                            verbose=1)


    ## logs and model
    if (args.history):
        import pickle
        with open('gila.hist', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

    if (args.plot):
        import matplotlib.pyplot as plt
        #  "Accuracy"
        plt.plot(history.history['loss'])
        plt.plot(history.history['acc_pos'])
        plt.plot(history.history['acc_neg'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.savefig('gila.png')
        plt.show()

    ## SAVE MODEL
    if (args.save_model!=None):
        save_json_model(model,args.save_model)










    ###############
