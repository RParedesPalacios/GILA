import numpy as np
import keras


from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.callbacks import LearningRateScheduler as LRS

from automodel import auto_model
from pretrainedmodel import pretrained_model
from generators import *
from files import *
from loaders import *

##################################
### EVAL CLASS MODELS
##################################
def eval_class_model(args):

    if (args.load_model==None):
        print("No model name (-load_model)")
        sys.exit(0)


    model=load_json_model(args.load_model)
    if (args.summary==True):
        model.summary()


    if (args.chan=="rgb"):
        print ("setting depth to RGB")
        CHAN=3
    else:
        print ("setting depth to Gray")
        CHAN=1

    Xt=0
    Lt=0
    if (args.tsfile!=None):
        print("Loading test file:",args.tsfile)
        [Xt,Lt]=load_list_file_class_to_numpy(args.trfile,args.height,args.width,CHAN,args.resize)
        num_classes=len(set(Lt))
        Lt = keras.utils.to_categorical(Lt, num_classes)
        numts=len(Lt)
    elif (args.tsdir!=None):
        print("Setting test dir to",args.tsdir)
        TEST=1
        gen=ImageDataGenerator().flow_from_directory(args.tsdir,target_size=(args.height,args.width),
            batch_size=args.batch,
            class_mode='categorical')
        numts=gen.samples

    batch_size=args.batch

    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])

    ts_steps=numts//batch_size

    print("Evaluating...")
    score=model.evaluate_generator(test_generator(args,Xt,Lt),ts_steps)
    print("Acc:%.2f%%" % (score[1]*100))



##################################
### TRAIN CLASS MODELS
##################################
def train_class_model(args):


    if (args.chan=="rgb"):
        print ("setting depth to RGB")
        CHAN=3
    else:
        print ("setting depth to Gray")
        CHAN=1

    X=0
    L=0
    TEST=0
    Xt=0
    Lt=0
    ######### DATA SOURCE
    if (args.trfile!=None):
        print("Loading training file:",args.trfile)
        [X,L]=load_list_file_class_to_numpy(args.trfile,args.height,args.width,CHAN,args.resize)
        num_classes=len(set(L))
        numtr=len(L)
        if (args.tsfile!=None):
            TEST=1
            print("Loading test file:",args.tsfile)
            [Xt,Lt]=load_list_file_class_to_numpy(args.tsfile,args.height,args.width,CHAN,args.resize)
            numts=len(Lt)
    else:
        print("Setting training dir to",args.trdir)
        gen=ImageDataGenerator().flow_from_directory(args.trdir,target_size=(args.height,args.width),
            batch_size=args.batch,
            class_mode='categorical')
        num_classes=len(set(gen.classes))
        numtr=gen.samples
        if (args.tsdir!=None):
            print("Setting test dir to",args.tsdir)
            TEST=1
            gen=ImageDataGenerator().flow_from_directory(args.tsdir,target_size=(args.height,args.width),
                batch_size=args.batch,
                class_mode='categorical')
            numts=gen.samples

    ######### MODELS
    PRETR=False

    if (args.load_model!=None):
        model=load_json_model(args.load_model)
    elif (args.model=="auto"):
        model=auto_model(args,num_classes)
    else:
        PRETR=True
        [base,model]=pretrained_model(args,num_classes)


    if (args.summary==True):
        from keras.utils import plot_model
        model.summary()
        plot_model(model, to_file='model.png')

    print ("Classification to",num_classes,"classes")

    if (args.trdir==None):
        L = keras.utils.to_categorical(L, num_classes)
        Lt = keras.utils.to_categorical(Lt, num_classes)


    #### GENERAL
    batch_size=args.batch
    print("Batch size",batch_size)
    epochs=args.epochs
    print("Epochs",epochs)
    if (PRETR):
        fepochs=args.fepochs
        print("Fepochs",fepochs)

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


    tr_steps=numtr//batch_size
    if (TEST):
        ts_steps=numts//batch_size


    #### PRE-TRAIN
    if (PRETR==True):
        if (args.flayer!=None):
            print("Freezing up to layer",args.flayer,"of the pre-trained model %d epochs" %(fepochs))
            for layer in base.layers:
                if (layer.name==args.flayer):
                    break
                else:
                    layer.trainable = False
        else:
            print("Freezing the pre-trained model %d epochs" %(fepochs))
            for layer in base.layers:
                layer.trainable = False

        model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

        if (TEST):
            model.fit_generator(training_generator(args,num_classes,X,L),
                            steps_per_epoch=tr_steps,
                            epochs=fepochs,
                            validation_data=test_generator(args,Xt,Lt),
                            validation_steps=ts_steps,
                            verbose=1)
        else:
            model.fit_generator(training_generator(args,num_classes,X,L),
                            steps_per_epoch=tr_steps,
                            epochs=fepochs,
                            verbose=1)

        for layer in base.layers:
            layer.trainable = True

        print("Setting all trainable")



    ## REGULAR TRAINING
    if (args.lra==True):
        e1=int(epochs*0.5)
        e2=int(epochs*0.75)
        print ("Learning rate annealing epochs: 0-->",e1,"-->",e2,"-->",epochs)
        print ("Learning rate annealing LR:",args.lr,"-->",args.lr/args.lra_scale,"-->",args.lr/(args.lra_scale*args.lra_scale))

        def scheduler(epoch):
            if epoch < e1:
                return args.lr
            elif epoch < e2:
                if (epoch==e1):
                    print ("===============================")
                    print ("New learning rate:",args.lr/10)
                    print ("===============================")
                return args.lr/args.lra_scale
            else:
                if (epoch==e2):
                    print ("===============================")
                    print ("New learning rate:",args.lr/100)
                    print ("===============================")
                return args.lr/(args.lra_scale*args.lra_scale)

        set_lr = LRS(scheduler)
        callbacks=[set_lr]
    else:
        print ("Learning rate=",args.lr,"no annealing")
        callbacks=[]

    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])


    if (TEST):
        history = model.fit_generator(training_generator(args,num_classes,X,L),
                            steps_per_epoch=tr_steps,
                            epochs=epochs,
                            validation_data=test_generator(args,Xt,Lt),
                            validation_steps=ts_steps,
                            callbacks=callbacks,
                            verbose=1)
    else:
        history = model.fit_generator(training_generator(args,num_classes,X,L),
                            steps_per_epoch=tr_steps,
                            epochs=epochs,
                            callbacks=callbacks,
                            verbose=1)

    if (args.history):
        import pickle
        with open('gila.hist', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

    if (args.plot):
        import matplotlib.pyplot as plt
        #  "Accuracy"
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('gila.png')
        plt.show()

    ## SAVE MODEL
    if (args.save_model!=None):
        save_json_model(model,args.save_model)
