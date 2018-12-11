from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.callbacks import LearningRateScheduler as LRS
from keras.callbacks import ModelCheckpoint


def set_tr_params(args):

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


    ## LR Scheduller
    if (args.lra==True):
        e1=int(epochs*0.5)
        e2=int(epochs*0.75)
        print ("Learning rate annealing epochs: 0-->",e1,"-->",e2,"-->",epochs)
        print ("Learning rate annealing LR:",args.lr,"-->",args.lr/args.lra_scale,"-->",args.lr/(args.lra_scale))

        def scheduler(epoch):
            if epoch < e1:
                return args.lr
            elif epoch < e2:
                if (epoch==e1):
                    print ("===============================")
                    print ("New learning rate:",args.lr/args.lra_scale)
                    print ("===============================")
                return args.lr/args.lra_scale
            else:
                if (epoch==e2):
                    print ("===============================")
                    print ("New learning rate:",args.lr/(args.lra_scale*args.lra_scale))
                    print ("===============================")
                return args.lr/(args.lra_scale*args.lra_scale)

        set_lr = LRS(scheduler)
        callbacks=[set_lr]
    else:
        print ("Learning rate=",args.lr,"no annealing")
        callbacks=[]

    ## Model checkpoint
    if (args.save_epochs):
        filepath=args.save_model+"_{epoch:03d}.h5"
        checkpoint = ModelCheckpoint(filepath,verbose=1, save_best_only=False)
        callbacks.append(checkpoint)

    return opt,callbacks
