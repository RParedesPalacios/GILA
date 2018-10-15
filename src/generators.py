from keras.preprocessing.image import ImageDataGenerator

#### TRAINING BALANCE GENERATOR
def training_generator(args,num_classes,X=0,L=0):
    #### DATA AUGMENTATION
    da_width=args.da_width/100.0
    da_height=args.da_height/100.0
    da_rotation=args.da_rotation
    da_zoom=args.da_zoom
    da_flip_v=False
    if (args.da_flip_v=="yes"):
        da_flip_v=True
    da_flip_h=False
    if (args.da_flip_h=="yes"): da_flip_h=True

    print("Data Augmentation: Width",da_width)
    print("Data Augmentation: Height",da_width)
    print("Data Augmentation: Rotation",da_rotation)
    print("Data Augmentation: Zoom [%.2f,%.2f]" %(1.0-da_zoom,1+da_zoom))
    print("Data Augmentation: FlipV",da_flip_v)
    print("Data Augmentation: FlipH",da_flip_h)

    if (args.balance=="yes"):
        print("Batch balance")

    datagen = ImageDataGenerator(rescale=1./255,
                                    width_shift_range=da_width,
                                    height_shift_range=da_height,
                                    horizontal_flip=da_flip_h,
                                    vertical_flip=da_flip_v,
                                    rotation_range=da_rotation,
                                    zoom=da_zoom)
    if (args.trdir==None):
        gen=datagen.flow(X,L,batch_size=args.batch)
    else:
        gen = datagen.flow_from_directory(args.trdir,
            target_size=(args.height,args.width),
            batch_size=args.batch,
            class_mode='categorical')

    while True:
        [Xi,Li] = gen.next()

        if (args.balance=="no"):
            yield Xi,Li
        else:
            Xib=Xi
            Lib=Li
            bs=Xi.shape[0]

            if (bs<num_classes):
                yield Xi,Li
            else:
                num=int(bs/num_classes)
                for i in range(num_classes):
                    j=0
                    k=0
                    enc=False
                    while (j<num):
                        if (Li[k,i]==1):
                            Xib[(i*num)+j,:]=Xi[k,:]
                            Lib[(i*num)+j,:]=0
                            Lib[(i*num)+j,i]=1
                            j=j+1
                            enc=True

                        k=k+1
                        if (k>=bs):
                            k=0
                            if (enc==False):
                                break
                    if (enc==False):
                        break

                if (enc==True):
                    yield Xib,Lib
                else:
                    yield Xi,Li


#### TEST GENERATOR
def test_generator(args,Xt=0,Lt=0):

    datagen = ImageDataGenerator(rescale=1./255)
    if (args.tsdir==None):
        gen=datagen.flow(Xt,Lt,batch_size=args.batch)
    else:
        gen = datagen.flow_from_directory(args.tsdir,
            target_size=(args.height,args.width),
            batch_size=args.batch,
            class_mode='categorical')
    while True:
        [Xi,Li] = gen.next()
        yield Xi,Li
