import keras
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121
from keras.applications.densenet import DenseNet169
from keras.applications.densenet import DenseNet201
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet import MobileNet
#from keras.applications.mobilenetv2 import MobileNetV2
from keras.layers import  Activation
from keras.layers.normalization import BatchNormalization as BN

from keras.layers import Input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

import numpy as np

def load_pretrained_model(args):

    h=args.height
    w=args.width
    if (args.chan=="rgb"):
        shape=(h,w,3)
    else:
        shape=(h,w,1)

    print("Setting input shape for pretrained model to",shape)

    input_tensor=Input(shape=shape)

    if (args.model=="vgg16"):
        load_model = VGG16(input_tensor=input_tensor,weights='imagenet', include_top=False)
    elif (args.model=="vgg19"):
        load_model = VGG19(input_tensor=input_tensor,weights='imagenet', include_top=False)
    elif (args.model=="resnet50"):
        load_model = ResNet50(input_tensor=input_tensor,weights='imagenet', include_top=False)
    elif (args.model=="densenet121"):
        load_model = DenseNet121(input_tensor=input_tensor,weights='imagenet', include_top=False)
    elif (args.model=="densenet169"):
        load_model = DenseNet169(input_tensor=input_tensor,weights='imagenet', include_top=False)
    elif (args.model=="densenet201"):
        load_model = DenseNet201(input_tensor=input_tensor,weights='imagenet', include_top=False)
    elif (args.model=="inceptionv3"):
        load_model = InceptionV3(input_tensor=input_tensor,weights='imagenet', include_top=False)
    elif (args.model=="inceptionresnetv2"):
        load_model = InceptionResNetV2(input_tensor=input_tensor,weights='imagenet', include_top=False)
    elif (args.model=="mobilenet"):
        load_model = MobileNet(input_tensor=input_tensor,weights='imagenet', include_top=False)
    elif (args.model=="mobilenetv2"):
        load_model = MobileNetV2(input_tensor=input_tensor,weights='imagenet', include_top=False)

    return load_model

def pretrained_model(args,num_classes):

    load_model=load_pretrained_model(args)

    x=load_model.output
    x=GlobalAveragePooling2D()(x)
    for i in range(args.predlayers):
        x=Dense(args.predsize)(x)
        x=BN()(x)
        x=Activation('relu')(x)

    predictions = Dense(num_classes, activation='softmax')(x)

    model=Model(inputs=load_model.input, outputs=predictions)


    return load_model,model
