from PIL import Image, ImageOps
import numpy as np
import sys

from pprint import pprint

from detect_tools import *

import keras.backend as K

######################################################################
################### DETECTION GENERATORS #############################
######################################################################
def detect_train_generator(args,maps,outm):
    ## read json annot files

    [images,imglen,boxes,boxlen,catdict,catlen]=load_annot_json(args.trannot)


    ## X,Y for trainig (data, targets) and A foro anchors
    lanchors=len(args.anchors)//2
    A=build_anchors(args,maps)

    ## Build X,Y
    [X,Y]=buil_XY(args,maps)



    ## Provide images and achors fitting with iou>0.5
    print("Start Generator....")
    gen = ImageDataGenerator()

    if (args.log):
        logfile = open("gila_log.txt", "w")
        logfile.write("============================\n")
        logfile.close()

    while True:

        for y in Y:
            y[:]=0.0
            y[:,:,:,catlen-1::catlen]=1


        if (args.log):
            logfile = open("gila_log.txt", "a")

        for b in range(args.batch):

            [img,ws,hs,imgname]=rand_image(args,images)

            ##DATA AUGMENTATION
            [img,dx,dy,scale,flip]=transform(args,img,gen)
            X[b,:]=img

            ## Load annotation of image, codification
            ## w.r.t an image of (args.height x args.width)
            anot=[]
            for box in boxes:
                 if (box['image_id']==imgname):
                    [x,y,w,h]=transform_box(args,box,ws,hs,dx,dy,scale,flip)
                    anot.append([catdict[box['category_id']],x,y,(x+w),(y+h)])
                    #cat,x1,y1,x2,y2

            match=0
            for an in anot:
                k=0
                setanchor=False
                for y in Y:
                    # scale annotations to maps and obtain center
                    scaley=float(args.height)/float(y.shape[1])
                    scalex=float(args.width)/float(y.shape[2])

                    cx=an[1]+(an[3]-an[1])/2
                    mx=int(min(max(cx/scalex,0),y.shape[2]-1))

                    cy=an[2]+(an[4]-an[2])/2
                    my=int(min(max(cy/scaley,0),y.shape[1]-1))

                    i=0
                    for j in range(lanchors):
                        score=iou([A[k][my,mx,i],A[k][my,mx,i+1],A[k][my,mx,i+2],A[k][my,mx,i+3]],
                                        [an[1],an[2],an[3],an[4]])

                        if (score>0.5):
                            setanchor=True
                            oclass=int(an[0])
                            y[b,my,mx,(j*catlen)+oclass]=1 # positive target
                            y[b,my,mx,((j+1)*catlen)-1]=0 # remove negative
                        i=i+4
                    k=k+1

                if (setanchor==True):
                    match=match+1


            if (args.log):
                logfile.write("Image %s - %f\n" %(imgname,(match*100.0)/len(anot)))
        if (args.log):
            logfile.write("============================\n")
            logfile.close()

        k=0
        Yr=[]
        for y in Y:
            Yr.append(Y[k].reshape((args.batch,-1,catlen)))
            #print(Yr[k].shape)
            k=k+1

        Yc=np.concatenate(Yr, axis=1)

        yield (X,Yc)





        ######
