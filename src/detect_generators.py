from PIL import Image, ImageOps
import numpy as np
import sys

from pprint import pprint

from detect_tools import *

import keras.backend as K

######################################################################
################### DETECTION GENERATORS #############################
######################################################################
def detect_train_generator(args,maps):
    ## read json annot files

    [images,imglen,boxes,boxlen,catdict,catlen]=load_annot_json(args.trannot)


    ## X,Y for trainig (data, targets) and A foro anchors
    lanchors=len(args.anchors)//2
    A=build_anchors(args,maps)

    ## Build X,Y
    [X,Y]=buil_XY(args,maps)

    output_dict={}
    k=0
    for m in maps:
     output_dict.update({m.name.replace('/Sigmoid:0',''):Y[k]})
     k=k+1

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


        logfile = open("gila_log.txt", "a")

        for b in range(args.batch):

            [img,ws,hs,imgname]=rand_image(args,images)

            ##DATA AUGMENTATION
            [img,dx,dy,scale,flip]=transform(args,img,gen)
            X[b,:]=img

            ## Load annotation of image, codification
            ## w.r.t an image of (args.height x args.width)
            anot=[]
            for all in boxes:
                 if (all['image_id']==imgname):
                    x,y,w,h=all['bbox']
                    x=x*ws
                    y=y*hs
                    w=w*ws
                    h=h*hs
                    #Apply transofmrs to the gt box
                    if (flip):
                        x=args.width-(x+w)

                    x=(x+dx)*scale
                    y=(y+dy)*scale
                    w=w*scale
                    h=h*scale
                    anot.append([catdict[all['category_id']],x,y,(x+w),(y+h)])
                    #cat,x1,y1,x2,y2
            match=0
            for an in anot:
                k=0
                setanchor=False
                max=0
                for y in Y:
                    # scale annotations to maps and obtain center
                    scaley=float(args.height)/float(y.shape[1])
                    scalex=float(args.width)/float(y.shape[2])

                    cx=an[1]+(an[3]-an[1])/2
                    mx=int(cx/scalex)
                    cy=an[2]+(an[4]-an[2])/2
                    my=int(cy/scaley)

                    i=0
                    for j in range(lanchors):
                        score=iou([A[k][my,mx,i],A[k][my,mx,i+1],A[k][my,mx,i+2],A[k][my,mx,i+3]],
                                        [an[1],an[2],an[3],an[4]])

                        if (score>0.5):
                            setanchor=True
                            oclass=int(an[0])
                            y[b,my,mx,(j*catlen)+oclass]=1.0
                        i=i+4
                    k=k+1
                if (setanchor==True):
                    match=match+1

            if (args.log):
                logfile.write("Image %s - %f\n" %(imgname,(match*100.0)/len(anot)))
        if (args.log):
            logfile.write("============================\n")
            logfile.close()
        yield (X,output_dict)





        ######


















            #########
