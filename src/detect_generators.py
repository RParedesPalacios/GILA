from PIL import Image, ImageOps, ImageDraw
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

    [images,imglen,boxes,boxlen,catdict,catlen,_]=load_annot_json(args.trannot)


    ## X,Y for trainig (data, targets) and A foro anchors
    lanchors=len(args.anchors)//2
    A=build_anchors(args,maps)

    ## Build X,Y
    [X,Y]=build_XY(args,maps)


    #### DATA AUGMENTATION
    da_width=args.da_width/100.0
    da_height=args.da_height/100.0
    da_zoom=args.da_zoom
    da_flip_h=False

    print("Data Modification: Rescale input values",args.da_rescale) ## In loader
    if (da_width!=0.0):
        print("Data Augmentation: Width [%.2f,%.2f]" %(-args.width*da_width,args.width*da_width))
    if (da_height!=0.0):
        print("Data Augmentation: Height [%.2f,%.2f]" %(-args.height*da_height,args.height*da_height))
    if (da_zoom!=0.0):
        print("Data Augmentation: Zoom [%.2f,%.2f]" %(1.0,1+da_zoom))
    if (args.da_flip_h==True):
        print("Data Augmentation: FlipH")

    ## Provide images and achors fitting with iou>0.5
    print("Start Generator....")
    gen = ImageDataGenerator()

    if (args.log):
        logfile = open("gila_log.txt", "w")
        logfile.write("============================\n")
        logfile.close()

    save_gt=1
    iou_thr=0.4

    while True:

        for y in Y:
            y[:]=0.0
            y[:,:,:,catlen-1::catlen]=1


        if (args.log):
            logfile = open("gila_log.txt", "a")

        match=0
        totan=0
        for b in range(args.batch):

            [img,ws,hs,id]=rand_image(args,images)

            ##DATA AUGMENTATION
            [img,dx,dy,scale,flip]=transform(args,img,gen)
            X[b,:]=img

            if (save_gt):
                modimg=Image.fromarray(np.uint8(img*255))
                draw=ImageDraw.Draw(modimg)


            ## Load annotation of image, codification
            ## w.r.t an image of (args.height x args.width)
            anot=[]
            for box in boxes:
                 if (int(box['image_id'])==int(id)):
                    ### I have done this: -dy,-dx,1.0/scale to obtain a correct
                    ### box displacement according to the image transform...
                    ### open issue in keras-preprocessing ...
                    [x,y,w,h]=transform_box(args,box,ws,hs,-dy,-dx,1.0/scale,flip)
                    anot.append([catdict[box['category_id']],x,y,(x+w),(y+h)])
                    if (save_gt):
                        draw.rectangle((x,y,(x+w),(y+h)), fill=None)
                    #cat,x1,y1,x2,y2





            totan+=len(anot)
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

                        if (score>iou_thr):
                            if (save_gt):
                                draw.rectangle((A[k][my,mx,i],A[k][my,mx,i+1],A[k][my,mx,i+2],A[k][my,mx,i+3]), fill=None)
                            setanchor=True
                            oclass=int(an[0])
                            y[b,my,mx,(j*catlen)+oclass]=1 # positive target
                            y[b,my,mx,((j+1)*catlen)-1]=0 # remove negative
                        i=i+4
                    k=k+1

                if (setanchor==True):
                    match=match+1

            if (save_gt):
                modimg.save("gt.jpg")



        # batch
        if (args.log):
            logfile.write("GT boxes matched with anchors IOU>%1.2f = %.2f%%\n" %(iou_thr,(match*100.0)/totan))
            logfile.write("============================\n")
            logfile.close()

        print("\nGT boxes matched with anchors IOU>%1.2f = %.2f%%\n" %(iou_thr,(match*100.0)/totan))

        k=0
        Yr=[]
        for y in Y:
            Yr.append(Y[k].reshape((args.batch,-1,catlen)))
            #print(Yr[k].shape)
            k=k+1

        Yc=np.concatenate(Yr, axis=1)

        yield (X,Yc)





        ######
