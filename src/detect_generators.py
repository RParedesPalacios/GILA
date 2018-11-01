import json
from PIL import Image, ImageOps
import numpy as np
import sys
from loaders import *
from pprint import pprint
import random
from keras import backend as K
import tensorflow as tf

def iou(box1,box2):
    ## (x1,y1,x2,y2)
    w1=box1[2]-box1[0]
    h1=box1[3]-box1[1]

    w2=box2[2]-box2[0]
    h2=box2[3]-box2[1]

    area1=w1*h1
    area2=w2*h2

    #print("Box1:",box1[0],box1[1],box1[2],box1[3]," - ",box1[0]+w1//2,box1[1]+h1//2)
    #print("Box2:",box2[0],box2[1],box2[2],box2[3]," - ",box2[0]+w2//2,box2[1]+h2//2)

    ##intersection
    dx=min(box1[2],box2[2])-max(box1[0],box2[0])
    dy=min(box1[3],box2[3])-max(box1[1],box2[1])
    if (dx>=0)and(dy>=0):
        inter=dx*dy
    else:
        return 0

    union=area1+area2-inter

    return float(inter)/float(union)



######################################################################
################### DETECTION GENERATORS #############################
######################################################################

def detect_train_generator(args,maps):
    ## read json annot files

    print("Loading JSON annotations file",args.trannot)
    with open(args.trannot) as f:
        data = json.load(f)
    f.close()

    cat=data['categories']
    pprint(data['categories'])
    size=len(cat)
    print ("Categories in annotation file:",size)

    catdict={}
    j=0
    for i in cat:
        catdict.update({cat[j]['id']:j})
        j=j+1

    print(catdict)

    databox=data['annotations']
    size=len(databox)
    print ("Boxes in annotation file:",size)

    dataimg=data['images']
    size=len(dataimg)
    print ("Images in annotation file:",size)


    ## X,Y for trainig (data, targets) and A foro anchors
    Y=[]
    A=[]
    for m in maps:
        Y.append(np.zeros((args.batch,m.shape[1],m.shape[2],m.shape[3])))
        A.append(np.zeros((m.shape[1],m.shape[2],len(cat)*len(args.anchors)*2)))


    ch=3
    if (args.chan=="gray"):
        ch=1
    X=np.zeros((args.batch,args.height,args.width,ch))


    ## args.anchors codification
    ## w.r.t an image of (args.height x args.width)
    lanchors=len(args.anchors)//2
    k=0
    for m in maps:
        print(m.shape[1],m.shape[2])
        scalex=float(args.width)/float(m.shape.as_list()[2])
        scaley=float(args.height)/float(m.shape.as_list()[1])
        for my in range(m.shape.as_list()[1]):
            cy=(float(my)+0.5)*scaley
            for mx in range(m.shape.as_list()[2]):
                cx=(float(mx)+0.5)*scalex
                i=0
                for c in range(len(cat)):
                    for j in range(lanchors):
                        w=args.anchors[2*j]*scalex
                        h=args.anchors[2*j+1]*scaley
                        A[k][my,mx,i]=cx-(w/2)     #x1
                        A[k][my,mx,i+1]=cy-(h/2)   #y1
                        A[k][my,mx,i+2]=cx+(w/2)   #x2
                        A[k][my,mx,i+3]=cy+(h/2)   #y2
                        i=i+4
        k=k+1


    #print(A[0][0,0,:])


    ## Provide images and achors fitting with iou>0.5
    print("Start Generator....")
    while True:

        for y in Y:
            y[:]=0.0

        tot=0
        match=0
        for b in range(args.batch):

            read=0
            while (read==0):
                r=random.randint(0, size-1)
                imgname=databox[r]['image_id']
                ### from COCO image id to file path
                fname=args.trdir+args.fprefix+str(imgname)+".jpg"

                try:
                    [x,ws,hs]=load_image_as_numpy(args,fname)
                    read=1
                except (FileNotFoundError, IOError):
                    read=0


            X[b,:]=x

            ## Load annotation of image, codification
            ## w.r.t an image of (args.height x args.width)
            anot=[]
            for all in databox:
                 if (all['image_id']==imgname):
                     #print(imgname, all['bbox'],catdict[all['category_id']])
                     x,y,w,h=all['bbox']
                     anot.append([catdict[all['category_id']],x*ws,y*hs,(x+w)*ws,(y+h)*hs])
                     #cat,x1,y1,x2,y2




            #print(yanot)
            for an in anot:
                #print("====================================================")
                tot=tot+1
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

                    shift=0
                    for sy in range(-shift,shift+1,1):
                        if ((my+sy)>=0)and((my+sy)<A[k].shape[0]):
                            for sx in range(-shift,shift+1,1):
                                if ((mx+sx)>=0)and((mx+sx)<A[k].shape[1]):
                                    # print("(",cx,",",cy,")")
                                    # print("***(",dx,",",dy,")")
                                    # print("(",mx+dx,",",my+dy,")")
                                    i=0
                                    for j in range(lanchors):
                                        #w=args.anchors[2*j]*scalex
                                        #h=args.anchors[2*j+1]*scaley
                                        #print("[",w,",",h,"]")
                                        score=iou([A[k][my+sy,mx+sx,i],A[k][my+sy,mx+sx,i+1],A[k][my+sy,mx+sx,i+2],A[k][my+sy,mx+sx,i+3]],
                                        [an[1],an[2],an[3],an[4]])

                                        if (score>max):
                                            max=score
                                        if (score>0.5):
                                            #print("anchor found")
                                            setanchor=True
                                            cls=int(an[0])
                                            y[b,my+sy,mx+sx,(cls*lanchors)+j]=1.0
                                        i=i+4
                    k=k+1


                if (setanchor==True):
                    match=match+1

        mpc=float(100*match)/float(tot)
        if (mpc<50):
            print("few anchors matched= %d %d %.2f%%" %(match,tot,mpc))


        yield X,Y





        ######


















            #########
