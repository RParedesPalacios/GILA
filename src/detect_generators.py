import json
from PIL import Image, ImageOps
import numpy as np
import sys
from loaders import *
from pprint import pprint
import random
from keras import backend as K

def iou(box1,box2):
    ## (x1,y1,x2,y2)
    w1=box1[3]-box1[1]
    h1=box1[4]-box1[2]

    w2=box2[3]-box2[1]
    h2=box2[4]-box2[2]

    area1=w1*h1
    area2=w2*h2

    ##intersection
    dx=min(box1[3],box2[3])-max(box1[1],box2[1])
    dy=min(box1[4],box2[4])-max(box1[2],box2[2])
    if (dx>=0)and(dy>=0):
        inter=dx*dy
    else:
        return 0

    union=area1+area2-inter

    return float(inter)/float(union)



######################################################################
################### DETECTION GENERATORS #############################
######################################################################

def detect_train_generator(args,maps,anchors):
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


    Y=[]
    A=[]
    for m in maps:
        Y.append(np.zeros((args.batch,m.shape[1],m.shape[2],m.shape[3])))
        A.append(np.zeros((m.shape[1],m.shape[2],len(anchors)*2)))
    X=np.zeros((args.batch,args.height,args.width,3))


    ## Anchors codification
    k=0
    for m in maps:
        print(m.shape[2])
        scalex=float(args.width)/float(m.shape.as_list()[2])
        scaley=float(args.height)/float(m.shape.as_list()[1])
        for my in range(m.shape.as_list()[1]):
            cy=(float(my)+0.5)*scaley
            for mx in range(m.shape.as_list()[2]):
                cx=(float(mx)+0.5)*scalex
                i=0
                for j in range(len(anchors)//2):
                    w=anchors[2*j]*scalex
                    h=anchors[2*j+1]*scaley
                    A[k][my,mx,i]=cx-(w/2)     #x1
                    A[k][my,mx,i+1]=cx+(w/2)   #x2
                    A[k][my,mx,i+2]=cy-(h/2)   #y1
                    A[k][my,mx,i+3]=cy+(h/2)   #y2
                    i=i+4
        k=k+1


    print(A[2][0,0,:])

    while True:
        yield X,Y

    # while True:
    #
    #     for i in range(args.batch):
    #         # r=random.randint(0, size-1)
    #         # imgname=databox[r]['image_id']
    #         #
    #         # fname=args.trdir+args.fprefix+str(imgname)+".jpg"
    #         # [x,ws,hs]=load_image_as_numpy(args,fname)
    #         #
    #         # c=0
    #         # for all in databox:
    #         #     if (all['image_id']==imgname):
    #         #         c=c+1
    #         # yanot= np.zeros((c, 5))
    #         #
    #         # c=0
    #         # for all in databox:
    #         #     if (all['image_id']==imgname):
    #         #         print(imgname, all['bbox'],catdict[all['category_id']])
    #         #         x,y,w,h=all['bbox']
    #         #         print(x*ws,y*hs,(x+w)*ws,(y+h)*hs)
    #         #         yanot[c,:]=[catdict[all['category_id']],x*ws,y*hs,(x+w)*ws,(y+h)*hs]
    #         #         c=c+1
    #
    #
    #
    #         ##print(yanot)
    #         ## batch
    #         #X[i,:]=x
    #         #Y[i,:]=y
    #


















            #########
