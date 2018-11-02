import json
from PIL import Image, ImageOps,ImageDraw
import numpy as np
import sys
from loaders import *
import random
from files import *
from pprint import pprint

######################################################################
################### DETECTION INFERENCE  #############################
######################################################################
#def eval_detect_model(args):

    ## # TODO:


def eval_detect_model(args):

    if (args.load_model==None):
        print("No model name to eval (-load_model)")
        sys.exit(0)


    model=load_json_model(args.load_model)
    if (args.summary==True):
        model.summary()


    ## read json annot files
    print("Loading JSON annotations file",args.tsannot)
    with open(args.tsannot) as f:
        data = json.load(f)
    f.close()

    cat=data['categories']
    print(data['categories'])
    size=len(cat)
    print ("Categories in annotation file:",size)
    lencat=size

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

    ## args.anchors codification
    ## w.r.t an image of (args.height x args.width)
    lanchors=len(args.anchors)//2

    print("Maps")
    maps=model.outputs

    A=[]
    for m in maps:
        A.append(np.zeros((m.shape[1],m.shape[2],len(cat)*len(args.anchors)*2)))

    k=0
    for m in maps:
        print(m.shape[1],"x",m.shape[2])
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

    #build batch
    ch=3
    if (args.chan=="gray"):
        ch=1
    X=np.zeros((args.batch,args.height,args.width,ch))
    names=[]
    for b in range(args.batch):
        read=0
        while (read==0):
            r=random.randint(0, size-1)
            imgname=databox[r]['image_id']
            ### from COCO image id to file path
            fname=args.tsdir+args.fprefix+str(imgname)+".jpg"
            try:
                [x,ws,hs]=load_image_as_numpy(args,fname)
                names.append(str(imgname))
                read=1
            except (FileNotFoundError, IOError):
                print("Warning:",fname,"not found")
                read=0

        X[b,:]=x

    print("Predict batch")
    ## get output maps
    Y=model.predict(X, args.batch)

    ## Draw detections
    for b in range(args.batch):
        fname=args.tsdir+args.fprefix+str(names[b])+".jpg"
        [x,ws,hs]=load_image_as_numpy(args,fname)
        img=Image.open(fname)
        draw=ImageDraw.Draw(img)
        k=0
        c=0
        for y in Y:
            for my in range(y.shape[1]):
                for mx in range(y.shape[2]):
                    for mz in range(y.shape[3]):
                        if (y[b,my,mx,mz]>0.5):
                            an=mz%lanchors
                            c=c+1
                            draw.rectangle(((A[k][my,mx,an]/ws,A[k][my,mx,an+1]/hs), (A[k][my,mx,an+2]/ws,A[k][my,mx,an+3]/hs)), fill=None)
                            #draw.rectangle((A[k][my,mx,an]/ws,A[k][my,mx,an+1]/hs),
                            #(A[k][my,mx,an+2]/ws,A[k][my,mx,an+3]/hs),fill="black")
            k=k+1
        print("Image",fname,"found",c,"boxes")
        
        fname=args.tsdir+args.fprefix+str(names[b])+"ANOT"+".jpg"
        img.save(fname)















######


















            #########
