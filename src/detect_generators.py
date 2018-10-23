import json
from PIL import Image, ImageOps
import numpy as np
import sys
from loaders import *
from pprint import pprint
import random

######################################################################
################### DETECTION GENERATORS #############################
######################################################################

def detect_train_generator(args):
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


    while True:

        for i in range(args.batch):
            r=random.randint(0, size-1)
            imgname=databox[r]['image_id']

            fname=args.trdir+args.fprefix+str(imgname)+".jpg"
            [x,ws,hs]=load_image_as_numpy(args,fname)

            c=0
            for all in databox:
                if (all['image_id']==imgname):
                    c=c+1
            yanot= np.zeros((c, 5))

            c=0
            for all in databox:
                if (all['image_id']==imgname):
                    print(imgname, all['bbox'],catdict[all['category_id']])
                    x,y,w,h=all['bbox']
                    print(x*ws,y*hs,(x+w)*ws,(y+h)*hs)
                    yanot[c,:]=[catdict[all['category_id']],x*ws,y*hs,(x+w)*ws,(y+h)*hs]
                    c=c+1

            ## Target codification
            #for i in range(c):

            print(yanot)
            ## batch
            #X[i,:]=x
            #Y[i,:]=y
