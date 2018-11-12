import json
import random
from loaders import *

from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import scipy.ndimage


#### DETECT TOOLS

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


def load_annot_json(filename):

    ######### ANNOT FILE
    print("Loading JSON annotations file",filename)
    with open(filename) as f:
         data = json.load(f)
    f.close()

    categories=data['categories']
    catlen=len(categories)
    print ("Categories in annotation file:",catlen)

    images=data['images']
    imglen=len(images)
    print ("Images in annotation file:",imglen)

    boxes=data['annotations']
    boxlen=len(boxes)
    print ("Boxes in annotation file:",boxlen)

    catdict={}
    j=0
    for i in categories:
        catdict.update({categories[j]['id']:j})
        j=j+1

    print(catdict)

    return images,imglen,boxes,boxlen,catdict,catlen


def build_anchors(args,maps):


    lanchors=len(args.anchors)//2
    A=[]
    for m in maps:
        A.append(np.zeros((m.shape[1],m.shape[2],lanchors*4)))

    ## args.anchors codification
    ## w.r.t an image of (args.height x args.width)
    k=0
    for m in maps:
        print(m.name,m.shape[1],"x",m.shape[2])
        scalex=float(args.width)/float(m.shape.as_list()[2])
        scaley=float(args.height)/float(m.shape.as_list()[1])
        for my in range(m.shape.as_list()[1]):
            cy=(float(my)+0.5)*scaley
            for mx in range(m.shape.as_list()[2]):
                cx=(float(mx)+0.5)*scalex
                i=0
                for j in range(lanchors):
                    w=args.anchors[2*j]*scalex
                    h=args.anchors[2*j+1]*scaley
                    A[k][my,mx,i]=cx-(w/2)     #x1
                    A[k][my,mx,i+1]=cy-(h/2)   #y1
                    A[k][my,mx,i+2]=cx+(w/2)   #x2
                    A[k][my,mx,i+3]=cy+(h/2)   #y2
                    i=i+4
        k=k+1
    return A


def buil_XY(args,maps):
    Y=[]
    for m in maps:
        Y.append(np.zeros((args.batch,m.shape[1],m.shape[2],m.shape[3])))

    ch=3
    if (args.chan=="gray"):
        ch=1
    X=np.zeros((args.batch,args.height,args.width,ch))

    return X,Y


def rand_image(args,images,tr=1):

    read=0
    while (read==0):
        r=random.randint(0, len(images)-1)
        imgname=images[r]['id']
        ### from COCO image id to file path
        if (tr==1):
            fname=args.trdir+args.fprefix+str(imgname)+".jpg"
        else:
            fname=args.tsdir+args.fprefix+str(imgname)+".jpg"
        try:
            [x,ws,hs]=load_image_as_numpy(args,fname)
            read=1
        except (FileNotFoundError, IOError):
            read=0
    return x,ws,hs,imgname


def transform(args,x,gen):

    #HORIZAONTAL FLIP
    flip=False
    if (args.da_flip_h):
        if random.randint(0,1):
            flip=True
            transform={'flip_horizontal':True}
            x=gen.apply_transform(x, transform)

    #HORIZAONTAL AND VERTICAL SHIFT
    dx=0
    dy=0
    if (args.da_width!=0.0)or(args.da_height!=0.0):
        dx=(args.da_width*args.width)//100
        dy=(args.da_width*args.width)//100
        dx=random.randint(-dx,dx)
        dy=random.randint(-dy,dy)
        transform={'tx':dx,'ty':dy}
        x=gen.apply_transform(x, transform)

    # SCALE
    scale=1.0
    if (args.da_zoom!=0.0):
        scale=random.uniform(1,1+args.da_zoom)
        transform={'zx':scale,'zy':scale}
        x=gen.apply_transform(x, transform)

    return x,dx,dy,scale,flip


























    #####
