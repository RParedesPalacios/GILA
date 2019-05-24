import json
import random
from loaders import *

from keras.preprocessing.image import *
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
    catlen=catlen+1 ## background class

    images=data['images']

    imglen=len(images)
    print ("Images in annotation file:",imglen)

    boxes=data['annotations']
    boxlen=len(boxes)
    print ("Boxes in annotation file:",boxlen)

    catdict={}
    j=0
    catnames=[]
    for c in categories:
        catdict.update({c['id']:j})
        catnames.append(c['name'])
        print(j,c['name'])
        j=j+1

    print(catdict)

    return images,imglen,boxes,boxlen,catdict,catlen,catnames


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


def build_XY(args,maps):
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

        imgname=images[r]['file_name']
        id=images[r]['id']

        if (tr==1):
            fname=args.trdir+"/"+imgname
        else:
            fname=args.tsdir+"/"+imgname
        try:
            [x,ws,hs]=load_image_as_numpy(args,fname)
            read=1
        except (FileNotFoundError, IOError):
            print("\nFailing to load "+fname)
            read=0
    return x,ws,hs,id


def transform(args,x,gen):

    transform={}
    #HORIZAONTAL FLIP
    flip=False
    if (args.da_flip_h):
        if random.randint(0,1):
            flip=True
            transform.update({'flip_horizontal':True})

    #HORIZAONTAL AND VERTICAL SHIFT
    dx=0
    dy=0
    if (args.da_width!=0.0)or(args.da_height!=0.0):
        dx=(args.da_width*args.width)//100
        dy=(args.da_height*args.height)//100
        dx=random.uniform(-dx,dx)
        dy=random.uniform(-dy,dy)
        transform.update({'tx':dx,'ty':dy})

    # SCALE
    scale=1.0
    if (args.da_zoom!=0.0):
        scale=random.uniform(1.0-args.da_zoom,1.0)
        transform.update({'zx':scale,'zy':scale})


    if (args.da_flip_h)or(args.da_width!=0.0)or(args.da_height!=0.0)or(args.da_zoom!=0.0):
        x=gen.apply_transform(x, transform)

    return x,dx,dy,scale,flip

def transform_box(args,box,ws,hs,dx,dy,scale,flip):

    x,y,w,h=box['bbox']
    # scale to width and height
    x=x*ws
    y=y*hs
    w=w*ws
    h=h*hs

    ## Keras apply_transform: shit-zoom-flip
    ## Apply the same transforms to the gt box
    w2=args.width/2
    h2=args.height/2
    x=w2+((x+dx)-w2)*scale
    y=h2+((y+dy)-h2)*scale
    w=w*scale
    h=h*scale

    if (flip):
        x=args.width-(x+w)

    return x,y,w,h

# Malisiewicz et al.
# from https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
# with scores
def non_max_suppression_fast(boxes, overlapThresh):
    if len(boxes) == 0:
        return []

	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

	# initialize the list of picked indexes
    pick = []

	# grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    sc = boxes[:,4]

	# compute the area of the bounding boxes and sort the bounding
	# boxes by the score
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(sc)

	# keep looping while some indexes still remain in the indexes
	# list
    while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked using the
	# integer data type
    return boxes[pick].astype("int")
























    #####
