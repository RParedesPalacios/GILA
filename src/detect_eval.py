from PIL import Image, ImageOps, ImageDraw
import sys
from loaders import *
from files import *
from detect_loss import *
from detect_tools import *

######################################################################
################### DETECTION INFERENCE  #############################
######################################################################

def eval_detect_model(args,model=None):


    if (model==None):
        if (args.load_model==None):
            print("No model name to eval (-load_model)")
            sys.exit(0)
        model=load_from_disk(args.load_model,hnm_loss,num_pos)


    if (args.summary==True):
        model.summary()

    [images,imglen,boxes,boxlen,catdict,catlen]=load_annot_json(args.tsannot)

    lanchors=len(args.anchors)//2
    maps=model.outputs
    A=build_anchors(args,maps)


    [X,Y]=buil_XY(args,maps)


    names=[]
    for b in range(args.batch):
        [x,ws,hs,imgname]=rand_image(args,images,0)
        names.append(str(imgname))
        X[b,:]=x

    print("Predict batch")
    ## get output maps
    Y=model.predict(X, args.batch)
    for y in Y:
        print(y.shape,np.max(y))
    ## Draw detections
    for b in range(args.batch):

        detect=[]
        k=0
        for y in Y:
            for my in range(y.shape[1]):
                for mx in range(y.shape[2]):
                    for mz in range(y.shape[3]):
                        if (y[b,my,mx,mz]>0.75):
                            z=4*(mz//catlen)
                            detect.append([my,mx,z,k,y[b,my,mx,mz]])
            k=k+1


        ## Select top
        detect=sorted(detect,key=lambda x: x[4],reverse=True)

        ## convert to image Boxes
        fname=args.tsdir+args.fprefix+str(names[b])+".jpg"
        [x,ws,hs]=load_image_as_numpy(args,fname)

        tot=min(200,len(detect))
        boxes=np.zeros((tot,5))
        #x1,y1,x2,y2
        for i in range(tot):
            y=detect[i][0]
            x=detect[i][1]
            z=detect[i][2]
            k=detect[i][3]
            boxes[i,0]=A[k][y,x,z]/ws
            boxes[i,1]=A[k][y,x,z+1]/hs
            boxes[i,2]=A[k][y,x,z+2]/ws
            boxes[i,3]=A[k][y,x,z+3]/hs
            boxes[i,4]=detect[i][4]


        ## non-maximum supression
        #boxes=non_max_suppression_fast(boxes, 0.5)

        ## Draw selected
        print(fname)
        img=Image.open(fname)
        draw=ImageDraw.Draw(img)
        for box in boxes:
            draw.rectangle((box[0],box[1],box[2],box[3]), fill=None)


        fname=args.tsdir+args.fprefix+str(names[b])+"ANOT"+".jpg"
        img.save(fname)















######


















            #########
