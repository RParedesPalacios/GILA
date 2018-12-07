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
        model=load_from_disk(args.load_model,hnm_loss,acc_pos,acc_neg)


    if (args.summary==True):
        model.summary()

    [images,imglen,boxes,boxlen,catdict,catlen]=load_annot_json(args.tsannot)

    lanchors=len(args.anchors)//2
    outputs=model.outputs
    maps=[]
    for layer in model.layers:
        maps.append(layer.output)
        if ("re_lu" in layer.name):
            maps=[]
        if ("reshape" in layer.name):
            break

    dmaps=[]
    for m in maps:
        if (min(m.shape[1],m.shape[2])>=args.minmap)and(max(m.shape[1],m.shape[2])<=args.maxmap):
            dmaps.append(m)
    print(dmaps)



    A=build_anchors(args,dmaps)


    [X,Y]=buil_XY(args,dmaps)



    names=[]
    for b in range(args.batch):
        [x,ws,hs,imgname]=rand_image(args,images,0)
        names.append(str(imgname))
        X[b,:]=x

    print("Predict batch")
    ## get output maps
    OY=model.predict(X, args.batch)


    for y in Y:
        print(y.shape)

    print(OY.shape)


    #sys.exit(1)
    ## Draw detections
    for b in range(args.batch):
        detect=[]
        k=0
        ant=0

        for y in Y:
            act=0
            block=y.shape[3]//catlen
            for my in range(y.shape[1]):
                for mx in range(y.shape[2]):
                    for mz in range(y.shape[3]):
                        act=act+1
                        c=mz%catlen
                        d=mz//catlen
                    #    if (y[b,my,mx,mz]>0.5):
                        if (OY[b,ant+my*(y.shape[2]*block)+mx*block+d,c]>0.5):
                            if (mz%catlen!=(catlen-1)): ## not background class
                                z=4*(mz//catlen)
                                detect.append([my,mx,z,k,y[b,my,mx,mz]])

            ant=ant+(act//catlen)
            k=k+1


        ## Select top
        detect=sorted(detect,key=lambda x: x[4],reverse=True)

        ## convert to image Boxes
        fname=args.tsdir+args.fprefix+str(names[b])+".jpg"
        [x,ws,hs]=load_image_as_numpy(args,fname)

        tot=min(2000,len(detect))
        boxes=np.zeros((tot,5))
        #x1,y1,x2,y2
        i=0
        for d in detect:
            y=d[0]
            x=d[1]
            z=d[2]
            k=d[3]
            boxes[i,0]=A[k][y,x,z]/ws
            boxes[i,1]=A[k][y,x,z+1]/hs
            boxes[i,2]=A[k][y,x,z+2]/ws
            boxes[i,3]=A[k][y,x,z+3]/hs
            boxes[i,4]=d[4]
            i=i+1

        ## non-maximum supression
        #boxes=non_max_suppression_fast(boxes, 0.5)

        ## Draw selected
        print(fname)
        img=Image.open(fname)
        draw=ImageDraw.Draw(img)
        for box in boxes:
            draw.rectangle((box[0],box[1],box[2],box[3]), fill=None)


        fname=args.tsdir+args.fprefix+str(names[b])+"ANNOT"+".jpg"
        img.save(fname)















######


















            #########
