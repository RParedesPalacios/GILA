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
        [x,ws,hs,img,imgname]=rand_image(args,images,0)
        names.append(str(imgname))
        X[b,:]=x

    print("Predict batch")
    ## get output maps
    Y=model.predict(X, args.batch)
    for y in Y:
        print(y.shape,np.max(y))
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
                            c=c+1
                            z=4*(mz//catlen)
                            draw.rectangle(((A[k][my,mx,z]/ws,A[k][my,mx,z+1]/hs), (A[k][my,mx,z+2]/ws,A[k][my,mx,z+3]/hs)), fill=None)

            k=k+1
        print("Image",fname,"found",c,"boxes")

        fname=args.tsdir+args.fprefix+str(names[b])+"ANOT"+".jpg"
        img.save(fname)















######


















            #########
