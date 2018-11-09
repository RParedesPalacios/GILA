from PIL import Image, ImageOps
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

    [images,imglen,boxes,boxlen,catdict,catlen]=load_annot_json(args.trannot)


    ## X,Y for trainig (data, targets) and A foro anchors
    lanchors=len(args.anchors)//2
    A=build_anchors(args,maps)

    ## Build X,Y
    [X,Y]=buil_XY(args,maps)

    output_dict={}
    k=0
    for m in maps:
     output_dict.update({m.name.replace('/Sigmoid:0',''):Y[k]})
     k=k+1

    ## Provide images and achors fitting with iou>0.5
    print("Start Generator....")
    while True:

        for y in Y:
            y[:]=0.0

        # to debug and check
        tot=0
        match=0
        v=np.zeros(len(Y))

        for b in range(args.batch):

            [x,ws,hs,imgname]=rand_image(args,images)
            X[b,:]=x

            ## Load annotation of image, codification
            ## w.r.t an image of (args.height x args.width)
            anot=[]
            for all in boxes:
                 if (all['image_id']==imgname):
                    x,y,w,h=all['bbox']
                    anot.append([catdict[all['category_id']],x*ws,y*hs,(x+w)*ws,(y+h)*hs])
                    #cat,x1,y1,x2,y2

            for an in anot:
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

                    #shift to search for neighborhood cells to place anchors
                    shift=1
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
                                            oclass=int(an[0])
                                            y[b,my+sy,mx+sx,(j*catlen)+oclass]=1.0
                                            v[k]=v[k]+1
                                        i=i+4
                    k=k+1


                if (setanchor==True):
                    match=match+1



        mpc=float(100*match)/float(tot)
        if (mpc<50):
            print("")
            print("Warning: few gt boxes matched= %d %d %.2f%%" %(match,tot,mpc))


        # print("----------------------")
        # print("Total",tot)
        # print("Match",match)
        # k=0
        # for y in Y:
        #     print(k,":",v[k],np.count_nonzero(Y[k]),np.sum(Y[k]))
        #     k=k+1
        # print("----------------------")



        yield (X,output_dict)





        ######


















            #########
