import tensorflow as tf
import keras.backend as K

def hnm_loss(y_true,y_pred):

    ## reshape to 1D vectors
    yt=tf.reshape(y_true,[-1])
    yp=tf.reshape(y_pred,[-1])

    ## Count positives
    pos=tf.cast(tf.count_nonzero(yt),dtype=tf.int32)
    ## split yp into positives and negatives
    mask=tf.to_int32(tf.greater(yt,0))
    data=tf.dynamic_partition(yp,mask,2)

    yp_n=data[0]
    yp_p=data[1]

    # hard negatives 
    neg=tf.maximum(1,3*pos)
    neg=tf.cast(neg,dtype=tf.int32)
    yp_n,ind=tf.nn.top_k(yp_n,neg,sorted=False)

    lenp=tf.size(yp_p)
    #lenn=tf.size(yp_n)

    #lenp=tf.Print(lenp,[lenp],"Pos:")
    #lenn=tf.Print(lenn,[lenn],"Neg:")

    ## Define targets (pos 1s and neg 0s)
    yt1=tf.ones([lenp], tf.float32)
    #yt2=tf.zeros([lenn], tf.float32)

    # loss
    return tf.reduce_mean(yt1-yp_p)+tf.reduce_mean(yp_n)

    #return tf.reduce_mean(yp_n)-tf.reduce_mean(yp_p) 

def dif_pos_neg(y_true, y_pred):
    ## reshape to 1D vectors
    yt=tf.reshape(y_true,[-1])
    yp=tf.reshape(y_pred,[-1])

    ## Count positives and define neg as pos
    pos=tf.cast(tf.count_nonzero(yt),dtype=tf.int32)
    if (pos==0):
        return 0.0
    else:
        mask=tf.to_int32(tf.greater(yt,0))
        data=tf.dynamic_partition(yp,mask,2)
        yp_n=data[0]
        yp_p=data[1]
        
        neg=tf.maximum(1,3*pos)
        neg=tf.cast(neg,dtype=tf.int32)
        yp_n,ind=tf.nn.top_k(yp_n,neg,sorted=False)
        
        return tf.reduce_mean(yp_p)-tf.reduce_mean(yp_n)






###
