import tensorflow as tf
import keras.backend as K

def get_pos_neg(y_true,y_pred):
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
    hnm=2.0
    neg=tf.maximum(1,tf.to_int32(hnm*tf.to_float(pos)))
    neg=tf.cast(neg,dtype=tf.int32)
    yp_n,ind=tf.nn.top_k(yp_n,neg,sorted=False)

    return yp_p,yp_n


def hnm_loss(y_true,y_pred):
    yp_p,yp_n=get_pos_neg(y_true,y_pred)

    lenp=tf.size(yp_p)
    lenn=tf.size(yp_n)

    ## Define targets (pos 1s and neg 0s)
    yt_p=tf.ones([lenp], tf.float32)
    yt_n=tf.zeros([lenn], tf.float32)

    myp=tf.concat([yp_p,yp_n],0)
    myt=tf.concat([yt_p,yt_n],0)
    # loss
    return tf.reduce_mean(tf.square(myt-myp))


def dif_pos_neg(y_true, y_pred):
    yp_p,yp_n=get_pos_neg(y_true,y_pred)
    return tf.reduce_mean(yp_p)-tf.reduce_mean(yp_n)

def score_pos(y_true, y_pred):
    yp_p,_=get_pos_neg(y_true,y_pred)
    return tf.reduce_mean(yp_p)

def score_neg(y_true, y_pred):
    _,yp_n=get_pos_neg(y_true,y_pred)
    return tf.reduce_mean(yp_n)



###
