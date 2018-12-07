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

    #yp_p = tf.maximum(yp_p, 1e-15)
    #yp_n = tf.maximum(yp_n, 1e-15)
    return -tf.reduce_mean(yp_p)+tf.reduce_mean(yp_n)

    #return -tf.log(yp_p)+tf.log(yp_n)

    #return tf.reduce_mean(yp_n)-tf.reduce_mean(yp_p)

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
