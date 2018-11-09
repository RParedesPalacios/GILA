import tensorflow as tf
import keras.backend as K

def hnm_loss(y_true,y_pred):

    ## reshape to 1D vectors
    yt=tf.reshape(y_true,[-1])
    yp=tf.reshape(y_pred,[-1])

    ## Count positives and define neg as 3*pos
    pos=tf.cast(tf.count_nonzero(yt),dtype=tf.int32)
    neg=3*pos
    neg=tf.cast(neg,dtype=tf.int32)

    zero=tf.constant(0, dtype=tf.float32)
    ## Gather postives
    indpos=tf.where(tf.not_equal(yt, zero))
    indpos=tf.cast(tf.reshape(indpos,[-1]),dtype=tf.int32)
    yp_p=tf.gather(yp,indpos)

    ## Gather hard negatives
    indneg=tf.where(tf.equal(yt, zero))
    indneg=tf.cast(tf.reshape(indneg,[-1]),dtype=tf.int32)
    yp_n=tf.gather(yp,indneg)
    yp_n,ind=tf.nn.top_k(yp_n,neg)
    mask=tf.greater(yp_n, 0.5)
    yp_n=tf.boolean_mask(yp_n, mask)

    ##Concat predicted both
    yp=tf.concat([yp_p,yp_n],0)

    ## Define targets (pos 1s and neg 0s)
    yt1=tf.ones([tf.size(yp_p)], tf.float32)
    yt2=tf.zeros([tf.size(yp_n)], tf.float32)
    yt=tf.concat([yt1, yt2], 0)

    anchor_loss = tf.reduce_mean(tf.square(yt - yp))

    return anchor_loss



def num_pos(y_true, y_pred):
    ## reshape to 1D vectors
    yt=tf.reshape(y_true,[-1])
    yp=tf.reshape(y_pred,[-1])


    ## Count positives and define neg
    pos=tf.cast(tf.count_nonzero(yt),dtype=tf.int32)
    neg=pos
    neg=tf.cast(neg,dtype=tf.int32)

    zero=tf.constant(0, dtype=tf.float32)
    ## Gather postives
    indpos=tf.where(tf.not_equal(yt, zero))
    indpos=tf.cast(tf.reshape(indpos,[-1]),dtype=tf.int32)
    yp_p=tf.gather(yp,indpos)

    ## Gather hard negatives
    indneg=tf.where(tf.equal(yt, zero))
    indneg=tf.cast(tf.reshape(indneg,[-1]),dtype=tf.int32)
    yp_n=tf.gather(yp,indneg)
    yp_n,ind=tf.nn.top_k(yp_n,neg)

    return tf.reduce_mean(yp_p)/tf.reduce_mean(yp_n)
