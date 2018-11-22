import tensorflow as tf
import keras.backend as K

def hnm_loss(y_true,y_pred):

    ## reshape to 1D vectors
    yt=tf.reshape(y_true,[-1])
    yp=tf.reshape(y_pred,[-1])

    ## Count positives and define neg as pos
    pos=tf.cast(tf.count_nonzero(yt),dtype=tf.int32)
    zero=tf.constant(0, dtype=tf.float32)
    pos=tf.Print(pos,[pos],"Pos=")

    ## Gather postives
    indpos=tf.where(tf.not_equal(yt, zero))
    indpos=tf.cast(tf.reshape(indpos,[-1]),dtype=tf.int32)
    yp_p=tf.gather(yp,indpos)

    ## Gather hard negatives
    indneg=tf.where(tf.equal(yt, zero))
    indneg=tf.cast(tf.reshape(indneg,[-1]),dtype=tf.int32)
    yp_n=tf.gather(yp,indneg)
    neg=tf.maximum(1,3*pos)
    neg=tf.cast(neg,dtype=tf.int32)
    yp_n,ind=tf.nn.top_k(yp_n,neg,sorted=True)

    lenp=tf.size(yp_p)
    lenn=tf.size(yp_n)

    ## Concat predicted both
    yp_p=tf.Print(yp_p,[yp_p],"Pos:")
    yp_n=tf.Print(yp_n,[yp_n],"Neg:")
    myp=tf.concat([yp_p,yp_n],0)

    ## Define targets (pos 1s and neg 0s)
    yt1=tf.ones([lenp], tf.float32)
    yt2=tf.zeros([lenn], tf.float32)
    myt=tf.concat([yt1, yt2], 0)

    ln=tf.cast(lenn,dtype=tf.float32)
    lp=tf.cast(lenp,dtype=tf.float32)

    #return tf.reduce_mean(tf.square(myt - myp))
    #return tf.reduce_mean(yt1-yp_p)+tf.reduce_mean(yp_n)
    return tf.reduce_mean(yp_n)

def num_pos(y_true, y_pred):
    ## reshape to 1D vectors
    yt=tf.reshape(y_true,[-1])
    yp=tf.reshape(y_pred,[-1])

    ## Count positives and define neg as pos
    pos=tf.cast(tf.count_nonzero(yt),dtype=tf.int32)
    if (pos==0):
        return 0.0
    else:
        zero=tf.constant(0, dtype=tf.float32)
        ## Gather postives
        indpos=tf.where(tf.not_equal(yt, zero))
        indpos=tf.cast(tf.reshape(indpos,[-1]),dtype=tf.int32)
        yp_p=tf.gather(yp,indpos)

        indneg=tf.where(tf.equal(yt, zero))
        indneg=tf.cast(tf.reshape(indneg,[-1]),dtype=tf.int32)
        yp_n=tf.gather(yp,indneg)

        neg=tf.maximum(1,3*pos)
        neg=tf.cast(neg,dtype=tf.int32)
        yp_n,ind=tf.nn.top_k(yp_n,neg,sorted=True)

        return tf.reduce_mean(yp_p)-tf.reduce_mean(yp_n)








###
