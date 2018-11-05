import tensorflow as tf
import keras.backend as K

def hnm_loss(y_true,y_pred):


    ## Count positives just to define neg as 3*pos
    yt=tf.reshape(y_true,[-1])
    pos=tf.cast(tf.count_nonzero(y_true),dtype=tf.int32)

    #neg=tf.maximum(pos,1)
    neg=pos
    neg=tf.cast(neg,dtype=tf.int32)

    zero=tf.constant(0, dtype=tf.float32)
    indpos=tf.where(tf.not_equal(y_true, zero))
    indpos=tf.cast(tf.reshape(indpos,[-1]),dtype=tf.int32)
    #indpos could be empty in some target maps

    indneg=tf.where(tf.equal(yt, zero))
    indneg=tf.cast(tf.reshape(indneg,[-1]),dtype=tf.int32)


    yp=tf.reshape(y_pred,[-1])

    ## Gather postives, could be empty
    ypgp=tf.gather(yp,indpos)

    ## Gather hard negatives
    ypgn=tf.gather(yp,indneg)
    ypgn,ind=tf.nn.top_k(ypgn,neg)

    ##Concat both
    ypg=tf.concat([ypgp,ypgn],0)
    indP=tf.concat([indpos,ind],0)

    ## Gather targets (pos 1s and neg 0s)
    ytg=tf.gather(yt,indP)

    anchor_loss = tf.reduce_mean(tf.square(ytg - ypg))

    #let the magic happen
    return anchor_loss



def num_pos(y_true, y_pred):
    return K.sum(y_true)
