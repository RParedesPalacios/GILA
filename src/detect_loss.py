import tensorflow as tf

def hnm_loss(y_true,y_pred):

    ## Count positives just to define neg as 3*pos
    yt=tf.reshape(y_true,[tf.size(y_true)])
    pos=tf.cast(tf.count_nonzero(yt)+1,dtype=tf.int32)
    neg=3*pos

    zero=tf.constant(0, dtype=tf.float32)
    indpos=tf.where(tf.not_equal(yt, zero))
    indpos=tf.cast(tf.reshape(indpos,[tf.size(indpos)]),dtype=tf.int32)
    #indpos could be empty in some target maps

    indneg=tf.where(tf.equal(yt, zero))
    indneg=tf.cast(tf.reshape(indneg,[tf.size(indneg)]),dtype=tf.int32)

    yp=tf.reshape(y_pred,[tf.size(y_pred)])
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

    pos=tf.cast(pos,dtype=tf.float32)
    neg=tf.cast(neg,dtype=tf.float32)
    anchor_loss = tf.div(tf.reduce_sum(tf.square(ytg - ypg)),(pos+neg))

    #let the magic happen
    return anchor_loss
