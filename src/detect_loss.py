import tensorflow as tf
import keras.backend as K

def get_pos_neg_cross(y_true,y_pred):

    cat=tf.shape(y_true)[2]
    zero=tf.constant(0, dtype=tf.float32)

    ## reshape to 1D vectors the positive part
    yt=tf.reshape(y_true,[-1,cat])
    yp=tf.reshape(y_pred,[-1,cat])

    ## Count positives and define neg as pos
    pos=tf.cast(tf.count_nonzero(yt[:,:cat-2]),dtype=tf.int32)

    ## Gather postives
    indpos=tf.where(tf.equal(yt[:,cat-1], 0.0))
    indpos=tf.cast(tf.reshape(indpos,[-1]),dtype=tf.int32)
    yp_p=tf.gather(yp,indpos)
    yt_p=tf.gather(yt,indpos)

    ## Gather postives
    indpos=tf.where(tf.not_equal(yt[:,cat-1], 0.0))
    indpos=tf.cast(tf.reshape(indpos,[-1]),dtype=tf.int32)
    yp_n=tf.gather(yp,indpos)
    yt_n=tf.gather(yt,indpos)

    neg=tf.maximum(1,3*pos)
    neg=tf.cast(neg,dtype=tf.int32)
    _,ind=tf.nn.top_k(tf.negative(yp_n[:,cat-1]),neg,sorted=True)
    ind=tf.cast(tf.reshape(ind,[-1]),dtype=tf.int32)
    yp_n=tf.gather(yp_n,ind)
    yt_n=tf.gather(yt_n,ind)


    return yp_p,yp_n,yt_p,yt_n

def get_pos_neg_log(y_true,y_pred):

    cat=tf.shape(y_true)[2]
    zero=tf.constant(0, dtype=tf.float32)

    ## reshape to 1D vectors the positive part
    yt=tf.reshape(y_true[:,:,:cat-2],[-1])
    yp=tf.reshape(y_pred[:,:,:cat-2],[-1])

    ## Count positives and define neg as pos
    pos=tf.cast(tf.count_nonzero(yt),dtype=tf.int32)

    #pos=tf.Print(pos,[pos],"Pos=")
    #pos=tf.Print(pos,[pos],"Pos=")

    ## Gather postives
    indpos=tf.where(tf.not_equal(yt, zero))
    indpos=tf.cast(tf.reshape(indpos,[-1]),dtype=tf.int32)
    yp_p=tf.gather(yp,indpos)


    ## reshape to 1D vectors the negative part
    yt=tf.reshape(y_true[:,:,cat-1],[-1])
    yp=tf.reshape(y_pred[:,:,cat-1],[-1])

    ## Gather hard negatives
    indneg=tf.where(tf.not_equal(yt, zero))
    indneg=tf.cast(tf.reshape(indneg,[-1]),dtype=tf.int32)
    yp_n=tf.gather(yp,indneg)
    neg=tf.maximum(1,2*pos)
    neg=tf.cast(neg,dtype=tf.int32)
    yp_n,ind=tf.nn.top_k(tf.negative(yp_n),neg,sorted=True)

    yp_n=tf.negative(yp_n)
    return yp_p,yp_n



def hnm_loss(y_true,y_pred):
    ## we have to remove last softmax to use this loss
    #yp_p,yp_n,yt_p,yt_n=get_pos_neg_cross(y_true,y_pred)
    #return tf.losses.softmax_cross_entropy(yt_n,yp_n)+tf.losses.softmax_cross_entropy(yt_p,yp_p)

    yp_p,yp_n=get_pos_neg_log(y_true,y_pred)
    return -tf.reduce_mean(tf.log(yp_p))-tf.reduce_mean(tf.log(yp_n))


def acc_pos(y_true, y_pred):

    cat=tf.shape(y_true)[2]
    zero=tf.constant(0, dtype=tf.float32)

    ytrue=tf.reshape(y_true,[-1,cat])
    ypred=tf.reshape(y_pred,[-1,cat])

    ind=tf.where(tf.equal(ytrue[:,cat-1], zero))
    ind=tf.cast(tf.reshape(ind,[-1]),dtype=tf.int32)

    ytrue=tf.gather(ytrue,ind)
    ypred=tf.gather(ypred,ind)

    return K.mean(K.equal(K.argmax(ytrue, axis=-1), K.argmax(ypred, axis=-1)))

def acc_neg(y_true, y_pred):
    cat=tf.shape(y_true)[2]
    zero=tf.constant(0, dtype=tf.float32)

    ytrue=tf.reshape(y_true,[-1,cat])
    ypred=tf.reshape(y_pred,[-1,cat])

    ind=tf.where(tf.not_equal(ytrue[:,cat-1], zero))
    ind=tf.cast(tf.reshape(ind,[-1]),dtype=tf.int32)

    ytrue=tf.gather(ytrue,ind)
    ypred=tf.gather(ypred,ind)

    return K.mean(K.equal(K.argmax(ytrue, axis=-1), K.argmax(ypred, axis=-1)))




###
