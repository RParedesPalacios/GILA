import tensorflow as tf
import keras.backend as K

def get_pos_neg(y_true,y_pred):

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
    neg=tf.maximum(1,3*pos)
    neg=tf.cast(neg,dtype=tf.int32)
    yp_n,ind=tf.nn.top_k(tf.negative(yp_n),neg,sorted=True)

    yp_n=tf.negative(yp_n)
    return yp_p,yp_n

def hnm_loss(y_true,y_pred):
    #return tf.nn.softmax_cross_entropy_with_logits_v2(labels = y_true, logits = y_pred)
    ## reshape to 1D vectors

    yp_p,yp_n=get_pos_neg(y_true,y_pred)

    #
    # cat=tf.shape(y_true)[2]
    # zero=tf.constant(0, dtype=tf.float32)
    #
    # y_true=tf.reshape(y_true,[-1,cat])
    # y_pred=tf.reshape(y_pred,[-1,cat])
    #
    # neg_true=y_true[:,cat-1]
    # pos_true=y_true[:,0:cat-2]
    #
    # neg_pred=y_pred[:,cat-1]
    # pos_pred=y_pred[:,0:cat-2]
    #
    # ##positive and negatives indices
    # p_true=tf.reduce_max(pos_true,1)
    # indpos=tf.where(tf.not_equal(p_true, zero))
    # indneg=tf.where(tf.equal(p_true, zero))
    #
    #
    # yp_p=tf.gather_nd(y_pred, [indpos])
    # yt_p=tf.gather_nd(y_true, [indpos])
    #
    # ##negative block
    # yp_n=tf.gather_nd(y_pred, [indneg])
    # yt_n=tf.gather_nd(y_true, [indneg])
    #
    # y_true=tf.concat([yt_p,yt_n],1)
    # y_pred=tf.concat([yp_p,yp_n],1)

    return -tf.reduce_mean(tf.log(yp_p))-tf.reduce_mean(tf.log(yp_n))

    #return -tf.losses.softmax_cross_entropy(onehot_labels=yt_p,logits=yp_p)
    #yp_p,yp_n=get_pos_neg(y_true,y_pred)

    #yp_p = tf.maximum(yp_p, 1e-15)
    #yp_n = tf.maximum(yp_n, 1e-15)
    #

    #return -tf.log(yp_p)+tf.log(yp_n)

    #return tf.reduce_mean(yp_n)-tf.reduce_mean(yp_p)

def dif_pos_neg(y_true, y_pred):
    yp_p,yp_n=get_pos_neg(y_true,y_pred)
    return tf.reduce_mean(yp_p)-tf.reduce_mean(yp_n)

def score_pos(y_true, y_pred):
    yp_p,yp_n=get_pos_neg(y_true,y_pred)
    return tf.reduce_mean(yp_p)

def score_neg(y_true, y_pred):
    yp_p,yp_n=get_pos_neg(y_true,y_pred)
    return tf.reduce_mean(yp_n)



###
