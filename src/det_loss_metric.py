import keras.backend as K

def ssd_loss(mapT,mapP):

    #depth=anchors*(cat+1)

    #for m in mapT:
