from keras.models import model_from_json
from keras.models import load_model

def load_json_model(basename):
    filename=basename+".json"
    json_file = open(filename, 'r')
    model_json = json_file.read()
    json_file.close()

    model = model_from_json(model_json)

    filename=basename+".h5"
    model.load_weights(filename)
    print("Loaded model from disk")
    return model


def save_json_model(model,basename):
    model_json = model.to_json()
    filename=basename+".json"
    with open(filename, "w") as json_file:
        json_file.write(model_json)

    filename=basename+".h5"
    model.save_weights(filename)
    print("Saved model to disk")


def load_from_disk(basename,custom_loss=None,custom_metric=None):
    filename=basename+".h5"
    if (custom_loss==None):
        model=load_model(filename)
    else:
        model=load_model(filename,custom_objects={'hnm_loss': custom_loss,'num_pos':custom_metric})
    print("Loaded model from disk")
    return model


def save_to_disk(model,basename):
    filename=basename+".h5"
    model.save(filename)
    print("Saved model to disk")
