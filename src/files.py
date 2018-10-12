from keras.models import model_from_json

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
    fmodel_json = model.to_json()
    filename=args.save_model+".json"
    with open(filename, "w") as json_file:
        json_file.write(model_json)
    
    filename=args.save_model+".h5"
    model.save_weights(filename)
    print("Saved model to disk")


