import keras
import pickle
import pandas as pd

PREDICTORS = ['Value', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12']

TARGET = ["Class"]


def save_model(model, columns, output_dir = "../output"):
    model.save(f"{output_dir}/deep-ae-model")

    file = open(f"{output_dir}/columns.mapping", "wb")
    pickle.dump(columns, file)
    file.close()

    return True


def load_model(model_path, output_dir ="../output" ):
    model = None
    try:
        model = keras.models.load_model(model_path)
    except:
        print("Please enter correct path")
        exit(0)

    file = open(f"{output_dir}/columns.mapping", "rb")
    columns = pickle.load(file)
    file.close()

    return model, columns

