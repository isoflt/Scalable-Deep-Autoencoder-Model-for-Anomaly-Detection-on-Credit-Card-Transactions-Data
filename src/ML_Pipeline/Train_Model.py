from . import Utils
from tensorflow import keras
from tensorflow.keras import layers


# Function to train ML model
def train(model, x_train, y_train, batch_size=128, epochs=20):
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    return model


# Function to initiate model and training data
def fit(data, loss="mse",learning_rate=0.01):
    columns = data.columns

    x_train = data.drop(Utils.TARGET, axis=1).values
    y_train = data[Utils.TARGET].values

    print(x_train.shape, y_train.shape)

    model = keras.Sequential(
        [
            keras.Input(shape=(13,)),
            layers.Dense(13, activation="relu"),
            layers.Dense(6, activation="relu"),
            layers.Dense(6, activation="relu"),
            layers.Dense(13, activation="linear"),
        ]
    )
    optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    model.compile(loss=loss, optimizer=optimizer)
    print(model.summary())

    model = train(model, x_train, x_train)

    return model, columns
