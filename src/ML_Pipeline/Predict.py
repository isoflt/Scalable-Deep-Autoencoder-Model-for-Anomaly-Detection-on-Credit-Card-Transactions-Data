import numpy as np


def init(test_data, model, columns):
    # Preprocess test data to do categorical encoding and match columns expected
    # by the model
    columns = list(columns)
    columns.remove("Class")
    test_data = test_data[columns]
    test_data = test_data.loc[:, ~test_data.columns.duplicated()]

    x_test = test_data.values
    predict = model.predict(x_test)

    difference_array = np.subtract(predict, x_test)
    squared_array = np.square(difference_array)
    mse = squared_array.mean()
    return mse
