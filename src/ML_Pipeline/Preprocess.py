import pandas as pd

# Function to cleanup data, convert to numerical variables and impute wherever required
def cleanup(data):
    data = data.fillna(data.mean())
    data = data.drop("Timestamp", axis=1)
    return data


# Function to normalize data between 0 and 1
def normalize(data, is_train, output_dir='../output'):
    min_val = data.min()
    max_val = data.max()

    if is_train:
        min_val.to_pickle(f"{output_dir}/min_val.pkl")
        max_val.to_pickle(f"{output_dir}/max_val.pkl")
    else:
        min_val = pd.read_pickle(f"{output_dir}/min_val.pkl")
        max_val = pd.read_pickle(f"{output_dir}/max_val.pkl")

    normalized_df = (data - min_val) / (max_val - min_val)
    return normalized_df

# Function to call dependent functions
def apply(data, is_train):
    print("Preprocessing started....")

    data = cleanup(data)
    print("Data cleanup completed....")

    data = normalize(data, is_train)
    print("Normalization completed....")

    data = data.loc[:, ~data.columns.duplicated()]

    print("Preprocessing completed....")
    return data
