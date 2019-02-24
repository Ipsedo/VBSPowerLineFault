import pandas as pd
import pickle as pkl
import numpy as np

if __name__ == "__main__":
    data_root = "./data/"
    train_meta_df = pd.read_csv(data_root + 'metadata_train.csv')
    label = train_meta_df.values[:, 3].reshape(-1, 3)

    positives = np.argwhere((label[:, 0] == 1) | (label[:, 1] == 1) | (label[:, 2] == 1))
    negatives = np.argwhere((label[:, 0] == 0) & (label[:, 1] == 0) & (label[:, 2] == 0))

    file_name = "positive_numpy_array.pkl"
    pkl.dump(positives, open(data_root + file_name, "wb"))

    file_name = "negative_numpy_array.pkl"
    pkl.dump(negatives, open(data_root + file_name, "wb"))
