import sys
from tqdm import tqdm
import pandas as pd
import pyarrow.parquet as pq
from preprocess.frequencies import *
import pickle as pkl


def load_preprocess_data(data_root, nb_data_train, nb_data_test):
    nom_fichier = data_root + 'train.parquet'

    print("Loading data...")
    # just load train/test data
    train_meta_df = pd.read_csv(data_root + 'metadata_train.csv')
    # test_meta_df = pd.read_csv(data_root + 'metadata_test.csv')
    # set index, it makes the data access much faster
    # train_meta_df = train_meta_df.set_index(['id_measurement', 'phase'])

    nb_canaux = 3

    positives = pkl.load(open(data_root + "positive_numpy_array.pkl", "rb"))
    negatives = pkl.load(open(data_root + "negative_numpy_array.pkl", "rb"))

    np.random.shuffle(positives)
    np.random.shuffle(negatives)

    ratio_positive = 1 / 4
    nb_positive = int((nb_data_test + nb_data_train) * ratio_positive)
    nb_negative = nb_data_train + nb_data_test - nb_positive

    to_select = []
    for i in range(nb_positive):
        to_select.append(positives[i] * 3)
        to_select.append(positives[i] * 3 + 1)
        to_select.append(positives[i] * 3 + 2)
    for i in range(nb_negative):
        to_select.append(negatives[i] * 3)
        to_select.append(negatives[i] * 3 + 1)
        to_select.append(negatives[i] * 3 + 2)

    to_select = list(np.sort(np.asarray(to_select)).reshape(-1))

    signals = pq.read_table(nom_fichier, columns=[str(i) for i in to_select]).to_pandas()
    signals = np.array(signals)

    signals = signals.T.reshape((nb_data_train + nb_data_test, nb_canaux, 800000))

    print("Converting signal to float...")
    signals = signals.astype(np.float)

    print("Processing signals...")
    sys.stdout.flush()

    for d in tqdm(range(nb_data_train + nb_data_test)):
        for c in range(nb_canaux):
            signals[d, c, :] = normalize(filter_signal(signals[d, c, :], 1e8))

    sys.stdout.flush()
    print("Signal processed !")

    print("signal shape :", signals.shape)
    print(signals.dtype)
    print(train_meta_df.shape)

    target = train_meta_df.values[:(nb_data_train + nb_data_test) * nb_canaux, 3]
    target = target.reshape(nb_data_train + nb_data_test, nb_canaux)

    return {"signals": signals, "targets": target}
