#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 14:40:50 2019

@author: cabanal & berrien
"""
from models.conv_model import *
from preprocess.load_data import *
from torchnet.meter.aucmeter import AUCMeter
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle as pkl

if __name__ == "__main__":

    nb_data_train = 250
    nb_data_test = 50
    nb_canaux = 3

    data_root = "./data/"

    data = load_preprocess_data(data_root, nb_data_train, nb_data_test)

    signals = data["signals"]
    target = data["targets"]

    train_x = signals[:nb_data_train, :, :]
    train_y = target[:nb_data_train, :]

    test_x = signals[nb_data_train:nb_data_train + nb_data_test, :, :]
    test_y = target[nb_data_train:nb_data_train + nb_data_test, :]

    print(test_x.shape)
    print(test_y.shape)

    nb_epoch = 15
    batch_size = 1
    nb_batch = int(nb_data_train / batch_size)

    model = ConvModel()
    loss_fn = nn.BCELoss()

    model.cuda()
    loss_fn.cuda()

    optim = th.optim.Adagrad(model.parameters(), lr=1e-3)

    losses = []
    roc_aucs = []

    for e in range(nb_epoch):
        sum_loss = 0
        model.train()
        for i in tqdm(range(nb_batch)):
            i_min = i * batch_size
            i_max = (i + 1) * batch_size
            i_max = nb_data_train if i_max > nb_data_train else i_max

            x, y = train_x[i_min:i_max, :, :], train_y[i_min:i_max, :]
            x, y = th.Tensor(x).cuda(), th.Tensor(y).type(th.float).cuda().view(-1)

            out = model(x).view(-1)
            # weights = th.where(y == 1, th.Tensor([1e2]).cuda(), th.Tensor([1e-2]).cuda()).cuda()
            # loss_fn.weight = weights
            loss = loss_fn(out, y)

            loss.backward()
            optim.step()

            sum_loss += loss.item()

        sys.stdout.flush()

        losses.append(sum_loss / nb_batch)
        print("Epoch %d, loss = %f" % (e, sum_loss / nb_batch))
        sys.stdout.flush()

        auc_meters = {i: AUCMeter() for i in range(nb_canaux)}

        model.eval()
        batch_size_test = 2
        nb_batch_test = int(nb_data_test / batch_size_test)
        for i in tqdm(range(nb_batch_test)):
            i_min = i * batch_size_test
            i_max = (i + 1) * batch_size_test
            i_max = nb_data_test if i_max > nb_data_test else i_max

            x, y = test_x[i_min:i_max, :, :], test_y[i_min:i_max, :]
            x, y = th.Tensor(x).cuda(), th.Tensor(y).type(th.float)

            out = model(x).detach().cpu()

            for c in range(nb_canaux):
                auc_meters[c].add(out[:, c], y[:, c])

        sys.stdout.flush()

        res_roc_auc = []

        for c in range(nb_canaux):
            print("Canal(%d), roc auc = %f" % (c, auc_meters[c].value()[0]))
            res_roc_auc.append(auc_meters[c].value()[0])

        roc_aucs.append(res_roc_auc)
        print("")
        sys.stdout.flush()

    plt.plot(losses, c="b", label="loss values")
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.title("Loss values during train")
    plt.legend()
    plt.show()

    roc_aucs = np.asarray(roc_aucs)

    plt.plot(roc_aucs[:, 0], c="b", label="Canal 1")
    plt.plot(roc_aucs[:, 1], c="r", label="Canal 2")
    plt.plot(roc_aucs[:, 2], c="g", label="Canal 3")
    plt.xlabel("Epoch")
    plt.ylabel("ROC AUC")
    plt.title("ROC AUC values on dev set")
    plt.legend()
    plt.show()

    to_save = {"model": model.cpu(), "criterion": loss_fn.cpu(), "optim": optim}
    pkl.dump(to_save, open("model_class_repartition.pkl", "wb"))

"""
donnees = pq.read_pandas(nom_fichier, columns=[str(i) for i in range(9)]).to_pandas()

donnees = donnees.T

longueur = 1000

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5), sharey=True)
for i in range(3):
    sns.lineplot(x=donnees.columns[:longueur], y=donnees.iloc[i, :longueur],
                ax=ax1, label=["phase:" + str(train_meta_df.iloc[i, :].phase)])

liste_fd = [i for i, j in enumerate(train_meta_df['target']) if j == 1]
liste_nd = [i for i, j in enumerate(train_meta_df['target']) if j == 0]
"""

"""
signals = np.array(signals).T.reshape((999//3, 3, 800000))
filtered = filter_signal(signals[0, 0, :], threshold=1e3)

plt.figure(figsize=(15, 10))
plt.plot(signals[0, 0, :], label='Raw')
plt.plot(filtered, label='Filtered')
plt.legend()
plt.title("FFT Denoising with threshold = 1e3", size=15)
plt.show()
"""


"""
def filter_signal(signal, threshold=1e8):
    fourier = rfft(signal)
    frequencies = rfftfreq(signal.size, d=20e-3/signal.size)
    fourier[frequencies > threshold] = 0
    return irfft(fourier)


def applat(signal, seuil):
    return signal - filter_signal(signal, threshold=seuil)


def select_1(signal, coef):
    ecart_type = np.median(np.absolute(signal))
    signal[np.absolute(signal) < coef * ecart_type] = 0
    return signal


def select_2(signal, nombre):
    ecart = np.sort(np.absolute(signal))[-nombre]
    signal[np.absolute(signal) < ecart] = 0
    return signal


def filtre_decharge(signal, largeur):
    suppression = signal == 200
    
    for i in range(largeur):
        positif = signal > 0
        negatif = signal < 0
        j = i + 1
        a_supprimer = (positif[j:] & negatif[:-j]) | (positif[:-j] & negatif[j:])
        suppression[:-j][a_supprimer] = True
        suppression[j:][a_supprimer] = True
    
    signal[suppression] = 0
    return signal


def combinaison(signal, j):
    return select_2(filtre_decharge(select_1(applat(signal, 1e3), 2), j), nombre=100)


def fourier_ponctuel(signal, seuil):
    points = np.argwhere(np.absolute(signal) > seuil)
    temps = np.linspace(0, np.pi, num=100)
    points = points.reshape(-1, 1)
    temps = temps.reshape(1, -1)
    pt = np.matmul(points, temps)
    amplitude = np.sqrt(np.sum(np.cos(pt), axis=0) ** 2 + np.sum(np.sin(pt), axis=0) ** 2)
    
    return amplitude


i = 0
plt.figure(figsize=(15, 10))
plt.plot(applat(signals[liste_nd[i], :100], seuil=1e3), label='Applati n° {}, ND'.format(liste_nd[i]))
plt.legend()
plt.title("Signal réduit", size=15)
plt.show()

for i in range(6):
        plt.figure(figsize=(15, 10))
        plt.plot(applat(signals[liste_nd[i], :], seuil=1e3), label='Applati n° {}, ND'.format(liste_nd[i]))
        plt.legend()
        plt.title("Signal réduit", size=15)
        plt.show()
        
        for j in range(5):
            plt.figure(figsize=(15, 10))
            plt.plot(combinaison(signals[liste_nd[i], :], j), label='Applati n° {}, ND'.format(liste_nd[i]))
            plt.legend()
            plt.title("Signal filtré", size=15)
            plt.show()


for i in range(3):
        plt.figure(figsize=(15, 10))
        plt.plot(applat(signals[liste_fd[i], :], seuil=1e3), label='Applati n° {}, FD'.format(liste_fd[i]))
        plt.legend()
        plt.title("Signal réduit", size=15)
        plt.show()

        for j in range(5):
            plt.figure(figsize=(15, 10))
            plt.plot(combinaison(signals[liste_fd[i], :], j), label='Applati n° {}, FD'.format(liste_fd[i]))
            plt.legend()
            plt.title("Signal filtré", size=15)
            plt.show()


plt.figure(figsize=(15, 10))
plt.plot(fourier_ponctuel(Applat(signals[liste_nd[9], :], 1e3), 2), label='Frequence n° {}, FD'.format(liste_fd[0]))
plt.legend()
plt.title("Signal filtré", size=15)
plt.show()


# AA, PT = fourier_ponctuel(applat(signals[liste_fd[0], :], 1e3), 8)

"""
