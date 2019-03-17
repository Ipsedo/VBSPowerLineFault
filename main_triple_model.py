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
    nb_data_test = 100
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

    models = []
    for i in range(nb_canaux):
        model = SmallConvModel()
        loss_fn = nn.BCELoss()

        model.cuda()
        loss_fn.cuda()

        optim = th.optim.Adagrad(model.parameters(), lr=1e-4)

        models.append({"model": model, "criterion": loss_fn, "optim": optim})

    losses = []
    roc_aucs = []

    for e in range(nb_epoch):
        sum_loss = 0

        for c in range(nb_canaux):
            models[c]["model"].train()

        for i in tqdm(range(nb_batch)):
            i_min = i * batch_size
            i_max = (i + 1) * batch_size
            i_max = nb_data_train if i_max > nb_data_train else i_max

            x, y = train_x[i_min:i_max, :, :], train_y[i_min:i_max, :]
            x, y = th.Tensor(x).cuda(), th.Tensor(y).type(th.float).cuda()

            for c in range(nb_canaux):
                out = models[c]["model"](x).view(-1)
                loss = models[c]["criterion"](out, y[:, c])

                loss.backward()
                models[c]["optim"].step()

                sum_loss += loss.item() / 3

        sys.stdout.flush()

        losses.append(sum_loss / nb_batch)
        print("Epoch %d, loss = %f" % (e, sum_loss / nb_batch))
        sys.stdout.flush()

        auc_meters = {i: AUCMeter() for i in range(nb_canaux)}

        for c in range(nb_canaux):
            models[c]["model"].eval()

        batch_size_test = 2
        nb_batch_test = int(nb_data_test / batch_size_test)
        for i in tqdm(range(nb_batch_test)):
            i_min = i * batch_size_test
            i_max = (i + 1) * batch_size_test
            i_max = nb_data_test if i_max > nb_data_test else i_max

            x, y = test_x[i_min:i_max, :, :], test_y[i_min:i_max, :]
            x, y = th.Tensor(x).cuda(), th.Tensor(y).type(th.float)

            for c in range(nb_canaux):
                out = models[c]["model"](x).detach().cpu()

                auc_meters[c].add(out, y[:, c])

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

    for i in range(nb_canaux):
        models[i]["model"].cpu()
        models[i]["criterion"].cpu()

    pkl.dump(models, open("triple_model_class_repartition.pkl", "wb"))

