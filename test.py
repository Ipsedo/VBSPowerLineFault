import pickle as pkl
from preprocess.load_data import *
from sklearn.metrics import matthews_corrcoef as mcc
from tqdm import tqdm
import torch as th
from torchnet.meter import AUCMeter

if __name__ == "__main__":
    saved_model_path = "./res/saved_models/"
    model_file_name = "model_class_repartition_best.pkl"

    print("Loading model...")
    model_data = pkl.load(open(saved_model_path + model_file_name, "rb"))

    model = model_data["model"]
    loss_fn = model_data["criterion"]
    optim = model_data["optim"]

    nb_data_test = 100
    nb_canaux = 3

    data_root = "./data/"

    data = load_preprocess_data(data_root, 0, nb_data_test)

    signals = data["signals"]
    target = data["targets"]

    res = np.zeros(target.shape)
    auc_c0 = AUCMeter()
    auc_c1 = AUCMeter()
    auc_c2 = AUCMeter()

    print("Testing model...")
    model.eval()
    for i in tqdm(range(nb_data_test)):
        out = model(th.Tensor(signals[None, i, :, :])).detach().numpy()

        auc_c0.add(out[None, 0, 0], target[None, i, 0])
        auc_c1.add(out[None, 0, 1], target[None, i, 1])
        auc_c2.add(out[None, 0, 2], target[None, i, 2])

        res[i, 0] = 1 if out[0, 0] > 0.5 else -1
        res[i, 1] = 1 if out[0, 1] > 0.5 else -1
        res[i, 2] = 1 if out[0, 2] > 0.5 else -1

    target = np.where(target == 1, 1, -1)

    mcc_canal_0 = mcc(target[:, 0], res[:, 0])
    mcc_canal_1 = mcc(target[:, 1], res[:, 1])
    mcc_canal_2 = mcc(target[:, 2], res[:, 2])

    print("\nMCC")
    print("Canal 0 : %d" % (mcc_canal_0,))
    print("Canal 1 : %d" % (mcc_canal_1,))
    print("Canal 2 : %d" % (mcc_canal_2,))

    print("\nROC AUC")
    print("Canal 0 %f" % (auc_c0.value()[0]))
    print("Canal 1 %f" % (auc_c1.value()[0]))
    print("Canal 2 %f" % (auc_c2.value()[0]))



