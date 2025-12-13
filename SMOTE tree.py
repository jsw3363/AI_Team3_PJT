# stabled SMOTE with tree

import csv
import os
import random
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import tree
from sklearn.metrics import (
    roc_auc_score, recall_score, confusion_matrix,
    brier_score_loss, matthews_corrcoef
)

warnings.filterwarnings('ignore')
np.set_printoptions(suppress=True)


# 전역 설정
target_defect_ratio = 0.5
neighbor = 5
classifier = "tree"


# Stable SMOTE 
class stable_SMOTE:
    def __init__(self, z_nearest=5):
        self.z_nearest = z_nearest
    
    def fit_sample(self, x_dataset, y_dataset):
        x_dataset = pd.DataFrame(x_dataset)
        x_dataset = x_dataset.rename(
            columns={0:"wmc",1:"dit",2:"noc",3:"cbo",4:"rfc",5:"lcom",6:"ca",7:"ce",
                     8:"npm",9:"lcom3",10:"loc",11:"dam",12:"moa",13:"mfa",14:"cam",
                     15:"ic",16:"cbm",17:"amc",18:"max_cc",19:"avg_cc",20:"bug"}
        )

        defective_instance = x_dataset[x_dataset["bug"] > 0]
        clean_instance = x_dataset[x_dataset["bug"] == 0]

        defective_number = len(defective_instance)
        need_number = int((target_defect_ratio * len(x_dataset) - defective_number) /
                          (1 - target_defect_ratio))
        if need_number <= 0:
            return False

        total_pair = []
        generated_dataset = []

        number_on_each_instance = need_number / defective_number
        rround = number_on_each_instance / self.z_nearest

        while rround >= 1:
            for index, row in defective_instance.iterrows():
                temp = defective_instance.copy(deep=True)
                dist = ((row - temp) ** 2).sum(axis=1) ** 0.5
                temp["distance"] = dist
                neighbors = temp.sort_values("distance")[1:self.z_nearest+1]
                for a in neighbors.index:
                    total_pair.append(tuple(sorted([index, a])))
            rround -= 1

        need_number1 = need_number - len(total_pair)
        number_on_each_instance = need_number1 / defective_number

        for index, row in defective_instance.iterrows():
            temp = defective_instance.copy(deep=True)
            dist = ((row - temp) ** 2).sum(axis=1) ** 0.5
            temp["distance"] = dist
            neighbors = temp.sort_values("distance")[1:self.z_nearest+1]
            for a in neighbors.sort_values("distance", ascending=False)\
                              .head(int(number_on_each_instance)).index:
                total_pair.append(tuple(sorted([index, a])))

        residue = need_number - len(total_pair)
        residue_instance = defective_instance.sample(n=residue)

        for index, row in residue_instance.iterrows():
            temp = defective_instance.copy(deep=True)
            dist = ((row - temp) ** 2).sum(axis=1) ** 0.5
            temp["distance"] = dist
            neighbor = temp.sort_values("distance").iloc[-1]
            total_pair.append(tuple(sorted([index, neighbor.name])))

        for (i, j), cnt in Counter(total_pair).items():
            r1 = defective_instance.loc[i]
            r2 = defective_instance.loc[j]
            samples = np.linspace(r1, r2, cnt + 2)[1:-1]
            generated_dataset.extend(samples.tolist())

        gen_df = pd.DataFrame(generated_dataset, columns=x_dataset.columns)
        return pd.concat([clean_instance, defective_instance, gen_df])



# Bootstrap 분리
def separate_data(data):
    data = np.array(data)
    idx = np.random.randint(0, len(data), len(data))
    train_idx = np.unique(idx)
    test_idx = np.setdiff1d(np.arange(len(data)), train_idx)
    return data[train_idx], data[test_idx]



# 결과 저장 폴더
OUT_DIR = "output_data/SMOTE_tree/"
os.makedirs(OUT_DIR, exist_ok=True)

def open_writer(filename):
    f = open(OUT_DIR + filename, "w", newline="")
    w = csv.writer(f)
    w.writerow(
        ["inputfile"] + [""]*10 +
        ["min","lower","avg","median","upper","max","variance"]
    )
    return w


auc_writer = open_writer("5auc_smote_result_on_tree.csv")
balance_writer = open_writer("5balance_smote_result_on_tree.csv")
recall_writer = open_writer("5recall_smote_result_on_tree.csv")
pf_writer = open_writer("5pf_smote_result_on_tree.csv")
brier_writer = open_writer("5brier_smote_result_on_tree.csv")
mcc_writer = open_writer("5mcc_smote_result_on_tree.csv")

stable_auc_writer = open_writer("5auc_stable_smote_result_on_tree.csv")
stable_balance_writer = open_writer("5balance_stable_smote_result_on_tree.csv")
stable_recall_writer = open_writer("5recall_stable_smote_result_on_tree.csv")
stable_pf_writer = open_writer("5pf_stable_smote_result_on_tree.csv")
stable_brier_writer = open_writer("5brier_stable_smote_result_on_tree.csv")
stable_mcc_writer = open_writer("5mcc_stable_smote_result_on_tree.csv")



# 메인 실험 루프
for inputfile in os.listdir("input_data/"):
    print("Processing:", inputfile)
    print("Start:", time.asctime())

    dataset = pd.read_csv("input_data/" + inputfile)
    dataset = dataset.drop(columns=["name","version","name.1"])

    defect_ratio = len(dataset[dataset["bug"] > 0]) / len(dataset)
    if defect_ratio > 0.45:
        print(inputfile, " defect ratio larger than 0.45")
        continue

    dataset["bug"] = (dataset["bug"] > 0).astype(int)

    for col in dataset.columns:
        dataset[col] = (dataset[col] - dataset[col].min()) / (dataset[col].max() - dataset[col].min())

    for _ in range(10):
        auc_l, bal_l, rec_l, pf_l, bri_l, mcc_l = [], [], [], [], [], []
        s_auc_l, s_bal_l, s_rec_l, s_pf_l, s_bri_l, s_mcc_l = [], [], [], [], [], []

        train, test = separate_data(dataset)
        while (
            len(train[train[:, -1] == 0]) == 0 or
            len(test[test[:, -1] == 1]) == 0 or
            len(train[train[:, -1] == 1]) <= neighbor or
            len(test[test[:, -1] == 0]) == 0 or
            len(train[train[:, -1] == 1]) >= len(train[train[:, -1] == 0])
        ):
            train, test = separate_data(dataset)

        for _ in range(10):
            X_tr, y_tr = train[:, :-1], train[:, -1]
            X_te, y_te = test[:, :-1], test[:, -1]

            smote = SMOTE(k_neighbors=neighbor)
            X_os, y_os = smote.fit_resample(X_tr, y_tr)

            clf = tree.DecisionTreeClassifier(random_state=42)
            clf.fit(X_os, y_os)
            pred = clf.predict(X_te)

            tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
            pf = fp / (fp + tn)

            auc_l.append(roc_auc_score(y_te, pred))
            rec_l.append(recall_score(y_te, pred))
            bal_l.append(1 - (((pf)**2 + (1 - recall_score(y_te, pred))**2)/2)**0.5)
            pf_l.append(pf)
            bri_l.append(brier_score_loss(y_te, pred))
            mcc_l.append(matthews_corrcoef(y_te, pred))

            ssm = stable_SMOTE(neighbor)
            s_train = np.array(ssm.fit_sample(train, y_tr))

            clf.fit(s_train[:, :-1], s_train[:, -1])
            s_pred = clf.predict(X_te)

            tn, fp, fn, tp = confusion_matrix(y_te, s_pred).ravel()
            s_pf = fp / (fp + tn)

            s_auc_l.append(roc_auc_score(y_te, s_pred))
            s_rec_l.append(recall_score(y_te, s_pred))
            s_bal_l.append(1 - (((s_pf)**2 + (1 - recall_score(y_te, s_pred))**2)/2)**0.5)
            s_pf_l.append(s_pf)
            s_bri_l.append(brier_score_loss(y_te, s_pred))
            s_mcc_l.append(matthews_corrcoef(y_te, s_pred))

        def write(writer, arr, name):
            row = arr + [
                min(arr),
                np.percentile(arr,25),
                np.mean(arr),
                np.median(arr),
                np.percentile(arr,75),
                max(arr),
                np.std(arr,ddof=1)
            ]
            writer.writerow([inputfile + " " + name] + row)

        write(auc_writer, auc_l, "auc")
        write(balance_writer, bal_l, "balance")
        write(recall_writer, rec_l, "recall")
        write(pf_writer, pf_l, "pf")
        write(brier_writer, bri_l, "brier")
        write(mcc_writer, mcc_l, "mcc")

        write(stable_auc_writer, s_auc_l, "auc")
        write(stable_balance_writer, s_bal_l, "balance")
        write(stable_recall_writer, s_rec_l, "recall")
        write(stable_pf_writer, s_pf_l, "pf")
        write(stable_brier_writer, s_bri_l, "brier")
        write(stable_mcc_writer, s_mcc_l, "mcc")
