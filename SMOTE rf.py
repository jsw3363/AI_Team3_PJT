# stabled SMOTE with rf
import csv
import os
import random
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    recall_score,
    brier_score_loss,
    matthews_corrcoef
)

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)


# 전역 설정
target_defect_ratio = 0.5
neighbor = 5
classifier = "rf"


# Stable SMOTE
class stable_SMOTE:
    def __init__(self, z_nearest=5):
        self.z_nearest = z_nearest

    def fit_sample(self, x_dataset, y_dataset):
        x_dataset = pd.DataFrame(x_dataset)
        x_dataset = x_dataset.rename(
            columns={
                0:"wmc",1:"dit",2:"noc",3:"cbo",4:"rfc",5:"lcom",6:"ca",7:"ce",8:"npm",
                9:"lcom3",10:"loc",11:"dam",12:"moa",13:"mfa",14:"cam",15:"ic",16:"cbm",
                17:"amc",18:"max_cc",19:"avg_cc",20:"bug"
            }
        )

        defective_instance = x_dataset[x_dataset["bug"] > 0]
        clean_instance = x_dataset[x_dataset["bug"] == 0]

        need_number = int(
            (target_defect_ratio * len(x_dataset) - len(defective_instance)) /
            (1 - target_defect_ratio)
        )

        if need_number <= 0:
            return x_dataset

        total_pair = []
        generated_dataset = []

        number_on_each_instance = need_number / len(defective_instance)
        rround = number_on_each_instance / self.z_nearest

        
        
        while rround >= 1:
            for index, row in defective_instance.iterrows():
                temp = defective_instance.copy(deep=True)
                distance = ((row - temp) ** 2).sum(axis=1) ** 0.5
                temp["distance"] = distance
                neighbors = temp.sort_values(by="distance").iloc[1:self.z_nearest + 1]
                for a in neighbors.index:
                    pair = sorted([index, a])
                    total_pair.append(pair)
            rround -= 1

        
        # Residue handling
        
        residue = need_number - len(total_pair)
        residue = min(residue, len(defective_instance))

        residue_instances = defective_instance.sample(n=residue)

        for index, row in residue_instances.iterrows():
            temp = defective_instance.copy(deep=True)
            distance = ((row - temp) ** 2).sum(axis=1) ** 0.5
            temp["distance"] = distance
            neighbor_idx = temp.sort_values(by="distance").iloc[1].name
            pair = sorted([index, neighbor_idx])
            total_pair.append(pair)

        
        
        
        counter = Counter(map(tuple, total_pair))
        for (i, j), cnt in counter.items():
            r1 = defective_instance.loc[i]
            r2 = defective_instance.loc[j]
            samples = np.linspace(r1, r2, cnt + 2)[1:-1]
            generated_dataset.extend(samples)

        gen_df = pd.DataFrame(generated_dataset)
        gen_df.columns = x_dataset.columns

        return pd.concat([clean_instance, defective_instance, gen_df])


# Bootstrap split
def separate_data(original_data):
    data = np.array(original_data)
    size = len(data)
    train_idx = [random.randint(0, size - 1) for _ in range(size)]
    train_idx = list(set(train_idx))
    test_idx = list(set(range(size)) - set(train_idx))
    return data[train_idx], data[test_idx]


# Output directory
OUT_DIR = "output_data/SMOTE_rf/"
os.makedirs(OUT_DIR, exist_ok=True)

def open_writer(name):
    f = open(f"{OUT_DIR}{neighbor}{name}_result_on_{classifier}.csv", "w", newline="")
    w = csv.writer(f)
    w.writerow([
        "inputfile","","","","","","","","","","",
        "min","lower","avg","median","upper","max","variance"
    ])
    return w

auc_w = open_writer("auc_smote")
bal_w = open_writer("balance_smote")
rec_w = open_writer("recall_smote")
pf_w = open_writer("pf_smote")
brier_w = open_writer("brier_smote")
mcc_w = open_writer("mcc_smote")

s_auc_w = open_writer("auc_stable_smote")
s_bal_w = open_writer("balance_stable_smote")
s_rec_w = open_writer("recall_stable_smote")
s_pf_w = open_writer("pf_stable_smote")
s_brier_w = open_writer("brier_stable_smote")
s_mcc_w = open_writer("mcc_stable_smote")


# Main experiment
for inputfile in os.listdir("input_data/"):
    print(f"\nProcessing: {inputfile}")
    print("Start:", time.asctime())

    dataset = pd.read_csv("input_data/" + inputfile)
    dataset = dataset.drop(columns=["name", "version", "name.1"])

    if (dataset["bug"] > 0).mean() > 0.45:
        print("Skip (high defect ratio)")
        continue

    dataset["bug"] = (dataset["bug"] > 0).astype(int)

    for col in dataset.columns:
        mx, mn = dataset[col].max(), dataset[col].min()
        dataset[col] = (dataset[col] - mn) / (mx - mn)

    for _ in range(10):
        metrics = {k: [] for k in [
            "auc","bal","rec","pf","brier","mcc",
            "s_auc","s_bal","s_rec","s_pf","s_brier","s_mcc"
        ]}

        train, test = separate_data(dataset.values)
        X_tr, y_tr = train[:, :-1], train[:, -1]
        X_te, y_te = test[:, :-1], test[:, -1]

        for _ in range(10):
            sm = SMOTE(k_neighbors=neighbor)
            X_os, y_os = sm.fit_resample(X_tr, y_tr)

            clf = RandomForestClassifier()
            clf.fit(X_os, y_os)
            pred = clf.predict(X_te)

            tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
            pf = fp / (fp + tn)

            metrics["auc"].append(roc_auc_score(y_te, pred))
            metrics["rec"].append(recall_score(y_te, pred))
            metrics["pf"].append(pf)
            metrics["bal"].append(1 - (((pf)**2 + (1-metrics["rec"][-1])**2)/2)**0.5)
            metrics["brier"].append(brier_score_loss(y_te, pred))
            metrics["mcc"].append(matthews_corrcoef(y_te, pred))

            ssm = stable_SMOTE(neighbor)
            sdata = np.array(ssm.fit_sample(train, y_tr))
            clf.fit(sdata[:, :-1], sdata[:, -1])
            sp = clf.predict(X_te)

            tn, fp, fn, tp = confusion_matrix(y_te, sp).ravel()
            spf = fp / (fp + tn)

            metrics["s_auc"].append(roc_auc_score(y_te, sp))
            metrics["s_rec"].append(recall_score(y_te, sp))
            metrics["s_pf"].append(spf)
            metrics["s_bal"].append(1 - (((spf)**2 + (1-metrics["s_rec"][-1])**2)/2)**0.5)
            metrics["s_brier"].append(brier_score_loss(y_te, sp))
            metrics["s_mcc"].append(matthews_corrcoef(y_te, sp))

        def write_row(w, key, label):
            v = metrics[key]
            row = v + [
                min(v),
                np.percentile(v, 25),
                np.mean(v),
                np.median(v),
                np.percentile(v, 75),
                max(v),
                np.std(v)
            ]
            w.writerow([inputfile + " " + label] + row)

        write_row(auc_w, "auc", "auc")
        write_row(s_auc_w, "s_auc", "auc")
        write_row(bal_w, "bal", "balance")
        write_row(s_bal_w, "s_bal", "balance")
        write_row(rec_w, "rec", "recall")
        write_row(s_rec_w, "s_rec", "recall")
        write_row(pf_w, "pf", "pf")
        write_row(s_pf_w, "s_pf", "pf")
        write_row(brier_w, "brier", "brier")
        write_row(s_brier_w, "s_brier", "brier")
        write_row(mcc_w, "mcc", "mcc")
        write_row(s_mcc_w, "s_mcc", "mcc")
