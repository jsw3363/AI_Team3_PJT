# stabled BORDERLINE with rf
from imblearn.over_sampling import BorderlineSMOTE
import os
import numpy as np
import pandas as pd
import csv
import warnings
import random
import time
from collections import Counter

from sklearn import neighbors, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, recall_score,
    confusion_matrix, matthews_corrcoef,
    brier_score_loss
)

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)


# 전역 설정
classifier = "rf"
neighbor = 5
target_defect_ratio = 0.5


# Stable Borderline SMOTE 
class stable_SMOTE:
    def __init__(self, z_nearest=5):
        self.z_nearest = z_nearest

    def fit_sample(self, x_dataset, y_dataset):
        x_dataset = pd.DataFrame(x_dataset)
        x_dataset = x_dataset.rename(
            columns={0:"wmc",1:"dit",2:"noc",3:"cbo",4:"rfc",5:"lcom",6:"ca",7:"ce",8:"npm",
                     9:"lcom3",10:"loc",11:"dam",12:"moa",13:"mfa",14:"cam",15:"ic",
                     16:"cbm",17:"amc",18:"max_cc",19:"avg_cc",20:"bug"}
        )

        defective_instance = x_dataset[x_dataset["bug"] > 0]
        clean_instance = x_dataset[x_dataset["bug"] == 0]

        need_number = int((target_defect_ratio * len(x_dataset) - len(defective_instance)) / (1 - target_defect_ratio))
        if need_number <= 0:
            return pd.concat([clean_instance, defective_instance])

        select_column = x_dataset.columns[:-1]
        container = pd.DataFrame(columns=x_dataset.columns)

        for _, h in defective_instance.iterrows():
            dist = ((defective_instance[select_column] - h[select_column])**2).sum(axis=1)**0.5
            neighbors = defective_instance.assign(distance=dist).sort_values("distance")[1:self.z_nearest+1]
            majority = len(neighbors[neighbors["bug"] == 0])
            if 2 < majority < self.z_nearest:
                container = pd.concat([container, h.to_frame().T])

        if len(container) == 0:
            return pd.concat([clean_instance, defective_instance])

        total_pair = []
        num_each = need_number / len(container)

        for idx, row in container.iterrows():
            dist = ((defective_instance[select_column] - row[select_column])**2).sum(axis=1)**0.5
            neighbors = defective_instance.assign(distance=dist).sort_values("distance")[1:self.z_nearest+1]
            for nidx in neighbors.index[:int(num_each)+1]:
                total_pair.append(sorted([idx, nidx]))

        generated = []
        for (i, j), cnt in Counter(map(tuple, total_pair)).items():
            a, b = defective_instance.loc[i], defective_instance.loc[j]
            samples = np.linspace(a, b, cnt+2)[1:-1]
            generated.extend(samples.tolist())

        generated_df = pd.DataFrame(generated, columns=x_dataset.columns)
        return pd.concat([clean_instance, defective_instance, generated_df])


# Bootstrap 분리
def separate_data(data):
    data = np.array(data)
    idx = np.random.randint(0, len(data), len(data))
    train = data[np.unique(idx)]
    test = data[list(set(range(len(data))) - set(idx))]
    return train, test


# 출력 경로
OUT_DIR = "output_data/BORDERLINE_rf/"
os.makedirs(OUT_DIR, exist_ok=True)

def open_csv(name):
    f = open(OUT_DIR + f"{neighbor}{name}_borderline_result_on_{classifier}.csv", "w", newline="")
    w = csv.writer(f)
    w.writerow(["inputfile","","","","","","","","","","","min","lower","avg","median","upper","max","variance"])
    return f, w

auc_f, auc_w = open_csv("auc")
bal_f, bal_w = open_csv("balance")
rec_f, rec_w = open_csv("recall")
pf_f, pf_w = open_csv("pf")
brier_f, brier_w = open_csv("brier")
mcc_f, mcc_w = open_csv("mcc")

s_auc_f, s_auc_w = open_csv("auc_stable")
s_bal_f, s_bal_w = open_csv("balance_stable")
s_rec_f, s_rec_w = open_csv("recall_stable")
s_pf_f, s_pf_w = open_csv("pf_stable")
s_brier_f, s_brier_w = open_csv("brier_stable")
s_mcc_f, s_mcc_w = open_csv("mcc_stable")


# 메인 실험
for inputfile in os.listdir("input_data/"):
    dataset = pd.read_csv("input_data/" + inputfile)
    dataset = dataset.drop(columns=["name","version","name.1"])

    dataset["bug"] = (dataset["bug"] > 0).astype(int)

    for c in dataset.columns:
        dataset[c] = (dataset[c] - dataset[c].min()) / (dataset[c].max() - dataset[c].min())

    for _ in range(10):
        auc_r=[]; bal_r=[]; rec_r=[]; pf_r=[]; brier_r=[]; mcc_r=[]
        s_auc_r=[]; s_bal_r=[]; s_rec_r=[]; s_pf_r=[]; s_brier_r=[]; s_mcc_r=[]

        train, test = separate_data(dataset.values)

        for _ in range(10):
            X_tr, y_tr = train[:,:-1], train[:,-1]
            X_te, y_te = test[:,:-1], test[:,-1]

            sm = BorderlineSMOTE(k_neighbors=neighbor)
            Xs, ys = sm.fit_resample(X_tr, y_tr)

            clf = RandomForestClassifier()
            clf.fit(Xs, ys)
            pred = clf.predict(X_te)

            tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
            auc_r.append(roc_auc_score(y_te, pred))
            rec = recall_score(y_te, pred)
            rec_r.append(rec)
            pf = fp/(fp+tn)
            pf_r.append(pf)
            bal_r.append(1-(((pf)**2+(1-rec)**2)/2)**0.5)
            brier_r.append(brier_score_loss(y_te, pred))
            mcc_r.append(matthews_corrcoef(y_te, pred))

            ssm = stable_SMOTE(neighbor)
            st = np.array(ssm.fit_sample(train, y_tr))
            clf.fit(st[:,:-1], st[:,-1])
            sp = clf.predict(X_te)

            tn, fp, fn, tp = confusion_matrix(y_te, sp).ravel()
            s_auc_r.append(roc_auc_score(y_te, sp))
            s_rec = recall_score(y_te, sp)
            s_rec_r.append(s_rec)
            s_pf = fp/(fp+tn)
            s_pf_r.append(s_pf)
            s_bal_r.append(1-(((s_pf)**2+(1-s_rec)**2)/2)**0.5)
            s_brier_r.append(brier_score_loss(y_te, sp))
            s_mcc_r.append(matthews_corrcoef(y_te, sp))

        def write_row(w, name, arr):
            arr2 = arr + [min(arr), np.percentile(arr,25), np.mean(arr),
                          np.median(arr), np.percentile(arr,75), max(arr), np.std(arr)]
            w.writerow([inputfile+" "+name] + arr2)

        write_row(auc_w,"auc",auc_r)
        write_row(bal_w,"balance",bal_r)
        write_row(rec_w,"recall",rec_r)
        write_row(pf_w,"pf",pf_r)
        write_row(brier_w,"brier",brier_r)
        write_row(mcc_w,"mcc",mcc_r)

        write_row(s_auc_w,"auc",s_auc_r)
        write_row(s_bal_w,"balance",s_bal_r)
        write_row(s_rec_w,"recall",s_rec_r)
        write_row(s_pf_w,"pf",s_pf_r)
        write_row(s_brier_w,"brier",s_brier_r)
        write_row(s_mcc_w,"mcc",s_mcc_r)
