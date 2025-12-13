# stabled SMOTE with knn

import csv
import os
import random
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import neighbors
from sklearn.metrics import (
    roc_auc_score, recall_score,
    confusion_matrix, matthews_corrcoef,
    brier_score_loss
)

warnings.filterwarnings('ignore')
np.set_printoptions(suppress=True)


# 전역 설정
target_defect_ratio = 0.5
neighbor = 5
classifier = "knn"


# Stable SMOTE 
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

        defective = x_dataset[x_dataset["bug"] > 0]
        clean = x_dataset[x_dataset["bug"] == 0]

        need_number = int((target_defect_ratio * len(x_dataset) - len(defective)) /
                          (1 - target_defect_ratio))
        if need_number <= 0:
            return pd.concat([clean, defective])

        total_pair = []
        number_on_each = need_number / len(defective)

        rround = number_on_each / self.z_nearest
        while rround >= 1:
            for idx, row in defective.iterrows():
                temp = defective.copy()
                temp["dist"] = np.sqrt(((temp.iloc[:,:-1] - row.iloc[:-1]) ** 2).sum(axis=1))
                nn = temp.sort_values("dist").iloc[1:self.z_nearest+1]
                for j in nn.index:
                    total_pair.append(sorted([idx, j]))
            rround -= 1

        residue = need_number - len(total_pair)
        if residue > 0:
            residue = min(residue, len(defective))
            residue_df = defective.sample(n=residue)
            for idx, row in residue_df.iterrows():
                temp = defective.copy()
                temp["dist"] = np.sqrt(((temp.iloc[:,:-1] - row.iloc[:-1]) ** 2).sum(axis=1))
                nn = temp.sort_values("dist").iloc[-1:]
                for j in nn.index:
                    total_pair.append(sorted([idx, j]))

        generated = []
        cnt = Counter(tuple(p) for p in total_pair)
        for (i, j), n in cnt.items():
            r1 = defective.loc[i]
            r2 = defective.loc[j]
            samples = np.linspace(r1, r2, n+2)[1:-1]
            generated.extend(samples)

        gen_df = pd.DataFrame(generated, columns=x_dataset.columns)
        return pd.concat([clean, defective, gen_df])



# Bootstrap split
def separate_data(data):
    data = np.array(data)
    idx = np.random.randint(0, len(data), len(data))
    train_idx = np.unique(idx)
    test_idx = np.setdiff1d(np.arange(len(data)), train_idx)
    return data[train_idx], data[test_idx]



# CSV 초기화
os.makedirs("output_data/SMOTE_knn", exist_ok=True)

def open_csv(name):
    f = open(f"output_data/SMOTE_knn/{neighbor}{name}_smote_result_on_{classifier}.csv",
             "w", newline="")
    w = csv.writer(f)
    w.writerow(["inputfile"] + [""]*10 +
               ["min","lower","avg","median","upper","max","variance"])
    return w

auc_w = open_csv("auc")
balance_w = open_csv("balance")
recall_w = open_csv("recall")
pf_w = open_csv("pf")
brier_w = open_csv("brier")
mcc_w = open_csv("mcc")

stable_auc_w = open_csv("auc_stable")
stable_balance_w = open_csv("balance_stable")
stable_recall_w = open_csv("recall_stable")
stable_pf_w = open_csv("pf_stable")
stable_brier_w = open_csv("brier_stable")
stable_mcc_w = open_csv("mcc_stable")



# Main loop
for file in os.listdir("input_data"):
    print("Processing:", file)
    print("Start:", time.asctime())

    data = pd.read_csv("input_data/" + file)
    data = data.drop(columns=["name","version","name.1"])
    data["bug"] = (data["bug"] > 0).astype(int)

    for col in data.columns:
        mx, mn = data[col].max(), data[col].min()
        if mx != mn:
            data[col] = (data[col] - mn) / (mx - mn)

    for _ in range(10):
        train, test = separate_data(data.values)
        Xtr, ytr = train[:,:-1], train[:,-1]
        Xte, yte = test[:,:-1], test[:,-1]

        auc_l=[]; bal_l=[]; rec_l=[]; pf_l=[]; bri_l=[]; mcc_l=[]
        s_auc_l=[]; s_bal_l=[]; s_rec_l=[]; s_pf_l=[]; s_bri_l=[]; s_mcc_l=[]

        for _ in range(10):
            sm = SMOTE(k_neighbors=neighbor)
            Xr, yr = sm.fit_resample(Xtr, ytr)

            clf = neighbors.KNeighborsClassifier(n_neighbors=5)
            clf.fit(Xr, yr)
            pr = clf.predict(Xte)

            tn, fp, fn, tp = confusion_matrix(yte, pr).ravel()
            rec = recall_score(yte, pr)
            pf = fp / (fp + tn)
            bal = 1 - np.sqrt(((pf)**2 + (1-rec)**2)/2)

            auc_l.append(roc_auc_score(yte, pr))
            rec_l.append(rec)
            pf_l.append(pf)
            bal_l.append(bal)
            bri_l.append(brier_score_loss(yte, pr))
            mcc_l.append(matthews_corrcoef(yte, pr))

            ssm = stable_SMOTE(neighbor)
            sdata = np.array(ssm.fit_sample(train, ytr))
            Xs, ys = sdata[:,:-1], sdata[:,-1]

            s_clf = neighbors.KNeighborsClassifier(n_neighbors=5)
            s_clf.fit(Xs, ys)
            spr = s_clf.predict(Xte)

            tn, fp, fn, tp = confusion_matrix(yte, spr).ravel()
            rec = recall_score(yte, spr)
            pf = fp / (fp + tn)
            bal = 1 - np.sqrt(((pf)**2 + (1-rec)**2)/2)

            s_auc_l.append(roc_auc_score(yte, spr))
            s_rec_l.append(rec)
            s_pf_l.append(pf)
            s_bal_l.append(bal)
            s_bri_l.append(brier_score_loss(yte, spr))
            s_mcc_l.append(matthews_corrcoef(yte, spr))

        def write(w, arr):
            q1, q3 = np.percentile(arr,[25,75])
            w.writerow([file]+arr+[min(arr),q1,np.mean(arr),
                        np.median(arr),q3,max(arr),np.std(arr)])

        write(auc_w, auc_l); write(balance_w, bal_l); write(recall_w, rec_l)
        write(pf_w, pf_l); write(brier_w, bri_l); write(mcc_w, mcc_l)

        write(stable_auc_w, s_auc_l); write(stable_balance_w, s_bal_l)
        write(stable_recall_w, s_rec_l); write(stable_pf_w, s_pf_l)
        write(stable_brier_w, s_bri_l); write(stable_mcc_w, s_mcc_l)
