# stabled ADASYN with TREE
from imblearn.over_sampling import ADASYN
import os
import numpy as np
import pandas as pd
import csv
import warnings
import random
import time

from sklearn.metrics import (
    roc_auc_score, recall_score, confusion_matrix,
    brier_score_loss, matthews_corrcoef
)
from sklearn import tree
from collections import Counter

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)


# 전역 설정
neighbor = 5
target_defect_ratio = 0.5
classifier = "tree"


# Stable ADASYN 알고리즘
class stable_ADASYN:
    def __init__(self, z_nearest=5):
        self.z_nearest = z_nearest

    def fit_sample(self, x_dataset, y_dataset):
        x_dataset = pd.DataFrame(x_dataset)
        x_dataset = x_dataset.rename(
            columns={0:"wmc",1:"dit",2:"noc",3:"cbo",4:"rfc",5:"lcom",6:"ca",7:"ce",
                     8:"npm",9:"lcom3",10:"loc",11:"dam",12:"moa",13:"mfa",14:"cam",
                     15:"ic",16:"cbm",17:"amc",18:"max_cc",19:"avg_cc",20:"bug"}
        )

        clean_dataset = x_dataset[x_dataset["bug"] == 0]
        defect_dataset = x_dataset[x_dataset["bug"] > 0]

        need_number = int(
            (target_defect_ratio * len(x_dataset) - len(defect_dataset)) /
            (1 - target_defect_ratio)
        )

        select_column = x_dataset.columns[:-1]
        total_ratio = 0
        total_pair = []
        generated_dataset = []

        for _, row in x_dataset.iterrows():
            if row["bug"] == 0:
                continue
            temp = row.drop("bug")
            dist = ((x_dataset[select_column] - temp) ** 2).sum(axis=1) ** 0.5
            nn = x_dataset.assign(distance=dist).sort_values("distance").iloc[1:self.z_nearest+1]
            total_ratio += len(nn[nn["bug"] == 0]) / self.z_nearest

        for idx, row in defect_dataset.iterrows():
            temp = row.drop("bug")
            dist = ((x_dataset[select_column] - temp) ** 2).sum(axis=1) ** 0.5
            nn = x_dataset.assign(distance=dist).sort_values("distance").iloc[1:self.z_nearest+1]

            ratio = len(nn[nn["bug"] == 0]) / self.z_nearest
            gen_num = round((ratio / total_ratio) * need_number)

            defect_dist = ((defect_dataset[select_column] - temp) ** 2).sum(axis=1) ** 0.5
            nn_def = defect_dataset.assign(distance=defect_dist)\
                                    .sort_values("distance").iloc[1:self.z_nearest+1]

            r = gen_num / self.z_nearest
            while r >= 1:
                for i in nn_def.index:
                    total_pair.append(tuple(sorted([idx, i])))
                r -= 1

            rest = round(r * self.z_nearest)
            for i in nn_def.sort_values("distance", ascending=False).head(rest).index:
                total_pair.append(tuple(sorted([idx, i])))

        for (i, j), cnt in Counter(total_pair).items():
            p1 = defect_dataset.loc[i]
            p2 = defect_dataset.loc[j]
            samples = np.linspace(p1, p2, cnt + 2)[1:-1]
            generated_dataset.extend(samples.tolist())

        gen_df = pd.DataFrame(generated_dataset, columns=x_dataset.columns)
        return pd.concat([clean_dataset, defect_dataset, gen_df])



# Bootstrap 분리
def separate_data(data):
    data = np.array(data)
    idx = np.random.randint(0, len(data), len(data))
    train_idx = np.unique(idx)
    test_idx = np.setdiff1d(np.arange(len(data)), train_idx)
    return data[train_idx], data[test_idx]



# 결과 저장 폴더
OUT_DIR = "output_data/ADASYN_tree/"
os.makedirs(OUT_DIR, exist_ok=True)

def open_writer(filename):
    f = open(OUT_DIR + filename, "w", newline="")
    w = csv.writer(f)
    w.writerow(
        ["inputfile"] + [""] * 10 +
        ["min", "lower", "avg", "median", "upper", "max", "variance"]
    )
    return w


auc_writer = open_writer("5auc_adasyn_result_on_tree.csv")
balance_writer = open_writer("5balance_adasyn_result_on_tree.csv")
recall_writer = open_writer("5recall_adasyn_result_on_tree.csv")
pf_writer = open_writer("5pf_adasyn_result_on_tree.csv")
brier_writer = open_writer("5brier_adasyn_result_on_tree.csv")
mcc_writer = open_writer("5mcc_adasyn_result_on_tree.csv")

stable_auc_writer = open_writer("5auc_stable_adasyn_result_on_tree.csv")
stable_balance_writer = open_writer("5balance_stable_adasyn_result_on_tree.csv")
stable_recall_writer = open_writer("5recall_stable_adasyn_result_on_tree.csv")
stable_pf_writer = open_writer("5pf_stable_adasyn_result_on_tree.csv")
stable_brier_writer = open_writer("5brier_stable_adasyn_result_on_tree.csv")
stable_mcc_writer = open_writer("5mcc_stable_adasyn_result_on_tree.csv")



# 메인 실험 루프
for inputfile in os.listdir("input_data/"):
    print("Processing:", inputfile)
    print("Start:", time.asctime())

    dataset = pd.read_csv("input_data/" + inputfile)
    dataset = dataset.drop(columns=["name", "version", "name.1"])

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

            
            # ADASYN
            adasyn = ADASYN(n_neighbors=neighbor)
            X_os, y_os = adasyn.fit_resample(X_tr, y_tr)

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

            
            # Stable ADASYN
            sad = stable_ADASYN(neighbor)
            s_train = np.array(sad.fit_sample(train, y_tr))

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
                np.percentile(arr, 25),
                np.mean(arr),
                np.median(arr),
                np.percentile(arr, 75),
                max(arr),
                np.std(arr, ddof=1)
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
