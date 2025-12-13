# stabled ADASYN with knn

from imblearn.over_sampling import ADASYN
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score
import csv
import warnings
from sklearn import neighbors
from sklearn.metrics import confusion_matrix
from sklearn.metrics import brier_score_loss, matthews_corrcoef
import random
from collections import Counter
import time

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)


# 전역 설정
classifier = "knn"
neighbor = 5
target_defect_ratio = 0.5


# Stable ADASYN 
class stable_ADASYN:
    def __init__(self, z_nearest=5):
        self.z_nearest = z_nearest

    def fit_sample(self, x_dataset, y_dataset):
        x_dataset = pd.DataFrame(x_dataset)
        x_dataset = x_dataset.rename(
            columns={0:"wmc",1:"dit",2:"noc",3:"cbo",4:"rfc",5:"lcom",6:"ca",7:"ce",8:"npm",
                     9:"lcom3",10:"loc",11:"dam",12:"moa",13:"mfa",14:"cam",15:"ic",
                     16:"cbm",17:"amc",18:"max_cc",19:"avg_cc",20:"bug"}
        )

        clean = x_dataset[x_dataset["bug"] == 0]
        defect = x_dataset[x_dataset["bug"] > 0]

        need_number = int((target_defect_ratio * len(x_dataset) - len(defect)) / (1 - target_defect_ratio))
        total_pair = []
        generated = []

        cols = x_dataset.columns[:-1]

        total_ratio = 0
        for _, row in defect.iterrows():
            temp = x_dataset[cols]
            dist = ((row[cols] - temp) ** 2).sum(axis=1) ** 0.5
            temp = temp.copy()
            temp["bug"] = x_dataset["bug"]
            temp["dist"] = dist
            near = temp.sort_values("dist").iloc[1:self.z_nearest+1]
            total_ratio += len(near[near["bug"] == 0]) / self.z_nearest

        for idx, row in defect.iterrows():
            temp = x_dataset[cols]
            dist = ((row[cols] - temp) ** 2).sum(axis=1) ** 0.5
            temp = temp.copy()
            temp["bug"] = x_dataset["bug"]
            temp["dist"] = dist
            near = temp.sort_values("dist").iloc[1:self.z_nearest+1]

            ratio = len(near[near["bug"] == 0]) / self.z_nearest
            gen_n = round((ratio / total_ratio) * need_number)

            defect_only = defect.copy()
            d = ((row - defect_only) ** 2).sum(axis=1) ** 0.5
            defect_only["dist"] = d
            nn = defect_only.sort_values("dist").iloc[1:self.z_nearest+1]

            rround = gen_n / self.z_nearest
            while rround >= 1:
                for a in nn.index:
                    total_pair.append(sorted([idx, a]))
                rround -= 1

            k = round(rround * self.z_nearest)
            nn = nn.sort_values("dist", ascending=False).iloc[:k]
            for a in nn.index:
                total_pair.append(sorted([idx, a]))

        counter = Counter(map(tuple, total_pair))
        for (i, j), cnt in counter.items():
            r1 = defect.loc[i]
            r2 = defect.loc[j]
            samples = np.linspace(r1, r2, cnt + 2)[1:-1]
            generated.extend(samples)

        gen_df = pd.DataFrame(generated, columns=x_dataset.columns)
        return pd.concat([clean, defect, gen_df])


# Bootstrap split
def separate_data(data):
    data = np.array(data)
    idx = [random.randint(0, len(data)-1) for _ in range(len(data))]
    train = data[list(set(idx))]
    test = np.delete(data, list(set(idx)), axis=0)
    return train, test


# Output directory
OUT_DIR = "output_data/ADASYN_knn/"
os.makedirs(OUT_DIR, exist_ok=True)

def writer(name):
    f = open(f"{OUT_DIR}{neighbor}{name}_result_on_{classifier}.csv", "w", newline="")
    w = csv.writer(f)
    w.writerow(["inputfile","","","","","","","","","","",
                "min","lower","avg","median","upper","max","variance"])
    return w

auc_w = writer("auc_adasyn")
bal_w = writer("balance_adasyn")
rec_w = writer("recall_adasyn")
pf_w  = writer("pf_adasyn")
brier_w = writer("brier_adasyn")
mcc_w = writer("mcc_adasyn")

s_auc_w = writer("auc_stable_adasyn")
s_bal_w = writer("balance_stable_adasyn")
s_rec_w = writer("recall_stable_adasyn")
s_pf_w  = writer("pf_stable_adasyn")
s_brier_w = writer("brier_stable_adasyn")
s_mcc_w = writer("mcc_stable_adasyn")


# Main loop
for file in os.listdir("input_data/"):
    print("Processing:", file)
    print("Start:", time.asctime())

    df = pd.read_csv("input_data/" + file)
    df = df.drop(columns=["name","version","name.1"])
    if (df["bug"] > 0).mean() > 0.45:
        continue

    df["bug"] = (df["bug"] > 0).astype(int)
    for c in df.columns:
        df[c] = (df[c] - df[c].min()) / (df[c].max() - df[c].min())

    for _ in range(10):
        metrics = {k: [] for k in
                   ["auc","bal","rec","pf","brier","mcc",
                    "s_auc","s_bal","s_rec","s_pf","s_brier","s_mcc"]}

        train, test = separate_data(df.values)
        Xtr, ytr = train[:,:-1], train[:,-1]
        Xte, yte = test[:,:-1], test[:,-1]

        for _ in range(10):
            ad = ADASYN(n_neighbors=neighbor)
            Xos, yos = ad.fit_resample(Xtr, ytr)

            clf = neighbors.KNeighborsClassifier(n_neighbors=5)
            clf.fit(Xos, yos)
            p = clf.predict(Xte)

            tn, fp, fn, tp = confusion_matrix(yte, p).ravel()
            pf = fp / (fp + tn)

            metrics["auc"].append(roc_auc_score(yte, p))
            metrics["rec"].append(recall_score(yte, p))
            metrics["pf"].append(pf)
            metrics["bal"].append(1 - (((pf)**2 + (1-metrics["rec"][-1])**2)/2)**0.5)
            metrics["brier"].append(brier_score_loss(yte, p))
            metrics["mcc"].append(matthews_corrcoef(yte, p))

            sad = stable_ADASYN(neighbor)
            sdata = np.array(sad.fit_sample(train, ytr))
            clf.fit(sdata[:,:-1], sdata[:,-1])
            sp = clf.predict(Xte)

            tn, fp, fn, tp = confusion_matrix(yte, sp).ravel()
            spf = fp / (fp + tn)

            metrics["s_auc"].append(roc_auc_score(yte, sp))
            metrics["s_rec"].append(recall_score(yte, sp))
            metrics["s_pf"].append(spf)
            metrics["s_bal"].append(1 - (((spf)**2 + (1-metrics["s_rec"][-1])**2)/2)**0.5)
            metrics["s_brier"].append(brier_score_loss(yte, sp))
            metrics["s_mcc"].append(matthews_corrcoef(yte, sp))

        def write(w, k, tag):
            v = metrics[k]
            row = v + [min(v), np.percentile(v,25), np.mean(v),
                       np.median(v), np.percentile(v,75),
                       max(v), np.std(v)]
            w.writerow([file+" "+tag] + row)

        write(auc_w,"auc","auc"); write(s_auc_w,"s_auc","auc")
        write(bal_w,"bal","balance"); write(s_bal_w,"s_bal","balance")
        write(rec_w,"rec","recall"); write(s_rec_w,"s_rec","recall")
        write(pf_w,"pf","pf"); write(s_pf_w,"s_pf","pf")
        write(brier_w,"brier","brier"); write(s_brier_w,"s_brier","brier")
        write(mcc_w,"mcc","mcc"); write(s_mcc_w,"s_mcc","mcc")
