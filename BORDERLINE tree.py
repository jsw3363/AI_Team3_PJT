# stabled BORDERLINE with tree

from imblearn.over_sampling import BorderlineSMOTE
import os
import numpy as np
import pandas as pd
import csv
import warnings
import random
import time

from sklearn.metrics import (
    roc_auc_score, recall_score, confusion_matrix,
    matthews_corrcoef, brier_score_loss
)
from sklearn import tree
from collections import Counter

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=196)
np.set_printoptions(suppress=True)

# =========================
# Global settings
# =========================
classifier = "tree"
neighbor = 5
target_defect_ratio = 0.5

# =========================
# Stable Borderline SMOTE (절대 수정 금지)
# =========================
class stable_SMOTE:
    def __init__(self, z_nearest=5):
        self.z_nearest = z_nearest

    def fit_sample(self, x_dataset, y_dataset):
        total_pair = []
        x_dataset = pd.DataFrame(x_dataset)
        x_dataset = x_dataset.rename(columns={
            0:"wmc",1:"dit",2:"noc",3:"cbo",4:"rfc",5:"lcom",6:"ca",7:"ce",8:"npm",
            9:"lcom3",10:"loc",11:"dam",12:"moa",13:"mfa",14:"cam",15:"ic",
            16:"cbm",17:"amc",18:"max_cc",19:"avg_cc",20:"bug"
        })

        defective = x_dataset[x_dataset["bug"] > 0]
        clean = x_dataset[x_dataset["bug"] == 0]

        need_number = int((target_defect_ratio * len(x_dataset) - len(defective)) / (1 - target_defect_ratio))
        if need_number <= 0:
            return pd.concat([clean, defective])

        select_cols = x_dataset.columns[:-1]
        container = pd.DataFrame(columns=x_dataset.columns)

        # STEP 1: borderline instance selection
        for _, row in x_dataset.iterrows():
            if row["bug"] == 0:
                continue
            dist = np.sqrt(((row[select_cols] - x_dataset[select_cols]) ** 2).sum(axis=1))
            nn = x_dataset.iloc[dist.argsort()[1:self.z_nearest+1]]
            maj = len(nn[nn["bug"] == 0])
            if maj == 5 or maj < 2.5:
                continue
            container = pd.concat([container, row.to_frame().T])

        if len(container) == 0:
            return pd.concat([clean, defective])

        # STEP 2–3: pair selection + interpolation
        total_pair = []
        per_instance = need_number / len(container)

        for idx, row in container.iterrows():
            dist = np.sqrt(((row[select_cols] - defective[select_cols]) ** 2).sum(axis=1))
            nn = defective.iloc[dist.argsort()[1:self.z_nearest+1]]
            for nidx in nn.index[:int(per_instance)]:
                total_pair.append(sorted([idx, nidx]))

        result = Counter(tuple(p) for p in total_pair)
        generated = []

        for (i, j), cnt in result.items():
            a = defective.loc[i]
            b = defective.loc[j]
            inter = np.linspace(a, b, cnt+2)[1:-1]
            generated.extend(inter.tolist())

        gen_df = pd.DataFrame(generated, columns=x_dataset.columns)
        return pd.concat([clean, defective, gen_df])

# =========================
# Bootstrap split
# =========================
def separate_data(data):
    data = np.array(data)
    idx = np.random.randint(0, len(data), len(data))
    train = data[np.unique(idx)]
    test = data[list(set(range(len(data))) - set(idx))]
    return train, test

# =========================
# CSV writers
# =========================
os.makedirs("output_data/BORDERLINE_tree", exist_ok=True)

def open_writer(name):
    f = open(
        f"output_data/BORDERLINE_tree/{neighbor}{name}_borderline_result_on_tree.csv",
        "w",
        newline=""
    )
    w = csv.writer(f)
    w.writerow(["inputfile"] + [""]*10 + ["min","lower","avg","median","upper","max","variance"])
    return f, w

auc_f, auc_w = open_writer("auc")
bal_f, bal_w = open_writer("balance")
rec_f, rec_w = open_writer("recall")
pf_f, pf_w = open_writer("pf")
bri_f, bri_w = open_writer("brier")
mcc_f, mcc_w = open_writer("mcc")

sauc_f, sauc_w = open_writer("auc_stable")
sbal_f, sbal_w = open_writer("balance_stable")
srec_f, srec_w = open_writer("recall_stable")
spf_f, spf_w = open_writer("pf_stable")
sbri_f, sbri_w = open_writer("brier_stable")
smcc_f, smcc_w = open_writer("mcc_stable")

# =========================
# Main experiment loop
# =========================
for inputfile in os.listdir("input_data"):
    print(inputfile)

    df = pd.read_csv("input_data/" + inputfile)
    df = df.drop(columns=["name","version","name.1"])

    df["bug"] = (df["bug"] > 0).astype(int)
    df = (df - df.min()) / (df.max() - df.min())

    for _ in range(10):
        auc_r, bal_r, rec_r, pf_r, bri_r, mcc_r = [],[],[],[],[],[]
        sauc_r, sbal_r, srec_r, spf_r, sbri_r, smcc_r = [],[],[],[],[],[]

        train, test = separate_data(df.values)
        Xtr, ytr = train[:,:-1], train[:,-1]
        Xte, yte = test[:,:-1], test[:,-1]

        for _ in range(10):
            # ----- Borderline SMOTE -----
            sm = BorderlineSMOTE(k_neighbors=neighbor, kind="borderline-1")
            Xs, ys = sm.fit_resample(Xtr, ytr)

            clf = tree.DecisionTreeClassifier(random_state=0)
            clf.fit(Xs, ys)
            p = clf.predict(Xte)

            tn, fp, fn, tp = confusion_matrix(yte, p).ravel()
            pf = fp/(fp+tn)

            auc_r.append(roc_auc_score(yte, p))
            rec_r.append(recall_score(yte, p))
            bal_r.append(1 - (((pf)**2 + (1-recall_score(yte,p))**2)/2)**0.5)
            bri_r.append(brier_score_loss(yte,p))
            mcc_r.append(matthews_corrcoef(yte,p))
            pf_r.append(pf)

            # ----- Stable Borderline SMOTE -----
            st = stable_SMOTE()
            st_tr = st.fit_sample(train, ytr)
            st_tr = st_tr.values  # ★ 핵심 수정 (DataFrame → numpy)

            Xs2, ys2 = st_tr[:,:-1], st_tr[:,-1]

            clf.fit(Xs2, ys2)
            p2 = clf.predict(Xte)

            tn, fp, fn, tp = confusion_matrix(yte, p2).ravel()
            pf2 = fp/(fp+tn)

            sauc_r.append(roc_auc_score(yte,p2))
            srec_r.append(recall_score(yte,p2))
            sbal_r.append(1 - (((pf2)**2 + (1-recall_score(yte,p2))**2)/2)**0.5)
            sbri_r.append(brier_score_loss(yte,p2))
            smcc_r.append(matthews_corrcoef(yte,p2))
            spf_r.append(pf2)

        def write(w, name, r):
            q1,q3 = np.percentile(r,[25,75])
            w.writerow([name]+r+[min(r),q1,np.mean(r),np.median(r),q3,max(r),np.var(r)])

        write(auc_w,inputfile+" auc",auc_r)
        write(bal_w,inputfile+" balance",bal_r)
        write(rec_w,inputfile+" recall",rec_r)
        write(pf_w,inputfile+" pf",pf_r)
        write(bri_w,inputfile+" brier",bri_r)
        write(mcc_w,inputfile+" mcc",mcc_r)

        write(sauc_w,inputfile+" auc",sauc_r)
        write(sbal_w,inputfile+" balance",sbal_r)
        write(srec_w,inputfile+" recall",srec_r)
        write(spf_w,inputfile+" pf",spf_r)
        write(sbri_w,inputfile+" brier",sbri_r)
        write(smcc_w,inputfile+" mcc",smcc_r)
