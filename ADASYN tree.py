from imblearn.over_sampling import ADASYN
import os
import numpy as np
import pandas as pd
import csv
import warnings
import random

from sklearn.metrics import (
    roc_auc_score, recall_score, confusion_matrix,
    brier_score_loss, matthews_corrcoef
)
from sklearn import tree
from collections import Counter

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)

# ===============================
# 전역 설정
# ===============================
neighbor = 5
target_defect_ratio = 0.5

# ===============================
# Stable ADASYN (절대 수정 ❌)
# ===============================
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

        clean_dataset = x_dataset[x_dataset["bug"] == 0]
        defect_dataset = x_dataset[x_dataset["bug"] > 0]

        defect_number = len(defect_dataset)
        need_number = int((target_defect_ratio * len(x_dataset) - defect_number) / (1 - target_defect_ratio))

        select_column = x_dataset.columns[:-1].tolist()
        total_ratio = 0
        total_pair = []
        generated_dataset = []

        for index, row in x_dataset.iterrows():
            if row["bug"] == 0:
                continue
            euclid = ((row[select_column] - x_dataset[select_column]) ** 2).sum(axis=1) ** 0.5
            nearest = euclid.sort_values().iloc[1:self.z_nearest+1]
            majority = x_dataset.loc[nearest.index]["bug"].value_counts().get(0, 0)
            total_ratio += majority / self.z_nearest

        for index, row in defect_dataset.iterrows():
            euclid = ((row[select_column] - x_dataset[select_column]) ** 2).sum(axis=1) ** 0.5
            nearest = euclid.sort_values().iloc[1:self.z_nearest+1]
            majority = x_dataset.loc[nearest.index]["bug"].value_counts().get(0, 0)
            ratio = (majority / self.z_nearest) / total_ratio
            single_need = round(ratio * need_number)

            defect_dist = ((row[select_column] - defect_dataset[select_column]) ** 2).sum(axis=1) ** 0.5
            neighbors = defect_dist.sort_values().iloc[1:self.z_nearest+1].index.tolist()

            for _ in range(single_need):
                a = random.choice(neighbors)
                total_pair.append(tuple(sorted([index, a])))

        result = Counter(total_pair)

        for (i, j), cnt in result.items():
            p1 = defect_dataset.loc[i]
            p2 = defect_dataset.loc[j]
            points = np.linspace(p1, p2, cnt + 2)[1:-1]
            generated_dataset.extend(points)

        final_generated = pd.DataFrame(generated_dataset, columns=x_dataset.columns)
        return pd.concat([clean_dataset, defect_dataset, final_generated])

# ===============================
# Bootstrap 분리
# ===============================
def separate_data(original_data):
    data = np.array(original_data)
    size = len(data)
    idx = np.random.randint(0, size, size)
    train = data[list(set(idx))]
    test = np.delete(data, list(set(idx)), axis=0)
    return train, test

# ===============================
# CSV 설정 (SVM 코드 동일)
# ===============================
OUT_DIR = "output_data/ADASYN_tree/"
os.makedirs(OUT_DIR, exist_ok=True)

def open_csv(fname):
    f = open(OUT_DIR + fname, 'w', newline='')
    w = csv.writer(f)
    w.writerow(["inputfile","","","","","","","","","","","min","lower","avg","median","upper","max","variance"])
    return f, w

auc_f, auc_w = open_csv(f"{neighbor}auc_adasyn_tree.csv")
bal_f, bal_w = open_csv(f"{neighbor}balance_adasyn_tree.csv")
rec_f, rec_w = open_csv(f"{neighbor}recall_adasyn_tree.csv")
pf_f, pf_w = open_csv(f"{neighbor}pf_adasyn_tree.csv")
brier_f, brier_w = open_csv(f"{neighbor}brier_adasyn_tree.csv")
mcc_f, mcc_w = open_csv(f"{neighbor}mcc_adasyn_tree.csv")

s_auc_f, s_auc_w = open_csv(f"{neighbor}auc_stable_adasyn_tree.csv")
s_bal_f, s_bal_w = open_csv(f"{neighbor}balance_stable_adasyn_tree.csv")
s_rec_f, s_rec_w = open_csv(f"{neighbor}recall_stable_adasyn_tree.csv")
s_pf_f, s_pf_w = open_csv(f"{neighbor}pf_stable_adasyn_tree.csv")
s_brier_f, s_brier_w = open_csv(f"{neighbor}brier_stable_adasyn_tree.csv")
s_mcc_f, s_mcc_w = open_csv(f"{neighbor}mcc_stable_adasyn_tree.csv")

# ===============================
# 요약 기록 함수 (빈 row 안전)
# ===============================
def write_summary(writer, name, row):
    if len(row) == 0:
        return
    q1, q3 = np.percentile(row, [25, 75])
    writer.writerow(
        [name,"","","","","","","","","","",
         min(row), q1, np.mean(row), np.median(row), q3, max(row), np.std(row, ddof=1)]
    )

# ===============================
# 메인 실험 루프
# ===============================
for inputfile in os.listdir("input_data/"):
    print(inputfile)

    dataset = pd.read_csv("input_data/" + inputfile)
    dataset = dataset.drop(columns=["name", "version", "name.1"])
    dataset["bug"] = (dataset["bug"] > 0).astype(int)

    for col in dataset.columns:
        dataset[col] = (dataset[col] - dataset[col].min()) / (dataset[col].max() - dataset[col].min())

    for _ in range(10):

        auc_row, bal_row, rec_row, pf_row, brier_row, mcc_row = [], [], [], [], [], []
        s_auc_row, s_bal_row, s_rec_row, s_pf_row, s_brier_row, s_mcc_row = [], [], [], [], [], []

        train_data, test_data = separate_data(dataset)
        train_x, train_y = train_data[:, :-1], train_data[:, -1]
        test_x, test_y = test_data[:, :-1], test_data[:, -1]

        for _ in range(10):

            # ADASYN 불가 split skip
            if np.sum(train_y == 1) <= neighbor:
                continue

            # -------- ADASYN --------
            smote = ADASYN(n_neighbors=neighbor)
            sx, sy = smote.fit_resample(train_x, train_y)

            clf = tree.DecisionTreeClassifier()
            clf.fit(sx, sy)
            pred = clf.predict(test_x)

            tn, fp, fn, tp = confusion_matrix(test_y, pred).ravel()

            rec = recall_score(test_y, pred)
            pf = fp / (fp + tn)

            auc_row.append(roc_auc_score(test_y, pred))
            rec_row.append(rec)
            pf_row.append(pf)
            bal_row.append(1 - (((pf)**2 + (1 - rec)**2) / 2) ** 0.5)
            brier_row.append(brier_score_loss(test_y, pred))
            mcc_row.append(matthews_corrcoef(test_y, pred))

            # -------- Stable ADASYN --------
            stable = stable_ADASYN(neighbor)
            stable_train = np.array(stable.fit_sample(train_data, train_y))
            sx, sy = stable_train[:, :-1], stable_train[:, -1]

            clf = tree.DecisionTreeClassifier()
            clf.fit(sx, sy)
            spred = clf.predict(test_x)

            tn, fp, fn, tp = confusion_matrix(test_y, spred).ravel()

            s_rec = recall_score(test_y, spred)
            s_pf = fp / (fp + tn)

            s_auc_row.append(roc_auc_score(test_y, spred))
            s_rec_row.append(s_rec)
            s_pf_row.append(s_pf)
            s_bal_row.append(1 - (((s_pf)**2 + (1 - s_rec)**2) / 2) ** 0.5)
            s_brier_row.append(brier_score_loss(test_y, spred))
            s_mcc_row.append(matthews_corrcoef(test_y, spred))

        # ===== CSV 요약 기록 =====
        write_summary(auc_w, inputfile+" auc", auc_row)
        write_summary(bal_w, inputfile+" balance", bal_row)
        write_summary(rec_w, inputfile+" recall", rec_row)
        write_summary(pf_w, inputfile+" pf", pf_row)
        write_summary(brier_w, inputfile+" brier", brier_row)
        write_summary(mcc_w, inputfile+" mcc", mcc_row)

        write_summary(s_auc_w, inputfile+" auc", s_auc_row)
        write_summary(s_bal_w, inputfile+" balance", s_bal_row)
        write_summary(s_rec_w, inputfile+" recall", s_rec_row)
        write_summary(s_pf_w, inputfile+" pf", s_pf_row)
        write_summary(s_brier_w, inputfile+" brier", s_brier_row)
        write_summary(s_mcc_w, inputfile+" mcc", s_mcc_row)
