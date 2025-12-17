# ============================================================
# Stable ADASYN with SVM (Bank Marketing - bank.csv)
# ------------------------------------------------------------
# [수정사항]
# 1) 입력 데이터: new_input_data/bank.csv (UCI Bank Marketing 10% / sep=";")
# 2) 출력 폴더: new_output_data/new_ADASYN_svm/
# 3) 저장 파일명: ..._bank_marketing.csv 가 뒤에 붙도록 수정
#
# [기존 흐름 유지]
# - out-of-sample bootstrap 분리
# - GridSearchCV로 best C, kernel 탐색
# - outer 10회, inner 10회 반복
# - 일반 ADASYN vs Stable ADASYN 비교
# ============================================================

from imblearn.over_sampling import ADASYN
import os
import numpy as np
import pandas as pd
import csv
import warnings
import random
import time
from collections import Counter

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    confusion_matrix,
    brier_score_loss,
    matthews_corrcoef
)
from sklearn import svm
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)

# -------------------------
# 전역 설정
# -------------------------
classifier = "svm"
neighbor = 5
tuned_parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10, 100, 1000]}
target_defect_ratio = 0.5  # minority ratio to 0.5

INPUT_CSV_PATH = "new_input_data/bank.csv"         # ✅ 입력 고정
OUTPUT_DIR = "new_output_data/new_ADASYN_svm/"      # ✅ 출력 고정
dataset_tag = "bank_marketing"                      # ✅ 파일명 suffix
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------
# Bank Marketing 전처리
# -------------------------
def load_and_preprocess_bank(csv_path: str) -> pd.DataFrame:
    # bank.csv는 sep=";" / target 컬럼은 y (yes/no)
    df = pd.read_csv(csv_path, sep=";")

    if "y" not in df.columns:
        raise ValueError(f"'y' column not found in {csv_path}. columns={df.columns.tolist()}")

    # y -> bug(0/1), y 제거
    df["bug"] = (df["y"] == "yes").astype(int)
    df = df.drop(columns=["y"])

    # 범주형 -> LabelEncoder
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Min-Max scaling (bug 제외)
    feat_cols = [c for c in df.columns if c != "bug"]
    for col in feat_cols:
        cmax, cmin = df[col].max(), df[col].min()
        if cmax != cmin:
            df[col] = (df[col] - cmin) / (cmax - cmin)
        else:
            df[col] = 0.0

    df["bug"] = df["bug"].astype(int)
    return df


# -------------------------
# Stable ADASYN (원 코드 구조 유지 + 컬럼 일반화)
# -------------------------
class stable_ADASYN:
    def __init__(self, z_nearest=5):
        self.z_nearest = z_nearest

    def fit_sample(self, x_dataset, y_dataset=None):
        """
        입력: x_dataset은 (N, D+1) 형태로 마지막 컬럼이 bug(0/1)라고 가정
        반환: DataFrame (clean + defect + synthetic)
        """
        x_dataset = pd.DataFrame(x_dataset).copy()

        # 마지막 컬럼을 bug로 가정하고, feature는 f0.. 로 이름 부여
        n_cols = x_dataset.shape[1]
        feat_idx = list(range(n_cols - 1))
        x_dataset = x_dataset.rename(columns={**{i: f"f{i}" for i in feat_idx}, n_cols - 1: "bug"})

        clean_dataset = x_dataset[x_dataset["bug"] == 0]
        defect_dataset = x_dataset[x_dataset["bug"] > 0]
        defect_number = len(defect_dataset)

        # 목표 비율(0.5)에 맞춰 생성할 샘플 수
        need_number = int((target_defect_ratio * len(x_dataset) - defect_number) / (1 - target_defect_ratio))
        if need_number <= 0:
            return False

        select_column = [c for c in x_dataset.columns if c != "bug"]
        total_ratio = 0.0

        total_pair = []
        generated_dataset = []

        # 각 결함 샘플 ratio 계산 (주변 정상 샘플 비율)
        for _, row in x_dataset.iterrows():
            if row["bug"] == 0:
                continue

            copy_dataset = x_dataset[select_column].copy(deep=True)
            copy_bug = x_dataset["bug"].copy()

            row_feat = row.drop(labels=["bug"]).to_numpy(dtype=float)
            X_all = copy_dataset.to_numpy(dtype=float)

            dist = np.sqrt(((X_all - row_feat) ** 2).sum(axis=1))
            tmp = copy_dataset.copy()
            tmp["distance"] = dist
            tmp["bug"] = copy_bug.values

            sort_dataset = tmp.sort_values(by="distance", ascending=True)
            nearest_dataset = sort_dataset.iloc[1:self.z_nearest + 1]
            majority_number = len(nearest_dataset[nearest_dataset["bug"] == 0])
            ratio = majority_number / self.z_nearest
            total_ratio += ratio

        # total_ratio가 0이면(주변에 정상 이웃이 거의 없다면) 안전 처리
        if total_ratio == 0:
            return pd.concat([clean_dataset, defect_dataset], ignore_index=True)

        # 각 결함 샘플별 생성할 합성 샘플 결정
        for index, row in defect_dataset.iterrows():
            copy_dataset = x_dataset[select_column].copy(deep=True)
            copy_bug = x_dataset["bug"].copy()

            row_feat = row.drop(labels=["bug"]).to_numpy(dtype=float)
            X_all = copy_dataset.to_numpy(dtype=float)

            dist = np.sqrt(((X_all - row_feat) ** 2).sum(axis=1))
            tmp = copy_dataset.copy()
            tmp["distance"] = dist
            tmp["bug"] = copy_bug.values

            sort_dataset = tmp.sort_values(by="distance", ascending=True)
            nearest_dataset = sort_dataset.iloc[1:self.z_nearest + 1]
            majority_number = len(nearest_dataset[nearest_dataset["bug"] == 0])
            ratio = majority_number / self.z_nearest

            normalized_ratio = ratio / total_ratio
            single_need_number = round(normalized_ratio * need_number)

            # 결함 샘플들 중에서 k-최근접 이웃 찾기
            copy_defect_dataset = defect_dataset.copy(deep=True)
            # bug 포함한 거리(원 코드 방식) 유지
            dist_def = np.sqrt(((copy_defect_dataset.to_numpy(dtype=float) - row.to_numpy(dtype=float)) ** 2).sum(axis=1))
            copy_defect_dataset["distance"] = dist_def
            sort_defect_dataset = copy_defect_dataset.sort_values(by="distance", ascending=True)
            neighbors = sort_defect_dataset.iloc[1:self.z_nearest + 1].drop(columns=["distance"])

            # 각 이웃과의 페어 생성 (결정적 방식)
            rround = single_need_number / self.z_nearest
            while rround >= 1:
                for a, _r in neighbors.iterrows():
                    pair = [index, a]
                    pair.sort()
                    total_pair.append(pair)
                rround -= 1

            number_on_each_instance = round(rround * self.z_nearest)
            if number_on_each_instance > 0:
                # nearest 중 먼쪽부터 선택
                copy_defect_dataset2 = defect_dataset.copy(deep=True)
                dist_def2 = np.sqrt(((copy_defect_dataset2.to_numpy(dtype=float) - row.to_numpy(dtype=float)) ** 2).sum(axis=1))
                copy_defect_dataset2["distance"] = dist_def2
                neighbors2 = copy_defect_dataset2.sort_values(by="distance", ascending=True).iloc[1:self.z_nearest + 1]
                neighbors2 = neighbors2.sort_values(by="distance", ascending=False).drop(columns=["distance"])
                target_sample_instance = neighbors2.iloc[0:number_on_each_instance]
                for a, _r in target_sample_instance.iterrows():
                    pair = [index, a]
                    pair.sort()
                    total_pair.append(pair)

        # 페어별 생성 횟수 집계 및 합성 샘플 생성
        total_pair_tuple = [tuple(l) for l in total_pair]
        pair_counter = Counter(total_pair_tuple)

        for (row1_index, row2_index), generated_num in pair_counter.items():
            row1 = defect_dataset.loc[row1_index]
            row2 = defect_dataset.loc[row2_index]
            generated_instances = np.linspace(row1, row2, int(generated_num) + 2)
            generated_instances = generated_instances[1:-1]
            for w in generated_instances.tolist():
                generated_dataset.append(w)

        final_generated_dataset = pd.DataFrame(generated_dataset, columns=x_dataset.columns)
        result = pd.concat([clean_dataset, defect_dataset, final_generated_dataset], ignore_index=True)
        return result


# -------------------------
# Out-of-sample bootstrap 분리
# -------------------------
def separate_data(original_data):
    original_data = np.array(original_data).tolist()
    size = len(original_data)
    train_dataset = []
    train_index = []

    for _ in range(size):
        idx = random.randint(0, size - 1)
        train_dataset.append(original_data[idx])
        train_index.append(idx)

    original_index = list(range(size))
    train_index = list(set(train_index))
    test_index = list(set(original_index).difference(set(train_index)))

    original_data = np.array(original_data)
    train_dataset = original_data[train_index]
    test_dataset = original_data[test_index]
    return train_dataset, test_dataset


# -------------------------
# 결과 저장용 CSV 파일 생성 (✅ 출력경로/파일명 suffix 적용)
# -------------------------
auc_file = open(OUTPUT_DIR + f"{neighbor}auc_adasyn_result_on_{classifier}_{dataset_tag}.csv", "w", newline="")
auc_writer = csv.writer(auc_file)
auc_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

balance_file = open(OUTPUT_DIR + f"{neighbor}balance_adasyn_result_on_{classifier}_{dataset_tag}.csv", "w", newline="")
balance_writer = csv.writer(balance_file)
balance_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

recall_file = open(OUTPUT_DIR + f"{neighbor}recall_adasyn_result_on_{classifier}_{dataset_tag}.csv", "w", newline="")
recall_writer = csv.writer(recall_file)
recall_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

pf_file = open(OUTPUT_DIR + f"{neighbor}pf_adasyn_result_on_{classifier}_{dataset_tag}.csv", "w", newline="")
pf_writer = csv.writer(pf_file)
pf_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max"])

brier_file = open(OUTPUT_DIR + f"{neighbor}brier_adasyn_result_on_{classifier}_{dataset_tag}.csv", "w", newline="")
brier_writer = csv.writer(brier_file)
brier_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

mcc_file = open(OUTPUT_DIR + f"{neighbor}mcc_adasyn_result_on_{classifier}_{dataset_tag}.csv", "w", newline="")
mcc_writer = csv.writer(mcc_file)
mcc_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

stable_auc_file = open(OUTPUT_DIR + f"{neighbor}auc_stable_adasyn_result_on_{classifier}_{dataset_tag}.csv", "w", newline="")
stable_auc_writer = csv.writer(stable_auc_file)
stable_auc_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

stable_balance_file = open(OUTPUT_DIR + f"{neighbor}balance_stable_adasyn_result_on_{classifier}_{dataset_tag}.csv", "w", newline="")
stable_balance_writer = csv.writer(stable_balance_file)
stable_balance_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

stable_recall_file = open(OUTPUT_DIR + f"{neighbor}recall_stable_adasyn_result_on_{classifier}_{dataset_tag}.csv", "w", newline="")
stable_recall_writer = csv.writer(stable_recall_file)
stable_recall_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

stable_pf_file = open(OUTPUT_DIR + f"{neighbor}pf_stable_adasyn_result_on_{classifier}_{dataset_tag}.csv", "w", newline="")
stable_pf_writer = csv.writer(stable_pf_file)
stable_pf_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

stable_brier_file = open(OUTPUT_DIR + f"{neighbor}brier_stable_adasyn_result_on_{classifier}_{dataset_tag}.csv", "w", newline="")
stable_brier_writer = csv.writer(stable_brier_file)
stable_brier_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

stable_mcc_file = open(OUTPUT_DIR + f"{neighbor}mcc_stable_adasyn_result_on_{classifier}_{dataset_tag}.csv", "w", newline="")
stable_mcc_writer = csv.writer(stable_mcc_file)
stable_mcc_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])


# =========================
# 메인 실행 (bank.csv 단일)
# =========================
if __name__ == "__main__":
    inputfile = "bank.csv"
    print(inputfile)
    start_time = time.asctime(time.localtime(time.time()))
    print("start_time:", start_time)

    dataset = load_and_preprocess_bank(INPUT_CSV_PATH)

    total_number = len(dataset)
    defect_ratio = len(dataset[dataset["bug"] > 0]) / total_number
    print("Total samples:", total_number)
    print("Minority ratio (bug=1):", round(defect_ratio, 4))

    # (원 코드 유지) defect ratio > 0.45면 skip
    if defect_ratio > 0.45:
        print(inputfile, " defect ratio larger than 0.45")
        raise SystemExit(0)

    # GridSearchCV용 validation split
    validation_train_data, validation_test_data = separate_data(dataset)
    while (
        len(validation_train_data[validation_train_data[:, -1] == 0]) == 0
        or len(validation_test_data[validation_test_data[:, -1] == 1]) == 0
        or len(validation_train_data[validation_train_data[:, -1] == 1]) <= neighbor
        or len(validation_test_data[validation_test_data[:, -1] == 0]) == 0
        or len(validation_train_data[validation_train_data[:, -1] == 1]) >= len(validation_train_data[validation_train_data[:, -1] == 0])
    ):
        validation_train_data, validation_test_data = separate_data(dataset)

    validation_train_x = validation_train_data[:, 0:-1]
    validation_train_y = validation_train_data[:, -1]

    print("GridSearchCV start...")
    validation_clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=5, n_jobs=-1)
    validation_clf.fit(validation_train_x, validation_train_y)
    best_parameter = validation_clf.best_params_
    best_c = best_parameter["C"]
    best_kernal = best_parameter["kernel"]
    print("Best params:", best_parameter)
    print("--------------------------------------------")

    # 10회 outer loop
    for j in range(10):
        auc_row, balance_row, recall_row, pf_row, brier_row, mcc_row = [], [], [], [], [], []
        stable_auc_row, stable_balance_row, stable_recall_row, stable_pf_row, stable_brier_row, stable_mcc_row = [], [], [], [], [], []

        train_data, test_data = separate_data(dataset)
        while (
            len(train_data[train_data[:, -1] == 0]) == 0
            or len(test_data[test_data[:, -1] == 1]) == 0
            or len(train_data[train_data[:, -1] == 1]) <= neighbor
            or len(test_data[test_data[:, -1] == 0]) == 0
            or len(train_data[train_data[:, -1] == 1]) >= len(train_data[train_data[:, -1] == 0])
        ):
            train_data, test_data = separate_data(dataset)

        train_x = train_data[:, 0:-1]
        train_y = train_data[:, -1]

        # 10회 inner loop
        for s in range(10):
            # 1) 일반 ADASYN
            adasyn = ADASYN(n_neighbors=neighbor)
            adasyn_train_x, adasyn_train_y = adasyn.fit_resample(train_x, train_y)

            test_x = test_data[:, 0:-1]
            test_y = test_data[:, -1]

            clf = svm.SVC(C=best_c, kernel=best_kernal)
            clf.fit(adasyn_train_x, adasyn_train_y)
            predict_result = clf.predict(test_x)

            tn, fp, fn, tp = confusion_matrix(test_y, predict_result).ravel()

            brier = brier_score_loss(test_y, predict_result)
            mcc = matthews_corrcoef(test_y, predict_result)
            recall = recall_score(test_y, predict_result)

            pf = fp / (tn + fp) if (tn + fp) > 0 else 0
            balance = 1 - (((0 - pf) ** 2 + (1 - recall) ** 2) / 2) ** 0.5
            auc = roc_auc_score(test_y, predict_result)

            auc_row.append(auc)
            balance_row.append(balance)
            recall_row.append(recall)
            pf_row.append(pf)
            brier_row.append(brier)
            mcc_row.append(mcc)

            # 2) Stable ADASYN
            stable_adasyn = stable_ADASYN(neighbor)
            stable_adasyn_train = stable_adasyn.fit_sample(train_data, train_y)

            if stable_adasyn_train is False:
                stable_auc_row.append(auc)
                stable_balance_row.append(balance)
                stable_recall_row.append(recall)
                stable_pf_row.append(pf)
                stable_brier_row.append(brier)
                stable_mcc_row.append(mcc)
                continue

            stable_adasyn_train = np.array(stable_adasyn_train)
            stable_adasyn_train_x = stable_adasyn_train[:, 0:-1]
            stable_adasyn_train_y = stable_adasyn_train[:, -1]

            stable_clf = svm.SVC(C=best_c, kernel=best_kernal)
            stable_clf.fit(stable_adasyn_train_x, stable_adasyn_train_y)
            stable_predict_result = stable_clf.predict(test_x)

            stn, sfp, sfn, stp = confusion_matrix(test_y, stable_predict_result).ravel()

            stable_brier = brier_score_loss(test_y, stable_predict_result)
            stable_mcc = matthews_corrcoef(test_y, stable_predict_result)
            stable_recall = recall_score(test_y, stable_predict_result)

            stable_pf = sfp / (stn + sfp) if (stn + sfp) > 0 else 0
            stable_balance = 1 - (((0 - stable_pf) ** 2 + (1 - stable_recall) ** 2) / 2) ** 0.5
            stable_auc = roc_auc_score(test_y, stable_predict_result)

            stable_auc_row.append(stable_auc)
            stable_balance_row.append(stable_balance)
            stable_recall_row.append(stable_recall)
            stable_pf_row.append(stable_pf)
            stable_brier_row.append(stable_brier)
            stable_mcc_row.append(stable_mcc)

        # ---- 통계 기록(원 코드 방식 유지) ----
        def stats_append(row, ddof_1=False):
            mx = max(row); mn = min(row)
            avg = np.mean(row)
            med = np.median(row)
            q = np.percentile(row, (25, 75), method="midpoint")
            lo, up = q[0], q[1]
            var = np.std(row, ddof=1) if ddof_1 else np.std(row)
            return row + [mn, lo, avg, med, up, mx, var]

        # brier
        out = stats_append(brier_row, ddof_1=True); out.insert(0, inputfile + " brier"); brier_writer.writerow(out)
        out = stats_append(stable_brier_row, ddof_1=True); out.insert(0, inputfile + " brier"); stable_brier_writer.writerow(out)

        # auc
        out = stats_append(auc_row, ddof_1=True); out.insert(0, inputfile + " auc"); auc_writer.writerow(out)
        out = stats_append(stable_auc_row, ddof_1=True); out.insert(0, inputfile + " auc"); stable_auc_writer.writerow(out)

        # balance
        out = stats_append(balance_row, ddof_1=True); out.insert(0, inputfile + " balance"); balance_writer.writerow(out)
        out = stats_append(stable_balance_row, ddof_1=True); out.insert(0, inputfile + " balance"); stable_balance_writer.writerow(out)

        # recall
        out = stats_append(recall_row, ddof_1=True); out.insert(0, inputfile + " recall"); recall_writer.writerow(out)
        out = stats_append(stable_recall_row, ddof_1=True); out.insert(0, inputfile + " recall"); stable_recall_writer.writerow(out)

        # pf (원 코드: pf_file은 variance 헤더가 없어서 6개만)
        def stats_pf(row):
            mx = max(row); mn = min(row)
            avg = np.mean(row)
            med = np.median(row)
            q = np.percentile(row, (25, 75), method="midpoint")
            lo, up = q[0], q[1]
            return row + [mn, lo, avg, med, up, mx]

        out = stats_pf(pf_row); out.insert(0, inputfile + " pf"); pf_writer.writerow(out)

        # stable pf는 헤더에 variance가 있으므로 그대로 variance까지 기록
        out = stats_append(stable_pf_row, ddof_1=True); out.insert(0, inputfile + " pf"); stable_pf_writer.writerow(out)

        # mcc
        out = stats_append(mcc_row, ddof_1=False); out.insert(0, inputfile + " mcc"); mcc_writer.writerow(out)
        out = stats_append(stable_mcc_row, ddof_1=False); out.insert(0, inputfile + " mcc"); stable_mcc_writer.writerow(out)

        print(f"[Outer {j+1}/10] done")

    # 파일 닫기
    auc_file.close()
    balance_file.close()
    recall_file.close()
    pf_file.close()
    brier_file.close()
    mcc_file.close()

    stable_auc_file.close()
    stable_balance_file.close()
    stable_recall_file.close()
    stable_pf_file.close()
    stable_brier_file.close()
    stable_mcc_file.close()

    end_time = time.asctime(time.localtime(time.time()))
    print("============================================")
    print("Experiment completed!")
    print("Start:", start_time)
    print("End  :", end_time)
    print("Saved to:", OUTPUT_DIR)
    print("============================================")
