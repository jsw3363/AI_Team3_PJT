# ============================================================
# Stable BORDERLINE-SMOTE with SVM (Bank Marketing - bank.csv)
# ------------------------------------------------------------
# ✔ 입력 데이터: new_input_data/bank.csv  (UCI Bank Marketing 10% / sep=";")
# ✔ 출력 폴더: new_output_data/new_BORDERLINE_svm/
# ✔ 기존 코드 흐름(부트스트랩 분리, 10x10 반복, 지표 계산/CSV 저장)은 유지
# ✔ PROMISE 결함데이터 형식(bug 컬럼)을 그대로 쓰기 위해:
#   - 타깃 y(yes/no) -> bug(0/1)로 변환 후 y 제거
#   - 범주형 feature는 LabelEncoder로 수치화
# ============================================================

from imblearn.over_sampling import BorderlineSMOTE

import os
import csv
import time
import random
import warnings
from collections import Counter

import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
    brier_score_loss
)

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)

# -------------------------
# 전역 설정
# -------------------------
classifier = "svm"
neighbor = 5
tuned_parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10, 100, 1000]}
target_defect_ratio = 0.5

INPUT_CSV_PATH = "new_input_data/bank.csv"  # ✅ 여기 고정
OUTPUT_DIR = "new_output_data/new_BORDERLINE_svm/"  # ✅ 여기 고정
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------
# Stable Borderline (커스텀) 구현
# -------------------------
class stable_SMOTE:
    """
    기존 'Stable Borderline' 로직을 Bank feature 컬럼에 맞게 일반화한 버전.
    - 입력: x_dataset (numpy array or DataFrame) 마지막 컬럼이 bug(0/1)
    - 반환: (clean + defect + generated) DataFrame
    """
    def __init__(self, z_nearest=5):
        self.z_nearest = z_nearest

    def fit_sample(self, x_dataset, y_dataset=None):
        x_dataset = pd.DataFrame(x_dataset).copy()

        # 마지막 컬럼을 bug로 가정
        feature_cols = list(range(x_dataset.shape[1] - 1))
        x_dataset = x_dataset.rename(columns={**{i: f"f{i}" for i in feature_cols}, x_dataset.shape[1]-1: "bug"})

        # 결함/정상 분리
        defective_instance = x_dataset[x_dataset["bug"] > 0]
        clean_instance = x_dataset[x_dataset["bug"] == 0]
        defective_number = len(defective_instance)

        # 목표 비율 달성을 위한 필요 샘플 수
        need_number = int((target_defect_ratio * len(x_dataset) - defective_number) / (1 - target_defect_ratio))
        if need_number <= 0:
            return False

        # STEP 1: Borderline 결함 인스턴스 식별
        # - 최근접 이웃 중 정상(majority) 비율이 "절반 이상 but 전부는 아님"인 경우만 사용
        container = pd.DataFrame(columns=[*x_dataset.columns])

        X_feat_all = x_dataset.drop(columns=["bug"]).to_numpy(dtype=float)
        y_all = x_dataset["bug"].to_numpy(dtype=float)

        k = self.z_nearest

        for idx, row in x_dataset.iterrows():
            if row["bug"] == 0:
                continue

            x = row.drop(labels=["bug"]).to_numpy(dtype=float)
            # 유클리드 거리
            dist = np.sqrt(((X_feat_all - x) ** 2).sum(axis=1))
            # 자기 자신 제외하고 k개
            nn_idx = np.argsort(dist)[1:k+1]
            majority_number = np.sum(y_all[nn_idx] == 0)

            # 원본 코드 조건 일반화:
            # majority==k(전부 정상) 이거나 majority<k/2(정상이 절반 미만)이면 skip
            if majority_number == k or majority_number < (k / 2):
                continue

            container = pd.concat([container, row.to_frame().T], ignore_index=False)

        involve_defective_number = len(container)
        if involve_defective_number == 0:
            # borderline 후보가 없으면 그냥 원본(증강 없이) 반환
            return pd.concat([clean_instance, defective_instance])

        # STEP 2: 각 인스턴스당 생성할 샘플 수 계산 및 이웃 페어 선택
        generated_dataset = []
        total_pair = []

        number_on_each_instance = need_number / involve_defective_number

        # 첫 번째 라운드: 모든 최근접 이웃과 페어링
        rround = number_on_each_instance / k
        while rround >= 1:
            for index, row in container.iterrows():
                temp_defective_instance = defective_instance.copy(deep=True)

                # 거리 계산 (bug 포함해도 동일 차원이라 원본 스타일 유지하되, 사실상 feature 기반이 더 정확)
                subtraction = row - temp_defective_instance
                square = subtraction ** 2
                row_sum = square.apply(lambda s: s.sum(), axis=1)
                distance = row_sum ** 0.5

                temp_defective_instance["distance"] = distance
                temp_defective_instance = temp_defective_instance.sort_values(by="distance", ascending=True)
                neighbors = temp_defective_instance.iloc[1:k+1]  # 자기 자신 제외

                for a, _r in neighbors.iterrows():
                    selected_pair = [index, a]
                    selected_pair.sort()
                    total_pair.append(selected_pair)

            rround -= 1

        # 두 번째 라운드: 추가 필요 샘플을 위한 먼 이웃 선택
        need_number1 = need_number - len(total_pair)
        number_on_each_instance = need_number1 / involve_defective_number

        for index, row in container.iterrows():
            temp_defective_instance = defective_instance.copy(deep=True)

            subtraction = row - temp_defective_instance
            square = subtraction ** 2
            row_sum = square.apply(lambda s: s.sum(), axis=1)
            distance = row_sum ** 0.5

            temp_defective_instance["distance"] = distance
            temp_defective_instance = temp_defective_instance.sort_values(by="distance", ascending=True)

            neighbors = temp_defective_instance.iloc[1:k+1]
            neighbors = neighbors.sort_values(by="distance", ascending=False)  # nearest 중 가장 먼 쪽부터

            take_n = int(number_on_each_instance)
            if take_n <= 0:
                continue

            target_sample_instance = neighbors.iloc[0:take_n].drop(columns="distance")

            for a, _r in target_sample_instance.iterrows():
                selected_pair = [index, a]
                selected_pair.sort()
                total_pair.append(selected_pair)

        # 세 번째 라운드: 잔여 샘플 처리
        residue_number = need_number - len(total_pair)
        if residue_number > 0:
            residue_defective_instance = container.sample(n=residue_number, replace=True, random_state=None)
            for index, row in residue_defective_instance.iterrows():
                temp_defective_instance = defective_instance.copy(deep=True)

                subtraction = row - temp_defective_instance
                square = subtraction ** 2
                row_sum = square.apply(lambda s: s.sum(), axis=1)
                distance = row_sum ** 0.5

                temp_defective_instance["distance"] = distance
                temp_defective_instance = temp_defective_instance.sort_values(by="distance", ascending=True)

                neighbors = temp_defective_instance.iloc[1:k+1]
                target_sample_instance = neighbors.iloc[-1:]  # 가장 먼 1개

                for a in target_sample_instance.index:
                    selected_pair = [index, a]
                    selected_pair.sort()
                    total_pair.append(selected_pair)

        # STEP 3: 선택된 페어 간 선형 보간으로 합성 샘플 생성
        total_pair_tuple = [tuple(l) for l in total_pair]
        result = Counter(total_pair_tuple)

        for (row1_index, row2_index), generated_num in result.items():
            row1 = defective_instance.loc[row1_index]
            row2 = defective_instance.loc[row2_index]

            generated_instances = np.linspace(row1, row2, int(generated_num) + 2)
            generated_instances = generated_instances[1:-1]  # 양 끝 제거
            for w in generated_instances.tolist():
                generated_dataset.append(w)

        final_generated_dataset = pd.DataFrame(generated_dataset, columns=x_dataset.columns)

        # 결합
        result = pd.concat([clean_instance, defective_instance, final_generated_dataset], ignore_index=True)
        return result


# -------------------------
# Out-of-Sample Bootstrap 분리
# -------------------------
def separate_data(original_data):
    original_data = np.array(original_data).tolist()
    size = len(original_data)
    train_dataset = []
    train_index = []

    for _ in range(size):
        index = random.randint(0, size - 1)
        train_dataset.append(original_data[index])
        train_index.append(index)

    original_index = list(range(size))
    train_index = list(set(train_index))
    test_index = list(set(original_index).difference(set(train_index)))

    original_data = np.array(original_data)
    train_dataset = original_data[train_index]
    test_dataset = original_data[test_index]
    return train_dataset, test_dataset


# -------------------------
# Bank Marketing 전처리
# -------------------------
def load_and_preprocess_bank(csv_path: str) -> pd.DataFrame:
    # bank.csv는 sep=";" / 타깃 컬럼 'y'
    df = pd.read_csv(csv_path, sep=";")

    if "y" not in df.columns:
        raise ValueError(f"'y' column not found in {csv_path}. columns={df.columns.tolist()}")

    # y -> bug(0/1), 그리고 y 제거
    df["bug"] = (df["y"] == "yes").astype(int)
    df = df.drop(columns=["y"])

    # 범주형 -> LabelEncoding
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Min-Max Scaling (bug 제외)
    feature_cols = [c for c in df.columns if c != "bug"]
    for col in feature_cols:
        cmax, cmin = df[col].max(), df[col].min()
        if cmax != cmin:
            df[col] = (df[col] - cmin) / (cmax - cmin)
        else:
            df[col] = 0.0

    # bug는 0/1 그대로
    df["bug"] = df["bug"].astype(int)
    return df


# -------------------------
# 결과 저장 CSV 초기화
# -------------------------
# 파일명 뒤에 붙일 태그
dataset_tag = "bank_marketing"

auc_file = open(OUTPUT_DIR + f"{neighbor}auc_borderline_result_on_{classifier}_{dataset_tag}.csv", "w", newline="")
auc_writer = csv.writer(auc_file)
auc_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

balance_file = open(OUTPUT_DIR + f"{neighbor}balance_borderline_result_on_{classifier}_{dataset_tag}.csv", "w", newline="")
balance_writer = csv.writer(balance_file)
balance_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

recall_file = open(OUTPUT_DIR + f"{neighbor}recall_borderline_result_on_{classifier}_{dataset_tag}.csv", "w", newline="")
recall_writer = csv.writer(recall_file)
recall_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

pf_file = open(OUTPUT_DIR + f"{neighbor}pf_borderline_result_on_{classifier}_{dataset_tag}.csv", "w", newline="")
pf_writer = csv.writer(pf_file)
pf_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max"])

brier_file = open(OUTPUT_DIR + f"{neighbor}brier_borderline_result_on_{classifier}_{dataset_tag}.csv", "w", newline="")
brier_writer = csv.writer(brier_file)
brier_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

mcc_file = open(OUTPUT_DIR + f"{neighbor}mcc_borderline_result_on_{classifier}_{dataset_tag}.csv", "w", newline="")
mcc_writer = csv.writer(mcc_file)
mcc_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

stable_auc_file = open(OUTPUT_DIR + f"{neighbor}auc_stable_borderline_result_on_{classifier}_{dataset_tag}.csv", "w", newline="")
stable_auc_writer = csv.writer(stable_auc_file)
stable_auc_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

stable_balance_file = open(OUTPUT_DIR + f"{neighbor}balance_stable_borderline_result_on_{classifier}_{dataset_tag}.csv", "w", newline="")
stable_balance_writer = csv.writer(stable_balance_file)
stable_balance_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

stable_recall_file = open(OUTPUT_DIR + f"{neighbor}recall_stable_borderline_result_on_{classifier}_{dataset_tag}.csv", "w", newline="")
stable_recall_writer = csv.writer(stable_recall_file)
stable_recall_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

stable_pf_file = open(OUTPUT_DIR + f"{neighbor}pf_stable_borderline_result_on_{classifier}_{dataset_tag}.csv", "w", newline="")
stable_pf_writer = csv.writer(stable_pf_file)
stable_pf_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

stable_brier_file = open(OUTPUT_DIR + f"{neighbor}brier_stable_borderline_result_on_{classifier}_{dataset_tag}.csv", "w", newline="")
stable_brier_writer = csv.writer(stable_brier_file)
stable_brier_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

stable_mcc_file = open(OUTPUT_DIR + f"{neighbor}mcc_stable_borderline_result_on_{classifier}_{dataset_tag}.csv", "w", newline="")
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

    # 로드 + 전처리
    dataset = load_and_preprocess_bank(INPUT_CSV_PATH)

    total_number = len(dataset)
    defect_ratio = len(dataset[dataset["bug"] > 0]) / total_number
    print("Total samples:", total_number)
    print("Minority ratio (bug=1):", round(defect_ratio, 4))
    print("Majority ratio (bug=0):", round(1 - defect_ratio, 4))

    # (원 코드 유지) 결함 비율이 0.45 이상이면 skip
    if defect_ratio > 0.45:
        print(inputfile, " defect ratio larger than 0.45")
        raise SystemExit(0)

    # x/y 분리 (원 코드 흐름 유지)
    x = dataset.drop(columns="bug")
    y = dataset["bug"]

    # GridSearchCV 최적 파라미터 탐색 (부트스트랩 split 기반)
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

    # 10회의 독립적인 실험 반복 (외부 루프)
    for j in range(10):
        auc_row, balance_row, recall_row, pf_row, brier_row, mcc_row = [], [], [], [], [], []
        stable_auc_row, stable_balance_row, stable_recall_row, stable_pf_row, stable_brier_row, stable_mcc_row = [], [], [], [], [], []

        # Train/Test 분리 (유효성 검증 포함)
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

        # 10회 반복 실험 (내부 루프)
        for s in range(10):
            # 1) 기본 BorderlineSMOTE
            smote = BorderlineSMOTE(k_neighbors=neighbor, kind='borderline-1')
            smote_train_x, smote_train_y = smote.fit_resample(train_x, train_y)

            test_x = test_data[:, 0:-1]
            test_y = test_data[:, -1]

            clf = svm.SVC(C=best_c, kernel=best_kernal)
            clf.fit(smote_train_x, smote_train_y)
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

            # 2) Stable Borderline(커스텀)
            stable_smote = stable_SMOTE(z_nearest=neighbor)
            stable_smote_train = stable_smote.fit_sample(train_data, train_y)

            # 실패 시 기본 결과로 대체(원 코드 스타일과 유사하게 안전 처리)
            if stable_smote_train is False:
                stable_auc_row.append(auc)
                stable_balance_row.append(balance)
                stable_recall_row.append(recall)
                stable_pf_row.append(pf)
                stable_brier_row.append(brier)
                stable_mcc_row.append(mcc)
                continue

            stable_smote_train = np.array(stable_smote_train)
            stable_smote_train_x = stable_smote_train[:, 0:-1]
            stable_smote_train_y = stable_smote_train[:, -1]

            stable_clf = svm.SVC(C=best_c, kernel=best_kernal)
            stable_clf.fit(stable_smote_train_x, stable_smote_train_y)
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

        # ---- 통계 저장 (원 코드 포맷 유지) ----
        def write_stats(writer, row, tag):
            mx = max(row); mn = min(row); avg = np.mean(row); med = np.median(row)
            q = np.percentile(row, (25, 75), method='midpoint')
            lo, up = q[0], q[1]
            var = np.std(row)
            out = row + [mn, lo, avg, med, up, mx, var]
            out.insert(0, inputfile + f" {tag}")
            writer.writerow(out)

        # PF는 variance 컬럼이 원래 없음(원 코드 유지)
        def write_stats_pf(writer, row, tag):
            mx = max(row); mn = min(row); avg = np.mean(row); med = np.median(row)
            q = np.percentile(row, (25, 75), method='midpoint')
            lo, up = q[0], q[1]
            out = row + [mn, lo, avg, med, up, mx]
            out.insert(0, inputfile + f" {tag}")
            writer.writerow(out)

        write_stats(brier_writer, brier_row, "brier")
        write_stats(stable_brier_writer, stable_brier_row, "brier")

        write_stats(auc_writer, auc_row, "auc")
        write_stats(stable_auc_writer, stable_auc_row, "auc")

        write_stats(balance_writer, balance_row, "balance")
        write_stats(stable_balance_writer, stable_balance_row, "balance")

        write_stats(recall_writer, recall_row, "recall")
        write_stats(stable_recall_writer, stable_recall_row, "recall")

        write_stats_pf(pf_writer, pf_row, "pf")
        # 원래 stable_pf_file은 variance가 있었는데 헤더와 불일치 가능성이 있어
        # 기존 코드 출력 형식 유지하려면 stable도 variance 포함 버전(write_stats) 쓰고 싶다면 아래 줄을 write_stats로 바꿔도 됨.
        write_stats(stable_pf_writer, stable_pf_row, "pf")

        write_stats(mcc_writer, mcc_row, "mcc")
        write_stats(stable_mcc_writer, stable_mcc_row, "mcc")

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
