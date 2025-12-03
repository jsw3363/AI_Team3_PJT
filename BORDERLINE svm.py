# stabled BORDERLINE

from imblearn.over_sampling import SMOTE, ADASYN
# SMOTE -> kind 파라미터 지원 중단으로 BorderlineSMOTE 이용
from imblearn.over_sampling import BorderlineSMOTE

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
import csv
import warnings
from sklearn import neighbors
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn import svm
from sklearn.metrics import brier_score_loss
import random
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from sklearn import tree
import time
from sklearn.model_selection import GridSearchCV
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=196)

# 전역 설정: 분류기 및 하이퍼파라미터 설정
classifier_for_selection = {"svm": svm.SVC(gamma='auto'), "knn": neighbors.KNeighborsClassifier(), "rf": RandomForestClassifier(), "tree": tree.DecisionTreeClassifier()}
classifier = "svm"
np.set_printoptions(suppress=True)
# fold = 5
neighbor = 5
tuned_parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10, 100, 1000]}
target_defect_ratio = 0.5

# Stable SMOTE 구현 : Borderline-SMOTE로 사용할 것
class stable_SMOTE:
    def __init__(self, z_nearest=5):
        self.z_nearest = z_nearest
    #     self.x = x
    #     self.y = y
    #     self.k = k
    #     self.distance_metric = distance_metric
    def fit_sample(self, x_dataset, y_dataset):
        total_pair = []
        # print(k_nearest)
        x_dataset = pd.DataFrame(x_dataset)
        x_dataset = x_dataset.rename(
            columns={0: "wmc", 1: "dit", 2: "noc", 3: "cbo", 4: "rfc", 5: "lcom", 6: "ca", 7: "ce", 8: "npm",
                     9: "lcom3", 10: "loc", 11: "dam", 12: "moa", 13: "mfa", 14: "cam", 15: "ic", 16: "cbm", 17: "amc",
                     18: "max_cc", 19: "avg_cc", 20: "bug"}
        )
        # 결함/정상 인스턴스 분리
        defective_instance = x_dataset[x_dataset["bug"] > 0]
        clean_instance = x_dataset[x_dataset["bug"] == 0]
        defective_number = len(defective_instance)
        clean_number = len(clean_instance)

        # 목표 결함 비율 달성을 위한 필요 샘플 수 계산
        need_number = int((target_defect_ratio * len(x_dataset) - defective_number) / (1 - target_defect_ratio))  # clean_number - defective_number
        if need_number <= 0:
            return False
        generated_dataset = []
        synthetic_dataset = pd.DataFrame()

        select_column = ["wmc", "dit", "noc", "cbo", "rfc", "lcom", "ca", "ce", "npm", "lcom3", "loc", "dam", "moa",
                         "mfa", "cam", "ic", "cbm", "amc", "max_cc", "avg_cc"]
        count = 0
        container = pd.DataFrame(
            columns=["wmc", "dit", "noc", "cbo", "rfc", "lcom", "ca", "ce", "npm", "lcom3", "loc", "dam", "moa",
                     "mfa", "cam", "ic", "cbm", "amc", "max_cc", "avg_cc", "bug"])


        # STEP 1: Borderline 결함 인스턴스 식별
        # - 최근접 이웃 중 정상 인스턴스가 적정 비율인 경우만 선택
        for g, h in x_dataset.iterrows():
            if h["bug"] == 0:
                continue
            copy_dataset = x_dataset.copy(deep=True)
            copy_dataset = copy_dataset[select_column]
            copy_bug = x_dataset["bug"]
            # 유클리드 거리 계산
            euclidean = (h - copy_dataset) ** 2
            euclidean_sum = euclidean.apply(lambda s: s.sum(), axis=1)
            euclidean_distance = np.sqrt(euclidean_sum)
            copy_dataset["distance"] = euclidean_distance
            copy_dataset["bug"] = copy_bug
            euclidean = copy_dataset.sort_values(by="distance", ascending=True)
            euclidean = euclidean.iloc[1:self.z_nearest + 1]
            # Borderline 조건 확인 (최근접 이웃 중 정상 인스턴스 비율)
            majority_number = len(euclidean[euclidean["bug"] == 0])
            if majority_number == 5 or majority_number < 2.5:
                continue
            count = count + 1
            h = pd.DataFrame(h)
            container = pd.concat([container, h.T])
        involve_defective_number = len(container)
        if involve_defective_number == 0:
            result = pd.concat([clean_instance, defective_instance])
            return result
        # STEP 2: 각 인스턴스당 생성할 샘플 수 계산 및 이웃 페어 선택
        number_on_each_instance = need_number / involve_defective_number
        total_pair = []

        # 첫 번째 라운드: 모든 최근접 이웃과 페어링
        rround = number_on_each_instance / self.z_nearest
        while rround >= 1:
            # for index, row in defective_instance.iterrows():
            for index, row in container.iterrows():
                temp_defective_instance = defective_instance.copy(deep=True)
                subtraction = row - temp_defective_instance
                square = subtraction ** 2
                row_sum = square.apply(lambda s: s.sum(), axis=1)
                distance = row_sum ** 0.5
                temp_defective_instance["distance"] = distance
                temp_defective_instance = temp_defective_instance.sort_values(by="distance", ascending=True)
                neighbors = temp_defective_instance[1:self.z_nearest + 1]
                for a, r in neighbors.iterrows():
                    selected_pair = [index, a]
                    selected_pair.sort()
                    total_pair.append(selected_pair)
            rround = rround - 1
        # 두 번째 라운드: 추가 필요 샘플을 위한 먼 이웃 선택
        need_number1 = need_number - len(total_pair)
        number_on_each_instance = need_number1 / involve_defective_number

        # for index, row in defective_instance.iterrows():
        for index, row in container.iterrows():
            temp_defective_instance = defective_instance.copy(deep=True)
            subtraction = row - temp_defective_instance
            square = subtraction ** 2
            row_sum = square.apply(lambda s: s.sum(), axis=1)
            distance = row_sum ** 0.5

            temp_defective_instance["distance"] = distance
            temp_defective_instance = temp_defective_instance.sort_values(by="distance", ascending=True)
            neighbors = temp_defective_instance[1:self.z_nearest + 1]
            neighbors = neighbors.sort_values(by="distance", ascending=False)  # 这里取nearest neighbor里最远的
            target_sample_instance = neighbors[0: int(number_on_each_instance)]
            target_sample_instance = target_sample_instance.drop(columns="distance")
            for a, r in target_sample_instance.iterrows():
                selected_pair = [index, a]
                selected_pair.sort()
                total_pair.append(selected_pair)
        # 세 번째 라운드: 잔여 샘플 처리
        temp_defective_instance = defective_instance.copy(deep=True)
        residue_number = need_number - len(total_pair)
        residue_defective_instance = container.sample(n=residue_number)# temp_defective_instance.sample(n=residue_number)
        for index, row in residue_defective_instance.iterrows():
            temp_defective_instance = defective_instance.copy(deep=True)
            subtraction = row - temp_defective_instance
            square = subtraction ** 2
            row_sum = square.apply(lambda s: s.sum(), axis=1)
            distance = row_sum ** 0.5

            temp_defective_instance["distance"] = distance
            temp_defective_instance = temp_defective_instance.sort_values(by="distance", ascending=True)
            neighbors = temp_defective_instance[1:self.z_nearest + 1]
            target_sample_instance = neighbors[-1:]
            for a in target_sample_instance.index:
                selected_pair = [index, a]
                selected_pair.sort()
                total_pair.append(selected_pair)
        # STEP 3: 선택된 페어 간 선형 보간으로 합성 샘플 생성
        total_pair_tuple = [tuple(l) for l in total_pair]
        result = Counter(total_pair_tuple)
        result_number = len(result)
        result_keys = result.keys()
        result_values = result.values()
        for f in range(result_number):
            current_pair = list(result_keys)[f]
            row1_index = current_pair[0]
            row2_index = current_pair[1]
            row1 = defective_instance.loc[row1_index]
            row2 = defective_instance.loc[row2_index]
            generated_num = list(result_values)[f]
            # 두 인스턴스 간 선형 보간
            generated_instances = np.linspace(row1, row2, generated_num + 2)
            generated_instances = generated_instances[1:-1]
            generated_instances = generated_instances.tolist()
            for w in generated_instances:
                generated_dataset.append(w)

        final_generated_dataset = pd.DataFrame(generated_dataset)
        final_generated_dataset = final_generated_dataset.rename(
            columns={0: "wmc", 1: "dit", 2: "noc", 3: "cbo", 4: "rfc", 5: "lcom", 6: "ca", 7: "ce", 8: "npm",
                     9: "lcom3", 10: "loc", 11: "dam", 12: "moa", 13: "mfa", 14: "cam", 15: "ic", 16: "cbm", 17: "amc",
                     18: "max_cc", 19: "avg_cc", 20: "bug"}
        )
        # 원본 데이터와 합성 데이터 결합
        result = pd.concat([clean_instance, defective_instance, final_generated_dataset])
        return result
    
# Out-of-Sample Bootstrap으로 Train/Test 데이터 분리
def separate_data(original_data):
    '''
    Bootstrap 샘플링으로 훈련/테스트 세트 생성
    - 복원 추출로 N개 샘플 생성 (훈련 세트)
    - 선택되지 않은 약 36.8% 샘플을 테스트 세트로 사용
    '''
    original_data = np.array(original_data).tolist()
    size = len(original_data)
    train_dataset = []
    train_index = []
    # Bootstrap 샘플링 (복원 추출)
    for i in range(size):
        index = random.randint(0, size - 1)
        train_instance = original_data[index]
        train_dataset.append(train_instance)
        train_index.append(index)
    # 테스트 세트: 선택되지 않은 인스턴스
    original_index = [z for z in range(size)]
    train_index = list(set(train_index))
    test_index = list(set(original_index).difference(set(train_index)))
    original_data = np.array(original_data)
    train_dataset = original_data[train_index]
    # original_data = pd.DataFrame(original_data)
    # original_data = original_data.rename(
    #     columns={0: "wmc", 1: "dit", 2: "noc", 3: "cbo", 4: "rfc", 5: "lcom", 6: "ca", 7: "ce", 8: "npm",
    #              9: "lcom3", 10: "loc", 11: "dam", 12: "moa", 13: "mfa", 14: "cam", 15: "ic", 16: "cbm", 17: "amc",
    #              18: "max_cc", 19: "avg_cc", 20: "bug"})
    # train_dataset = pd.DataFrame(train_dataset)
    # train_dataset = train_dataset.rename(
    #     columns={0: "wmc", 1: "dit", 2: "noc", 3: "cbo", 4: "rfc", 5: "lcom", 6: "ca", 7: "ce", 8: "npm",
    #              9: "lcom3", 10: "loc", 11: "dam", 12: "moa", 13: "mfa", 14: "cam", 15: "ic", 16: "cbm", 17: "amc",
    #              18: "max_cc", 19: "avg_cc", 20: "bug"}
    # )
    test_dataset = original_data[test_index]
    return train_dataset, test_dataset


# 결과 저장을 위한 CSV 파일 초기화
# measure = "pf"
# for measure in ["auc", "balance", "recall", "pf", "brier"]:
auc_file = open('output_data/BORDERLINE_svm/'+str(neighbor)+'auc_borderline_result_on_'+classifier+'.csv', 'w', newline='')
auc_writer = csv.writer(auc_file)
auc_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

balance_file = open('output_data/BORDERLINE_svm/'+str(neighbor)+'balance_borderline_result_on_'+classifier+'.csv', 'w', newline='')
balance_writer = csv.writer(balance_file)
balance_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

recall_file = open('output_data/BORDERLINE_svm/'+str(neighbor)+'recall_borderline_result_on_'+classifier+'.csv', 'w', newline='')
recall_writer = csv.writer(recall_file)
recall_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

pf_file = open('output_data/BORDERLINE_svm/'+str(neighbor)+'pf_borderline_result_on_'+classifier+'.csv', 'w', newline='')
pf_writer = csv.writer(pf_file)
pf_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max"])

brier_file = open('output_data/BORDERLINE_svm/'+str(neighbor)+'brier_borderline_result_on_'+classifier+'.csv', 'w', newline='')
brier_writer = csv.writer(brier_file)
brier_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

mcc_file = open('output_data/BORDERLINE_svm/'+str(neighbor)+'mcc_borderline_result_on_'+classifier+'.csv', 'w', newline='')
mcc_writer = csv.writer(mcc_file)
mcc_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

stable_auc_file = open('output_data/BORDERLINE_svm/'+str(neighbor)+'auc_stable_borderline_result_on_'+classifier+'.csv', 'w',
              newline='')
stable_auc_writer = csv.writer(stable_auc_file)
stable_auc_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

stable_balance_file = open('output_data/BORDERLINE_svm/'+str(neighbor)+'balance_stable_borderline_result_on_'+classifier+'.csv', 'w',
              newline='')
stable_balance_writer = csv.writer(stable_balance_file)
stable_balance_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

stable_recall_file = open('output_data/BORDERLINE_svm/'+str(neighbor)+'recall_stable_borderline_result_on_'+classifier+'.csv', 'w',
              newline='')
stable_recall_writer = csv.writer(stable_recall_file)
stable_recall_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

stable_pf_file = open('output_data/BORDERLINE_svm/'+str(neighbor)+'pf_stable_borderline_result_on_'+classifier+'.csv', 'w',
              newline='')
stable_pf_writer = csv.writer(stable_pf_file)
stable_pf_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

stable_brier_file = open('output_data/BORDERLINE_svm/'+str(neighbor)+'brier_stable_borderline_result_on_'+classifier+'.csv', 'w',
              newline='')
stable_brier_writer = csv.writer(stable_brier_file)
stable_brier_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

stable_mcc_file = open('output_data/BORDERLINE_svm/'+str(neighbor)+'mcc_stable_borderline_result_on_'+classifier+'.csv', 'w',
              newline='')
stable_mcc_writer = csv.writer(stable_mcc_file)
stable_mcc_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

# 메인 실험 루프: 각 데이터셋에 대해 실험 수행
for inputfile in os.listdir("input_data/"):
    print(inputfile)
    start_time = time.asctime(time.localtime(time.time()))
    print("start_time: ", start_time)
    # 데이터 로드 및 전처리
    dataset = pd.read_csv("input_data/" + inputfile)
    dataset = dataset.drop(columns="name")
    dataset = dataset.drop(columns="version")
    dataset = dataset.drop(columns="name.1")
    # defect = dataset[dataset["bug"] > 0]
    # clean = dataset[dataset["bug"] == 0]
    # defect_number = len(defect)
    # clean_number = len(clean)
    total_number = len(dataset)
    defect_ratio = len(dataset[dataset["bug"] > 0]) / total_number

    # 결함 비율이 0.45 이상인 데이터셋 제외
    if defect_ratio > 0.45:
        print(inputfile, " defect ratio larger than 0.45")
        continue
    # 결함 레이블 이진화 (bug > 0 -> 1)    
    for j in range(total_number):
        if dataset.loc[j, "bug"] > 0:
            dataset.loc[j, "bug"] = 1
    # 정규화: Min-Max Scaling
    cols = list(dataset.columns)
    for col in cols:
        column_max = dataset[col].max()
        column_min = dataset[col].min()
        dataset[col] = (dataset[col] - column_min) / (column_max - column_min)
    x = dataset.drop(columns="bug")
    y = dataset["bug"]

    # GridSearchCV를 위한 최적 하이퍼파라미터 탐색
    validation_train_data, validation_test_data = separate_data(dataset)
    while len(validation_train_data[validation_train_data[:, -1] == 0]) == 0 or len(
            validation_test_data[validation_test_data[:, -1] == 1]) == 0 or len(
            validation_train_data[validation_train_data[:, -1] == 1]) <= neighbor or len(
            validation_test_data[validation_test_data[:, -1] == 0]) == 0 or len(
            validation_train_data[validation_train_data[:, -1] == 1]) >= len(
            validation_train_data[validation_train_data[:, -1] == 0]):
        validation_train_data, validation_test_data = separate_data(dataset)
    validation_train_x = validation_train_data[:, 0:-1]
    validation_train_y = validation_train_data[:, -1]
    print("location")
    # GridSearchCV로 최적 파라미터 탐색
    validation_clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=5, n_jobs=-1)
    validation_clf.fit(validation_train_x, validation_train_y)
    best_parameter = validation_clf.best_params_
    best_c = best_parameter["C"]
    best_kernal = best_parameter["kernel"]
    print(best_parameter)
    print("--------------------------------------------")
     
    # 10회의 독립적인 실험 반복
    for j in range(10):
        auc_row = []
        balance_row = []
        recall_row = []
        pf_row = []
        brier_row = []
        mcc_row = []

        stable_auc_row = []
        stable_balance_row = []
        stable_recall_row = []
        stable_pf_row = []
        stable_brier_row = []
        stable_mcc_row = []
        # Train/Test 분리 (유효성 검증 포함)
        train_data, test_data = separate_data(dataset)
        while len(train_data[train_data[:, -1] == 0]) == 0 or len(test_data[test_data[:, -1] == 1]) == 0 or len(
                train_data[train_data[:, -1] == 1]) <= neighbor or len(test_data[test_data[:, -1] == 0]) == 0 or len(
                train_data[train_data[:, -1] == 1]) >= len(train_data[train_data[:, -1] == 0]):
            train_data, test_data = separate_data(dataset)
        # print(len(train_data[train_data[:, -1] == 1]))
        train_x = train_data[:, 0:-1]
        # print(train_x)
        train_y = train_data[:, -1]

        # 10회 반복 실험 (내부 루프) - 각 SMOTE 방법 비교
        # (line354) smote = SMOTE(k_neighbors=neighbor, kind="borderline1") # SMOTE에서 kind 사용 중단되어 대체
        # -> smote = BorderlineSMOTE(k_neighbors=neighbor, kind='borderline-1')
        for s in range(10):
            # 1. 기본 BorderlineSMOTE 적용 및 평가
            smote = BorderlineSMOTE(k_neighbors=neighbor, kind='borderline-1')
            smote_train_x, smote_train_y = smote.fit_resample(train_x, train_y)
            test_x = test_data[:, 0:-1]
            test_y = test_data[:, -1]

            # SVM 모델 학습 및 예측
            # svc = classifier_for_selection[classifier]
            # clf = GridSearchCV(svc, tuned_parameters, cv=5, n_jobs=-1)
            clf = svm.SVC(C=best_c, kernel=best_kernal)
            clf.fit(smote_train_x, smote_train_y)
            predict_result = clf.predict(test_x)
            print("predict result:", predict_result[0])
            print(time.asctime(time.localtime(time.time())))
            print("----------------------------------")
            # 성능 지표 계산
            true_negative, false_positive, false_negative, true_positive = confusion_matrix(test_y, predict_result).ravel()
            brier = brier_score_loss(test_y, predict_result)
            mcc = matthews_corrcoef(test_y, predict_result)
            recall = recall_score(test_y, predict_result)
            # total_recall = total_recall + recall
            pf = false_positive / (true_negative + false_positive)
            # total_pf = total_pf + pf
            balance = 1 - (((0 - pf) ** 2 + (1 - recall) ** 2) / 2) ** 0.5
            # total_balance = total_balance + balance
            auc = roc_auc_score(test_y, predict_result)
            # total_auc = total_auc + auc

            # 결과 저장
            auc_row.append(auc)
            balance_row.append(balance)
            recall_row.append(recall)
            pf_row.append(pf)
            brier_row.append(brier)
            mcc_row.append(mcc)

            ##################################################################
            ##################################################################
            # 2. Stable BorderlineSMOTE 적용 및 평가
            stable_smote = stable_SMOTE()

            stable_smote_train = stable_smote.fit_sample(train_data, train_y)
            stable_smote_train = np.array(stable_smote_train)

            stable_smote_train_x = stable_smote_train[:, 0:-1]
            stable_smote_train_y = stable_smote_train[:, -1]

            test_x = test_data[:, 0:-1]
            test_y = test_data[:, -1]

            # SVM 모델 학습 및 예측
            # stable_svc = classifier_for_selection[classifier]
            # stable_clf = GridSearchCV(stable_svc, tuned_parameters, cv=5, n_jobs=-1)
            stable_clf = svm.SVC(C=best_c, kernel=best_kernal)
            stable_clf.fit(stable_smote_train_x, stable_smote_train_y)
            stable_predict_result = stable_clf.predict(test_x)
            # 성능 지표 계산
            stable_true_negative, stable_false_positive, stable_false_negative, stable_true_positive = confusion_matrix(test_y,
                                                                                            stable_predict_result).ravel()

            stable_brier = brier_score_loss(test_y, stable_predict_result)
            stable_mcc = matthews_corrcoef(test_y, stable_predict_result)
            stable_recall = recall_score(test_y, stable_predict_result)
            # total_recall = total_recall + stable_recall
            stable_pf = stable_false_positive / (stable_true_negative + stable_false_positive)
            # total_pf = total_pf + pf
            stable_balance = 1 - (((0 - stable_pf) ** 2 + (1 - stable_recall) ** 2) / 2) ** 0.5
            # total_balance = total_balance + balance
            stable_auc = roc_auc_score(test_y, stable_predict_result)
            # total_auc = total_auc + auc

            # 결과 저장
            stable_auc_row.append(stable_auc)
            stable_balance_row.append(stable_balance)
            stable_recall_row.append(stable_recall)
            stable_pf_row.append(stable_pf)
            stable_brier_row.append(stable_brier)
            stable_mcc_row.append(stable_mcc)

        # 통계 지표 계산 및 CSV 파일에 기록
        # Brier Score (일반 SMOTE)
        max_brier = max(brier_row)
        min_brier = min(brier_row)
        avg_brier = np.mean(brier_row)
        median_brier = np.median(brier_row)
        quartile_brier = np.percentile(brier_row, (25, 75), interpolation='midpoint')
        lower_quartile_brier = quartile_brier[0]
        upper_quartile_brier = quartile_brier[1]
        variance_brier = np.std(brier_row)
        brier_row.append(min_brier)
        brier_row.append(lower_quartile_brier)
        brier_row.append(avg_brier)
        brier_row.append(median_brier)
        brier_row.append(upper_quartile_brier)
        brier_row.append(max_brier)
        brier_row.append(variance_brier)
        brier_row.insert(0, inputfile + " brier")
        brier_writer.writerow(brier_row)
        # Brier Score (Stable SMOTE)
        stable_max_brier = max(stable_brier_row)
        stable_min_brier = min(stable_brier_row)
        stable_avg_brier = np.mean(stable_brier_row)
        stable_median_brier = np.median(stable_brier_row)
        stable_quartile_brier = np.percentile(stable_brier_row, (25, 75), interpolation='midpoint')
        stable_lower_quartile_brier = stable_quartile_brier[0]
        stable_upper_quartile_brier = stable_quartile_brier[1]
        stable_variance_brier = np.std(stable_brier_row)
        stable_brier_row.append(stable_min_brier)
        stable_brier_row.append(stable_lower_quartile_brier)

        stable_brier_row.append(stable_avg_brier)
        stable_brier_row.append(stable_median_brier)

        stable_brier_row.append(stable_upper_quartile_brier)
        stable_brier_row.append(stable_max_brier)
        stable_brier_row.append(stable_variance_brier)
        stable_brier_row.insert(0, inputfile + " brier")
        stable_brier_writer.writerow(stable_brier_row)
        # AUC (일반 SMOTE)
        max_auc = max(auc_row)
        min_auc = min(auc_row)
        avg_auc = np.mean(auc_row)
        median_auc = np.median(auc_row)
        quartile_auc = np.percentile(auc_row, (25, 75), interpolation='midpoint')
        lower_quartile_auc = quartile_auc[0]
        upper_quartile_auc = quartile_auc[1]
        variance_auc = np.std(auc_row)
        auc_row.append(min_auc)
        auc_row.append(lower_quartile_auc)

        auc_row.append(avg_auc)
        auc_row.append(median_auc)

        auc_row.append(upper_quartile_auc)
        auc_row.append(max_auc)
        auc_row.append(variance_auc)
        auc_row.insert(0, inputfile + " auc")
        auc_writer.writerow(auc_row)
        # AUC (Stable SMOTE)
        stable_max_auc = max(stable_auc_row)
        stable_min_auc = min(stable_auc_row)
        stable_avg_auc = np.mean(stable_auc_row)
        stable_median_auc = np.median(stable_auc_row)
        stable_quartile_auc = np.percentile(stable_auc_row, (25, 75), interpolation='midpoint')
        stable_lower_quartile_auc = stable_quartile_auc[0]
        stable_upper_quartile_auc = stable_quartile_auc[1]
        stable_variance_auc = np.std(stable_auc_row)
        stable_auc_row.append(stable_min_auc)
        stable_auc_row.append(stable_lower_quartile_auc)
        stable_auc_row.append(stable_avg_auc)
        stable_auc_row.append(stable_median_auc)

        stable_auc_row.append(stable_upper_quartile_auc)
        stable_auc_row.append(stable_max_auc)
        stable_auc_row.append(stable_variance_auc)
        stable_auc_row.insert(0, inputfile + " auc")
        stable_auc_writer.writerow(stable_auc_row)

        # Balance (일반 SMOTE)
        max_balance = max(balance_row)
        min_balance = min(balance_row)
        avg_balance = np.mean(balance_row)
        median_balance = np.median(balance_row)
        quartile_balance = np.percentile(balance_row, (25, 75), interpolation='midpoint')
        lower_quartile_balance = quartile_balance[0]
        upper_quartile_balance = quartile_balance[1]
        variance_balance = np.std(balance_row)
        balance_row.append(min_balance)
        balance_row.append(lower_quartile_balance)
        balance_row.append(avg_balance)
        balance_row.append(median_balance)

        balance_row.append(upper_quartile_balance)
        balance_row.append(max_balance)
        balance_row.append(variance_balance)
        balance_row.insert(0, inputfile + " balance")
        balance_writer.writerow(balance_row)
        # Balance (Stable SMOTE)
        stable_max_balance = max(stable_balance_row)
        stable_min_balance = min(stable_balance_row)
        stable_avg_balance = np.mean(stable_balance_row)
        stable_median_balance = np.median(stable_balance_row)
        stable_quartile_balance = np.percentile(stable_balance_row, (25, 75), interpolation='midpoint')
        stable_lower_quartile_balance = stable_quartile_balance[0]
        stable_upper_quartile_balance = stable_quartile_balance[1]
        stable_variance_balance = np.std(stable_balance_row)
        stable_balance_row.append(stable_min_balance)
        stable_balance_row.append(stable_lower_quartile_balance)
        stable_balance_row.append(stable_avg_balance)
        stable_balance_row.append(stable_median_balance)

        stable_balance_row.append(stable_upper_quartile_balance)
        stable_balance_row.append(stable_max_balance)
        stable_balance_row.append(stable_variance_balance)
        stable_balance_row.insert(0, inputfile + " balance")
        stable_balance_writer.writerow(stable_balance_row)

        # Recall (일반 SMOTE)
        max_recall = max(recall_row)
        min_recall = min(recall_row)
        avg_recall = np.mean(recall_row)
        median_recall = np.median(recall_row)
        quartile_recall = np.percentile(recall_row, (25, 75), interpolation='midpoint')
        lower_quartile_recall = quartile_recall[0]
        upper_quartile_recall = quartile_recall[1]
        variance_recall = np.std(recall_row)
        recall_row.append(min_recall)
        recall_row.append(lower_quartile_recall)
        recall_row.append(avg_recall)
        recall_row.append(median_recall)

        recall_row.append(upper_quartile_recall)
        recall_row.append(max_recall)
        recall_row.append(variance_recall)
        recall_row.insert(0, inputfile + " recall")
        recall_writer.writerow(recall_row)
        # Recall (Stable SMOTE)
        stable_max_recall = max(stable_recall_row)
        stable_min_recall = min(stable_recall_row)
        stable_avg_recall = np.mean(stable_recall_row)
        stable_median_recall = np.median(stable_recall_row)
        stable_quartile_recall = np.percentile(stable_recall_row, (25, 75), interpolation='midpoint')
        stable_lower_quartile_recall = stable_quartile_recall[0]
        stable_upper_quartile_recall = stable_quartile_recall[1]
        stable_variance_recall = np.std(stable_recall_row)
        stable_recall_row.append(stable_min_recall)
        stable_recall_row.append(stable_lower_quartile_recall)
        stable_recall_row.append(stable_avg_recall)
        stable_recall_row.append(stable_median_recall)

        stable_recall_row.append(stable_upper_quartile_recall)
        stable_recall_row.append(stable_max_recall)
        stable_recall_row.append(stable_variance_recall)
        stable_recall_row.insert(0, inputfile + " recall")
        stable_recall_writer.writerow(stable_recall_row)

        # PF (일반 SMOTE)
        max_pf = max(pf_row)
        min_pf = min(pf_row)
        avg_pf = np.mean(pf_row)
        median_pf = np.median(pf_row)
        quartile_pf = np.percentile(pf_row, (25, 75), interpolation='midpoint')
        lower_quartile_pf = quartile_pf[0]
        upper_quartile_pf = quartile_pf[1]
        variance_pf = np.std(pf_row)
        pf_row.append(min_pf)
        pf_row.append(lower_quartile_pf)
        pf_row.append(avg_pf)
        pf_row.append(median_pf)

        pf_row.append(upper_quartile_pf)
        pf_row.append(max_pf)
        pf_row.append(variance_pf)
        pf_row.insert(0, inputfile + " pf")
        pf_writer.writerow(pf_row)
        # PF (Stable SMOTE)
        stable_max_pf = max(stable_pf_row)
        stable_min_pf = min(stable_pf_row)
        stable_avg_pf = np.mean(stable_pf_row)
        stable_median_pf = np.median(stable_pf_row)
        stable_quartile_pf = np.percentile(stable_pf_row, (25, 75), interpolation='midpoint')
        stable_lower_quartile_pf = stable_quartile_pf[0]
        stable_upper_quartile_pf = stable_quartile_pf[1]
        stable_variance_pf = np.std(stable_pf_row)
        stable_pf_row.append(stable_min_pf)
        stable_pf_row.append(stable_lower_quartile_pf)
        stable_pf_row.append(stable_avg_pf)
        stable_pf_row.append(stable_median_pf)

        stable_pf_row.append(stable_upper_quartile_pf)
        stable_pf_row.append(stable_max_pf)
        stable_pf_row.append(stable_variance_pf)
        stable_pf_row.insert(0, inputfile + " pf")
        stable_pf_writer.writerow(stable_pf_row)

        # MCC (일반 SMOTE)
        max_mcc = max(mcc_row)
        min_mcc = min(mcc_row)
        avg_mcc = np.mean(mcc_row)
        median_mcc = np.median(mcc_row)
        quartile_mcc = np.percentile(mcc_row, (25, 75), interpolation='midpoint')
        lower_quartile_mcc = quartile_mcc[0]
        upper_quartile_mcc = quartile_mcc[1]
        variance_mcc = np.std(mcc_row)
        mcc_row.append(min_mcc)
        mcc_row.append(lower_quartile_mcc)
        mcc_row.append(avg_mcc)
        mcc_row.append(median_mcc)

        mcc_row.append(upper_quartile_mcc)
        mcc_row.append(max_mcc)
        mcc_row.append(variance_mcc)
        mcc_row.insert(0, inputfile + " mcc")
        mcc_writer.writerow(mcc_row)

        # MCC (Stable SMOTE)
        stable_max_mcc = max(stable_mcc_row)
        stable_min_mcc = min(stable_mcc_row)
        stable_avg_mcc = np.mean(stable_mcc_row)
        stable_median_mcc = np.median(stable_mcc_row)
        stable_quartile_mcc = np.percentile(stable_mcc_row, (25, 75), interpolation='midpoint')
        stable_lower_quartile_mcc = stable_quartile_mcc[0]
        stable_upper_quartile_mcc = stable_quartile_mcc[1]
        stable_variance_mcc = np.std(stable_mcc_row)
        stable_mcc_row.append(stable_min_mcc)
        stable_mcc_row.append(stable_lower_quartile_mcc)
        stable_mcc_row.append(stable_avg_mcc)
        stable_mcc_row.append(stable_median_mcc)

        stable_mcc_row.append(stable_upper_quartile_mcc)
        stable_mcc_row.append(stable_max_mcc)
        stable_mcc_row.append(stable_variance_mcc)
        stable_mcc_row.insert(0, inputfile + " mcc")
        stable_mcc_writer.writerow(stable_mcc_row)



    # single_writer.writerow(auc_row)
    # single_writer.writerow(pf_row)
    # single_writer.writerow(recall_row)
    # single_writer.writerow(pf_row)
    # single_writer.writerow([])
