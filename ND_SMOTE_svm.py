# Stable SMOTE with SVM - UCI Bank Marketing Dataset
# 원본 코드를 Bank Marketing 데이터셋에 맞게 수정

import csv
import random
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import svm
from sklearn.metrics import brier_score_loss, matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')
np.set_printoptions(suppress=True)

# 전역 설정: 분류기 및 하이퍼파라미터 설정
target_defect_ratio = 0.5
neighbor = 5
tuned_parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10, 100, 1000]}
classifier = "svm"

# Stable SMOTE 클래스: 안정적인 오버샘플링 알고리즘
class stable_SMOTE:
    def __init__(self, z_nearest=5):
        self.z_nearest = z_nearest
    
    def fit_sample(self, x_dataset, y_dataset, feature_names):
        """
        Stable SMOTE 알고리즘의 핵심 로직:
        1. 소수 클래스/다수 클래스 인스턴스 분리 및 필요 샘플 수 계산
        2. 각 소수 클래스 인스턴스에서 최근접 이웃과 페어링
        3. 선형 보간을 통한 합성 샘플 생성
        """
        x_dataset = pd.DataFrame(x_dataset)
        
        # feature_names에 'target' 추가하여 컬럼명 설정
        column_names = list(feature_names) + ['target']
        x_dataset.columns = column_names
        
        total_pair = []
        
        # 소수 클래스(yes=1)/다수 클래스(no=0) 인스턴스 분리
        minority_instance = x_dataset[x_dataset["target"] == 1]
        majority_instance = x_dataset[x_dataset["target"] == 0]
        minority_number = len(minority_instance)
        majority_number = len(majority_instance)
        
        # 목표 비율 달성을 위한 필요 샘플 수 계산
        need_number = int((target_defect_ratio * len(x_dataset) - minority_number) / (1 - target_defect_ratio))
        if need_number <= 0:
            return False
        
        generated_dataset = []
        number_on_each_instance = need_number / minority_number
        total_pair = []

        # STEP 1: 첫 번째 라운드 - 모든 최근접 이웃과 페어링
        rround = number_on_each_instance / self.z_nearest
        while rround >= 1:
            for index, row in minority_instance.iterrows():
                temp_minority_instance = minority_instance.copy(deep=True)
                subtraction = row - temp_minority_instance
                square = subtraction ** 2
                row_sum = square.apply(lambda s: s.sum(), axis=1)
                distance = row_sum ** 0.5
                temp_minority_instance["distance"] = distance
                temp_minority_instance = temp_minority_instance.sort_values(by="distance", ascending=True)
                neighbors = temp_minority_instance[1:self.z_nearest + 1]
                for a, r in neighbors.iterrows():
                    selected_pair = [index, a]
                    selected_pair.sort()
                    total_pair.append(selected_pair)
            rround = rround - 1
        
        # STEP 2: 두 번째 라운드 - 추가 필요 샘플을 위한 먼 이웃 선택
        need_number1 = need_number - len(total_pair)
        number_on_each_instance = need_number1 / minority_number

        for index, row in minority_instance.iterrows():
            temp_minority_instance = minority_instance.copy(deep=True)
            subtraction = row - temp_minority_instance
            square = subtraction ** 2
            row_sum = square.apply(lambda s: s.sum(), axis=1)
            distance = row_sum ** 0.5

            temp_minority_instance["distance"] = distance
            temp_minority_instance = temp_minority_instance.sort_values(by="distance", ascending=True)
            neighbors = temp_minority_instance[1:self.z_nearest + 1]
            neighbors = neighbors.sort_values(by="distance", ascending=False)
            target_sample_instance = neighbors[0: int(number_on_each_instance)]
            target_sample_instance = target_sample_instance.drop(columns="distance")
            for a, r in target_sample_instance.iterrows():
                selected_pair = [index, a]
                selected_pair.sort()
                total_pair.append(selected_pair)
        
        # STEP 3: 세 번째 라운드 - 잔여 샘플 처리
        temp_minority_instance = minority_instance.copy(deep=True)
        residue_number = need_number - len(total_pair)
        if residue_number > 0:
            residue_minority_instance = temp_minority_instance.sample(n=residue_number, replace=True)
            for index, row in residue_minority_instance.iterrows():
                temp_minority_instance2 = minority_instance.copy(deep=True)
                subtraction = row - temp_minority_instance2
                square = subtraction ** 2
                row_sum = square.apply(lambda s: s.sum(), axis=1)
                distance = row_sum ** 0.5

                temp_minority_instance2["distance"] = distance
                temp_minority_instance2 = temp_minority_instance2.sort_values(by="distance", ascending=True)
                neighbors = temp_minority_instance2[1:self.z_nearest + 1]
                target_sample_instance = neighbors[-1:]
                for a in target_sample_instance.index:
                    selected_pair = [index, a]
                    selected_pair.sort()
                    total_pair.append(selected_pair)
        
        # STEP 4: 선택된 페어 간 선형 보간으로 합성 샘플 생성
        total_pair_tuple = [tuple(l) for l in total_pair]
        result = Counter(total_pair_tuple)
        result_number = len(result)
        result_keys = result.keys()
        result_values = result.values()
        
        for f in range(result_number):
            current_pair = list(result_keys)[f]
            row1_index = current_pair[0]
            row2_index = current_pair[1]
            row1 = minority_instance.loc[row1_index]
            row2 = minority_instance.loc[row2_index]
            generated_num = list(result_values)[f]
            # 두 인스턴스 간 선형 보간
            generated_instances = np.linspace(row1, row2, generated_num + 2)
            generated_instances = generated_instances[1:-1]
            generated_instances = generated_instances.tolist()
            for w in generated_instances:
                generated_dataset.append(w)
        
        final_generated_dataset = pd.DataFrame(generated_dataset)
        final_generated_dataset.columns = column_names
        
        # 원본 데이터와 합성 데이터 결합
        result = pd.concat([majority_instance, minority_instance, final_generated_dataset])
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
    test_dataset = original_data[test_index]
    return train_dataset, test_dataset

def load_bank_marketing_data():
    """
    UCI Bank Marketing 10% dataset (bank.csv) 로드 (로컬 파일)
    - 경로: new_input_data/bank.csv
    - 반환 형태(X: DataFrame, y: DataFrame, metadata)는 기존과 동일 유지
    """
    csv_path = "new_input_data/bank.csv"

    df = pd.read_csv(csv_path, sep=";")  # UCI bank.csv는 세미콜론 구분자

    if "y" not in df.columns:
        raise ValueError(f"'y' column not found in {csv_path}. columns={df.columns.tolist()}")

    X = df.drop(columns=["y"])
    y = df[["y"]]  # preprocess에서 y.iloc[:,0] 쓰므로 DataFrame 유지

    metadata = {
        "source": "local bank.csv (10%)",
        "path": csv_path,
        "rows": len(df),
        "inputs": X.shape[1],
    }
    return X, y, metadata


def preprocess_bank_marketing(X, y):
    """
    Bank Marketing 데이터 전처리
    - 범주형 변수 인코딩
    - 타겟 변수 이진화 (yes=1, no=0)
    - Min-Max 정규화
    """
    # 복사본 생성
    X_processed = X.copy()
    
    # 범주형 컬럼과 수치형 컬럼 분리
    categorical_cols = X_processed.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Label Encoding for categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_processed[col] = le.fit_transform(X_processed[col].astype(str))
        label_encoders[col] = le
    
    # 타겟 변수 처리 (yes=1, no=0)
    y_processed = y.copy()
    if y_processed.iloc[:, 0].dtype == 'object':
        y_processed = (y_processed.iloc[:, 0] == 'yes').astype(int)
    else:
        y_processed = y_processed.iloc[:, 0]
    
    # 데이터프레임 결합
    dataset = X_processed.copy()
    dataset['target'] = y_processed.values
    
    # Min-Max Scaling
    cols = list(dataset.columns)
    for col in cols:
        column_max = dataset[col].max()
        column_min = dataset[col].min()
        if column_max != column_min:
            dataset[col] = (dataset[col] - column_min) / (column_max - column_min)
        else:
            dataset[col] = 0
    
    feature_names = X_processed.columns.tolist()
    
    return dataset, feature_names

# 메인 실행부
if __name__ == "__main__":
    print("="*60)
    print("Stable SMOTE with SVM - UCI Bank Marketing Dataset")
    print("="*60)
    
    start_time = time.asctime(time.localtime(time.time()))
    print(f"Start time: {start_time}")
    
    # 출력 디렉토리 설정
    output_dir = 'new_output_data/new_SMOTE_svm/'
    
    # 결과 저장을 위한 CSV 파일 초기화
    # 일반 SMOTE 결과 파일
    auc_file = open(output_dir + str(neighbor) + 'auc_smote_result_on_' + classifier + '_bank_marketing.csv', 'w', newline='')
    auc_writer = csv.writer(auc_file)
    auc_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

    balance_file = open(output_dir + str(neighbor) + 'balance_smote_result_on_' + classifier + '_bank_marketing.csv', 'w', newline='')
    balance_writer = csv.writer(balance_file)
    balance_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

    recall_file = open(output_dir + str(neighbor) + 'recall_smote_result_on_' + classifier + '_bank_marketing.csv', 'w', newline='')
    recall_writer = csv.writer(recall_file)
    recall_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

    pf_file = open(output_dir + str(neighbor) + 'pf_smote_result_on_' + classifier + '_bank_marketing.csv', 'w', newline='')
    pf_writer = csv.writer(pf_file)
    pf_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max"])

    brier_file = open(output_dir + str(neighbor) + 'brier_smote_result_on_' + classifier + '_bank_marketing.csv', 'w', newline='')
    brier_writer = csv.writer(brier_file)
    brier_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

    mcc_file = open(output_dir + str(neighbor) + 'mcc_smote_result_on_' + classifier + '_bank_marketing.csv', 'w', newline='')
    mcc_writer = csv.writer(mcc_file)
    mcc_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

    # Stable SMOTE 결과 파일
    stable_auc_file = open(output_dir + str(neighbor) + 'auc_stable_smote_result_on_' + classifier + '_bank_marketing.csv', 'w', newline='')
    stable_auc_writer = csv.writer(stable_auc_file)
    stable_auc_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

    stable_balance_file = open(output_dir + str(neighbor) + 'balance_stable_smote_result_on_' + classifier + '_bank_marketing.csv', 'w', newline='')
    stable_balance_writer = csv.writer(stable_balance_file)
    stable_balance_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

    stable_recall_file = open(output_dir + str(neighbor) + 'recall_stable_smote_result_on_' + classifier + '_bank_marketing.csv', 'w', newline='')
    stable_recall_writer = csv.writer(stable_recall_file)
    stable_recall_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

    stable_pf_file = open(output_dir + str(neighbor) + 'pf_stable_smote_result_on_' + classifier + '_bank_marketing.csv', 'w', newline='')
    stable_pf_writer = csv.writer(stable_pf_file)
    stable_pf_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

    stable_brier_file = open(output_dir + str(neighbor) + 'brier_stable_smote_result_on_' + classifier + '_bank_marketing.csv', 'w', newline='')
    stable_brier_writer = csv.writer(stable_brier_file)
    stable_brier_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

    stable_mcc_file = open(output_dir + str(neighbor) + 'mcc_stable_smote_result_on_' + classifier + '_bank_marketing.csv', 'w', newline='')
    stable_mcc_writer = csv.writer(stable_mcc_file)
    stable_mcc_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

    # 데이터 로드
    print("\nLoading Bank Marketing dataset...")
    try:
        X, y, metadata = load_bank_marketing_data()
        print(f"Dataset loaded successfully!")
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
    except Exception as e:
        print(f"Error loading from UCI repository: {e}")
        print("Please ensure you have internet connection and ucimlrepo package installed.")
        print("You can install it with: pip install ucimlrepo")
        exit(1)
    
    # 데이터 전처리
    print("\nPreprocessing data...")
    dataset, feature_names = preprocess_bank_marketing(X, y)
    
    inputfile = "bank_marketing"
    total_number = len(dataset)
    minority_ratio = len(dataset[dataset["target"] == 1]) / total_number
    
    print(f"Total samples: {total_number}")
    print(f"Minority class ratio (yes): {minority_ratio:.4f}")
    print(f"Majority class ratio (no): {1-minority_ratio:.4f}")
    print(f"Features: {feature_names}")
    
    x = dataset.drop(columns="target")
    y = dataset["target"]
    
    # GridSearchCV를 위한 최적 하이퍼파라미터 탐색
    print("\nFinding optimal hyperparameters with GridSearchCV...")
    validation_train_data, validation_test_data = separate_data(dataset)
    
    # 유효한 데이터 분할 확인
    max_attempts = 100
    attempts = 0
    while (len(validation_train_data[validation_train_data[:, -1] == 0]) == 0 or 
           len(validation_test_data[validation_test_data[:, -1] == 1]) == 0 or 
           len(validation_train_data[validation_train_data[:, -1] == 1]) <= neighbor or 
           len(validation_test_data[validation_test_data[:, -1] == 0]) == 0 or 
           len(validation_train_data[validation_train_data[:, -1] == 1]) >= len(validation_train_data[validation_train_data[:, -1] == 0])):
        validation_train_data, validation_test_data = separate_data(dataset)
        attempts += 1
        if attempts > max_attempts:
            print("Warning: Could not find valid data split after maximum attempts")
            break
    
    validation_train_x = validation_train_data[:, 0:-1]
    validation_train_y = validation_train_data[:, -1]
    
    # GridSearchCV로 최적 파라미터 탐색
    validation_clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=5, n_jobs=-1)
    validation_clf.fit(validation_train_x, validation_train_y)
    best_parameter = validation_clf.best_params_
    print(f"Best parameters: {best_parameter}")
    print("-" * 60)
    best_c = best_parameter["C"]
    best_kernel = best_parameter["kernel"]
    
    # 10회 반복 실험 (외부 루프)
    for j in range(10):
        print(f"\nOuter iteration {j+1}/10")
        
        # 각 반복마다 성능 지표 저장용 리스트 초기화
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
        attempts = 0
        while (len(train_data[train_data[:, -1] == 0]) == 0 or 
               len(test_data[test_data[:, -1] == 1]) == 0 or 
               len(train_data[train_data[:, -1] == 1]) <= neighbor or 
               len(test_data[test_data[:, -1] == 0]) == 0 or 
               len(train_data[train_data[:, -1] == 1]) >= len(train_data[train_data[:, -1] == 0])):
            train_data, test_data = separate_data(dataset)
            attempts += 1
            if attempts > max_attempts:
                break
        
        train_x = train_data[:, 0:-1]
        train_y = train_data[:, -1]
        
        # 10회 반복 실험 (내부 루프) - 각 SMOTE 방법 비교
        for s in range(10):
            print(f"  Inner iteration {s+1}/10", end='\r')
            
            # 1. 기본 SMOTE 적용 및 평가
            smote = SMOTE(k_neighbors=neighbor)
            smote_train_x, smote_train_y = smote.fit_resample(train_x, train_y)
            test_x = test_data[:, 0:-1]
            test_y = test_data[:, -1]

            # SVM 모델 학습 및 예측
            clf = svm.SVC(C=best_c, kernel=best_kernel)
            clf.fit(smote_train_x, smote_train_y)
            predict_result = clf.predict(test_x)
            
            # 성능 지표 계산
            tn, fp, fn, tp = confusion_matrix(test_y, predict_result).ravel()
            brier = brier_score_loss(test_y, predict_result)
            recall = recall_score(test_y, predict_result)
            mcc = matthews_corrcoef(test_y, predict_result)
            pf = fp / (tn + fp) if (tn + fp) > 0 else 0
            balance = 1 - (((0 - pf) ** 2 + (1 - recall) ** 2) / 2) ** 0.5
            auc = roc_auc_score(test_y, predict_result)

            # 결과 저장
            auc_row.append(auc)
            balance_row.append(balance)
            recall_row.append(recall)
            pf_row.append(pf)
            brier_row.append(brier)
            mcc_row.append(mcc)

            # 2. Stable SMOTE 적용 및 평가
            stable_smote = stable_SMOTE(neighbor)
            stable_smote_train = stable_smote.fit_sample(train_data, train_y, feature_names)
            
            if stable_smote_train is False:
                # Stable SMOTE 실패 시 기본 SMOTE 결과 사용
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

            # SVM 모델 학습 및 예측
            stable_clf = svm.SVC(C=best_c, kernel=best_kernel)
            stable_clf.fit(stable_smote_train_x, stable_smote_train_y)
            stable_predict_result = stable_clf.predict(test_x)
            
            # 성능 지표 계산
            stable_tn, stable_fp, stable_fn, stable_tp = confusion_matrix(test_y, stable_predict_result).ravel()
            stable_brier = brier_score_loss(test_y, stable_predict_result)
            stable_recall = recall_score(test_y, stable_predict_result)
            stable_mcc = matthews_corrcoef(test_y, stable_predict_result)
            stable_pf = stable_fp / (stable_tn + stable_fp) if (stable_tn + stable_fp) > 0 else 0
            stable_balance = 1 - (((0 - stable_pf) ** 2 + (1 - stable_recall) ** 2) / 2) ** 0.5
            stable_auc = roc_auc_score(test_y, stable_predict_result)

            # 결과 저장
            stable_auc_row.append(stable_auc)
            stable_balance_row.append(stable_balance)
            stable_recall_row.append(stable_recall)
            stable_pf_row.append(stable_pf)
            stable_brier_row.append(stable_brier)
            stable_mcc_row.append(stable_mcc)

        print(f"  Inner iteration 10/10 - Done")
        
        # 통계 지표 계산 및 CSV 파일에 기록
        def calculate_stats(row):
            return {
                'max': max(row),
                'min': min(row),
                'avg': np.mean(row),
                'median': np.median(row),
                'quartile': np.percentile(row, (25, 75), interpolation='midpoint'),
                'variance': np.std(row)
            }
        
        # Brier Score (일반 SMOTE)
        stats = calculate_stats(brier_row)
        brier_row.extend([stats['min'], stats['quartile'][0], stats['avg'], stats['median'], stats['quartile'][1], stats['max'], stats['variance']])
        brier_row.insert(0, inputfile + " brier")
        brier_writer.writerow(brier_row)

        # Brier Score (Stable SMOTE)
        stats = calculate_stats(stable_brier_row)
        stable_brier_row.extend([stats['min'], stats['quartile'][0], stats['avg'], stats['median'], stats['quartile'][1], stats['max'], stats['variance']])
        stable_brier_row.insert(0, inputfile + " brier")
        stable_brier_writer.writerow(stable_brier_row)

        # AUC (일반 SMOTE)
        stats = calculate_stats(auc_row)
        auc_row.extend([stats['min'], stats['quartile'][0], stats['avg'], stats['median'], stats['quartile'][1], stats['max'], stats['variance']])
        auc_row.insert(0, inputfile + " auc")
        auc_writer.writerow(auc_row)
        
        # AUC (Stable SMOTE)
        stats = calculate_stats(stable_auc_row)
        stable_auc_row.extend([stats['min'], stats['quartile'][0], stats['avg'], stats['median'], stats['quartile'][1], stats['max'], stats['variance']])
        stable_auc_row.insert(0, inputfile + " auc")
        stable_auc_writer.writerow(stable_auc_row)

        # Balance (일반 SMOTE)
        stats = calculate_stats(balance_row)
        balance_row.extend([stats['min'], stats['quartile'][0], stats['avg'], stats['median'], stats['quartile'][1], stats['max'], stats['variance']])
        balance_row.insert(0, inputfile + " balance")
        balance_writer.writerow(balance_row)

        # Balance (Stable SMOTE)
        stats = calculate_stats(stable_balance_row)
        stable_balance_row.extend([stats['min'], stats['quartile'][0], stats['avg'], stats['median'], stats['quartile'][1], stats['max'], stats['variance']])
        stable_balance_row.insert(0, inputfile + " balance")
        stable_balance_writer.writerow(stable_balance_row)

        # Recall (일반 SMOTE)
        stats = calculate_stats(recall_row)
        recall_row.extend([stats['min'], stats['quartile'][0], stats['avg'], stats['median'], stats['quartile'][1], stats['max'], stats['variance']])
        recall_row.insert(0, inputfile + " recall")
        recall_writer.writerow(recall_row)

        # Recall (Stable SMOTE)
        stats = calculate_stats(stable_recall_row)
        stable_recall_row.extend([stats['min'], stats['quartile'][0], stats['avg'], stats['median'], stats['quartile'][1], stats['max'], stats['variance']])
        stable_recall_row.insert(0, inputfile + " recall")
        stable_recall_writer.writerow(stable_recall_row)

        # PF (일반 SMOTE)
        stats = calculate_stats(pf_row)
        pf_row.extend([stats['min'], stats['quartile'][0], stats['avg'], stats['median'], stats['quartile'][1], stats['max'], stats['variance']])
        pf_row.insert(0, inputfile + " pf")
        pf_writer.writerow(pf_row)

        # PF (Stable SMOTE)
        stats = calculate_stats(stable_pf_row)
        stable_pf_row.extend([stats['min'], stats['quartile'][0], stats['avg'], stats['median'], stats['quartile'][1], stats['max'], stats['variance']])
        stable_pf_row.insert(0, inputfile + " pf")
        stable_pf_writer.writerow(stable_pf_row)

        # MCC (일반 SMOTE)
        stats = calculate_stats(mcc_row)
        mcc_row.extend([stats['min'], stats['quartile'][0], stats['avg'], stats['median'], stats['quartile'][1], stats['max'], stats['variance']])
        mcc_row.insert(0, inputfile + " mcc")
        mcc_writer.writerow(mcc_row)

        # MCC (Stable SMOTE)
        stats = calculate_stats(stable_mcc_row)
        stable_mcc_row.extend([stats['min'], stats['quartile'][0], stats['avg'], stats['median'], stats['quartile'][1], stats['max'], stats['variance']])
        stable_mcc_row.insert(0, inputfile + " mcc")
        stable_mcc_writer.writerow(stable_mcc_row)

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
    print("\n" + "="*60)
    print(f"Experiment completed!")
    print(f"Start time: {start_time}")
    print(f"End time: {end_time}")
    print(f"Results saved to: {output_dir}")
    print("="*60)