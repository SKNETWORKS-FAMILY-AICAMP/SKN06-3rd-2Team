# SKN06-3rd-2Team
강채연, 김동명, 박창규, 백하은, 홍준

## <팀명>

### 팀원

| 강채연 | 김동명| 박창규 | 백하은| 홍준 |
| --- | --- | --- | --- | --- |
| <> | <> | <> | <> | <> |
|성능 테스트 및 최적화|데이터 수집 및 전처리|RAG 체인 설계 및 최적화|벡터 데이터베이스 구축|RAG 체인 설계 및 최적화|

</br>

### RAG 기반 - 맛집 정보를 알려주는 어플리케이션 ![small BR](https://github.com/user-attachments/assets/9d9b741c-075b-4316-aeb3-91135fc85acb)![small BR](https://github.com/user-attachments/assets/f5c8c9cf-e5a7-41ad-acf2-36e3d24f374b)![small BR](https://github.com/user-attachments/assets/516ef109-240f-4142-b0da-dbe3ed72ce9e)



### 🍴 개발 기간

2024.12.23 ~ 2024.12.26 (총 3일)

### 🍴 개요

우리나라 최초의 맛집 “블루리본”</br>
"도시를 대표하는 세계적인 맛집 가이드북: 블루리본 서베이"에게 **뭐든지 물어보세요!**</br>
자 이제 나만의 맛집을 찾아 떠나볼까요?

### 🍴 목표

블루리본 서베이 웹사이트의 정보를 학습하여 사용자가 찾는 맛집 정보를 조회할 수 있게 도와주는 어플리케이션 개발

</br>

#### 🍴 Stacks

![Discord](https://img.shields.io/badge/discord-5865F2?style=for-the-badge&logo=discord&logoColor=white)

![Python](https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Numpy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=NumPy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
</br>
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=Git&logoColor=white)
![Github](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=GitHub&logoColor=white)
<br/> <>
<br/> <>
<br/> <>
<br/> <>
<br/> <>
<br/> <>


#### 🍴 Requirements

pandas == <> <br/>
numpy == <> <br/>
<> == <> <br/>

## 데이터 준비 및 분석

### 🎣 1. 데이터 수집

        import requests
        import time

        restaurants = []  # 데이터 수집할 리스트 생성

        total_pages = 576 # 전체 페이지 수를 설정
        url_template = "https://www.bluer.co.kr/api/v1/restaurants?page={page}&size=30&query=&foodType=&foodTypeDetail=&feature=&location=&locationDetail=&area=&areaDetail=&priceRange=&ribbonType=&recommended=false&isSearchName=false&tabMode=single&searchMode=ribbonType&zone1=&zone2=&zone2Lat=&zone2Lng="

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept": "application/hal+json",
            "x-requested-with": "XMLHttpRequest"
        }

        # 크롤링 진행
        for page in range(total_pages):
            url = url_template.format(page=page)
            response = requests.get(url, headers=headers)
    
            if response.status_code == 200:
                data = response.json()
                restaurants.extend(data["_embedded"]["restaurants"])
                print(f"Page {page + 1}/{total_pages} collected successfully.")
            else:
                print(f"Failed to fetch page {page + 1}. Status code: {response.status_code}")

        # 각 요청 사이에 시간 간격 추가
            time.sleep(2)  # 2초 간격

        # 전체 데이터를 저장
        import json
        with open("restaurants.json", "w", encoding="utf-8") as f:
            json.dump(restaurants, f, ensure_ascii=False, indent=4)

        print("전체 페이지 크롤링 완료. JSON 파일로 저장됨.")


### 🔪 2. 불필요 칼럼 삭제 및 정규화
> 결측치, 중복값, TMI (위도/경도, 업주명) 등
> 

```

#### 1. 전처리

import pandas as pd
from pandas import json_normalize

# JSON 파일 읽기
with open("restaurants.json", "r", encoding="utf-8") as f:
    restaurants = json.load(f)

# Pandas DataFrame으로 변환
df = pd.DataFrame(restaurants)

# 기존 전처리 코드 적용
# 필요 없는 컬럼 제거
columns_to_drop = ["createdDate", "id", "timeInfo", "gps", "tags", "status", "bookStatus",
                   "buzUsername", "business", "pageView", "brandMatchStatus", "brandRejectReason",
                   "orderDescending", "foodTypeDetails", "countEvaluate", "bookmark", "features",
                   "feature107", "brandBranches", "foodTypes", "brandHead", "firstImage", "firstLogoImage"]
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Nested JSON 컬럼 정규화
nested_columns = ["headerInfo", "defaultInfo", "statusInfo", "juso", "review", "etcInfo","_links"]
for column in nested_columns:
    if column in df.columns:
        n = pd.json_normalize(df[column])  # 정규화
        n.columns = [f"{column}_{subcol}" for subcol in n.columns]  # 열 이름 접두사 추가
        df = pd.concat([df.drop(columns=[column]), n], axis=1)  # 기존 열 삭제 후 결합

# 필요 없는 sub_columns 제거
sub_columns_to_drop = ["headerInfo_nickname", "headerInfo_year", "headerInfo_ribbonTypeByOrdinal",
                       "defaultInfo_websiteFacebook", "statusInfo_storeType", "statusInfo_openEra",
                       "statusInfo_newOpenDate", "juso_roadAddrPart2", "juso_jibunAddr", "juso_zipNo",
                       "juso_admCd", "juso_detBdNmList", "juso_zone2_1", "juso_zone2_2", "juso_map_1",
                       "juso_map_2", "review_readerReview", "review_businessReview", "review_editorReview",
                       "etcInfo_toilet", "etcInfo_toiletEtc", "etcInfo_chain", "etcInfo_close", "etcInfo_renewal",
                       "etcInfo_appYn", "etcInfo_projectNo", "etcInfo_reviewerRecommend", "etcInfo_onlySiteView",
                       "etcInfo_history", "etcInfo_mainMemo", "_links_self.href", "_links_restaurant_href",
                       "_links_restaurant_templated", "_links_childrenRestaurants.href", "_links_childrenRestaurants.templated"
                       "_links_evaluates.href", "_links_relativeBusinessOrder.href", "_links_parentRestaurant.href",
                       "_links_parentRestaurant.templated", "_links_reports.href", "_links"
                       ""]
df = df.drop(columns=[col for col in sub_columns_to_drop if col in df.columns])

# 데이터 저장
df.to_csv("nested_all_restaurants.csv", index=False)
print("전처리 테스트 완료. CSV 파일 저장됨.")


```
### 🍡 3. 형식 일치화
> 년도, 웹사이트 주소 기입 여부, 결측치 표기
>

```

import numpy as np
import re

# 빈칸이나 "없음"으로 표기된 칸을 None으로 변경
df.replace(["", "없음"], None, inplace=True)

# foodDetailTypes의 리스트 해제
# 각 행의 값이 리스트라면 이를 문자열로 변환 (쉼표로 구분된 문자열로 병합)
if "foodDetailTypes" in df.columns:
    df["foodDetailTypes"] = df["foodDetailTypes"].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)

if "headerInfo_ribbonType" in df.columns:
    df["headerInfo_ribbonType"] = df["headerInfo_ribbonType"].map(ribbon_mapping)

# 웹사이트 열 통합
if "defaultInfo_website" in df.columns and "defaultInfo_websiteInstagram" in df.columns:
    def merge_websites(row):
        websites = [site for site in [row.get("defaultInfo_website"), row.get("defaultInfo_websiteInstagram")] if site]
        return ", ".join(websites) if websites else None

    df["defaultInfo_website_combined"] = df.apply(merge_websites, axis=1)
    df.drop(columns=["defaultInfo_website", "defaultInfo_websiteInstagram"], inplace=True)
    df.rename(columns={"defaultInfo_website_combined": "defaultInfo_website"}, inplace=True)

# statusInfo_openDate 형식 변환 (년도 4자리로 추출 후 "2024년" 형식으로 표기)
def extract_year_with_suffix(date):
    if isinstance(date, str):
        # 패턴 매칭으로 숫자 4자리 추출
        match = re.search(r'\d{4}', date)
        if match:
            return f"{int(match.group(0))}년"  # "2024년" 형식으로 반환
    return None

if "statusInfo_openDate" in df.columns:
    df["statusInfo_openDate"] = df["statusInfo_openDate"].apply(extract_year_with_suffix)

# 최종 결과를 CSV 파일로 저장
df.to_csv("cleaned_all_restaurants.csv", index=False)
print("수정된 최종 CSV 파일 저장 완료!")


```

### 🥩 4. 산출물 정리
> 웹사이트 크롤링 데이터 : blueRibbon.csv </br>
> 필요없는 column 제거 후 데이터 : nested_restaurants.csv </br>
> 정규화 및 전처리 후 데이터 : cleaned_all_restaurants.csv </br>

</br>
</br>
</br>
</br>
</br>
</br>
</br>
</br>
</br>
</br>
</br>
</br>
</br>
</br>
</br>
</br>
</br>
</br>
</br>
</br>
</br>

#### 전처리 파이프라인 저장
```
import joblib
import os

os.makedirs('models', exist_ok=True)
joblib.dump(
    preprocessor_pipeline,     # 저장할 모델/전처리기
    "models/preprocessor.pkl"  # 저장경로. pickle로 저장된다.
)
```
## 모델링

### ✔️ 모델 선정하기

데이터와 어울리는 7개의 모델들은 뽑아 어떤 모델이 적합할지 확인해 보기로 했다.

- 평가

```
  from tqdm import tqdm

  from sklearn.linear_model import LogisticRegression
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.ensemble import GradientBoostingClassifier
  from xgboost import XGBClassifier, plot_importance
  from sklearn.svm import SVC
  from sklearn.neighbors import KNeighborsClassifier

  import matplotlib.pyplot as plt
  models = {
      # Logistic Regression model
      "Logistic Regression": LogisticRegression(),
      # Decision Tree model
      "Decision Tree Classifier": DecisionTreeClassifier(),
      # Random Forest model
      "Random Forest": RandomForestClassifier(),
      # Gradient Boosting model
      "Gradient Boosting": GradientBoostingClassifier(),
      # XGBoost model
      "XGBoost": XGBClassifier(),
      # SVM(Support Vector Machine)
      "SVC": SVC(),
      # KNN(K-Nearest Neighbors)
      "KNeighborsClassifier": KNeighborsClassifier(),
  }


  for name, model in tqdm(models.items(), desc="Training Models", total=len(models)):
      # 모델 훈련
      model.fit(X_train, y_train)
      # 모델 평가
      score = model.score(X_test, y_test)
      # 모델 검증
      model_pred = model.predict(X_test)
      # 모델 정확도
      tqdm.write(f">>> {name} : 정확도 {score:.2%}\n")

```

- 결과

```python
>>> Logistic Regression : 정확도 87.90%

>>> Decision Tree Classifier : 정확도 94.02%

>>> Random Forest : 정확도 95.65%

>>> Gradient Boosting : 정확도 96.15%

>>> XGBoost : 정확도 96.74%

>>> SVC : 정확도 84.20%

>>> KNeighborsClassifier : 정확도 90.47%
```

#### ⭐ 선정 결과

- LogisticRegression
- DecisionTreeClassifier (✔️) - 김동명
- RandomForestClassifier (✔️) - 임연경
- GradientBoostingClassifier (✔️) - 박유나
- xgboost (✔️) - 공인용
- SVC
- KNeighborsClassifier

7개의 모델 중 4개의 모델이 우수한 편이었고, 각자 모델 한개씩 맡아서 모델링을 하기로 했다.

### ✔️ 머신 러닝 모델

#### 1. Decision Tree Classifier : 정확도 93.78%

- 주요 파라미터

  > criterion: 노드 분할 기준
  >
  > max_depth: 각 결정 트리의 최대 깊이를 설정
  >
  > min_samples_split: 노드를 분할하기 위한 최소 샘플 수
  >
  > min_samples_leaf: 리프 노드의 최소 샘플 수
  >
  > max_features: 각 트리가 학습할 때마다 사용할 특성(feature)의 수

  ```

  from sklearn.tree import DecisionTreeClassifier

  # 1. 학습 및 예측
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



  tree = DecisionTreeClassifier()

  tree.fit(X_train, y_train)

  # 2. 모델 평가
  # Train set + Test set 평가
  y_train_pred_tree = tree.predict(X_train)
  y_train_proba_tree= tree.predict_proba(X_train)[:, 1]

  y_test_pred_tree = tree.predict(X_test)
  y_test_proba_tree= tree.predict_proba(X_test)[:, 1]

  # 혼동 행렬 시각화 (테스트 데이터)
  cm_test = confusion_matrix(y_test, y_test_pred_tree)
  plt.figure(figsize=(6, 4))
  sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", cbar=False)
  plt.xlabel("예측")
  plt.ylabel("정답")
  plt.title("Confusion Matrix - Decision Tree (Test Set)")
  plt.show()

  evaluate("Train - Decision Tree", y_train, y_train_pred_tree, y_train_proba_tree)
  evaluate("Test - Decision Tree", y_test, y_test_pred_tree, y_test_proba_tree)

  # 3. 특성 중요도 계산 및 시각화
  fi = tree.feature_importances_
  fi_series = pd.Series(fi, index=df.drop(columns="churn").columns).sort_values(ascending=False)

  # 특성 중요도 시각화
  plt.figure(figsize=(10, 6))
  sns.barplot(x=fi_series, y=fi_series.index)
  plt.title("Feature Importances in Decision Tree")
  plt.xlabel("Importance")
  plt.ylabel("Feature")
  plt.show()

  # 4. 최적의 매개변수 구하기 - GridSearchCV
  params = {
      'criterion': ['gini', 'entropy'],  # 노드 분할 기준
      'max_depth': [None, 10, 20, 30],   # 각 결정 트리의 최대 깊이를 설정
      'min_samples_split': [2, 10, 20],  # 노드를 분할하기 위한 최소 샘플 수
      'min_samples_leaf': [1, 5, 10],    # 리프 노드의 최소 샘플 수
      'max_features': [None, 'sqrt', 'log2']  # 각 트리가 학습할 때마다 사용할 특성(feature)의 수
  }

  gs_tree = GridSearchCV(
      estimator=tree,
      param_grid=params,
      scoring=scoring,
      refit='accuracy',
      cv=5,
      n_jobs=-1,
  )

  gs_tree.fit(X_train, y_train)

  # 5. Best Model: 최적의 하이파라미터로 만든 모델
  best_param_tree = gs_tree.best_params_
  best_model_tree = gs_tree.best_estimator_

  best_y_pred_tree = best_model_tree.predict(X_test)
  best_y_proba_tree= best_model_tree.predict_proba(X_test)[:, 1]

  ```

#### 2. Random Forest : 정확도 95.65%

- 주요 파라미터

  > n_estimators: 부스팅 단계의 수 = 모델이 생성할 트리 개수
  >
  > max_depth: 각 결정 트리의 최대 깊이를 설정
  >
  > max_features: 각 트리가 학습할 때마다 사용할 특성(feature)의 수

  ```
  from sklearn.ensemble import RandomForestClassifier

  # 1. 학습 및 예측
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  rf = RandomForestClassifier()

  rf.fit(X_train, y_train)

  # 2. 모델 평가
  # Train set + Test set 평가
  y_train_pred_rf = rf.predict(X_train)
  y_train_proba_rf= rf.predict_proba(X_train)[:, 1]

  y_test_pred_rf = rf.predict(X_test)
  y_test_proba_rf= rf.predict_proba(X_test)[:, 1]

  # 혼동 행렬 시각화 (테스트 데이터)
  cm_test = confusion_matrix(y_test, y_test_pred_rf)
  plt.figure(figsize=(6, 4))
  sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", cbar=False)
  plt.xlabel("예측")
  plt.ylabel("정답")
  plt.title("Confusion Matrix - Random Forest (Test Set)")
  plt.show()

  evaluate("Train - Random Forest", y_train, y_train_pred_rf, y_train_proba_rf)
  evaluate("Test - Random Forest", y_test, y_test_pred_rf, y_test_proba_rf)

  # 3. 특성 중요도 계산 및 시각화
  fi = rf.feature_importances_
  fi_series = pd.Series(fi, index=df.drop(columns="churn").columns).sort_values(ascending=False)

  # 특성 중요도 시각화
  plt.figure(figsize=(10, 6))
  sns.barplot(x=fi_series, y=fi_series.index)
  plt.title("Feature Importances in Random Forest")
  plt.xlabel("Importance")
  plt.ylabel("Feature")
  plt.show()

  # 4. 최적의 매개변수 구하기 - GridSearchCV
  params = {
      'n_estimators': [100, 200, 300],    # 결정 트리(Decision Tree)의 개수
      'max_depth': [5, 10, 15],           # 각 결정 트리의 최대 깊이를 설정
      'max_features': ['sqrt', 'log2']    # 각 트리가 학습할 때마다 사용할 특성(feature)의 수
  }
  gs_rf = GridSearchCV(
      estimator=rf,
      param_grid=params,
      scoring=scoring,
      refit='accuracy',
      cv=5,
      n_jobs=-1,
  )

  gs_rf.fit(X_train, y_train)

  # 5. Best Model: 최적의 하이파라미터로 만든 모델
  best_param_rf = gs_rf.best_params_
  best_model_rf = gs_rf.best_estimator_

  best_y_pred_rf = best_model_rf.predict(X_test)
  best_y_proba_rf= best_model_rf.predict_proba(X_test)[:, 1]

  ```

#### 3. Gradient Boosting : 정확도 96.79%

- 주요 파라미터

  > n_estimators: 부스팅 단계의 수 = 모델이 생성할 트리 개수
  >
  > learning_rate: 학습률
  >
  > max_depth: 각 결정 트리의 최대 깊이를 설정
  >
  > subsample: 각 트리 학습에 사용되는 샘플의 비율

  ```
  from sklearn.ensemble import GradientBoostingClassifier

  # 1. 학습 및 예측
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  gb = GradientBoostingClassifier()

  gb.fit(X_train, y_train)

  # 2. 모델 평가
  # Train set + Test set 평가
  y_train_pred_gb = gb.predict(X_train)
  y_train_proba_gb= gb.predict_proba(X_train)[:, 1]

  y_test_pred_gb = gb.predict(X_test)
  y_test_proba_gb= gb.predict_proba(X_test)[:, 1]

  # 혼동 행렬 시각화 (테스트 데이터)
  cm_test = confusion_matrix(y_test, y_test_pred_gb)
  plt.figure(figsize=(6,4))
  sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", cbar=False)
  plt.xlabel("예측")
  plt.ylabel("정답")
  plt.title("Confusion Matrix - Gradient Boosting (Test Set)")
  plt.show()

  evaluate("Train - Gradient Booting", y_train, y_train_pred_gb, y_train_proba_gb)
  evaluate("Test - Gradient Booting", y_test, y_test_pred_gb, y_test_proba_gb)

  # 3. 특성 중요도 계산 및 시각화
  fi = gb.feature_importances_
  fi_series = pd.Series(fi, index=df.drop(columns="churn").columns).sort_values(ascending=False)

  # 특성 중요도 시각화
  plt.figure(figsize=(10, 6))
  sns.barplot(x=fi_series, y=fi_series.index)
  plt.title("Feature Importances in Gradient Boosting")
  plt.xlabel("Importance")
  plt.ylabel("Feature")
  plt.show()

  # 4. 최적의 매개변수 구하기 - GridSearchCV
  params = {
      "n_estimators": [100, 200, 300],  #  부스팅 단계의 수 = 모델이 생성할 트리 개수
      "learning_rate": [0.1],  # 학습률
      "max_depth": [1, 2, 3, 4, 5],  # 각 결정 트리의 최대 깊이를 설정
      "subsample": [0.5, 0.7],  # 샘플링 비율
  }

  gs_gb = GridSearchCV(
      estimator=gb,
      param_grid=params,
      scoring=scoring,
      refit='accuracy',
      cv=5,
      n_jobs=-1,
  )

  gs_gb.fit(X_train, y_train)

  # 5. Best Model: 최적의 하이파라미터로 만든 모델
  best_param_gb = gs_gb.best_params_
  best_model_gb = gs_gb.best_estimator_

  best_y_pred_gb = best_model_gb.predict(X_test)
  best_y_proba_gb= best_model_gb.predict_proba(X_test)[:, 1]

  ```

#### 4. XGBoost : 정확도 97.19%

- 주요 파라미터

  > max_depth: 각 결정 트리의 최대 깊이를 설정
  >
  > learning_rate: 학습률
  >
  > n_estimators: 부스팅 단계의 수 = 모델이 생성할 트리 개수
  >
  > subsample: 각 트리의 훈련에 사용되는 샘플 비율
  >
  > colsample_bytree: 각 트리의 훈련에 사용되는 피처 비율
  >
  > gamma: 노드 분할에 대한 최소 손실 감소
  >
  > reg_alpha: L1 정규화
  >
  > reg_lambda: L2 정규화

  ```
  from xgboost import XGBClassifier

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  xgb = XGBClassifier()

  xgb.fit(X_train, y_train)

  # 2. 모델 평가
  # Train set + Test set 평가
  y_train_pred_xgb = xgb.predict(X_train)
  y_train_proba_xgb= xgb.predict_proba(X_train)[:, 1]

  y_test_pred_xgb = xgb.predict(X_test)
  y_test_proba_xgb= xgb.predict_proba(X_test)[:, 1]

  # 혼동 행렬 시각화 (테스트 데이터)
  cm_test = confusion_matrix(y_test, y_test_pred_xgb)
  plt.figure(figsize=(6, 4))
  sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", cbar=False)
  plt.xlabel("예측")
  plt.ylabel("정답")
  plt.title("Confusion Matrix - XGBoost (Test Set)")
  plt.show()

  evaluate("Train - XGBoost", y_train, y_train_pred_xgb, y_train_proba_xgb)
  evaluate("Test - XGBoost", y_test, y_test_pred_xgb, y_test_proba_xgb)

  # 3. 특성 중요도 계산 및 시각화
  fi = xgb.feature_importances_
  fi_series = pd.Series(fi, index=df.drop(columns="churn").columns).sort_values(ascending=False)

  # 특성 중요도 시각화
  plt.figure(figsize=(10, 6))
  sns.barplot(x=fi_series, y=fi_series.index)
  plt.title("Feature Importances in XGBoost")
  plt.xlabel("Importance")
  plt.ylabel("Feature")
  plt.show()

  # 4. 최적의 매개변수 구하기 - GridSearchCV
  params = {
      "max_depth":[1, 2, 3, 4, 5],            # 각 결정 트리의 최대 깊이를 설정
      'learning_rate': [0.1],                 # 학습률
      'n_estimators': [100, 200, 300],        # 부스팅 단계의 수 = 모델이 생성할 트리 개수
      'subsample': [0.5, 0.7],                # 각 트리의 훈련에 사용되는 샘플 비율
      'colsample_bytree': [0.5, 0.7, 1.0],    # 각 트리의 훈련에 사용되는 피처 비율
      'gamma': [0, 0.1],                      # 노드 분할에 대한 최소 손실 감소
      'reg_alpha': [0],                       # L1 정규화
      'reg_lambda': [0.1]                     # L2 정규화
  }
  gs_xgb = GridSearchCV(
      estimator=xgb,
      param_grid=params,
      scoring=scoring,
      refit='accuracy',
      cv=5,
      n_jobs=-1,
  )

  gs_xgb.fit(X_train, y_train)

  # 5. 튜닝 : Best Model 찾기
  best_param_xgb = gs_xgb.best_params_
  best_model_xgb = gs_xgb.best_estimator_

  best_y_pred_xgb = best_model_xgb.predict(X_test)
  best_y_proba_xgb= best_model_xgb.predict_proba(X_test)[:, 1]

  ```

| 머신러닝 방법    | Decision Tree Classifier                                                                                                                                                                                                                | Random Forest                                                                                                                                                                                                                                             | Gradient Boosting                                                                                                                                                                                           | XGBoost                                                                                                                                                                                                    |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Confusion Matrix | <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/%EA%B2%B0%EC%A0%95%EB%82%98%EB%AC%B4-cm.png" alt="image" width="200" height="200"/>                                                              | <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/%EB%9E%9C%EB%8D%A4%ED%8F%AC%EB%A0%88%EC%8A%A4%ED%8A%B8-cm.png" width="200" height="200"/>                                                                          | <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/gradient-cm.png" alt="image" width="200" height="200"/>                                                              | <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/XGboost-cm.png" alt="image" width="200" height="200"/>                                                              |
| 결과             | <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/%EA%B2%B0%EC%A0%95%EB%82%98%EB%AC%B4-%EA%B2%B0%EA%B3%BC.png" alt="image" width="300" height="150"/>                                              | <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/%EB%9E%9C%EB%8D%A4%ED%8F%AC%EB%A0%88%EC%8A%A4%ED%8A%B8-%EA%B2%B0%EA%B3%BC.png" width="300" height="150"/>                                                          | <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/gradient-%EA%B2%B0%EA%B3%BC.png" alt="image" width="300" height="150"/>                                              | <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/XGboost-%EA%B2%B0%EA%B3%BC.png" alt="image" width="300" height="150"/>                                              |
| 특성중요도       | <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/%EA%B2%B0%EC%A0%95%EB%82%98%EB%AC%B4-%ED%8A%B9%EC%84%B1%EC%A4%91%EC%9A%94%EB%8F%84.png" alt="image" width="300" height="150"/>                   | <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/%EB%9E%9C%EB%8D%A4%ED%8F%AC%EB%A0%88%EC%8A%A4%ED%8A%B8-%ED%8A%B9%EC%84%B1%EC%A4%91%EC%9A%94%EB%8F%84.png" width="300" height="150"/>                               | <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/gradient-%ED%8A%B9%EC%84%B1%EC%A4%91%EC%9A%94%EB%8F%84.png" alt="image" width="300" height="150"/>                   | <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/XGboost-%ED%8A%B9%EC%84%B1%EC%A4%91%EC%9A%94%EB%8F%84.png" alt="image" width="300" height="150"/>                   |
| 하이퍼파라미터   | <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/%EA%B2%B0%EC%A0%95%EB%82%98%EB%AC%B4-%ED%95%98%EC%9D%B4%ED%8D%BC%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0.png" alt="image" width="200" height="160"/> | <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/%EB%9E%9C%EB%8D%A4%ED%8F%AC%EB%A0%88%EC%8A%A4%ED%8A%B8-%ED%95%98%EC%9D%B4%ED%8D%BC%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0.png" alt="image" width="200" height="100"/> | <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/gradient-%ED%95%98%EC%9D%B4%ED%8D%BC%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0.png" alt="image" width="200" height="150"/> | <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/XGBoost-%ED%95%98%EC%9D%B4%ED%8D%BC%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0.png" alt="image" width="200" height="150"/> |

### ✔️ 모델 평가

```
# 여러 평가 지표 설정
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score),
    'auc': make_scorer(roc_auc_score)
}

model_box = pd.DataFrame(columns=['decision_tree', 'random_forest', 'gradient_boosting', 'xgboost'],
                            index = ['accuracy','precision','recall','f1 score','auc'])

def evaluate(title, y_real, y_pred, y_prob):
    acc = accuracy_score(y_real, y_pred)
    pre = precision_score(y_real, y_pred)
    rec = recall_score(y_real, y_pred)
    f1 = f1_score(y_real, y_pred)
    auc = roc_auc_score(y_real, y_prob)

    print(f"======= {title} =======")
    print('Accuracy : {:.6f}'.format(acc)) # 정확도 : 예측이 정답과 얼마나 정확한가
    print('Precision : {:.6f}'.format(pre)) # 정밀도 : 예측한 것 중에서 정답의 비율
    print('Recall : {:.6f}'.format(rec)) # 재현율 : 정답 중에서 예측한 것의 비율
    print('F1 score : {:.6f}'.format(f1)) # 정밀도와 재현율의 (조화)평균 - 정밀도와 재현율이 비슷할수록 높은 점수
    print('auc: {:.6f}'.format(auc))


    score_list = [acc,pre,rec,f1,auc]
    score_box = np.array(score_list)

    return score_box
```
| 모델 종류 | Decision Tree | Random Forest | Gradient Boosting | XGBoost|
|---------|---------------|---------------|-------------------|--------|
| Accuracy | 0.943196| 0.961348|0.974315|0.988392|
| Precision | 0.797872 | 0.957543| 0.954774|0.973059|
| Recall | 0.865385|0.799397| 0.880989|0.953416|
|F1 score | 0.830258|0.871352| 0.916399|0.963137|
|AUC|0.964163|0.993737|0.994633|0.998675|

### ✔️ 최고 성능 모델

🏆 XGBOOST

<br/>

## 모델 저장

하이퍼파라미터 튜닝을 통해 각 모델별 best params 를 통해 만든 best model들을 .pkl 파일로 저장.

```
import os
import joblib

directory = 'model/'
os.makedirs(directory, exist_ok=True)

joblib.dump(best_model_tree, os.path.join(directory, 'best_tree.pkl'))
joblib.dump(best_model_rf, os.path.join(directory, 'best_rf.pkl'))
joblib.dump(best_model_gb, os.path.join(directory, 'best_gb.pkl'))
joblib.dump(best_model_xgb, os.path.join(directory, 'best_xgb.pkl'))

# 저장된 모델과 파라미터 불러오기
model_tree = joblib.load('model/best_tree.pkl')
model_rf = joblib.load('model/best_rf.pkl')
model_gb = joblib.load('model/best_gb.pkl')
model_xgb = joblib.load('model/best_xgb.pkl')
```

## Streamlit


![image](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-2nd-4Team/blob/main/report/%EC%8A%A4%ED%8A%B8%EB%A6%BC%EB%A6%BF%20%EC%8B%A4%ED%96%89%20%ED%99%94%EB%A9%B4.png)

## 팀원 회고
김동명
> 끝까지 열심히 작업해주신 팀원분들 감사해요!
>
박유나
> 데이터 수집 단계에서 결측치, 이상치, 피처 엔지니어링을 충분히 다뤄볼 수 있는 데이터를 찾는 데에 주안점을 두었는데 팀원들과 같은 방향성을 공유하며 적절한 데이터를 선정할 수 있었어서 좋았습니다.
>
> 그리고 이번 프로젝트를 진행하며 프로젝트의 모든 과정을 다 함께하며 학습해보자는 목표가 있었는데 비록 시간이 상대적으로 걸리긴 했지만 그만큼 유익한 경험이되지 않았나 생각합니다. 저는 그 과정에서 방향을 제시하며 팀원들을 이끌어주었는데, 이 과정에서 오히려 제가 더 많이 배울 수 있었습니다.
>
> 특히 파이프라인 작업 중 예상과 다른 출력으로 인해 다양한 버그를 마주하며 많은 것을 배울 수 있어서 좋았습니다. 팀원들에게 설명하고 문제를 해결하는 과정이 큰 배움의 기회였던 것 같습니다.
>
> 또한, 이번 프로젝트에서는 Streamlit을 활용하여 시각화와 인터페이스를 구축하는 작업을 맡았습니다. 처음 예상보다 다루기 쉬웠고, 흥미롭게 동작한다고 생각했습니다. 사용성이 좋아 자주 활용할 것 같습니다.
>
임연경
> 저는 팀원들과 같이 머신러닝을 학습하는 과정을 맡고 파이프라인 제작을 위해 노력했지만 끝내 실패 하였습니다. 하지만 이 실패 속에서 문제를 해결하려는 끊임없는 시도와 학습은 제 역량을 더 커질 수 있도록 만들어 주었고, 다음에는 더 나은 결과를 만들어낼 자신감을 얻을 수 있었습니다.
>
