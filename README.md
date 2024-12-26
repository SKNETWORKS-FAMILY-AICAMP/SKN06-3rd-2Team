# SKN06-3rd-2Team
강채연, 김동명, 박창규, 백하은, 홍준

## <흑흑요리사>

### 팀원

| 강채연 | 김동명| 박창규 | 백하은| 홍준 |
| --- | --- | --- | --- | --- |
| <> | <> | <> | ![강록이형](https://github.com/user-attachments/assets/1e278c6c-7b68-4a57-8a6f-4097f5faa2c2)
 | <> |
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


#### 🍴 Requirements

pandas == <> </br>
numpy == <> </br>
config == 0.5.1 </br>
langchain == 0.3.13 </br>
chromadb == 0.5.23 </br>

</br></br>

## 1. 데이터 준비 및 분석

### 🎣 1) 데이터 수집

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


### 🔪 2) 불필요 칼럼 삭제 및 정규화
> 결측치, 중복값, TMI (위도/경도, 업주명) 등
> 

```

# 전처리

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
### 🍡 3) 형식 일치화
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

### 🥩 4) 산출물 정리
> 웹사이트 크롤링 데이터 : blueRibbon.csv </br>
> 필요없는 column 제거 후 데이터 : nested_all_restaurants.csv </br>
> 정규화 및 전처리 후 데이터 : cleaned_all_restaurants.csv </br>

</br></br>

## 2. 모델링

### 🍖 1) embedding_vector 생성
> text_splitter, embeddings 를 사용하여 데이터를 분해 및 저장
> 결과 : vector_store에 39074 개의 문서 생성

```
# 데이터 불러오기 및 저장
data = pd.read_csv('data/cleaned_all_restaurants.csv')


# 모든 데이터를 활용하도록 문서화

documents = []
for i, row in data.iterrows():
    # 텍스트 내용 (각 행 전체를 하나의 문서로 취급)
    page_content = "\n".join([f"{col}: {val}" for col, val in row.items()])
    
    # Document 생성
    doc = Document(page_content=page_content)
    documents.append(doc)

print(f"총 {len(documents)}개의 문서가 생성되었습니다.")


# 분리
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name=MODEL_NAME, 
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)
splited_docs = splitter.split_documents(documents)

print("데이터 분리 완료:", len(splited_docs), end='\n\n')


# Vector store 저장
embedding_model = OpenAIEmbeddings(
    model=EMBEDDING_NAME
)

# Persist directory 없는 경우 생성
if not os.path.exists(PERSIST_DIRECTORY):
    os.makedirs(PERSIST_DIRECTORY)

# 연결 + document 추가
vector_store = Chroma.from_documents(
    documents= splited_docs,
    embedding=embedding_model,
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIRECTORY
)

print("vecor_store에 splited_docs 저장완료")
```

### 🍖 2) config로부터 설정 값 입력
> chunk_size : 500
> chunk_overlap : 100
> 
```
# setting

CHUNK_SIZE = config.chunk_size
CHUNK_OVERLAP = config.chunk_overlap

MODEL_NAME  = config.model_name
EMBEDDING_NAME = config.embedding_name

COLLECTION_NAME = config.collection_name
PERSIST_DIRECTORY = config.persist_directory

```

### 🍖 3) vector_store 에 저장
```
data = pd.read_csv("final_merged_result.csv")


# 모든 데이터를 활용하도록 문서화
data.fillna("", inplace=True)  # NaN 값 처리

documents = []
for i, row in data.iterrows():
    # 텍스트 내용 (각 행 전체를 하나의 문서로 취급)
    page_content = "\n".join([f"{col}: {val}" for col, val in row.items()])
    metadata = row.to_dict()
    # Document 생성
    doc = Document(page_content=page_content, metadata=metadata)
    
    documents.append(doc)

print(f"총 {len(documents)}개의 문서가 생성되었습니다.")


# 분리
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name=MODEL_NAME, 
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)
splited_docs = splitter.split_documents(documents)

print("데이터 분리 완료:", len(splited_docs), end='\n\n')


# Vector store 저장
embedding_model = OpenAIEmbeddings(
    model=EMBEDDING_NAME
)

# Persist directory 없는 경우 생성
if not os.path.exists(PERSIST_DIRECTORY):
    os.makedirs(PERSIST_DIRECTORY)

# 연결 + document 추가
vector_store = Chroma.from_documents(
    documents= splited_docs,
    embedding=embedding_model,
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIRECTORY
)

print("vecor_store에 splited_docs 저장완료")
```

> 저장된 내용 확인
    
```
COLLECTION_NAME = "bluer_db_openai"
PERSIST_DIRECTORY = "vector_store/chroma/bluer_db"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
MODEL_NAME = 'gpt-4o-mini'

# vector store 연결
vector_store = Chroma(
    embedding_function=embedding_model,
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIRECTORY
)

# 저장된 데이터 내용 확인
documents = vector_store._collection.get()['documents']
metadatas = vector_store._collection.get()['metadatas']

print(f"Documents: {documents[:5]}") 
print(f"Metadatas: {metadatas[:5]}")
print(vector_store._collection.count())
```
</br></br>
## 3. GPT 모델, Prompt, Retriever 생성
```
vector_store = Chroma(
    embedding_function=embedding_model,
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIRECTORY
)


# GPT Model 생성
model = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0 
)


# Retriever 생성 - "Map Reduce" 방식 - 더 정확한 답변
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k":5, "fetch_k":10, "lambda_mult":0.5}
)


# # Prompt Template 생성
# prompt_template = ChatPromptTemplate(
#     [
#         ("system", "당신은 한국의 블루리본 서베이 전문가입니다. 질문에 자세히 답해주세요."),
#         MessagesPlaceholder("history"), 
#         ("human", "{query}")
#     ]
# )

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a professinal for blue ribbon survey in korea. please reply correct anwser from exact information."),
    ("human", "{question}")
])


# Chain 생성

# retrieval_qa = RetrievalQA.from_chain_type(
#     llm=model,
#     retriever=retriever,
#     return_source_documents=True,  # Include source documents in response
# )

chain = ({'content': retriever, 'question':RunnablePassthrough()} | prompt_template | model | StrOutputParser() )

# response = retrieval_qa({"query": QUERY})

# print("응답:", response["result"])
```

</br></br>
## 😋 모델 평가

| 질문 | 서초동에 가족과 함께 갈만한 식당을 소개해줘 |
| ------- |-------|
| 응답 | ~~ |
| ------- |-------|
| 질문 | 청담동에 블루리본 2개 이상인 일식당을 소개해줘|
| ------- |-------|
| 응답 | ~~ |

###  최고 성능 모델![small BR](https://github.com/user-attachments/assets/9d9b741c-075b-4316-aeb3-91135fc85acb)![small BR](https://github.com/user-attachments/assets/9d9b741c-075b-4316-aeb3-91135fc85acb)![small BR](https://github.com/user-attachments/assets/9d9b741c-075b-4316-aeb3-91135fc85acb)

🏆 <-------------------->

</br>

### 모델 저장
>


```

```

</br></br>

## 팀원 회고
강채연
> 
>
김동명
>

박창규
>

백하은
>

홍준
>
