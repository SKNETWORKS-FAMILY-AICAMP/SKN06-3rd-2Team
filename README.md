# SKN06-3rd-2Team
강채연, 김동명, 박창규, 백하은, 홍준

## <흑흑요리사>

### 팀원

| 강채연 | 김동명| 박창규 | 백하은| 홍준 |
| --- | --- | --- | --- | --- |
| ![IE001847808_STD](https://github.com/user-attachments/assets/c2f2176b-7c40-4769-b3a4-7dbef5565109) |![돌아이](https://github.com/user-attachments/assets/eac6eb13-7166-409a-9f04-defaeb629cfe)|![100](https://github.com/user-attachments/assets/89f5e5ed-6412-44bf-99e9-13bef9668415)|![강록이형](https://github.com/user-attachments/assets/3f3a4698-0df0-4028-8a9c-d8ced265a568)|![눈감성재](https://github.com/user-attachments/assets/19d76aae-223c-44b1-9fb9-57fd0991293f)|
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
pandas == 2.2.3 </br>
numpy == 1.26.4 </br>
config == 0.5.1 </br>
langchain == 0.3.13 </br>
chromadb == 0.5.23 </br>
</br></br>

## 🦴 0. 모델 구조
![structure](https://github.com/user-attachments/assets/628a2d6c-4033-4640-8008-118a8e4e6dba)


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
> 웹사이트 크롤링 데이터 : restaurants.json </br>
> 정규화 및 전처리 후 데이터 : final_restaurant.csv </br>
</br></br>
## 2. 모델링
### 🍖 1) embedding_vector 생성
> 결과 : vector_store에 39074 개의 문서 생성
```
# 데이터 불러오기 및 저장
data = pd.read_csv('data/final_restaurant.csv', low_memory=False)
# 모든 데이터를 활용하도록 문서화
doc_list = []
for _, info in data.iterrows():
    doc_list.append(Document(page_content=str(dict(info)), metadata=dict(info)))
print(f"총 {len(doc_list)}개의 문서가 생성되었습니다.")
# Vector store 저장
embedding_model = OpenAIEmbeddings(
    model=EMBEDDING_NAME
)
# Persist directory 없는 경우 생성
if not os.path.exists(PERSIST_DIRECTORY):
    os.makedirs(PERSIST_DIRECTORY)
# 연결 + document 추가
vector_store = Chroma.from_documents(
    documents= doc_list,
    embedding=embedding_model,
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIRECTORY
)
print("vecor_store에 저장완료")
```
</br>

### 🍖 2) config로부터 설정 값 입력
> setting
```
MODEL_NAME  = config.model_name
EMBEDDING_NAME = config.embedding_name
COLLECTION_NAME = config.collection_name
PERSIST_DIRECTORY = config.persist_directory
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
### 🍖 3) GPT 모델, Prompt, Retriever 생성
```
vector_store = Chroma(
    embedding_function=embedding_model,
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIRECTORY
)
# GPT Model 생성
model = ChatOpenAI(
    model='gpt-4o',
    temperature=0 
)
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 50,
        "fetch_k": 200,
        "lambda_mult": 0.5,
        # "filters": {"리본개수": {"$gte": 0}}
    }
)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", dedent("""
        당신은 한국의 식당을 소개하는 인공지능 비서입니다. 
        반드시 질문에 대해서 [context]에 주어진 내용을 바탕으로 답변을 해주세요. 
        질문에 '리본개수', '평점', '몇 개'라는 키워드가 포함된 경우, [context]에서 "리본개수" 항목을 확인해 답변하세요.
        리본개수는 평점과 같은 의미를 가집니다.
        [context]
        {context}
    """)),
    ("human", "{question}")
])
# Chain 생성
def content_from_doc(docs:list[Document]):
    return "\n\n".join([d.page_content for d in docs])
chain =  {'context': retriever  | RunnableLambda(content_from_doc), 'question': RunnablePassthrough()}  | prompt_template | model | StrOutputParser()
```
</br>

### 🥩 4) 산출물 정리
> vector_store

</br>
</br>




## 😋 3. 모델 평가

|질문|서울 서대문구 근처 음식점 중 리본 개수가 2개인 맛집을 추천해줘.|
|---|---|
|응답|서울 서대문구 근처에서 리본개수가 2개인 맛집으로는 '카덴'이 있습니다. 이곳은 일식주점, 이자카야로 정호영 셰프의 일식 요리를 즐길 수 있는 곳입니다. 모둠사시미를 비롯해 잘 구운 생선구이 등 일품 요리를 맛볼 수 있으며, 제철 재료를 사용하기 때문에 메뉴는 주기적으로 바뀝니다. 마주 보고 앉는 다치 자리 외에 별도로 구분된 룸도 갖추고 있습니다.
|질문|프랑스식을 추천해주세요|
|응답|<img src="https://github.com/user-attachments/assets/2b3c86dd-4ee0-4776-9ed1-9445f3d06359" alt="image" width="50%">|
</br>

### 1) 🥘 평가 결과, 결과물
```
Recall: 0.66
F1 Score: 0.68
```
</br>
> evaluation_results.csv </br>


### 2) 🤔 Discussion
> 방법 : 무작위로 하나의 식당(Document)에 대한 정보를 뽑아서 예상 질문 다섯개를 만들어 성능 평가. 총 100개 ( 20개 식당 X 5개의 질문)
> 결과


1. 특정 단어에 대한 질문을 어려워 함.</br>
- 연중무휴 혹은 무휴 등</br>
- > filter 추가로 개선</br>

```
Sample 23:   
Question: 서오릉메카다슬기의 휴무일은 언제인가요?   
Reference: 연중무휴   
Response: 죄송하지만, [context]에 서오릉메카다슬기에 대한 정보는 없습니다. 따라서 서오릉메카다슬기의 휴무일에 대한 정보를 제공할 수 없습니다. 다른 식당에 대한 정보를 원하시면 말씀해 주세요.   
Context Used: 휴무일: 연중무휴   
```

</br>

2. 답을 알고 있지만, 인지를 못함.</br>

```
Sample 95:   
Question: 대저할매국수의 주소는 어디인가요?   
Reference: 부산광역시 강서구 대저중앙로 337   
Response: 죄송하지만, [context]에 제공된 정보에는 대저할매국수에 대한 정보가 포함되어 있지 않습니다. 부산광역시 강서구 대저중앙로 337에 위치한 대저할매국수의 주소는 해당 주소일 가능성이 있습니다. 추가적인 정보는 다른 출처를 참고하시기 바랍니다.   
Context Used: 부산광역시 강서구 대저중앙로 337   
```


</br></br>

## 5. 팀원 회고     
강채연
> 플젝하면서 맛집 많이 알게되어서 조앗다 우하하
</br>

김동명
> 크롤링이 선녀였네..
</br>

박창규
>
</br>

백하은
>
</br>

홍준
> 어디로 가야하오...

