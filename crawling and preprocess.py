import os
import requests
import time
import json
import pandas as pd
import numpy as np
import re
from dotenv import load_dotenv

# .env 파일에서 환경 변수 불러오기
load_dotenv()

class DataCrawler:
    def __init__(self, total_pages, output_path):
        self.total_pages = total_pages
        self.url_template = os.getenv("URL_TEMPLATE")
        self.headers = {
            "User-Agent": os.getenv("USER_AGENT"),
            "Accept": os.getenv("ACCEPT"),
            "x-requested-with": os.getenv("X_REQUESTED_WITH")
        }
        self.output_path = output_path

    def crawl(self):
        restaurants = []
        for page in range(self.total_pages):
            url = self.url_template.format(page=page)
            response = requests.get(url, headers=self.headers)

            if response.status_code == 200:
                data = response.json()
                restaurants.extend(data["_embedded"]["restaurants"])
                print(f"Page {page + 1}/{self.total_pages} collected successfully.")
            else:
                print(f"Failed to fetch page {page + 1}. Status code: {response.status_code}")

            time.sleep(2)  # 2초 간격

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(restaurants, f, ensure_ascii=False, indent=4)
        print("Data crawling completed and saved to JSON.")


class DataPreprocessor:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def preprocess(self):
        with open(self.input_path, "r", encoding="utf-8") as f:
            restaurants = json.load(f)

        df = pd.DataFrame(restaurants)

        # Drop unnecessary columns
        columns_to_drop = ["createdDate", "timeInfo", "gps", "tags", "status",
                           "buzUsername", "business", "pageView", "brandMatchStatus", "brandRejectReason",
                           "orderDescending", "foodTypeDetails", "foodDetailTypes", "countEvaluate", "bookmark", "features",
                           "feature107", "brandBranches", "brandHead", "firstImage", "firstLogoImage",
                           "_links"]
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

        # Normalize nested columns
        nested_columns = ["headerInfo", "defaultInfo", "statusInfo", "juso", "review", "etcInfo"]
        for column in nested_columns:
            if column in df.columns:
                n = pd.json_normalize(df[column])
                n.columns = [f"{column}_{subcol}" for subcol in n.columns]
                df = pd.concat([df.drop(columns=[column]), n], axis=1)

        if "foodTypes" in df.columns:
            df["foodTypes"] = df["foodTypes"].apply(lambda x: ", ".join(x) if isinstance(x, list) else None)

        # Drop additional columns
        sub_columns_to_drop = ["headerInfo_nickname", "headerInfo_year", "headerInfo_ribbonTypeByOrdinal", "headerInfo_nameEN", "headerInfo_nameCN", "headerInfo_bookYear",
                               "defaultInfo_websiteFacebook", "defaultInfo_chefName", "statusInfo_storeType", "statusInfo_openEra", "statusInfo_openDate",
                               "juso_roadAddrPart2", "juso_jibunAddr", "juso_zipNo",
                               "juso_admCd", "juso_bdNm", "juso_buldMnnm", "juso_buldSlno", "juso_detBdNmList", "juso_zone2_1", "juso_zone2_2", "juso_map_1",
                               "juso_map_2", "juso_detailAddress", "juso_emdNm", "juso_engAddr", "juso_liNm",
                               "review_readerReview", "review_businessReview", "review_editorReview",
                               "etcInfo_toilet", "etcInfo_toiletEtc", "etcInfo_close", "etcInfo_interior",
                               "etcInfo_appYn", "etcInfo_projectNo", "etcInfo_reviewerRecommend", "etcInfo_onlySiteView",
                               "etcInfo_history", "etcInfo_mainMemo"]
        df = df.drop(columns=[col for col in sub_columns_to_drop if col in df.columns])

        # Replace empty strings and "없음" with None
        df.replace(["", "없음"], None, inplace=True)

        # Merge website columns
        if "defaultInfo_website" in df.columns and "defaultInfo_websiteInstagram" in df.columns:
            df["defaultInfo_website"] = df[["defaultInfo_website", "defaultInfo_websiteInstagram"]].apply(
                lambda x: ", ".join(filter(None, x)), axis=1
            )
            df.drop(columns=["defaultInfo_websiteInstagram"], inplace=True)

        # Extract year with suffix
        if "statusInfo_openDate" in df.columns:
            df["statusInfo_openDate"] = df["statusInfo_openDate"].apply(self.extract_year_with_suffix)

        # Save cleaned data
        df.to_csv(self.output_path, index=False, encoding="utf-8-sig")
        print("Preprocessed data saved to CSV.")

    @staticmethod
    def extract_year_with_suffix(date):
        if isinstance(date, str):
            match = re.search(r'\d{4}', date)
            if match:
                return f"{int(match.group(0))}년"
        return None


if __name__ == "__main__":
    # 크롤링
    crawler = DataCrawler(total_pages=576, output_path="data/restaurants.json")
    crawler.crawl()

    # 전처리
    preprocessor = DataPreprocessor(input_path="data/restaurants.json", output_path="data/cleaned_all_restaurants.csv")
    preprocessor.preprocess()

