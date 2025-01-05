# -*- coding: cp949 -*-


import pandas as pd
import requests
import os
from tkinter import Tk, filedialog, messagebox, StringVar, Toplevel, Label, Radiobutton, Button
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("process_log.log"),
        logging.StreamHandler()
    ]
)

# 네이버 API 설정
HEADERS = {
    'X-Naver-Client-Id': 'qw3_o0F8S2wII1JC3WjM',
    'X-Naver-Client-Secret': 'aGqOT3DXuf'
}

def select_file():
    """파일 선택 다이얼로그를 열고 파일 경로를 반환"""
    Tk().withdraw()  # Tkinter 기본 창 숨기기
    file_path = filedialog.askopenfilename(
        title="Select CSV File",
        filetypes=[("CSV files", "*.csv")]
    )
    if not file_path:
        raise FileNotFoundError("No file selected.")
    return file_path

def select_card_type():
    """라디오 버튼을 이용해 카드사를 선택하도록 GUI 제공"""
    root = Tk()
    root.title("Select Card Type")
    root.geometry("300x200")

    card_type = StringVar(value="")

    def submit():
        root.destroy()

    Label(root, text="Select Card Type:", font=("Arial", 14)).pack(pady=10)
    Radiobutton(root, text="Hyundai Card", variable=card_type, value="hyundai", font=("Arial", 12)).pack(anchor="w")
    Radiobutton(root, text="KB Card", variable=card_type, value="kb", font=("Arial", 12)).pack(anchor="w")
    Button(root, text="Submit", command=submit, font=("Arial", 12)).pack(pady=20)

    root.mainloop()
    if not card_type.get():
        raise ValueError("No card type selected.")
    return card_type.get()

def get_category(store_name):
    """가맹점명을 네이버 API를 사용하여 카테고리로 분류"""
    try:
        url = f'https://openapi.naver.com/v1/search/local.json?query={store_name}&display=1'
        response = requests.get(url, headers=HEADERS, verify=False)  # SSL 인증서 검증 비활성화
        if response.status_code == 200:
            result = response.json()
            if result['items']:
                return result['items'][0].get('category', '기타')
        return '기타'
    except Exception as e:
        logging.error(f"Failed to get category for store {store_name}: {e}")
        return '기타'

def process_dataframe(df, card_type, file_index):
    """DataFrame을 처리하고 카테고리를 추가한 후 엑셀로 저장"""
    try:
        # 카드사에 따라 열 이름 및 시작 위치 결정
        if card_type == "hyundai":
            store_column = '가맹점명'
            df = df.iloc[2:].reset_index(drop=True)  # 현대 카드 데이터는 3행부터 시작
        elif card_type == "kb":
            store_column = '가맹점 정보'
            df = df.iloc[8:].reset_index(drop=True)  # KB 카드 데이터는 9행부터 시작
        else:
            raise ValueError("Invalid card type for column selection.")

        if store_column not in df.columns:
            raise ValueError(f"Store column '{store_column}' not found in DataFrame.")

        # 중복 가맹점명 제거 후 병렬 처리
        unique_stores = df[store_column].dropna().unique()

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(get_category, store): store for store in unique_stores}
            for future in as_completed(futures):
                store = futures[future]
                category = future.result()
                df.loc[df[store_column] == store, '카테고리'] = category

        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file_path = os.path.join(os.path.expanduser("~"), "Desktop", f"anal_{file_index}_{timestamp}.csv")
        df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
        logging.info(f"Results saved to {output_file_path}")
    except Exception as e:
        logging.error(f"Error processing DataFrame: {e}")
        raise

def main():
    try:
        # CSV 파일 선택 및 로드
        csv_file = select_file()
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_file, encoding='cp949')

        # 카드사 선택
        card_type = select_card_type()

        # 데이터 처리
        process_dataframe(df, card_type, 1)

        messagebox.showinfo("Success", "Processing completed successfully!")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    main()
