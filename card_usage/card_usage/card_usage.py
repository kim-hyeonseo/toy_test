# -*- coding: cp949 -*-


import pandas as pd
import requests
import os
from tkinter import Tk, filedialog, messagebox, StringVar, Toplevel, Label, Radiobutton, Button
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from datetime import datetime

# �α� ����
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("process_log.log"),
        logging.StreamHandler()
    ]
)

# ���̹� API ����
HEADERS = {
    'X-Naver-Client-Id': 'qw3_o0F8S2wII1JC3WjM',
    'X-Naver-Client-Secret': 'aGqOT3DXuf'
}

def select_file():
    """���� ���� ���̾�α׸� ���� ���� ��θ� ��ȯ"""
    Tk().withdraw()  # Tkinter �⺻ â �����
    file_path = filedialog.askopenfilename(
        title="Select CSV File",
        filetypes=[("CSV files", "*.csv")]
    )
    if not file_path:
        raise FileNotFoundError("No file selected.")
    return file_path

def select_card_type():
    """���� ��ư�� �̿��� ī��縦 �����ϵ��� GUI ����"""
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
    """���������� ���̹� API�� ����Ͽ� ī�װ��� �з�"""
    try:
        url = f'https://openapi.naver.com/v1/search/local.json?query={store_name}&display=1'
        response = requests.get(url, headers=HEADERS, verify=False)  # SSL ������ ���� ��Ȱ��ȭ
        if response.status_code == 200:
            result = response.json()
            if result['items']:
                return result['items'][0].get('category', '��Ÿ')
        return '��Ÿ'
    except Exception as e:
        logging.error(f"Failed to get category for store {store_name}: {e}")
        return '��Ÿ'

def process_dataframe(df, card_type, file_index):
    """DataFrame�� ó���ϰ� ī�װ��� �߰��� �� ������ ����"""
    try:
        # ī��翡 ���� �� �̸� �� ���� ��ġ ����
        if card_type == "hyundai":
            store_column = '��������'
            df = df.iloc[2:].reset_index(drop=True)  # ���� ī�� �����ʹ� 3����� ����
        elif card_type == "kb":
            store_column = '������ ����'
            df = df.iloc[8:].reset_index(drop=True)  # KB ī�� �����ʹ� 9����� ����
        else:
            raise ValueError("Invalid card type for column selection.")

        if store_column not in df.columns:
            raise ValueError(f"Store column '{store_column}' not found in DataFrame.")

        # �ߺ� �������� ���� �� ���� ó��
        unique_stores = df[store_column].dropna().unique()

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(get_category, store): store for store in unique_stores}
            for future in as_completed(futures):
                store = futures[future]
                category = future.result()
                df.loc[df[store_column] == store, 'ī�װ�'] = category

        # ��� ����
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file_path = os.path.join(os.path.expanduser("~"), "Desktop", f"anal_{file_index}_{timestamp}.csv")
        df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
        logging.info(f"Results saved to {output_file_path}")
    except Exception as e:
        logging.error(f"Error processing DataFrame: {e}")
        raise

def main():
    try:
        # CSV ���� ���� �� �ε�
        csv_file = select_file()
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_file, encoding='cp949')

        # ī��� ����
        card_type = select_card_type()

        # ������ ó��
        process_dataframe(df, card_type, 1)

        messagebox.showinfo("Success", "Processing completed successfully!")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    main()
