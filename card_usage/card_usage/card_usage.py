import pandas as pd
import requests
import os
from tkinter import Tk, filedialog, messagebox, StringVar, Label, Radiobutton, Button
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

try:
    logging.info("Starting program execution.")

    # ���� ����
    Tk().withdraw()
    csv_file = filedialog.askopenfilename(
        title="Select CSV File",
        filetypes=[("CSV files", "*.csv")]
    )
    if not csv_file:
        raise FileNotFoundError("No file selected.")
    logging.info(f"Selected file: {csv_file}")

    # ���� �б�
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
        logging.info("File loaded successfully with UTF-8 encoding.")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_file, encoding='cp949')
        logging.info("File loaded successfully with CP949 encoding.")


    # ī��� ����
    root = Tk()
    root.title("Select Card Type")
    root.geometry("300x200")
    from tkinter import Tk, StringVar, Label, Radiobutton, Button, messagebox

    # �α� ����
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("process_log.log"),
            logging.StreamHandler()
        ]
    )

    # Tkinter ����
    root = Tk()
    root.title("Select Card Type")
    root.geometry("300x200")

    # StringVar �ʱ�ȭ
    card_type = StringVar(value="")

    def trace_card_type(*args):
        logging.info(f"Trace: card_type value changed to: {card_type.get()}")

    # Trace�� ���� StringVar �� ���� �����
    card_type.trace_add("write", trace_card_type)

    def submit():
        selected_value = card_type.get()
        logging.info(f"Submit clicked. Selected card_type: {selected_value}")
        if selected_value:
            root.destroy()
        else:
            messagebox.showerror("Error", "Please select a card type.")

    Label(root, text="Select Card Type:", font=("Arial", 14)).pack(pady=10)
    Radiobutton(root, text="Hyundai Card", variable=card_type, value="hyundai", font=("Arial", 12)).pack(anchor="w")
    Radiobutton(root, text="KB Card", variable=card_type, value="kb", font=("Arial", 12)).pack(anchor="w")
    Button(root, text="Submit", command=submit, font=("Arial", 12)).pack(pady=20)

    root.mainloop()

    selected_card_type = card_type.get()
    logging.info(f"Final selected card_type: {selected_card_type}")
    if not selected_card_type:
        raise ValueError("No card type selected.")
    logging.info(f"Selected card type: {selected_card_type}")


    # ������ ó��
    if selected_card_type == "hyundai":
        store_column = '��������'
        df = df.iloc[1:].reset_index(drop=True)  # ���� ī�� �����ʹ� 2����� ����
    elif selected_card_type == "kb":
        store_column = '������ ����'
        df = df.iloc[8:].reset_index(drop=True)  # KB ī�� �����ʹ� 9����� ����
    else:
        raise ValueError("Invalid card type for column selection.")

    logging.info(f"DataFrame columns: {df.columns.tolist()}")
    logging.info(f"First few rows of the DataFrame:\n{df.head()}")

    if store_column not in df.columns:
        raise ValueError(f"Store column '{store_column}' not found in DataFrame.")

    unique_stores = df[store_column].dropna().unique()
    logging.info(f"Found {len(unique_stores)} unique stores to process.")

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(
            lambda store: requests.get(
                f'https://openapi.naver.com/v1/search/local.json?query={store}&display=1',
                headers=HEADERS, verify=False
            ).json().get('items', [{}])[0].get('category', '��Ÿ'), store
        ): store for store in unique_stores}
        for future in as_completed(futures):
            store = futures[future]
            category = future.result()
            df.loc[df[store_column] == store, 'ī�װ���'] = category
            logging.info(f"Processed store: {store}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_path = os.path.join(os.path.expanduser("~"), "Desktop", f"anal_{timestamp}.csv")
    df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
    logging.info(f"Results saved to {output_file_path}")

    messagebox.showinfo("Success", "Processing completed successfully!")
    logging.info("Program execution completed successfully.")

except Exception as e:
    logging.error(f"An error occurred: {e}")
    messagebox.showerror("Error", str(e))
