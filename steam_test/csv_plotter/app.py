import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="CSV Data Plotter", layout="centered")

st.title("📊 CSV 업로드 → 자동 데이터 시각화")

uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 데이터 미리보기")
    st.dataframe(df.head())

    st.subheader("⚙️ Plot 설정")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if len(numeric_cols) < 1:
        st.warning("숫자형 컬럼이 없습니다.")
    else:
        plot_type = st.selectbox(
            "그래프 종류 선택",
            ["scatter", "line", "bar", "hist", "box"]
        )

        x_col = st.selectbox("X축 컬럼", df.columns)
        y_col = None

        if plot_type not in ["hist", "box"]:
            y_col = st.selectbox("Y축 컬럼", numeric_cols)

        st.subheader("📈 결과 그래프")

        fig, ax = plt.subplots()

        if plot_type == "scatter":
            sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)

        elif plot_type == "line":
            sns.lineplot(data=df, x=x_col, y=y_col, ax=ax)

        elif plot_type == "bar":
            sns.barplot(data=df, x=x_col, y=y_col, ax=ax)

        elif plot_type == "hist":
            sns.histplot(data=df, x=x_col, kde=True, ax=ax)

        elif plot_type == "box":
            sns.boxplot(data=df, x=x_col, ax=ax)

        st.pyplot(fig)



# streamlit run app.py
