import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ===== 설정 =====
MAX_FREE_SIZE_KB = 200
STRIPE_PAYMENT_URL = "https://buy.stripe.com/test_7sYcN5h1Q5g28xQfLE7ok00"

st.set_page_config(page_title="CSV Data Plotter", layout="centered")
st.title("CSV 업로드 → 자동 데이터 시각화")

uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file:
    file_size_kb = uploaded_file.size / 1024
    st.write(f"업로드 파일 크기: **{file_size_kb:.1f} KB**")

    # 유료 구간 (1회권)
    if file_size_kb > MAX_FREE_SIZE_KB:
        st.error("200KB를 초과한 파일은 1회권 결제가 필요합니다.")

        st.markdown(
            f"""
            ###  대용량 CSV 1회 이용권
            - 200KB 초과 CSV 1회 처리
            - 추가 회원가입 없음
            - 결제 후 바로 사용

            **[1회권 결제하기]({STRIPE_PAYMENT_URL})**
            """,
            unsafe_allow_html=True
        )

        st.info("결제 후 새로고침하여 다시 파일을 업로드하세요.")
        st.stop()  #  여기서 앱 실행 중단

    #  무료 구간
    df = pd.read_csv(uploaded_file)

    st.subheader("데이터 미리보기")
    st.dataframe(df.head())

    st.subheader("Plot 설정")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if not numeric_cols:
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