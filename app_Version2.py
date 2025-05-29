import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from feynman import run_feynman

st.title("AI Feynman 2.0 GUI (feynman本体版)")

st.write("""
AI Feynman (feynmanモジュール) を使って、アップロードしたデータから物理法則的な数式を発見します。
""")

uploaded_file = st.file_uploader("CSVファイルをアップロードしてください (先頭行: ヘッダ, 最後の列がY)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("データプレビュー:", df.head())
    columns = df.columns.tolist()

    x_cols = st.multiselect("説明変数 (X) を選んでください", columns, default=columns[:-1])
    y_col = st.selectbox("目的変数 (Y) を選んでください", columns, index=len(columns)-1)

    if st.button("AI Feynmanで数式発見！"):
        # データ整形
        X = df[x_cols].values
        y = df[y_col].values

        # AI FeynmanはCSVパスを要求するため、一時ファイル保存
        fname = "temp_data.csv"
        full_data = pd.concat([df[x_cols], df[[y_col]]], axis=1)
        full_data.to_csv(fname, index=False, header=False)

        # 入力変数名・出力変数名
        varnames = x_cols + [y_col]

        # feynmanでシンボリック回帰
        results = run_feynman(
            fname,
            varnames=varnames,
            BF_try_time=30,   # 探索時間（秒）
            polyfit_deg=5,    # 多項式次数
            NN_epochs=1000,   # ニューラルネット回帰のエポック数
            N_SAMPLES=5000    # サンプリング数
        )
        st.write("AI Feynmanによる最良数式：")
        st.latex(results['best_eq'])

        st.write("詳細結果：")
        st.json(results)

        # 推定値と実測値のグラフ
        y_pred = results['best_func'](X.T)
        fig, ax = plt.subplots()
        ax.scatter(y, y_pred, alpha=0.5)
        ax.set_xlabel("実測値")
        ax.set_ylabel("予測値")
        ax.set_title("実測値 vs 予測値 (AI Feynman)")
        st.pyplot(fig)

st.markdown("---")
st.write("※ 本ツールはAI Feynman本体を用いています。複雑な数式や大規模データの場合、計算時間が長くなる場合があります。")