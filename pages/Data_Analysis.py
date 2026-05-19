import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 页面标题
st.title('数据分析工具')
st.markdown('**欢迎来到交互式数据分析工具页面。**')

# 自定义按钮样式
st.markdown("""
    <style>
        .custom-button {
            display: inline-block;
            padding: 10px 20px;
            font-size: 18px;
            font-weight: bold;
            color: white;
            background-color: #4CAF50;
            border: none;
            border-radius: 10px;
            text-align: center;
            cursor: pointer;
            transition: 0.3s;
        }
        .custom-button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)


# 导航和功能简介
st.sidebar.title("导航菜单")

st.sidebar.markdown("请选择以下功能：")
st.sidebar.markdown('<a href="/AI-analysis" class="custom-button">🚀 进入 AI 数据分析助手</a>', unsafe_allow_html=True)
st.sidebar.markdown('<a href="/Data_Analysis" class="custom-button">📊 数据分析工具</a>', unsafe_allow_html=True)
st.sidebar.markdown('<a href="/Data_Washing" class="custom-button">📚 数据清洗工具</a>', unsafe_allow_html=True)
st.sidebar.markdown('<a href="/ML-Classification" class="custom-button">📈 分类模型</a>', unsafe_allow_html=True)
st.sidebar.markdown('<a href="/ML-Regression" class="custom-button">📉 回归模型</a>', unsafe_allow_html=True)

# 上传数据文件
uploaded_file = st.file_uploader("选择一个 CSV 文件，请确保文件已经进行过预处理。", type=["csv"])

if uploaded_file is not None:
    
    # 读取数据
    data = pd.read_csv(uploaded_file)
    st.write("数据预览：")
    st.write(data.head())
    
    # 选择分析类型
    analysis_type = st.selectbox("选择分析类型", ["描述性统计", "相关性分析", "数据可视化"])

    if analysis_type == "描述性统计":
        st.write("描述性统计：")
        st.write(data.describe())

    elif analysis_type == "相关性分析":
        st.write("相关性分析：")
        numeric_data = data.select_dtypes(include=[np.number])
        corr_matrix = numeric_data.corr()
        st.write(corr_matrix)
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    elif analysis_type == "数据可视化":
        st.write("数据可视化：")
        plot_type = st.selectbox("选择图表类型", ["散点图", "直方图", "箱线图"])
        if plot_type == "散点图":
            x = st.selectbox("选择X轴变量", data.columns)
            y = st.selectbox("选择Y轴变量", data.columns)
            fig, ax = plt.subplots()
            sns.scatterplot(data=data, x=x, y=y, ax=ax)
            st.pyplot(fig)
        elif plot_type == "直方图":
            column = st.selectbox("选择变量", data.columns)
            fig, ax = plt.subplots()
            sns.histplot(data[column], kde=True, ax=ax)
            st.pyplot(fig)
        elif plot_type == "箱线图":
            column = st.selectbox("选择变量", data.columns)
            fig, ax = plt.subplots()
            sns.boxplot(data[column], ax=ax)
            st.pyplot(fig)

