import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json


# 页面配置
st.set_page_config(page_title="BioInsight 平台", page_icon="🧬", layout="wide")

# 标题和介绍
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>欢迎来到 BioInsight 🧬</h1>
    <h3 style='text-align: center; color: gray;'>一站式生物医学数据分析平台</h3>
    <hr>
    """, unsafe_allow_html=True)

# 创建三个并列的列用于核心功能展示
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.image("images/1.png", width=100)
    st.markdown("**临床数据分析**")
    with st.expander("了解更多"):
        st.write("支持患者特征统计分析、生存分析、数据预处理等功能。")

with col2:
    st.image("images/2.png", width=100)
    st.markdown("**基因组数据分析**")
    with st.expander("了解更多"):
        st.write("支持变异检测、基因表达分析、GWAS 分析等功能。")

with col3:
    st.image("images/3.png", width=100)
    st.markdown("**单细胞数据分析**")
    with st.expander("了解更多"):
        st.write("支持细胞分类、UMAP/t-SNE 降维、差异表达分析等功能。")

with col4:
    st.image("images/4.png", width=100)
    st.markdown("**转录/蛋白/代谢组学分析**")
    with st.expander("了解更多"):
        st.write("支持代谢通路分析、蛋白互作分析等功能。")

st.markdown("<hr>", unsafe_allow_html=True)

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

st.subheader("功能简介")
st.markdown("""
- **AI数据分析助手**: 提供AI数据分析的能力。
- **数据分析工具**: 快速分析数据集的基本统计信息和可视化。
- **数据清洗工具**: 通过对缺失值、异常值等问题的修正，提升数据质量。
- **分类模型**: 使用机器学习算法构建分类模型。
- **回归模型**: 使用机器学习算法构建回归模型。
""")
st.markdown("<hr>", unsafe_allow_html=True)

# 简单问答功能
st.subheader("🧠 数据分析助手")
question = st.text_input("有问题？请在这里提问：", placeholder="例如：如何上传数据文件？")

if question:
    if "上传" in question:
        st.write("您可以通过左侧的 **'数据分析'** 功能上传 CSV 文件，工具会自动读取。")
    elif "清洗" in question or "处理" in question:
        st.write("您可以点击左侧导航中的 **数据清洗工具** 按钮，处理缺失值或删除异常值。")
    elif "模型" in question:
        st.write("根据任务需求，选择左侧 **分类模型** 或 **回归模型** 页面，构建机器学习模型。")
    else:
        st.write("目前没有明确的答案，请联系技术支持或浏览各页面的详细信息。")

# 底部信息
st.markdown("---")
st.markdown("💡 **提示**: 请确保上传的数据文件格式正确（如 CSV），并对数据进行必要的预处理。")


# cd 本main.py文件所在目录
# 运行   
# streamlit run main.py