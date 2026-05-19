import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from io import StringIO
import re

if 'df' not in st.session_state:
    st.session_state.df = None
if 'original_df' not in st.session_state:
    st.session_state.original_df = None

### 页面标题
st.title('数据清洗工具')
st.markdown('**欢迎来到交互式数据清洗工具页面。**')

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

### 数据上传
uploaded_file = st.file_uploader("选择一个 CSV 文件", type="csv")
if uploaded_file is not None:
    if 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
        st.session_state.last_uploaded_file = uploaded_file.name
        st.session_state.original_df = pd.read_csv(uploaded_file)
        st.session_state.df = st.session_state.original_df.copy()
    ### 展示数据
    st.subheader("原始数据预览：")
    st.dataframe(st.session_state.df.head())

    columns_info = {
        "Column Name": st.session_state.df.columns,
        "Non-Null Count": st.session_state.df.count(),
        "Data Type": st.session_state.df.dtypes
    }
    info_df = pd.DataFrame(columns_info)
    st.write("原始数据概述:")
    st.dataframe(info_df)

    ### 数据清洗功能
    st.subheader('数据清洗：')

    ### 缺失值处理
    missing_option = st.radio(
        '选择缺失值的处理方式：',
        ('不处理', '删除含缺失值的行', '填充缺失值')
    )
    st.session_state.df_mo_copy = st.session_state.df
    if missing_option == '不处理':
        pass
    elif missing_option == '删除含缺失值的行':
        delete_rows = st.radio(
            '选择删除含缺失值行的处理方式：',
            ('不处理', '删除有缺失值的行', '删除全为缺失值的行', '删除缺失值超过一定阈值的行')
        )
        if st.button('处理'):
            if delete_rows == '不处理':
                pass
            elif delete_rows == '删除有缺失值的行':
                st.session_state.df = st.session_state.df.dropna()
                st.success('已处理完含缺失值的行')
            elif delete_rows == '删除全为缺失值的行':
                st.session_state.df = st.session_state.df.dropna(how='all')
                st.success('已处理完含缺失值的行')
            elif delete_rows == '删除缺失值超过一定阈值的行':
                valve_value = st.number_input(
                    label='请输入对应阈值:',
                    format='%d'
                )
                if st.button('按照阈值处理'):
                    st.session_state.df = st.session_state.df.dropna(thresh=valve_value)
                    st.success('已处理完含缺失值的行')
    elif missing_option == '填充缺失值':
        selected_columns = st.multiselect('请选择需要处理的列：', st.session_state.df.columns)
        fill_value = st.radio(
            '选择填充缺失值的处理方式：',
            ('不处理', '零值填充', '列均值填充', '前向填充', '后向填充', '特殊值填充')
        )
        if st.button('处理'):
            if fill_value == '不处理':
                pass
            elif fill_value == '零值填充':
                st.session_state.df[selected_columns] = st.session_state.df[selected_columns].fillna(0)
                st.success('已填充缺失值')
            elif fill_value == '列均值填充':
                st.session_state.df[selected_columns] = st.session_state.df[selected_columns].apply(lambda x: x.fillna(x.mean()), axis=0)
                st.success('已填充缺失值')
            elif fill_value == '前向填充':
                st.session_state.df[selected_columns] = st.session_state.df[selected_columns].fillna(method='ffill')
                st.success('已填充缺失值')
            elif fill_value == '后向填充':
                st.session_state.df[selected_columns] = st.session_state.df[selected_columns].fillna(method='bfill')
                st.success('已填充缺失值')
            elif fill_value == '特殊值填充':
                special_value = st.number_input('请填写您的特殊值', format='%f')
                if st.button('提交'):
                    st.session_state.df[selected_columns] = st.session_state.df[selected_columns].fillna(special_value)
                    st.success('已填充缺失值')
    if st.button('撤回缺失值处理'):
        st.session_state.df = st.session_state.df_mo_copy
        st.success('已成功撤回上一级')
    
    ### 数据类型转换
    st.subheader('数据类型转换：')
    col_convert = st.multiselect('选择需要转换类型的列：', st.session_state.df.columns)
    dtype_option = st.radio(
        '选择目标数据类型：',
        ('字符串', '整数', '浮点数', '日期')
    )
    st.session_state.df_cc_copy = st.session_state.df
    if st.button('应用类型转换'):
        if dtype_option == '字符串':
            st.session_state.df[col_convert] = st.session_state.df[col_convert].astype('str')
        elif dtype_option == '整数':
            st.session_state.df[col_convert] = pd.to_numeric(st.session_state.df[col_convert], errors='coerce').astype('int')
        elif dtype_option == '浮点数':
            st.session_state.df[col_convert] = pd.to_numeric(st.session_state.df[col_convert], errors='coerce')
        elif dtype_option == '日期':
            st.session_state.df[col_convert] = pd.to_datetime(st.session_state.df[col_convert], errors='coerce')
        st.success(f'列{col_convert}已成功转换成{dtype_option}类型')
    if st.button('撤回类型转换处理'):
        st.session_state.df = st.session_state.df_cc_copy
        st.success('已成功撤回上一级')
    
    ### 分类数据编码
    st.subheader('分类数据编码')
    col_encode = st.multiselect('选择需要编码的列：', st.session_state.df.select_dtypes(include=['object', 'int']).columns)
    encode_type = st.radio(
        '选择编码方式：',
        ('独热编码', '标签编码')
    )
    st.session_state.df_ce_copy = st.session_state.df
    if st.button('应用编码'):
        if encode_type == '独热编码':
            for col in col_encode:
                encode = pd.get_dummies(st.session_state.df[col], prefix=col)
                st.session_state.df = pd.concat([st.session_state.df, encode], axis=1).drop(columns=[col])
        elif encode_type == '标签编码':
            st.session_state.df[col_encode] = st.session_state.df[col_encode].astype('category').cat.codes
        st.success(f'列{col_encode}已成功转换成{encode_type}类型')
    if st.button('撤回编码'):
        st.session_state.df = st.session_state.df_ce_copy
        st.success('已成功撤回上一级')

    ### 数据格式处理
    st.subheader('数据去格式处理：')
    col_format = st.multiselect('选择需要去格式处理的列：', st.session_state.df.select_dtypes(include=['object']).columns)
    st.session_state.df_cf_copy = st.session_state.df
    if st.button('去格式处理'):
        match = re.search(r'-?\d+(\.\d+)?')
        st.session_state.df[col_format] = st.session_state.df[col_format].apply(lambda x: float(re.search(r'-?\d+(\.\d+)?', x).group()) if re.search(r'-?\d+(\.\d+)?', x) else None)
        st.success(f'列{col_format}已成功去除数据的格式')
    if st.button('撤回去格式处理'):
        st.session_state.df = st.session_state.df_cf_copy
        st.success('已成功撤回上一级')

    ### 施工区

    ### 施工区

    st.subheader('处理后数据：')
    st.dataframe(st.session_state.df)

    columns_info = {
        "Column Name": st.session_state.df.columns,
        "Non-Null Count": st.session_state.df.count(),
        "Data Type": st.session_state.df.dtypes
    }
    info_df = pd.DataFrame(columns_info)
    st.write("处理后数据概述:")
    st.dataframe(info_df)

    if st.button('撤回'):
        st.session_state.df = st.session_state.original_df
        st.success('已成功撤回最初数据')
    
    st.markdown('### 下载清洗后的数据')
    new_df = st.session_state.df.to_csv(index=False).encode('gbk')
    st.download_button(
        label='下载数据为csv',
        data=new_df,
        file_name='Washed_Data.csv',
        mime='text/csv'
    )
