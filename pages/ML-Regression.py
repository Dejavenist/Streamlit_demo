import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

st.title("回归模型")

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

uploaded_file = st.file_uploader("上传数据文件 (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("数据预览:")
    st.write(df.head())

    # 用户选择目标列 (y)
    y_column = st.selectbox("选择目标列 (y)", df.columns)

    if y_column:
        # 用户选择特征列 (X)
        feature_columns = st.multiselect(
            "选择特征列 (X)", 
            [col for col in df.columns if col != y_column]
        )
        
        if feature_columns:
            # 提取 y 和特征列
            y = df[y_column].values  
            X = df[feature_columns]  
            
            # 检查是否有非数值列
            non_numeric_columns = X.select_dtypes(include=['object', 'category']).columns
            if not non_numeric_columns.empty:
                st.write("以下列将进行 One-Hot 编码:", list(non_numeric_columns))
                # 对非数值列进行 One-Hot 编码
                X = pd.get_dummies(X, columns=non_numeric_columns, drop_first=True)
            
            # 转换为 NumPy 数组
            X = X.values
            
            # 显示特征矩阵和目标列
            st.write("目标值 (y):", y[:5])  
            st.write("特征矩阵 (X):", X[:5])  

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            models = {
                "Linear": LinearRegression(),
                "Ridge": Ridge(alpha=1.0),
                "Lasso": Lasso(alpha=0.1),
                "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
                "Support Vector Regression (SVR)": SVR(),
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            }

            # 用户选择回归模型
            model_name = st.selectbox("选择回归模型", list(models.keys()))

            # 训练并评估模型
            if model_name:
                model = models[model_name]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # 计算评估指标
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                # 显示评估结果
                st.subheader(f"模型: {model_name}")
                st.write(f"**均方误差 (MSE):** {mse:.4f}")
                st.write(f"**R² 得分:** {r2:.4f}")

                # 用户选择可视化类型
                st.subheader("预测结果对比")
                chart_type = st.radio("选择图表类型", ("折线图", "散点图", "分组柱状图"))

                chart_data = pd.DataFrame({
                    "True Values": y_test,
                    "Predicted Values": y_pred
                })

                if chart_type == "折线图":
                    st.line_chart(chart_data)
                elif chart_type == "散点图":
                    st.write("散点图：")
                    fig, ax = plt.subplots()
                    ax.scatter(y_test, y_pred, alpha=0.7)
                    ax.set_xlabel("True Values")
                    ax.set_ylabel("Predicted Values")
                    ax.set_title("散点图")
                    st.pyplot(fig)
                elif chart_type == "分组柱状图":
                    st.write("分组柱状图：")
                    # 创建分组柱状图
                    x = np.arange(len(y_test))  # X轴的位置
                    width = 0.35  # 每组柱子的宽度

                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.bar(x - width / 2, y_test, width, label="True Values", color="blue")
                    ax.bar(x + width / 2, y_pred, width, label="Predicted Values", color="orange")

                    ax.set_xlabel("Sample Index")
                    ax.set_ylabel("Value")
                    ax.set_title("分组柱状图")
                    ax.legend()
                    st.pyplot(fig)
        else:
            st.warning("请选择至少一个特征列 (X)。")
    else:
        st.warning("请先选择目标列 (y)。")
