import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

st.title("分类模型")

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

            # 拆分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Support Vector Classifier (SVC)": SVC(probability=True),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "K-Nearest Neighbors (KNN)": KNeighborsClassifier(),
                "Naive Bayes": GaussianNB()
            }

            # 用户选择分类模型
            model_name = st.selectbox("选择分类模型", list(models.keys()))

            # 训练并评估模型
            if model_name:
                model = models[model_name]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

                # 计算评估指标
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                matrix = confusion_matrix(y_test, y_pred)

                # 显示评估结果
                st.subheader(f"模型: {model_name}")
                st.write(f"**准确率 (Accuracy):** {accuracy:.4f}")
                st.write("**分类报告:**")
                st.dataframe(pd.DataFrame(report).transpose())
                st.write("**混淆矩阵:**")
                st.write(matrix)

                # 可视化混淆矩阵
                st.subheader("混淆矩阵可视化")
                fig, ax = plt.subplots()
                cax = ax.matshow(matrix, cmap=plt.cm.Blues)
                plt.colorbar(cax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)


                # 可视化分类概率（如果支持）
                if y_proba is not None:
                    st.subheader("分类概率分布")
                    # 创建分类概率的 DataFrame
                    proba_df = pd.DataFrame(y_proba, columns=[f"Class {cls}" for cls in np.unique(y)])
                    max_samples = 25
                    if proba_df.shape[0] > max_samples:
                        proba_df = proba_df.head(max_samples)
                        y_test = y_test[:max_samples]
                    proba_df.index = [f"Sample {i+1}" for i in range(proba_df.shape[0])]

                    # 如果真实类别可用，将其添加为标签
                    true_classes = pd.Series(y_test, index=proba_df.index, name="True Class")

                    # 堆积柱状图
                    st.write("堆积柱状图显示每个样本在不同类别的概率分布，同时标注真实类别：")
                    fig, ax = plt.subplots(figsize=(12, 7))
                    proba_df.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')

                    # 在图表上标注真实类别
                    fontsize = max(6, 12 - len(proba_df) // 10)  # 样本越多字体越小
                    for i, true_class in enumerate(true_classes):
                        color = 'red' if true_class != model.predict([X_test[i]])[0] else 'green'
                        ax.text(
                            i, 1.02, f"True: {true_class}", 
                            ha='center', va='bottom', fontsize=fontsize, color=color, rotation=45  # 调整字体大小和角度
                        )

                    # 图表美化
                    ax.set_xlabel("Samples")
                    ax.set_ylabel("Probability")
                    ax.set_title("Stacked Bar Chart of Classification Probabilities with True Classes")
                    ax.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.xticks(rotation=45, ha='right')  # X轴样本标签倾斜
                    st.pyplot(fig)

                # 输入新数据并预测分类
                st.subheader("新数据预测")
                user_input = st.text_area(
                    "输入新数据点 (用逗号分隔特征值)", 
                    help="按照选择的特征顺序输入数值，如 5.1,3.5,1.4,0.2"
                )

                if user_input:
                    try:
                        # 解析输入的新数据
                        new_data = np.array(user_input.split(",")).astype(float).reshape(1, -1)
                        
                        # 检查特征维度是否匹配
                        if new_data.shape[1] != X_train.shape[1]:
                            st.error(f"输入的特征数量 ({new_data.shape[1]}) 与训练模型所需的特征数量 ({X_train.shape[1]}) 不匹配。")
                        else:
                            prediction = model.predict(new_data)
                            proba = model.predict_proba(new_data) if hasattr(model, "predict_proba") else None

                            st.write("**预测结果:**")
                            st.write(f"分类: {prediction[0]}")
                            if proba is not None:
                                st.write("分类概率:")
                                st.write(dict(zip([f"Class {cls}" for cls in np.unique(y)], proba[0])))
                    except Exception as e:
                        st.error(f"数据输入有误，请检查格式是否正确。错误信息: {e}")
        else:
            st.warning("请选择至少一个特征列 (X)。")
    else:
        st.warning("请先选择目标列 (y)。")
