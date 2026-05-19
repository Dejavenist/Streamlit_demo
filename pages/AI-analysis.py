import requests
import json
import time
import streamlit as st
import tempfile
import os
import pandas as pd

# API 配置
BOT_ID = "7483873851123023922"
API_TOKEN = "pat_ed7gleO1eX5m6T1xe0d8jl5vn2CYiuWDhw0ISDS0N0ZFknlH6U4UcGdB4gk7l40K"
UPLOAD_URL = "https://api.coze.cn/v1/files/upload"  # 文件上传接口URL
CHAT_URL = "https://api.coze.cn/v3/chat"  # 聊天接口URL

# 上传文件到 Coze
def upload_file_to_coze(file):
    headers = {
        "Authorization": f"Bearer {API_TOKEN}"
    }
    files = {'file': (file.name, file, 'application/octet-stream')}
    response = requests.post(UPLOAD_URL, headers=headers, files=files)
    
    if response.status_code == 200:
        response_json = response.json()
        st.write("File upload response:", json.dumps(response_json, indent=2, ensure_ascii=False))
        return response_json.get('data', {}).get('id')  # 安全地返回 file_id
    else:
        st.error("文件上传失败")
        return None

# 创建对话并引用文件
def create_conversation(file_id, question):
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    # 构建多模态内容
    object_string = [
        {
            "type": "text",
            "text": question  # 用户的问题
        },
        {
            "type": "file",
            "file_id": file_id  # 引用上传的文件
        }
    ]
    
    # 构建请求体
    data = {
        "bot_id": BOT_ID,
        "user_id": "123456789",  # 可以使用任意用户ID
        "stream": False,
        "auto_save_history": True,
        "additional_messages": [
            {
                "role": "user",
                "content": json.dumps(object_string),  # 将多模态内容转换为JSON字符串
                "content_type": "object_string"
            }
        ]
    }
    
    response = requests.post(CHAT_URL, headers=headers, json=data)
    return response.json()

# 获取聊天消息
def get_chat_messages(chat_id, conversation_id):
    url = f"{CHAT_URL}/message/list?chat_id={chat_id}&conversation_id={conversation_id}"
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }

    response = requests.get(url, headers=headers)
    return response.json()

# 主程序
def main(file, question):
    # 上传文件并获取 file_id
    file_id = upload_file_to_coze(file)
    if file_id:
        # 创建对话
        conversation_response = create_conversation(file_id, question)
        st.write("Conversation response:", json.dumps(conversation_response, indent=2, ensure_ascii=False))

        if "code" in conversation_response and conversation_response["code"] == 0 and "data" in conversation_response:
            chat_id = conversation_response["data"].get("id")
            conversation_id = conversation_response["data"].get("conversation_id")

            if chat_id and conversation_id:
                # 尝试获取聊天消息，最多重试5次
                messages_data = []  # 存储所有消息
                for attempt in range(5):
                    st.write(f"尝试获取消息，第 {attempt + 1} 次")
                    messages = get_chat_messages(chat_id, conversation_id)

                    if "code" in messages and messages["code"] == 0:
                        if "data" in messages and messages["data"]:
                            messages_data.extend(messages["data"])
                        else:
                            st.write("消息数据为空，等待15秒后重试")
                    else:
                        st.write("获取聊天消息失败或返回格式不正确，等待15秒后重试")

                    time.sleep(15)  # 等待15秒后重试

                # 处理所有获取到的消息
                found_answer = False
                max_follow_up = 3
                follow_up_count = 0

                for message in messages_data:
                    if message["role"] == "assistant" and message["type"] == "answer":
                        found_answer = True  # 标记找到回答
                        st.write("智能体的回答：", message["content"])
                    elif message["role"] == "assistant" and message["type"] == "follow_up" and found_answer:
                        st.write("智能体的后续问题：", message["content"])
                        follow_up_count += 1
                    if found_answer and follow_up_count == max_follow_up:
                        break
            else:
                st.write("chat_id 或 conversation_id 未在响应中找到")
        else:
            st.write("创建对话失败或返回格式不正确")
    else:
        st.write("无法获取 file_id")

# 设置Streamlit的页面标题
st.title('AI 数据分析助手')

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

# 上传文件
uploaded_file = st.file_uploader("选择一个文件", type=["doc", "docx", "xls", "xlsx", "ppt", "pptx", "pdf", "csv", "jpg", "png"])
user_question = st.text_input("问我任何关于文件的问题,我将在三分钟后回答")

if uploaded_file is not None and user_question:
    main(uploaded_file, user_question)
else:
    st.write("请上传一个文件并提出问题")