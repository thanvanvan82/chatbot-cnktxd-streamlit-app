import streamlit as st
import openai

openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("🏗️ Chatbot Kỹ thuật xây dựng")

user_question = st.text_input("Hỏi về kỹ thuật xây dựng:")

if user_question:
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Bạn là một trợ lý học tập chuyên ngành Công nghệ kỹ thuật xây dựng. Chỉ trả lời các câu hỏi liên quan đến lĩnh vực này. Nếu câu hỏi không liên quan, hãy lịch sự từ chối."},
            {"role": "user", "content": user_question}
        ]
    )
    st.write(response['choices'][0]['message']['content'])
