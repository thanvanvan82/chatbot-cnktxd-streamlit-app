import streamlit as st
from openai import OpenAI
import google.generativeai as genai
import time

# --- CẤU HÌNH API CHO CÁC NHÀ CUNG CẤP ---

# 1. Google Gemini
gemini_configured = False
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    gemini_configured = True
except KeyError:
    pass # Bỏ qua nếu không có key

# 2. Groq
groq_client = None
try:
    groq_client = OpenAI(
        api_key=st.secrets["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1"
    )
except KeyError:
    pass # Bỏ qua nếu không có key

# 3. OpenRouter
# THAY ĐỔI URL VÀ TÊN APP CỦA BẠN CHO PHÙ HỢP
YOUR_SITE_URL = "https://github.com/thanvanvan82/cvht-streamlit-app" 
YOUR_APP_NAME = "Chatbot Xay Dung TLU"

openrouter_client = None
try:
    openrouter_client = OpenAI(
        api_key=st.secrets["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        default_headers={"HTTP-Referer": YOUR_SITE_URL, "X-Title": YOUR_APP_NAME},
    )
except KeyError:
    pass # Bỏ qua nếu không có key

# --- HEADER ---
st.title("🏗️ Chatbot công nghệ kỹ thuật xây dựng")
st.markdown("*Powered by Multiple AI APIs*")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Cấu hình Model")

    # Chọn nhà cung cấp API
    api_provider = st.selectbox(
        "Chọn nhà cung cấp API:",
        ["Google Gemini", "Groq", "OpenRouter"],  
        index= 0,
        help="Mỗi nhà cung cấp có các model và giới hạn sử dụng khác nhau."
    )

    # Dictionary chứa các model cho từng nhà cung cấp
    models = {
        "Google Gemini": {
            "gemini-1.5-flash-latest": "Gemini 1.5 Flash (Nhanh, Tối ưu)",
            "gemini-1.5-pro-latest": "Gemini 1.5 Pro (Mạnh, Đa năng)",
        },
        "Groq": {
            "llama-3.1-8b-instant": "Llama 3.1 8B (Nhanh)",
            "llama-3.1-70b-versatile": "Llama 3.1 70B (Mạnh)",
            "mixtral-8x7b-32768": "Mixtral 8x7B",
            "gemma2-9b-it": "Gemma2 9B"
        },
        "OpenRouter": {
            "meta-llama/llama-3-8b-instruct:free": "Meta Llama 3 8B (Miễn phí)",
            "google/gemma-7b-it:free": "Google Gemma 7B (Miễn phí)",
            "mistralai/mistral-7b-instruct:free": "Mistral 7B Instruct (Miễn phí)",
            "microsoft/phi-3-medium-128k-instruct": "Microsoft Phi-3 Medium"
        }        
    }

    # Hiển thị dropdown model tương ứng
    provider_models = models.get(api_provider, {})
    if provider_models:
        selected_model = st.selectbox(
            f"Chọn model ({api_provider}):",
            options=list(provider_models.keys()),
            format_func=lambda x: provider_models[x]
        )
    else:
        st.warning("Nhà cung cấp này chưa có model nào được định nghĩa.")
        st.stop()

    max_tokens = st.slider("Max tokens:", 100, 4096, 1500)
    temperature = st.slider("Temperature:", 0.0, 1.0, 0.7, 0.1)
    
    st.header("📚 Chủ đề xây dựng")
    st.markdown("""
    - **Thiết kế:** Kết cấu, tính toán    
    - **Thi công:** Đất, đá, bê tông (móng, cột, dầm, sàn).
    """)

# --- GIAO DIỆN CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Nhập câu hỏi về công nghệ kỹ thuật xây dựng..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        try:
            start_time = time.time()
            system_message = """
            Bạn là một chuyên gia tư vấn công nghệ kỹ thuật xây dựng hàng đầu Việt Nam.
            CHUYÊN MÔN: Vật liệu xây dựng, kỹ thuật thi công, thiết kế kết cấu, tiêu chuẩn TCVN/QCVN, an toàn lao động.
            QUY TẮC: CHỈ trả lời câu hỏi về xây dựng. Trả lời chi tiết, có ví dụ, tham khảo TCVN, nhấn mạnh an toàn. Luôn dùng tiếng Việt.
            """
            
            # --- LOGIC GỌI API PHÂN NHÁNH ---
            
            
            # 1. Logic cho Google Gemini
            if api_provider == "Google Gemini":
                if not gemini_configured:
                    st.error("❌ Chưa cấu hình GOOGLE_API_KEY trong secrets!")
                    st.stop()

                with st.spinner("Gemini đang suy nghĩ..."):
                    model = genai.GenerativeModel(
                        model_name=selected_model,
                        system_instruction=system_message,
                        generation_config=genai.GenerationConfig(max_output_tokens=max_tokens, temperature=temperature)
                    )
                    
                    gemini_history = [
                        {"role": "model" if msg["role"] == "assistant" else "user", "parts": [msg["content"]]}
                        for msg in st.session_state.messages
                    ]
                    
                    chat_session = model.start_chat(history=gemini_history[:-1])
                    response = chat_session.send_message(gemini_history[-1]['parts'][0], stream=True)
                    
                    response_placeholder = st.empty()
                    full_response = ""
                    for chunk in response:
                        if chunk.text:
                            full_response += chunk.text
                            response_placeholder.markdown(full_response + "▌")
                    response_placeholder.markdown(full_response)
            
            # 2. Logic cho Groq và OpenRouter (cùng cấu trúc OpenAI)
            elif api_provider == "Groq" or api_provider == "OpenRouter":
                client = groq_client if api_provider == "Groq" else openrouter_client
                if not client:
                    st.error(f"❌ Chưa cấu hình API Key cho {api_provider} trong secrets!")
                    st.stop()

                with st.spinner(f"{api_provider} đang xử lý..."):
                    messages = [{"role": "system", "content": system_message}]
                    messages.extend(st.session_state.messages)
                    
                    response = client.chat.completions.create(
                        model=selected_model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=True
                    )
                    
                    response_placeholder = st.empty()
                    full_response = ""
                    for chunk in response:
                        if chunk.choices[0].delta.content is not None:
                            full_response += chunk.choices[0].delta.content
                            response_placeholder.markdown(full_response + "▌")
                    response_placeholder.markdown(full_response)

            # --- Xử lý chung sau khi có response ---
            end_time = time.time()
            processing_time = round(end_time - start_time, 2)
            model_display_name = provider_models.get(selected_model, selected_model)
            st.caption(f"⚡ {processing_time}s | {model_display_name} ({api_provider})")
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"❌ Lỗi API từ {api_provider}: {str(e)}")
            st.info("🔧 Vui lòng kiểm tra API Key trong secrets, hoặc thử lại sau.")

# Nút xóa chat
if st.session_state.messages:
    if st.button("🗑️ Xóa cuộc trò chuyện"):
        st.session_state.messages = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    🏗️ <strong>Chatbot công nghệ kỹ thuật xây dựng</strong> | Powered by Multiple AI APIs<br>
    ⚠️ Kết quả mang tính tham khảo - Cần xác minh với chuyên gia thực tế
</div>
""", unsafe_allow_html=True)