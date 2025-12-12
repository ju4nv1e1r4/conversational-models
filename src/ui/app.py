import streamlit as st
import requests

st.set_page_config(page_title="Compound AI PoC", page_icon="🤖")
st.title("🤖 Compound AI - Debug Console")
API_URL = "http://app:8002/chat" 

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Diga algo para o Gemini 2.0..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Pensando...")
        
        try:
            payload = {"message": prompt, "user_id": "demo_user"}
            response = requests.post("http://app:8002/chat", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                ai_response = data["response"]
                message_placeholder.markdown(ai_response)
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
            else:
                message_placeholder.error(f"Erro na API: {response.text}")
        except Exception as e:
             message_placeholder.error(f"Erro de conexão: {e}")