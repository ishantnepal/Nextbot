
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Page configuration
st.set_page_config(
    page_title="NeuroBot AI",
    page_icon="🤖",
    layout="centered"
)

# Custom styling
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: white;
}
.stChatMessage {
    border-radius: 12px;
}
h1 {
    color: #7c9cff;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 NeuroBot AI")
st.caption("Powered by LangChain + OpenRouter")

# API configuration
api_key = st.sidebar.text_input(
    "OpenRouter API Key",
    type="password"
)

model = st.sidebar.text_input(
    "Model ID",
    value="openai/gpt-4o-mini"
)

system_prompt = st.sidebar.text_area(
    "System Prompt",
    value="You are NeuroBot, a helpful and intelligent AI assistant."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize model
llm = None

if api_key:
    try:
        llm = ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.7,
            max_tokens=1000
        )
    except Exception as e:
        st.error(f"Model initialization error: {e}")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Message NeuroBot..."):

    if not api_key:
        st.warning("Please enter your OpenRouter API key in the sidebar.")

    elif not llm:
        st.error("Model is not initialized.")

    else:
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()

            try:
                history = [
                    SystemMessage(content=system_prompt)
                ]

                for msg in st.session_state.messages[:-1]:
                    if msg["role"] == "user":
                        history.append(HumanMessage(content=msg["content"]))
                    else:
                        history.append(AIMessage(content=msg["content"]))

                history.append(HumanMessage(content=prompt))

                response = llm.invoke(history)
                answer = response.content

                if isinstance(answer, list):
                    answer = "\n".join(
                        str(part.get("text", part))
                        if isinstance(part, dict) else str(part)
                        for part in answer
                    )

                response_placeholder.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                st.error(f"Error: {e}")

# Clear chat
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()
