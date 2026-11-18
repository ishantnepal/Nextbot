
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# PAGE CONFIG
st.set_page_config(
    page_title="NeuroBot AI",
    page_icon="🤖",
    layout="wide"
)

# CUSTOM PURPLE + BLACK THEME
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #19082e, #08050f 65%);
    color: #ffffff;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #160625, #080510);
    border-right: 1px solid #6824a8;
}

[data-testid="stHeader"] {
    background: transparent;
}

h1, h2, h3 {
    color: #ffffff !important;
}

.neuro-logo {
    text-align: center;
    padding: 25px 5px;
    border-bottom: 1px solid #6322a0;
    margin-bottom: 25px;
}

.neuro-logo .robot {
    font-size: 65px;
    filter: drop-shadow(0 0 15px #a855f7);
}

.neuro-logo h1 {
    font-size: 38px;
    font-weight: 900;
    background: linear-gradient(90deg, #ffffff, #bd65ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.neuro-logo p {
    color: #c9a7ed;
    letter-spacing: 2px;
}

.hero {
    text-align: center;
    padding: 35px 10px 20px;
}

.hero .icon {
    font-size: 75px;
    filter: drop-shadow(0 0 20px #a020f0);
}

.hero h1 {
    font-size: clamp(38px, 6vw, 65px);
    font-weight: 900;
    background: linear-gradient(90deg, #ffffff, #b44aff, #7928ca);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #c6a5e8;
    font-size: 18px;
    letter-spacing: 2px;
}

div[data-testid="stChatMessage"] {
    background: rgba(76, 24, 112, 0.20);
    border: 1px solid rgba(168, 85, 247, 0.35);
    border-radius: 16px;
    padding: 14px;
    margin-bottom: 12px;
}

.stChatInput textarea {
    background: #160c24 !important;
    color: white !important;
}

.stChatInput {
    border-color: #963cff !important;
}

.stButton button {
    background: linear-gradient(90deg, #7928ca, #b44aff);
    color: white;
    border: 1px solid #c084fc;
    border-radius: 12px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton button:hover {
    border-color: white;
    box-shadow: 0 0 15px #a855f7;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# SIDEBAR BRANDING
with st.sidebar:
    st.markdown("""
    <div class="neuro-logo">
        <div class="robot">🤖</div>
        <h1>NeuroBot AI</h1>
        <p>CHAT • EXPLORE • INNOVATE</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ AI SETTINGS")

    api_key = st.text_input(
        "OpenRouter API Key",
        type="password",
        placeholder="Enter your API key"
    )

    model = st.text_input(
        "Model ID",
        value="openai/gpt-4o-mini"
    )

    system_prompt = st.text_area(
        "System Prompt",
        value="You are NeuroBot, a helpful, intelligent and friendly AI assistant."
    )

    st.markdown("---")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("**⚡ Powered by**")
    st.markdown("LangChain + OpenRouter")
    st.caption("💜 Smarter Conversations. A Brighter Tomorrow.")

# MAIN HEADER
st.markdown("""
<div class="hero">
    <div class="icon">🤖</div>
    <h1>NeuroBot AI</h1>
    <p>Your Intelligent AI Companion</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# INITIALIZE CHAT HISTORY
if "messages" not in st.session_state:
    st.session_state.messages = []

# INITIALIZE MODEL
llm = None

if api_key:
    try:
        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.7,
            max_tokens=1000
        )
    except Exception as e:
        st.error(f"Model initialization error: {e}")

# WELCOME MESSAGE
if not st.session_state.messages:
    st.markdown(
        "<h3 style='text-align:center;color:#c084fc;'>"
        "✨ Ask me anything... I'm here to help!"
        "</h3>",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("💡 **Learn anything**")

    with col2:
        st.markdown("🚀 **Explore ideas**")

    with col3:
        st.markdown("💻 **Get coding help**")

# DISPLAY CHAT HISTORY
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# CHAT INPUT
if prompt := st.chat_input("Type your message to NeuroBot..."):

    if not api_key:
        st.warning("Please enter your OpenRouter API key in the sidebar.")

    elif not llm:
        st.error("AI model is not initialized.")

    else:
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                history = [
                    SystemMessage(content=system_prompt)
                ]

                for msg in st.session_state.messages[:-1]:
                    if msg["role"] == "user":
                        history.append(
                            HumanMessage(content=msg["content"])
                        )
                    else:
                        history.append(
                            AIMessage(content=msg["content"])
                        )

                history.append(HumanMessage(content=prompt))

                with st.spinner("NeuroBot is thinking..."):
                    response = llm.invoke(history)

                answer = response.content

                if isinstance(answer, list):
                    answer = "\n".join(
                        str(part.get("text", part))
                        if isinstance(part, dict)
                        else str(part)
                        for part in answer
                    )

                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.session_state.messages.pop()

