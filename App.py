import streamlit as st
import os
import uuid
import base64
import io
from datetime import datetime

from pypdf import PdfReader
import speech_recognition as sr

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="NeuroBot AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_NAME = "NeuroBot AI"
DEFAULT_MODEL = "openai/gpt-4o-mini"

SYSTEM_PROMPT = """
You are NeuroBot AI, a helpful, intelligent and friendly AI assistant.
Answer clearly and accurately. Explain complicated topics simply.
Help with programming, writing, creativity, and document/image analysis.
Use Markdown for readable answers and proper code blocks for code.
Admit uncertainty and never invent sources or facts.
Be conversational, useful, and concise when appropriate.
"""

# --------------------------------------------------
# CHATGPT-INSPIRED RESPONSIVE STYLING
# --------------------------------------------------
st.markdown("""
<style>
:root { color-scheme: light dark; }

[data-testid="stAppViewContainer"] { min-height: 100vh; }
[data-testid="stHeader"] {
    background: transparent;
}
[data-testid="stToolbar"] { right: 1rem; }

.block-container {
    max-width: 900px;
    padding-top: 1.4rem;
    padding-bottom: 7rem;
}
[data-testid="stSidebar"] {
    min-width: 265px;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.1rem;
}
.main-title {
    font-size: 1.25rem;
    font-weight: 650;
    letter-spacing: -0.035em;
    margin: 0 0 .25rem 0;
}
.top-subtitle {
    font-size: .88rem;
    opacity: .62;
    margin-bottom: 1.5rem;
}
.welcome-wrap {
    min-height: 42vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    padding: 1.5rem .5rem;
}
.welcome-title {
    font-size: clamp(1.65rem, 4vw, 2.25rem);
    font-weight: 650;
    letter-spacing: -.045em;
    margin-bottom: .5rem;
}
.welcome-caption {
    opacity: .62;
    font-size: .95rem;
}
.stChatMessage {
    border: 0;
    border-radius: 18px;
    padding: .8rem .95rem;
}
[data-testid="stChatInput"] {
    border-radius: 25px !important;
}
[data-testid="stChatInput"] textarea {
    font-size: 1rem !important;
}
button {
    border-radius: 12px !important;
}
[data-testid="stSidebar"] button {
    text-align: left;
}
div[data-testid="stStatusWidget"] { border-radius: 14px; }

@media (max-width: 700px) {
    .block-container {
        padding: .8rem .85rem 6rem .85rem;
    }
    [data-testid="stSidebar"] { min-width: 0; }
    .main-title { font-size: 1.12rem; }
    .welcome-wrap { min-height: 38vh; }
    .stChatMessage {
        padding: .65rem .7rem;
    }
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "chats" not in st.session_state:
    first_id = str(uuid.uuid4())
    st.session_state.chats = {
        first_id: {
            "title": "New chat",
            "messages": [],
            "created": datetime.now().strftime("%d %b %Y"),
        }
    }
    st.session_state.current_chat = first_id

if "current_chat" not in st.session_state:
    st.session_state.current_chat = next(iter(st.session_state.chats))

if "model_name" not in st.session_state:
    st.session_state.model_name = DEFAULT_MODEL

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def get_current_chat():
    return st.session_state.chats[st.session_state.current_chat]


def get_api_key():
    key = os.getenv("OPENROUTER_API_KEY", "")
    try:
        if not key:
            key = st.secrets.get("OPENROUTER_API_KEY", "")
    except Exception:
        pass
    return key


def create_model(api_key, model_name):
    return ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.7,
        streaming=True,
        max_tokens=4096,
        default_headers={
            "HTTP-Referer": "https://neurobot-ai.streamlit.app",
            "X-Title": APP_NAME,
        },
    )


def extract_document(uploaded_file):
    try:
        filename = uploaded_file.name.lower()
        if filename.endswith(".txt"):
            return uploaded_file.getvalue().decode("utf-8", errors="replace")
        if filename.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as error:
        st.error(f"Document error ({uploaded_file.name}): {error}")
    return ""


def prepare_image(uploaded_image):
    encoded = base64.b64encode(uploaded_image.getvalue()).decode("utf-8")
    mime_type = uploaded_image.type or "image/jpeg"
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
    }


def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(io.BytesIO(audio_file.getvalue())) as source:
        audio_data = recognizer.record(source)
    return recognizer.recognize_google(audio_data)


def prepare_messages(history, user_content=None):
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for item in history:
        if item["role"] == "user":
            messages.append(HumanMessage(content=item["content"]))
        elif item["role"] == "assistant":
            messages.append(AIMessage(content=item["content"]))
    if user_content is not None:
        messages.append(HumanMessage(content=user_content))
    return messages


# --------------------------------------------------
# SEND MESSAGE
# --------------------------------------------------
def send_message(user_text, api_key, model_name, uploaded_files=None):
    chat = get_current_chat()
    uploaded_files = uploaded_files or []

    image_files = [
        file for file in uploaded_files
        if (file.type or "").startswith("image/")
    ]
    document_files = [
        file for file in uploaded_files
        if file.name.lower().endswith((".pdf", ".txt"))
    ]

    document_text = ""
    for file in document_files:
        extracted = extract_document(file)
        if extracted:
            document_text += (
                f"\n\nDOCUMENT: {file.name}\n{extracted[:50000]}"
            )

    final_text = (user_text or "").strip()
    if document_text:
        final_text += (
            "\n\nPlease analyze these uploaded documents:\n" + document_text
        )
    if not final_text and image_files:
        final_text = "Please analyze the attached image(s)."
    if not final_text:
        final_text = "Please analyze the attached file(s)."

    if not api_key:
        st.error("Please add your OpenRouter API key in Settings.")
        return

    if image_files:
        content = [{"type": "text", "text": final_text}]
        content.extend(prepare_image(image) for image in image_files[:4])
    else:
        content = final_text

    display_text = (user_text or "").strip() or "📎 Uploaded attachment(s)"
    chat["messages"].append({
        "role": "user",
        "content": content,
        "display": display_text,
        "attachments": [file.name for file in uploaded_files],
    })

    if chat["title"] == "New chat":
        title_source = (user_text or "").strip() or "New conversation"
        chat["title"] = (
            title_source[:35] + "..." if len(title_source) > 35
            else title_source
        )

    with st.chat_message("user"):
        st.markdown(display_text)
        for image in image_files:
            st.image(image, width=240)
        for file in document_files:
            st.caption(f"📄 {file.name}")

    with st.chat_message("assistant"):
        try:
            llm = create_model(api_key, model_name)
            history = chat["messages"][:-1]
            messages = prepare_messages(history, user_content=content)
            response = st.write_stream(llm.stream(messages))
            chat["messages"].append({
                "role": "assistant",
                "content": response,
            })
        except Exception as error:
            chat["messages"].pop()
            st.error(f"AI response failed: {error}")


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 🤖 NeuroBot AI")
    st.caption("Your AI assistant")

    if st.button("＋  New chat", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {
            "title": "New chat",
            "messages": [],
            "created": datetime.now().strftime("%d %b %Y"),
        }
        st.session_state.current_chat = new_id
        st.rerun()

    st.divider()
    st.markdown("**Your chats**")

    for chat_id, chat_data in reversed(list(st.session_state.chats.items())):
        label = chat_data["title"] or "New chat"
        if st.button(
            "💬  " + label,
            key=f"chat_{chat_id}",
            use_container_width=True,
        ):
            st.session_state.current_chat = chat_id
            st.rerun()

    st.divider()
    with st.expander("⚙️ Settings"):
        saved_key = get_api_key()
        api_key = st.text_input(
            "OpenRouter API key",
            value=saved_key,
            type="password",
            help="Use Streamlit Secrets for a persistent, private key.",
        )
        model_name = st.text_input(
            "AI model",
            value=st.session_state.model_name,
            help="Enter an OpenRouter model ID.",
        )
        st.session_state.model_name = model_name.strip() or DEFAULT_MODEL
        st.caption("Appearance follows your Streamlit/browser theme setting.")

    chat_for_export = get_current_chat()
    if chat_for_export["messages"]:
        export_text = ""
        for message in chat_for_export["messages"]:
            export_text += (
                f"\n{message['role'].upper()}\n"
                f"{message.get('display', message['content'])}\n"
            )
        st.download_button(
            "⬇️ Export conversation",
            data=export_text,
            file_name="neurobot_conversation.txt",
            mime="text/plain",
            use_container_width=True,
        )

    if st.button("🗑️ Clear current chat", use_container_width=True):
        chat_for_export["messages"] = []
        chat_for_export["title"] = "New chat"
        st.rerun()

# --------------------------------------------------
# MAIN CHAT
# --------------------------------------------------
chat = get_current_chat()

top_left, top_right = st.columns([5, 1])
with top_left:
    st.markdown('<div class="main-title">NeuroBot AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="top-subtitle">A helpful space for ideas, questions and creativity.</div>',
        unsafe_allow_html=True,
    )
with top_right:
    st.caption(" ")

if not chat["messages"]:
    st.markdown("""
    <div class="welcome-wrap">
        <div class="welcome-title">What can I help with?</div>
        <div class="welcome-caption">
            Ask anything, explore an idea, write code, or attach a file.
        </div>
    </div>
    """, unsafe_allow_html=True)

# Render existing conversation
for message in chat["messages"]:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message.get("display", "Uploaded attachment"))
            for filename in message.get("attachments", []):
                st.caption(f"📎 {filename}")
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "image_url":
                        st.caption("🖼️ Image attached")
        else:
            st.markdown(message["content"])

# Unified composer: text + attachments + microphone
prompt = st.chat_input(
    "Message NeuroBot AI...",
    key="main_chat_input",
    accept_file="multiple",
    file_type=["png", "jpg", "jpeg", "webp", "pdf", "txt"],
    accept_audio=True,
    audio_sample_rate=16000,
)

if prompt:
    submitted_text = prompt.text or ""
    attached_files = list(prompt.files) if hasattr(prompt, "files") else []

    if getattr(prompt, "audio", None):
        try:
            with st.spinner("Transcribing your voice message..."):
                transcript = transcribe_audio(prompt.audio)
            submitted_text = (submitted_text + "\n" + transcript).strip()
        except Exception as error:
            st.error(f"Could not transcribe audio: {error}")

    if submitted_text.strip() or attached_files:
        send_message(
            submitted_text,
            api_key,
            st.session_state.model_name,
            attached_files,
        )
        st.rerun()
