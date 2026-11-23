import streamlit as st
import os
import uuid
import base64
import io
from datetime import datetime

from pypdf import PdfReader
import speech_recognition as sr

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="NeuroBot AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CHATGPT-INSPIRED STYLING
# --------------------------------------------------

st.markdown("""
<style>
    .block-container {
        max-width: 1050px;
        padding-top: 2rem;
        padding-bottom: 7rem;
    }

    [data-testid="stSidebar"] {
        min-width: 260px;
    }

    .main-title {
        font-size: 30px;
        font-weight: 700;
        letter-spacing: -1px;
    }

    .subtitle {
        opacity: 0.65;
        font-size: 14px;
    }

    .stChatMessage {
        border-radius: 12px;
    }

    button {
        border-radius: 10px !important;
    }

    @media (prefers-color-scheme: dark) {
        .main-title {
            color: #f5f5f5;
        }
    }

    @media (prefers-color-scheme: light) {
        .main-title {
            color: #222222;
        }
    }

    @media screen and (max-width: 600px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .main-title {
            font-size: 24px;
        }
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

APP_NAME = "NeuroBot AI"

SYSTEM_PROMPT = """
You are NeuroBot AI, a helpful, intelligent and friendly AI assistant.

Your responsibilities:
- Answer questions clearly and accurately.
- Explain complicated topics in simple language.
- Help with programming and debugging.
- Assist with writing and creativity.
- Analyze documents and images when provided.
- Use Markdown for readable answers.
- Format programming code in proper code blocks.
- Admit when you are uncertain.
- Never invent sources or facts.

Be conversational, useful and concise when appropriate.
"""

DEFAULT_MODEL = "openai/gpt-4o-mini"

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "chats" not in st.session_state:
    chat_id = str(uuid.uuid4())

    st.session_state.chats = {
        chat_id: {
            "title": "New chat",
            "messages": [],
            "created": datetime.now().strftime("%d %b %Y")
        }
    }

    st.session_state.current_chat = chat_id

if "current_chat" not in st.session_state:
    st.session_state.current_chat = next(
        iter(st.session_state.chats)
    )

if "voice_draft" not in st.session_state:
    st.session_state.voice_draft = ""

# --------------------------------------------------
# CURRENT CHAT
# --------------------------------------------------

def get_current_chat():
    return st.session_state.chats[
        st.session_state.current_chat
    ]


# --------------------------------------------------
# API KEY
# --------------------------------------------------

def get_api_key():
    key = os.getenv("OPENROUTER_API_KEY", "")

    try:
        if not key:
            key = st.secrets.get("OPENROUTER_API_KEY", "")
    except Exception:
        pass

    return key


# --------------------------------------------------
# LANGUAGE MODEL
# --------------------------------------------------

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
            "X-Title": "NeuroBot AI"
        }
    )


# --------------------------------------------------
# CONVERT CHAT HISTORY
# --------------------------------------------------

def prepare_messages(history, user_content=None):
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    for item in history:
        if item["role"] == "user":
            messages.append(
                HumanMessage(content=item["content"])
            )

        elif item["role"] == "assistant":
            messages.append(
                AIMessage(content=item["content"])
            )

    if user_content is not None:
        messages.append(
            HumanMessage(content=user_content)
        )

    return messages


# --------------------------------------------------
# DOCUMENT PROCESSING
# --------------------------------------------------

def extract_document(uploaded_file):
    try:
        if uploaded_file.name.lower().endswith(".txt"):
            return uploaded_file.getvalue().decode(
                "utf-8", errors="replace"
            )

        if uploaded_file.name.lower().endswith(".pdf"):
            reader = PdfReader(uploaded_file)

            text = []

            for page in reader.pages:
                text.append(page.extract_text() or "")

            return "\n".join(text)

    except Exception as error:
        st.error(f"Document error: {error}")

    return ""


# --------------------------------------------------
# IMAGE PROCESSING
# --------------------------------------------------

def prepare_image(uploaded_image):
    image_bytes = uploaded_image.getvalue()

    encoded = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    mime_type = uploaded_image.type or "image/jpeg"

    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{encoded}"
        }
    }


# --------------------------------------------------
# AI RESPONSE
# --------------------------------------------------

def send_message(
    user_text,
    api_key,
    model_name,
    image_file=None,
    document_files=None
):
    chat = get_current_chat()

    document_files = document_files or []

    if not user_text.strip() and not image_file and not document_files:
        return

    if not api_key:
        st.error(
            "Please enter your OpenRouter API key in Settings."
        )
        return

    # Process uploaded documents
    document_text = ""

    for file in document_files:
        extracted = extract_document(file)

        if extracted:
            document_text += (
                f"\n\nDOCUMENT: {file.name}\n"
                f"{extracted[:50000]}"
            )

    final_text = user_text.strip()

    if document_text:
        final_text += (
            "\n\nPlease analyze these uploaded documents:\n"
            + document_text
        )

    if not final_text and image_file:
        final_text = "Please analyze this image."

    # Prepare image content
    if image_file:
        content = [
            {"type": "text", "text": final_text},
            prepare_image(image_file)
        ]
    else:
        content = final_text

    # Save user message
    chat["messages"].append({
        "role": "user",
        "content": content,
        "display": user_text or "Uploaded attachment"
    })

    # Generate chat title
    if chat["title"] == "New chat":
        chat["title"] = (
            user_text[:35] + "..."
            if len(user_text) > 35
            else user_text or "New conversation"
        )

    # Display user message
    with st.chat_message("user"):
        st.markdown(
            user_text or "Please analyze my attachment."
        )

        if image_file:
            st.image(image_file, width=300)

        for file in document_files:
            st.caption(f"📎 {file.name}")

    # Generate AI response
    with st.chat_message("assistant"):
        try:
            llm = create_model(api_key, model_name)

            history = chat["messages"][:-1]

            messages = prepare_messages(
                history,
                user_content=content
            )

            response = st.write_stream(
                llm.stream(messages)
            )

            chat["messages"].append({
                "role": "assistant",
                "content": response
            })

        except Exception as error:
            chat["messages"].pop()

            st.error(
                f"AI response failed: {str(error)}"
            )


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("## 🤖 NeuroBot AI")

    st.caption("Your intelligent AI assistant")

    if st.button(
        "＋  New chat",
        use_container_width=True
    ):
        new_id = str(uuid.uuid4())

        st.session_state.chats[new_id] = {
            "title": "New chat",
            "messages": [],
            "created": datetime.now().strftime("%d %b %Y")
        }

        st.session_state.current_chat = new_id
        st.rerun()

    st.divider()

    st.markdown("### Your chats")

    for chat_id, chat_data in reversed(
        list(st.session_state.chats.items())
    ):

        if st.button(
            "💬  " + chat_data["title"],
            key=f"chat_{chat_id}",
            use_container_width=True
        ):
            st.session_state.current_chat = chat_id
            st.rerun()

    st.divider()

    # Settings
    with st.expander("⚙️ Settings"):

        api_key = st.text_input(
            "OpenRouter API Key",
            value=get_api_key(),
            type="password",
            help="Your API key is not saved to the source code."
        )

        model_name = st.text_input(
            "AI Model",
            value=DEFAULT_MODEL,
            help="Enter an OpenRouter model ID."
        )

        st.caption(
            "Choose a model that supports the features "
            "you want to use."
        )

    # Attachments
    with st.expander("＋ Attachments"):

        image_file = st.file_uploader(
            "Upload image",
            type=["png", "jpg", "jpeg", "webp"],
            key="image_upload"
        )

        document_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key="document_upload"
        )

    # Voice input
    with st.expander("🎙️ Voice input"):

        audio_file = st.audio_input(
            "Record your message"
        )

        if audio_file:
            if st.button(
                "Transcribe audio",
                use_container_width=True
            ):
                try:
                    recognizer = sr.Recognizer()

                    with sr.AudioFile(
                        io.BytesIO(audio_file.getvalue())
                    ) as source:

                        audio = recognizer.record(source)

                    transcript = recognizer.recognize_google(
                        audio
                    )

                    st.session_state.voice_draft = transcript

                    st.success("Transcription completed!")

                except Exception as error:
                    st.error(
                        f"Could not transcribe audio: {error}"
                    )

        st.text_area(
            "Voice transcript",
            key="voice_draft",
            height=90
        )

        if st.button(
            "Send voice message",
            use_container_width=True
        ):
            if st.session_state.voice_draft.strip():
                send_message(
                    st.session_state.voice_draft,
                    api_key,
                    model_name
                )

                st.session_state.voice_draft = ""
                st.rerun()

    st.divider()

    # Export conversation
    chat = get_current_chat()

    if chat["messages"]:
        export_text = ""

        for message in chat["messages"]:
            export_text += (
                f"\n{message['role'].upper()}\n"
                f"{message.get('display', message['content'])}\n"
            )

        st.download_button(
            "⬇️ Export conversation",
            data=export_text,
            file_name="neurobot_conversation.txt",
            mime="text/plain",
            use_container_width=True
        )

    if st.button(
        "🗑️ Clear current chat",
        use_container_width=True
    ):
        chat["messages"] = []
        chat["title"] = "New chat"
        st.rerun()


# --------------------------------------------------
# MAIN CHAT INTERFACE
# --------------------------------------------------

chat = get_current_chat()

st.markdown(
    '<div class="main-title">NeuroBot AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">How can I help you today?</div>',
    unsafe_allow_html=True
)

# Display welcome screen
if not chat["messages"]:

    st.write("")
    st.write("")

    st.markdown(
        "<h3 style='text-align:center;'>"
        "What can I help you with?"
        "</h3>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='text-align:center; opacity:0.6;'>"
        "Ask questions, explore ideas, write code or "
        "upload a document."
        "</p>",
        unsafe_allow_html=True
    )

# Display previous messages
for message in chat["messages"]:

    with st.chat_message(message["role"]):

        if message["role"] == "user":
            st.markdown(
                message.get(
                    "display",
                    "Uploaded attachment"
                )
            )

            content = message["content"]

            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "image_url":
                        st.caption("🖼️ Image attached")

        else:
            st.markdown(message["content"])


# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

prompt = st.chat_input(
    "Message NeuroBot AI...",
    key="main_chat_input"
)

if prompt:

    send_message(
        prompt,
        api_key,
        model_name,
        image_file,
        document_files
    )

    st.rerun()
