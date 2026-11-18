import streamlit as st
import uuid
import io
import os
import base64
import html
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

import speech_recognition as sr

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Preformatted
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.utils import simpleSplit


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="NeuroBot AI",
    page_icon="🤖",
    layout="wide"
)


# ==========================================
# SESSION STATE
# ==========================================
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

if "chats" not in st.session_state:
    first_id = str(uuid.uuid4())
    st.session_state.chats = {
        first_id: {
            "title": "New Chat",
            "messages": []
        }
    }
    st.session_state.active_chat = first_id

if "active_chat" not in st.session_state:
    st.session_state.active_chat = list(
        st.session_state.chats.keys()
    )[0]

if "voice_transcript" not in st.session_state:
    st.session_state.voice_transcript = ""


active_id = st.session_state.active_chat
active_chat = st.session_state.chats[active_id]
messages = active_chat["messages"]



# ==========================================
# PURPLE BLACK NEUROBOT THEME
# ==========================================

st.markdown("""
<style>
.stApp {
    background: #090611;
    color: #F5F0FF;
}

[data-testid="stSidebar"] {
    background: #100B1D;
    border-right: 1px solid #30204B;
}

h1, h2, h3, p, label {
    color: #F5F0FF !important;
}

.neuro-title {
    font-size: 38px;
    font-weight: 900;
    text-align: center;
    color: #B78AFF !important;
    text-shadow: 0 0 18px #8B5CF6;
}

.neuro-subtitle {
    text-align: center;
    color: #A99BBE !important;
}

.stButton button {
    background: linear-gradient(
        135deg, #7C3AED, #4C1D95
    );
    color: white;
    border: 1px solid #A78BFA;
    border-radius: 12px;
    transition: 0.3s;
}

.stButton button:hover {
    background: #9333EA;
    border-color: #C4B5FD;
    box-shadow: 0 0 15px #8B5CF6;
}

[data-testid="stChatMessage"] {
    background: #171124;
    border: 1px solid #33204F;
    border-radius: 16px;
    padding: 15px;
}

[data-testid="stChatInput"] {
    border: 1px solid #8B5CF6;
    border-radius: 15px;
}

[data-testid="stFileUploader"] {
    background: #171124;
    border: 1px dashed #8B5CF6;
    border-radius: 14px;
}

hr {
    border-color: #30204B;
}

</style>
""", unsafe_allow_html=True)



# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("# 🤖 NeuroBot AI")
    st.caption("Your intelligent AI assistant")

    st.divider()

    st.subheader("Settings")

    st.session_state.dark_mode = st.toggle(
        "Dark Mode",
        value=st.session_state.dark_mode
    )

    api_key = st.text_input(
        "OpenRouter API Key",
        type="password",
        value=os.getenv("OPENROUTER_API_KEY", ""),
        placeholder="Enter your API key"
    )

    model_name = st.text_input(
        "AI Model",
        value="openai/gpt-4o-mini"
    )

    code_mode = st.toggle(
        "Code Assistant Mode",
        value=False
    )

    st.divider()

    st.subheader("Chat History")

    if st.button("＋ New Chat", use_container_width=True):

        new_id = str(uuid.uuid4())

        st.session_state.chats[new_id] = {
            "title": "New Chat",
            "messages": []
        }

        st.session_state.active_chat = new_id
        st.rerun()

    for chat_id, chat in reversed(
        list(st.session_state.chats.items())
    ):

        if st.button(
            "💬 " + chat["title"][:28],
            key=chat_id,
            use_container_width=True
        ):
            st.session_state.active_chat = chat_id
            st.rerun()

    st.divider()

    if st.button("🗑️ Clear Current Chat", use_container_width=True):

        active_chat["messages"] = []
        active_chat["title"] = "New Chat"
        st.rerun()

    st.caption("Powered by LangChain + OpenRouter")


# ==========================================
# REFRESH THEME AFTER TOGGLE
# ==========================================

# The next Streamlit rerun applies the selected theme.


# ==========================================
# SYSTEM PROMPT
# ==========================================

if code_mode:

    system_prompt = """
    You are NeuroBot AI, an expert programming assistant.

    Your responsibilities:
    - Write clean and readable code.
    - Explain code step by step.
    - Find and fix programming errors.
    - Support Python, JavaScript, HTML, CSS and other languages.
    - Use Markdown code blocks.
    - Give practical examples.
    - Never pretend that code was executed if it was not.

    Be friendly, accurate and helpful.
    """

else:

    system_prompt = """
    You are NeuroBot AI, a smart and friendly AI assistant.

    Your responsibilities:
    - Answer questions clearly.
    - Explain difficult topics simply.
    - Help with learning, writing and creativity.
    - Assist with programming and general knowledge.
    - Use Markdown formatting when useful.

    Be conversational, helpful and accurate.
    """


# ==========================================
# IMAGE HANDLING
# ==========================================

st.markdown(
    '<div class="neuro-title">🤖 NeuroBot AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="neuro-subtitle">'
    'Your personal intelligent assistant'
    '</div>',
    unsafe_allow_html=True
)

uploaded_image = st.file_uploader(
    "📎 Upload an image (optional)",
    type=["png", "jpg", "jpeg", "webp"],
    help="Choose an image to ask NeuroBot about it."
)

image_data_url = None

if uploaded_image:

    image_bytes = uploaded_image.getvalue()

    if len(image_bytes) > 10 * 1024 * 1024:
        st.error("Image must be smaller than 10 MB.")
        uploaded_image = None

    else:
        encoded_image = base64.b64encode(
            image_bytes
        ).decode("utf-8")

        mime_type = uploaded_image.type

        image_data_url = (
            f"data:{mime_type};base64,{encoded_image}"
        )

        st.image(
            image_bytes,
            caption="Image ready to send",
            width=250
        )


# ==========================================
# VOICE INPUT
# ==========================================

with st.expander("🎤 Voice Input"):

    st.caption(
        "Record your question and convert it into text."
    )

    voice_language = st.selectbox(
        "Speech language",
        ["en-IN", "en-US", "hi-IN", "bn-IN"]
    )

    audio_file = st.audio_input(
        "Record your voice"
    )

    if audio_file:

        if st.button("Transcribe Recording"):

            try:

                recognizer = sr.Recognizer()

                audio = sr.AudioFile(
                    io.BytesIO(audio_file.getvalue())
                )

                with audio as source:
                    audio_data = recognizer.record(source)

                transcript = recognizer.recognize_google(
                    audio_data,
                    language=voice_language
                )

                st.session_state.voice_transcript = transcript

            except sr.UnknownValueError:
                st.error("Could not understand the recording.")

            except sr.RequestError:
                st.error(
                    "Speech recognition service unavailable. "
                    "Check your internet connection."
                )

            except Exception as e:
                st.error(f"Voice error: {e}")

    if st.session_state.voice_transcript:

        st.session_state.voice_transcript = st.text_area(
            "Recognized text",
            value=st.session_state.voice_transcript,
            key="voice_edit"
        )


# ==========================================
# DISPLAY CHAT HISTORY
# ==========================================

for message in messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("image"):

            image_bytes = base64.b64decode(
                message["image"].split(",")[1]
            )

            st.image(image_bytes, width=250)


# ==========================================
# EXPORT FUNCTIONS
# ==========================================

def create_txt(chat_messages):

    output = []

    for msg in chat_messages:

        role = msg["role"].upper()

        output.append(
            f"{role}:\n{msg['content']}\n"
        )

        if msg.get("image"):
            output.append("[Image attached]\n")

    return "\n".join(output)


def create_pdf(chat_messages):

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=letter
    )

    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph("NeuroBot AI Conversation", styles["Title"])
    )

    story.append(Spacer(1, 20))

    for msg in chat_messages:

        role = msg["role"].capitalize()

        story.append(
            Paragraph(
                f"<b>{role}</b>",
                styles["Heading3"]
            )
        )

        safe_text = html.escape(msg["content"])

        story.append(
            Paragraph(
                safe_text.replace("\n", "<br/>"),
                styles["BodyText"]
            )
        )

        if msg.get("image"):
            story.append(
                Paragraph("[Image attached]", styles["Italic"])
            )

        story.append(Spacer(1, 12))

    document.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# ==========================================
# CHAT EXPORT
# ==========================================

if messages:

    with st.expander("📥 Export Conversation"):

        txt_data = create_txt(messages)

        st.download_button(
            "Download TXT",
            data=txt_data,
            file_name="neurobot_chat.txt",
            mime="text/plain",
            use_container_width=True
        )

        pdf_data = create_pdf(messages)

        st.download_button(
            "Download PDF",
            data=pdf_data,
            file_name="neurobot_chat.pdf",
            mime="application/pdf",
            use_container_width=True
        )


# ==========================================
# AI RESPONSE FUNCTION
# ==========================================

def generate_response(prompt, image_url=None):

    if not api_key:

        st.error(
            "Please enter your OpenRouter API key "
            "in the sidebar."
        )

        return

    if not model_name.strip():

        st.error("Please enter a model ID.")
        return

    # Save user message

    user_message = {
        "role": "user",
        "content": prompt
    }

    if image_url:
        user_message["image"] = image_url

    messages.append(user_message)

    if active_chat["title"] == "New Chat":
        active_chat["title"] = prompt[:35] or "Image Chat"

    with st.chat_message("user"):

        st.markdown(prompt)

        if image_url:

            image_bytes = base64.b64decode(
                image_url.split(",")[1]
            )

            st.image(image_bytes, width=250)

    # Build LangChain messages

    lc_messages = [
        SystemMessage(content=system_prompt)
    ]

    for msg in messages:

        if msg["role"] == "user":

            if msg.get("image"):

                content = [
                    {
                        "type": "text",
                        "text": msg["content"]
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": msg["image"]
                        }
                    }
                ]

                lc_messages.append(
                    HumanMessage(content=content)
                )

            else:

                lc_messages.append(
                    HumanMessage(content=msg["content"])
                )

        elif msg["role"] == "assistant":

            lc_messages.append(
                AIMessage(content=msg["content"])
            )

    # Connect to OpenRouter

    try:

        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.7,
            max_tokens=2000,
            streaming=True,
            default_headers={
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "NeuroBot AI"
            }
        )

        with st.chat_message("assistant"):

            response_placeholder = st.empty()

            full_response = ""

            # Stream responses for typing animation

            for chunk in llm.stream(lc_messages):

                content = chunk.content

                if isinstance(content, str):

                    full_response += content

                elif isinstance(content, list):

                    for part in content:

                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                full_response += part.get("text", "")

                response_placeholder.markdown(
                    full_response + " ▌"
                )

            response_placeholder.markdown(full_response)

        messages.append({
            "role": "assistant",
            "content": full_response
        })

        st.rerun()

    except Exception as e:

        messages.pop()

        st.error(
            f"Unable to generate response: {str(e)}"
        )


# ==========================================
# CHAT INPUT
# ==========================================

prompt = st.chat_input(
    "Message NeuroBot AI..."
)

send_voice = st.button(
    "Send Voice Transcript",
    disabled=not bool(st.session_state.voice_transcript),
    use_container_width=False
)

if prompt:

    generate_response(
        prompt,
        image_data_url
    )

elif send_voice:

    voice_prompt = st.session_state.voice_transcript

    st.session_state.voice_transcript = ""

    generate_response(
        voice_prompt,
        image_data_url
                )
    
