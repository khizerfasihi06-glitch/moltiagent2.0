import os
import io
import re
import streamlit as st
import streamlit.components.v1 as components
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI

# ---------------------------------------------------------------------------
# UI basics
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Multi-personal AI Chatbot", layout="centered")

if os.path.exists("Image.png"):
    st.image("Image.png", width=150)

# Mistral API key
# SECURITY NOTE: don't hardcode secrets in source. Set MISTRAL_API_KEY as a real
# environment variable, or put it in .streamlit/secrets.toml as:
#   MISTRAL_API_KEY = "your-key-here"
# and it'll be picked up automatically below.
if "MISTRAL_API_KEY" not in os.environ:
    try:
        os.environ["MISTRAL_API_KEY"] = st.secrets["MISTRAL_API_KEY"]
    except Exception:
        st.error(
            "MISTRAL_API_KEY is not set. Add it to your environment or "
            ".streamlit/secrets.toml before running."
        )
        st.stop()


# Model factory (cached)
@st.cache_resource
def get_model():
    # adjust model name / temperature if needed
    return ChatMistralAI(model="mistral-small-2506", temperature=0.9)


model = get_model()

st.sidebar.title("Chat Setting")

persona_options = [
    "Amazon",
    "Arabic",
    "Web Design (HTML/CSS)",
    "Astronomy & Space Exploration",
    "BS in Cloud Computing, AI, Robotics, Cyber Security, and Data Science",
    "Calculus",
    "Chemical Engineering",
    "Civil Engineering",
    "Coding",
    "Cooking",
    "Creative Writing",
    "Criminology",
    "Data Analytics & BI",
    "Digital Marketing & SEO",
    "Electric Engineering",
    "Electronic Engineering",
    "English",
    "Entrepreneurship & Startups",
    "Ethics",
    "Facebook and Instagram post",
    "Financial Planning",
    "Fitness & Exercise Science",
    "French",
    "Game Design & Development",
    "German",
    "Hardware",
    "Hindi",
    "Indonesia",
    "Linear Algebra",
    "Maths",
    "Medical Patients",
    "Mental Health & Mindfulness",
    "Motivation",
    "Music Theory & Production",
    "News of war and other good things",
    "Nutrition & Dietetics",
    "Persian",
    "Petroleum",
    "Photography & Videography",
    "Physics",
    "Portuguese",
    "Project Management",
    "Public Speaking & Presenting",
    "Research",
    "Roman Urdu (Urdu written in English letters)",
    "Russian",
    "Software engineering",
    "Spanish",
    "Statistics",
    "Turkish",
    "UI/UX Product Design",
    "Urdu",
    "World History & Archaeology",
]

persona = st.sidebar.selectbox(
    label="Choose Which AI do you want:",
    options=persona_options,
)

st.sidebar.markdown("---")
st.sidebar.title("Paper / Document")

uploaded_file = st.sidebar.file_uploader(
    "Upload a text document or paper context:",
    type=["txt", "pdf", "py", "csv", "json"],
)

MAX_DOC_CHARS = 20_000  # keep the system prompt from ballooning / blowing the context window


def extract_pdf_text(raw_bytes: bytes) -> str:
    """Extract text from a PDF's raw bytes using pdfplumber."""
    import pdfplumber

    with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages)


document_context = ""
if uploaded_file is not None:
    try:
        raw = uploaded_file.getvalue()

        if uploaded_file.name.lower().endswith(".pdf"):
            document_context = extract_pdf_text(raw)
        else:
            # Text-like files: txt, py, csv, json
            try:
                document_context = raw.decode("utf-8")
            except UnicodeDecodeError:
                document_context = raw.decode("utf-8", errors="replace")

        if len(document_context) > MAX_DOC_CHARS:
            document_context = (
                document_context[:MAX_DOC_CHARS]
                + f"\n\n[... truncated, {len(document_context) - MAX_DOC_CHARS} more characters omitted ...]"
            )

        st.sidebar.success(f"Loaded: {uploaded_file.name} ({len(document_context)} characters)")
    except Exception as e:
        st.sidebar.error("Error reading file: " + str(e))

# Predefined persona configs (extend this dict as needed)
PERSONA_CONFIGS = {
    "Motivation": {
        "title": "Mindset Master AI",
        "subtitle": "Your 24/7 personal pocket cheerleader.",
        "input_placeholder": "What goal are you tackling today?",
        "system_prompt": "You are an enthusiastic, high-energy Motivational AI agent. Boost user confidence, combat self-doubt, and give actionable productivity advice.",
        "spinner": "Channeling pure motivation...",
    },
    "Cooking": {
        "title": "Chef de Partie AI",
        "subtitle": "Your culinary guide, recipe creator, and kitchen assistant.",
        "input_placeholder": "What ingredients do you have, or what do you want to cook?",
        "system_prompt": "You are an expert culinary chef AI. Provide clear recipes, cooking techniques, ingredient substitutions, and kitchen tips.",
        "spinner": "Sharpening the knives...",
    },
    "Coding": {
        "title": "StackOverflow Companion",
        "subtitle": "Your expert software engineer and debugger.",
        "input_placeholder": "Paste your code error or ask a programming question...",
        "system_prompt": (
            "You are an expert senior software engineer AI. Provide clean, secure, "
            "optimized code. Always put full code in a fenced code block with the "
            "correct language tag (e.g. ```python ... ```), and explain the logic "
            "clearly outside the code block."
        ),
        "spinner": "Compiling thoughts...",
    },
    "Amazon": {
        "title": "Marketplace Navigator",
        "subtitle": "Your expert advisor for Amazon AWS, FBA, and e-commerce growth.",
        "input_placeholder": "Ask about AWS architecture, FBA listing optimization, or SEO...",
        "system_prompt": "You are an Amazon ecosystem expert AI. Offer clear, step-by-step guidance on AWS cloud infrastructure and Amazon Seller strategies.",
        "spinner": "Optimizing listings and servers...",
    },
    "Web Design (HTML/CSS)": {
        "title": "Web Design Studio AI",
        "subtitle": "Describe a page and get ready-to-use HTML/CSS.",
        "input_placeholder": "Describe the web page or component you want (e.g. 'a pricing page with 3 tiers')...",
        "system_prompt": (
            "You are an expert front-end web designer AI. When the user asks for a page, "
            "component, or layout, respond with a SINGLE complete, self-contained HTML "
            "document (including inline <style> CSS, and inline <script> JS if needed) "
            "inside one ```html code block. Use modern, clean design. Do not split the "
            "code across multiple blocks. After the code block you may add a short plain-"
            "text explanation of the design choices."
        ),
        "spinner": "Sketching the layout...",
    },
    # add other specific configs as you prefer...
}


def make_default_config(name: str) -> dict:
    """Fallback config generator for any persona not in PERSONA_CONFIGS."""
    return {
        "title": f"{name} Assistant",
        "subtitle": f"A helpful {name} assistant.",
        "input_placeholder": f"Ask the {name} assistant something...",
        "system_prompt": f"You are a helpful and knowledgeable assistant specialized in: {name}. Provide accurate and concise answers.",
        "spinner": "Thinking...",
    }


def extract_code_blocks(text: str) -> list[tuple[str, str]]:
    """Find all fenced code blocks in an assistant reply.

    Returns a list of (language, code) tuples, e.g. [("python", "print(1)")].
    Language is lowercased; defaults to "txt" if the fence has no language tag.
    Also catches a raw <html>...</html> section with no fence, tagged as "html".
    """
    blocks = []
    for match in re.finditer(r"```(\w+)?\s*\n?(.*?)```", text, re.DOTALL):
        lang = (match.group(1) or "txt").strip().lower()
        code = match.group(2).strip()
        if code:
            blocks.append((lang, code))

    if not any(lang == "html" for lang, _ in blocks):
        raw = re.search(r"(<html.*?</html>)", text, re.DOTALL | re.IGNORECASE)
        if raw:
            blocks.append(("html", raw.group(1).strip()))

    return blocks


# Map a fenced code block's language tag to a file extension + download mime type
LANG_TO_FILE = {
    "python": (".py", "text/x-python"),
    "py": (".py", "text/x-python"),
    "html": (".html", "text/html"),
    "css": (".css", "text/css"),
    "javascript": (".js", "application/javascript"),
    "js": (".js", "application/javascript"),
    "json": (".json", "application/json"),
    "sql": (".sql", "text/plain"),
    "bash": (".sh", "text/x-sh"),
    "sh": (".sh", "text/x-sh"),
    "yaml": (".yaml", "text/yaml"),
    "yml": (".yaml", "text/yaml"),
    "txt": (".txt", "text/plain"),
}


# ---------------------------------------------------------------------------
# PDF export (download) helpers
# ---------------------------------------------------------------------------
def generate_pdf_bytes(text: str, title: str | None = None, monospace: bool = False) -> bytes:
    """Render plain text (chat reply or code block) into a downloadable PDF and
    return the raw bytes.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
    from xml.sax.saxutils import escape

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )
    styles = getSampleStyleSheet()
    story = []

    if title:
        story.append(Paragraph(escape(title), styles["Title"]))
        story.append(Spacer(1, 12))

    if monospace:
        # Preformatted preserves whitespace/newlines exactly -- ideal for code.
        code_style = ParagraphStyle(
            "Code",
            parent=styles["Code"],
            fontName="Courier",
            fontSize=9,
            leading=11,
            alignment=TA_LEFT,
        )
        story.append(Preformatted(text, code_style))
    else:
        body_style = ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontSize=11,
            leading=15,
        )
        for para in text.split("\n\n"):
            safe = escape(para).replace("\n", "<br/>")
            if safe.strip():
                story.append(Paragraph(safe, body_style))
                story.append(Spacer(1, 8))

    doc.build(story)
    return buffer.getvalue()


def render_code_tools(lang: str, code: str, key_suffix: str) -> None:
    """Show a live HTML preview (if applicable) and download buttons (native
    format + PDF) for a code block."""
    ext, mime = LANG_TO_FILE.get(lang, (".txt", "text/plain"))
    file_name = f"generated{ext}"

    if lang == "html":
        with st.expander("Preview generated page", expanded=True):
            components.html(code, height=500, scrolling=True)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label=f" Download {file_name}",
            data=code,
            file_name=file_name,
            mime=mime,
            key=f"download_{key_suffix}_{lang}",
        )
    with col2:
        pdf_bytes = generate_pdf_bytes(code, title=f"Generated {lang} code", monospace=True)
        st.download_button(
            label=" Download as PDF",
            data=pdf_bytes,
            file_name=f"generated_{lang}.pdf",
            mime="application/pdf",
            key=f"download_pdf_{key_suffix}_{lang}",
        )


def render_message_pdf_button(content: str, key_suffix: str) -> None:
    """Let the user download a full assistant reply as a PDF."""
    pdf_bytes = generate_pdf_bytes(content, title="AI Response")
    st.download_button(
        label=" Download reply as PDF",
        data=pdf_bytes,
        file_name="ai_response.pdf",
        mime="application/pdf",
        key=f"download_reply_pdf_{key_suffix}",
    )


current_config = PERSONA_CONFIGS.get(persona, make_default_config(persona))

# Build final system prompt including uploaded document context if present
final_system_prompt = current_config["system_prompt"]
if document_context:
    final_system_prompt += (
        "\n\n[Context Document]\nThe user has provided the following reference document to help you answer questions:\n"
        + document_context
        + "\n[END OF CONTEXT DOCUMENT]\n"
    )

# Initialize / reset session state when persona or uploaded doc changes
current_doc_name = uploaded_file.name if uploaded_file else None
needs_reset = (
    "current_persona" not in st.session_state
    or st.session_state.current_persona != persona
    or "last_doc_name" not in st.session_state
    or st.session_state.last_doc_name != current_doc_name
)

if needs_reset:
    st.session_state.current_persona = persona
    st.session_state.last_doc_name = current_doc_name
    st.session_state.messages = [SystemMessage(content=final_system_prompt)]

st.title(current_config["title"])
st.subheader(current_config["subtitle"])

if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = [SystemMessage(content=final_system_prompt)]

# Render chat history
for idx, msg in enumerate(st.session_state.messages):
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)
            render_message_pdf_button(msg.content, key_suffix=f"history_{idx}")
            for block_idx, (lang, code) in enumerate(extract_code_blocks(msg.content)):
                render_code_tools(lang, code, key_suffix=f"history_{idx}_{block_idx}")
    # SystemMessage is intentionally not rendered in the chat UI

# Chat input
if user_input := st.chat_input(current_config["input_placeholder"]):
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))

    with st.chat_message("assistant"):
        with st.spinner(current_config.get("spinner", "Thinking...")):
            try:
                response = model.invoke(st.session_state.messages)
                content = getattr(response, "content", None) or str(response)
                st.write(content)
                st.session_state.messages.append(AIMessage(content=content))

                render_message_pdf_button(content, key_suffix="latest")
                for block_idx, (lang, code) in enumerate(extract_code_blocks(content)):
                    render_code_tools(lang, code, key_suffix=f"latest_{block_idx}")
            except Exception as e:
                st.error("Failed to fetch response: " + str(e))
