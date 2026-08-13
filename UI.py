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
    """Extract text from a PDF's raw bytes using the poppler-utils `pdftotext`
    command-line tool (no Python PDF library required)."""
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_in:
        tmp_in.write(raw_bytes)
        tmp_in_path = tmp_in.name

    tmp_out_path = tmp_in_path.replace(".pdf", ".txt")

    try:
        subprocess.run(
            ["pdftotext", "-layout", tmp_in_path, tmp_out_path],
            check=True,
            capture_output=True,
        )
        with open(tmp_out_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    finally:
        for p in (tmp_in_path, tmp_out_path):
            if os.path.exists(p):
                os.remove(p)


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
    return the raw bytes, using wkhtmltopdf (via pdfkit) -- no Python PDF
    generation library required.
    """
    import pdfkit
    from xml.sax.saxutils import escape

    body_font = "'Courier New', monospace" if monospace else "Arial, sans-serif"
    safe_text = escape(text)
    heading_html = f"<h1>{escape(title)}</h1>" if title else ""

    html = f"""
    <html>
      <head>
        <meta charset="utf-8">
        <style>
          body {{ font-family: {body_font}; font-size: 12px; white-space: pre-wrap;
                  word-wrap: break-word; margin: 40px; }}
          h1 {{ font-family: Arial, sans-serif; font-size: 20px; }}
        </style>
      </head>
      <body>
        {heading_html}
        <div>{safe_text}</div>
      </body>
    </html>
    """

    options = {"quiet": ""}
    return pdfkit.from_string(html, False, options=options)


def render_code_tools(lang: str, code: str, key_suffix: str) -> None:
    """Show a live HTML preview (if applicable) and download buttons (native
    format + PDF) for a code block."""
    ext, mime = LANG_TO_FILE.get(lang, (".txt", "text/plain"))
    file_name = f"generated{ext}"

    if lang == "html":
        with st.expander("🌐 Preview generated page", expanded=True):
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
