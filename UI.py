import os
import io
import re
import streamlit as st
import streamlit.components.v1 as components
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI


st.set_page_config(page_title="Multi-Expert AI Console", page_icon="📡", layout="centered")

# ---------------------------------------------------------------------------
# Design system
# A "dispatch console" look: every persona is a channel you tune into.
# Deep ink background, warm brass accent, monospace channel labels.
# ---------------------------------------------------------------------------
DESIGN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root {
    --bg-app: #10151C;
    --bg-panel: #161D27;
    --bg-panel-2: #1D2530;
    --accent: #E8A33D;
    --accent-soft: #3A4A5C;
    --text-primary: #EDEFF2;
    --text-muted: #8B96A5;
    --border: #262F3B;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}

.stApp {
    background: radial-gradient(circle at top left, #141B24 0%, var(--bg-app) 55%);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--bg-panel);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: 0.02em;
}
section[data-testid="stSidebar"] label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted) !important;
}

/* Channel header block */
.console-header {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
    padding: 1.1rem 1.4rem;
    margin-bottom: 1.4rem;
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 10px;
    background: linear-gradient(135deg, var(--bg-panel) 0%, var(--bg-panel-2) 100%);
}
.console-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent);
}
.console-badge::before {
    content: "● ";
    color: var(--accent);
}
.console-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.9rem;
    line-height: 1.15;
    margin: 0;
    color: var(--text-primary);
}
.console-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    color: var(--text-muted);
    margin: 0;
}

/* Chat messages */
div[data-testid="stChatMessage"] {
    background-color: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.6rem 0.4rem;
    margin-bottom: 0.6rem;
}

/* Buttons */
.stButton > button, .stDownloadButton > button {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    background-color: transparent;
    color: var(--accent);
    border: 1px solid var(--accent-soft);
    border-radius: 6px;
    transition: all 0.15s ease-in-out;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background-color: var(--accent);
    color: #10151C;
    border-color: var(--accent);
}

/* Chat input */
[data-testid="stChatInput"] {
    border: 1px solid var(--border);
    border-radius: 10px;
    background-color: var(--bg-panel);
}

/* Scrollbar */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--bg-app); }
::-webkit-scrollbar-thumb { background: var(--accent-soft); border-radius: 8px; }

/* File uploader box */
[data-testid="stFileUploaderDropzone"] {
    background-color: var(--bg-panel-2);
    border: 1px dashed var(--border);
    border-radius: 8px;
}
</style>
"""
st.markdown(DESIGN_CSS, unsafe_allow_html=True)

if os.path.exists("Image.png"):
    st.image("Image.png", width=150)

# Mistral API key
# SECURITY NOTE: don't hardcode secrets in source. Set MISTRAL_API_KEY as a real
# environment variable, or put it in .streamlit/secrets.toml as:
#   MISTRAL_API_KEY = "your-key-here"
# and it'll be picked up automatically below.
if "MISTRAL_API_KEY" not in os.environ:
    try:
        os.environ["MISTRAL_API_KEY"] = "fE2OLrga4hpKHkGXCn8n5Ck35wCwIq0L"
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

st.sidebar.markdown(
    "<span style='font-family:IBM Plex Mono, monospace; font-size:0.75rem; "
    "letter-spacing:0.1em; color:#8B96A5; text-transform:uppercase;'>Console</span>",
    unsafe_allow_html=True,
)
st.sidebar.title("Select a Channel")

persona_options = [
    "Amazon",
    "Arabic",
    "Web Stack Development",
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
st.sidebar.title("Reference Document")

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
    "Web Stack Development": {
        "title": "Full-Stack Dev Studio AI",
        "subtitle": "Frontend, backend, databases, and deployment — end to end.",
        "input_placeholder": "Describe the app or feature you want (e.g. 'a login page with a Node/Express + MongoDB backend')...",
        "system_prompt": (
            "You are an expert full-stack web development AI, covering the entire "
            "web stack: frontend (HTML, CSS, JavaScript/TypeScript, and frameworks "
            "like React, Vue, or Svelte), backend (Node.js/Express, Python/Flask or "
            "Django, PHP, etc.), databases (SQL and NoSQL: PostgreSQL, MySQL, "
            "MongoDB, etc.), REST/GraphQL APIs, authentication, and basic deployment "
            "or DevOps guidance (Docker, environment variables, hosting).\n\n"
            "When the user asks for an app or feature:\n"
            "- Break the solution into the files it actually needs (e.g. index.html, "
            "styles.css, script.js, server.js, routes/*.js, models/*.py, schema.sql, "
            "requirements.txt, Dockerfile, .env.example), instead of forcing "
            "everything into one file.\n"
            "- Put each file's contents in its own fenced code block, tagged with "
            "the correct language (```html, ```css, ```javascript, ```python, "
            "```sql, ```bash, ```json, ```yaml, etc.), and start that block with a "
            "one-line comment naming the file it belongs to (e.g. `// server.js`).\n"
            "- If the request is small and purely front-end (a single static page or "
            "component with no backend), a single self-contained HTML file with "
            "inline <style>/<script> is fine.\n"
            "- After the code blocks, briefly explain the architecture, how the "
            "pieces connect, how to install dependencies and run the project, and "
            "call out important security or scalability considerations (input "
            "validation, secrets management, SQL injection, CORS, etc.).\n"
            "- Prefer modern, widely-used, well-documented tools and clean, "
            "production-quality code over exotic or deprecated approaches."
        ),
        "spinner": "Wiring up the stack...",
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
    "typescript": (".ts", "application/typescript"),
    "ts": (".ts", "application/typescript"),
    "json": (".json", "application/json"),
    "sql": (".sql", "text/plain"),
    "bash": (".sh", "text/x-sh"),
    "sh": (".sh", "text/x-sh"),
    "yaml": (".yaml", "text/yaml"),
    "yml": (".yaml", "text/yaml"),
    "dockerfile": ("Dockerfile", "text/plain"),
    "php": (".php", "application/x-httpd-php"),
    "txt": (".txt", "text/plain"),
}



@st.cache_data(show_spinner=False)
def generate_pdf_bytes(text: str, title: str | None = None, monospace: bool = False) -> bytes:
    """Render plain text (chat reply or code block) into a downloadable PDF and
    return the raw bytes, using matplotlib's built-in PDF backend -- no
    external PDF library or binary required.
    """
    import textwrap
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    page_width, page_height = 8.27, 11.69  # A4 inches
    font_family = "monospace" if monospace else "sans-serif"
    font_size = 8 if monospace else 10
    chars_per_line = 100 if monospace else 90
    lines_per_page = 60

    wrapped_lines: list[str] = []
    for raw_line in text.split("\n"):
        if raw_line.strip() == "":
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(
            textwrap.wrap(raw_line, width=chars_per_line, break_long_words=True)
            or [""]
        )

    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        page_lines = wrapped_lines
        first_page = True
        while page_lines or first_page:
            chunk, page_lines = page_lines[:lines_per_page], page_lines[lines_per_page:]
            fig = plt.figure(figsize=(page_width, page_height))
            fig.patch.set_visible(False)
            y = 0.97
            if first_page and title:
                fig.text(0.07, y, title, fontsize=14, fontweight="bold", family="sans-serif")
                y -= 0.05
            for line in chunk:
                fig.text(0.07, y, line, fontsize=font_size, family=font_family, va="top")
                y -= 0.016
            pdf.savefig(fig)
            plt.close(fig)
            first_page = False

    return buffer.getvalue()


def render_code_tools(lang: str, code: str, key_suffix: str) -> None:
    """Show a live HTML preview (if applicable) and download buttons (native
    format + PDF) for a code block."""
    ext, mime = LANG_TO_FILE.get(lang, (".txt", "text/plain"))
    file_name = ext if ext == "Dockerfile" else f"generated{ext}"

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

st.markdown(
    f"""
    <div class="console-header">
        <span class="console-badge">Active Channel &middot; {persona}</span>
        <p class="console-title">{current_config["title"]}</p>
        <p class="console-subtitle">{current_config["subtitle"]}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

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
