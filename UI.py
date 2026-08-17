import os
import io
import re
import streamlit as st
import streamlit.components.v1 as components
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI


st.set_page_config(page_title="Boarding: AI Experts", page_icon="🎫", layout="centered")

# ---------------------------------------------------------------------------
# Design system: "Boarding Pass"
# Picking a persona = boarding a flight into that expert's world. The header
# renders as a literal ticket stub with a channel number, a torn-perforation
# divider, and a rotated stamp. Everything else stays quiet around it.
# ---------------------------------------------------------------------------
DESIGN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

/* Theme source of truth: Streamlit's OWN live CSS variables
   (--background-color, --secondary-background-color, --text-color), which
   it updates automatically based on the in-app Settings -> theme choice
   (Light / Dark / a custom theme) - NOT the OS's prefers-color-scheme.
   Each var() below falls back to our original dark palette so nothing
   breaks if an older Streamlit build doesn't expose these variables.
   --ink, --marigold, --coral stay fixed on purpose: they're always the
   dark text sitting on the gold stub/navbar, regardless of theme. */
:root {
    --ink: #1B1230;
    --marigold: #2E86FF;
    --coral: #FF5F87;

    --bg-app: var(--background-color, #1B1230);
    --panel: var(--secondary-background-color, #241A3D);
    --text-primary: var(--text-color, #F5F1FA);

    /* Streamlit doesn't expose a third background shade or a "muted text"
       token, so derive them by mixing its live variables with our accent -
       these recompute automatically whenever --background-color /
       --text-color change. */
    --panel-2: color-mix(in srgb, var(--panel) 85%, var(--marigold) 15%);
    --text-muted: color-mix(in srgb, var(--text-primary) 65%, var(--bg-app) 35%);
    --border: color-mix(in srgb, var(--text-primary) 18%, var(--bg-app) 82%);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}

.stApp {
    background: var(--bg-app);
}

/* Smooth, quiet transition when the person toggles Light/Dark in Settings,
   instead of every panel snapping instantly. Respect reduced-motion. */
.stApp, section[data-testid="stSidebar"], .ticket, div[data-testid="stChatMessage"],
[data-testid="stFileUploaderDropzone"], [data-testid="stChatInput"] {
    transition: background-color 0.25s ease, border-color 0.25s ease, color 0.25s ease, box-shadow 0.25s ease;
}
@media (prefers-reduced-motion: reduce) {
    .stApp, section[data-testid="stSidebar"], .ticket, div[data-testid="stChatMessage"],
    [data-testid="stFileUploaderDropzone"], [data-testid="stChatInput"], .fire-emoji, .ticket-stub::before {
        transition: none !important;
        animation: none !important;
    }
}

/* Sidebar = departure board */
section[data-testid="stSidebar"] {
    background-color: var(--panel);
    border-right: 1px dashed var(--border);
}
section[data-testid="stSidebar"] h1 {
    font-family: 'Fraunces', serif;
    font-weight: 700;
    letter-spacing: 0.01em;
}
section[data-testid="stSidebar"] label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted) !important;
}

/* ---------------- Boarding pass header ---------------- */
.ticket {
    display: flex;
    margin-bottom: 1.6rem;
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 12px 30px color-mix(in srgb, var(--ink) 35%, transparent);
    background: var(--panel);
    position: relative;
}
.ticket-stub {
    flex: 0 0 92px;
    background: var(--marigold);
    color: var(--ink);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    font-family: 'IBM Plex Mono', monospace;
    padding: 0.8rem 0.4rem;
    position: relative;
    overflow: hidden;
}
.ticket-stub::before {
    content: "";
    position: absolute;
    top: 50%;
    left: 50%;
    width: 130px;
    height: 130px;
    transform: translate(-50%, -50%) scale(0.9);
    background: radial-gradient(circle, color-mix(in srgb, #FFFFFF 55%, transparent) 0%, transparent 70%);
    pointer-events: none;
    animation: spotlight-pulse 2.6s ease-in-out infinite;
}
.ticket-stub .no-label,
.ticket-stub .no-value {
    position: relative;
    z-index: 1;
}
@keyframes spotlight-pulse {
    0%, 100% { opacity: 0.5; transform: translate(-50%, -50%) scale(0.88); }
    50%      { opacity: 0.85; transform: translate(-50%, -50%) scale(1.05); }
}
.ticket-stub .no-label {
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    opacity: 0.75;
}
.ticket-stub .no-value {
    font-size: 1.6rem;
    font-weight: 600;
    line-height: 1.1;
}
.ticket-perf {
    flex: 0 0 0;
    border-left: 2px dashed var(--ink);
    position: relative;
}
.ticket-perf::before, .ticket-perf::after {
    content: "";
    position: absolute;
    width: 16px;
    height: 16px;
    background: var(--ink);
    border-radius: 50%;
    left: -9px;
}
.ticket-perf::before { top: -8px; }
.ticket-perf::after { bottom: -8px; }
.ticket-main {
    flex: 1;
    padding: 1.1rem 1.4rem;
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
    min-width: 0;
}
.ticket-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
}
.ticket-title {
    font-family: 'Fraunces', serif;
    font-weight: 700;
    font-size: 1.85rem;
    line-height: 1.12;
    margin: 0;
    color: var(--text-primary);
}
.ticket-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 0.92rem;
    color: var(--text-muted);
    margin: 0.1rem 0 0 0;
}
.ticket-stamp {
    position: absolute;
    top: 14px;
    right: 18px;
    transform: rotate(-9deg);
    border: 2px solid var(--coral);
    color: var(--coral);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    padding: 3px 9px;
    border-radius: 6px;
    text-transform: uppercase;
}
.ticket-barcode {
    display: flex;
    align-items: flex-end;
    gap: 2px;
    height: 20px;
    margin-top: 0.4rem;
    opacity: 0.6;
}
.ticket-barcode span {
    display: block;
    width: 2px;
    background: var(--text-muted);
}

/* ---------------- Top navbar ---------------- */
/* Sits in normal page flow, right above the ticket header - not fixed, so
   it never fights Streamlit's own header for space or blocks its buttons. */
.topnav {
    width: 100%;
    background: var(--marigold);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.7rem 1.4rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    box-shadow: 0 6px 18px rgba(0,0,0,0.25);
}
.topnav-brand {
    font-family: 'Fraunces', serif;
    font-weight: 700;
    font-size: 1.15rem;
    color: var(--ink);
    letter-spacing: 0.01em;
}
.topnav-status {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--ink);
    background: rgba(27, 18, 48, 0.12);
    padding: 4px 10px;
    border-radius: 999px;
}

/* Flickering fire emoji */
@keyframes fire-flicker {
    0%   { transform: scale(1) rotate(-2deg);   opacity: 1;    filter: drop-shadow(0 0 0px #FF9A3C); }
    20%  { transform: scale(1.08) rotate(3deg); opacity: 0.9;  filter: drop-shadow(0 0 4px #FF9A3C); }
    40%  { transform: scale(0.96) rotate(-4deg);opacity: 1;    filter: drop-shadow(0 0 2px #FF9A3C); }
    60%  { transform: scale(1.1) rotate(2deg);  opacity: 0.85; filter: drop-shadow(0 0 6px #FF9A3C); }
    80%  { transform: scale(0.98) rotate(-3deg);opacity: 1;    filter: drop-shadow(0 0 3px #FF9A3C); }
    100% { transform: scale(1) rotate(-2deg);   opacity: 1;    filter: drop-shadow(0 0 0px #FF9A3C); }
}
.fire-emoji {
    display: inline-block;
    animation: fire-flicker 1.4s infinite ease-in-out;
    transform-origin: center bottom;
}

/* Streamlit's own header stays exactly where Streamlit puts it - just make
   it transparent so it blends with our background. */
header[data-testid="stHeader"] {
    background: transparent;
}

/* Header toolbar icons (GitHub / Share / "..." menu) were invisible: dark
   SVGs on a dark background. Force them to use the theme's text color. */
header[data-testid="stHeader"] svg,
[data-testid="stToolbar"] svg,
[data-testid="stToolbarActions"] svg,
[data-testid="stDeployButton"] svg,
[data-testid="stStatusWidget"] svg {
    fill: var(--text-primary) !important;
    color: var(--text-primary) !important;
    opacity: 1 !important;
}
header[data-testid="stHeader"] button,
[data-testid="stToolbar"] button,
[data-testid="stToolbarActions"] button,
[data-testid="stDeployButton"] button {
    color: var(--text-primary) !important;
}
header[data-testid="stHeader"] button:hover svg {
    fill: var(--marigold) !important;
    color: var(--marigold) !important;
}

/* Chat messages */
div[data-testid="stChatMessage"] {
    background-color: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.6rem 0.4rem;
    margin-bottom: 0.6rem;
}
div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] li,
div[data-testid="stChatMessage"] span,
div[data-testid="stChatMessage"] div,
div[data-testid="stChatMessage"] .stMarkdown {
    color: var(--text-primary) !important;
}
div[data-testid="stChatMessage"] code {
    color: var(--marigold) !important;
    background-color: color-mix(in srgb, var(--marigold) 12%, transparent) !important;
}

/* Chat input text */
[data-testid="stChatInput"] textarea {
    color: #FFFFFF !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #FFFFFF !important;
    opacity: 0.75;
}

/* Any other body / markdown text in the main area */
div[data-testid="stAppViewContainer"] .stMarkdown p,
div[data-testid="stAppViewContainer"] .stMarkdown li,
div[data-testid="stAppViewContainer"] .stMarkdown span {
    color: var(--text-primary);
}

/* Buttons -> ticket-pill style */
.stButton > button, .stDownloadButton > button {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    background-color: transparent;
    color: var(--marigold);
    border: 1.5px dashed var(--marigold);
    border-radius: 999px;
    padding: 0.35rem 0.9rem;
    transition: all 0.15s ease-in-out;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background-color: var(--marigold);
    color: var(--ink);
    border-style: solid;
}

/* Chat input */
[data-testid="stChatInput"] {
    border: 1px solid var(--border);
    border-radius: 10px;
    background-color: var(--marigold);
}

/* Send button icon inside the chat input was hardcoded black - force it to
   white so it stays visible on the new blue input background. */
[data-testid="stChatInputSubmitButton"] svg,
[data-testid="stChatInput"] button svg {
    fill: #FFFFFF !important;
    color: #FFFFFF !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--bg-app); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 8px; }

/* File uploader box */
[data-testid="stFileUploaderDropzone"] {
    background-color: var(--panel-2);
    border: 1px dashed var(--border);
    border-radius: 8px;
}
[data-testid="stFileUploaderDropzoneInstructions"] small,
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzone"] small {
    color: var(--marigold) !important;
}

/* Visible keyboard focus - the custom palette can otherwise hide Streamlit's
   default focus ring, so make it explicit and consistent everywhere. */
.stButton > button:focus-visible,
.stDownloadButton > button:focus-visible,
[data-testid="stFileUploaderDropzone"]:focus-within,
[data-testid="stChatInput"]:focus-within,
div[data-baseweb="select"]:focus-within {
    outline: 2px solid var(--marigold) !important;
    outline-offset: 2px;
}
</style>
"""
st.markdown(DESIGN_CSS, unsafe_allow_html=True)
st.markdown(
    """
    <div class="topnav">
        <span class="topnav-brand"><span class="fire-emoji">🔥</span> Expert Boarding</span>
        <span class="topnav-status">Now Boarding</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if os.path.exists("water.png"):
    st.image("water.png", width=150)

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
    "letter-spacing:0.1em; color:var(--text-muted); text-transform:uppercase;'>Departures</span>",
    unsafe_allow_html=True,
)
st.sidebar.title("Choose Your Flight")

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
            "web stack: frontend (HTML, CSS, JavaScript), backend (Node.js/Express, "
            "Python/Flask or Django, PHP, etc.), databases (SQL and NoSQL: "
            "PostgreSQL, MySQL, MongoDB, etc.), REST/GraphQL APIs, authentication, "
            "and basic deployment or DevOps guidance (Docker, environment "
            "variables, hosting).\n\n"
            "FRONTEND RULE: always combine the HTML, CSS, and JavaScript into ONE "
            "single self-contained ```html code block — inline <style> for CSS and "
            "inline <script> for JS in the same file, never split across separate "
            "html/css/js blocks. This applies whether the frontend is standalone or "
            "paired with a backend.\n\n"
            "BACKEND/DATABASE RULE: if the request needs a backend or database, put "
            "those in their own separate fenced code blocks (e.g. server.js, "
            "routes/*.js, models/*.py, schema.sql, requirements.txt, Dockerfile, "
            ".env.example), each tagged with the correct language and starting with "
            "a one-line comment naming the file (e.g. `// server.js`), and explain "
            "how the single-page frontend calls those backend endpoints (fetch/axios "
            "URLs, ports, etc.).\n\n"
            "After the code blocks, briefly explain the architecture, how the "
            "pieces connect, how to install dependencies and run the project, and "
            "call out important security or scalability considerations (input "
            "validation, secrets management, SQL injection, CORS, etc.). Prefer "
            "modern, widely-used, well-documented tools and clean, "
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


USER_AVATAR = "🧑‍💻"
ASSISTANT_AVATAR = "🤖"


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

try:
    channel_no = f"{persona_options.index(persona) + 1:03d}"
except ValueError:
    channel_no = "N/A"

# Deterministic little "barcode" flourish so it's not identical for every persona
_bar_seed = sum(ord(c) for c in persona)
_bar_widths = [2 + ((_bar_seed * (i + 3)) % 5) for i in range(28)]
_barcode_html = "".join(
    f'<span style="width:{2 if i % 3 == 0 else 3}px; height:{w * 2}px;"></span>'
    for i, w in enumerate(_bar_widths)
)

st.markdown(
    f"""
    <div class="ticket">
        <div class="ticket-stub">
            <span class="no-label">CHANNEL</span>
            <span class="no-value">{channel_no}</span>
        </div>
        <div class="ticket-perf"></div>
        <div class="ticket-main">
            <span class="ticket-stamp">💧 On Air</span>
            <span class="ticket-eyebrow">Boarding &middot; {persona}</span>
            <p class="ticket-title">{current_config["title"]}</p>
            <p class="ticket-subtitle">{current_config["subtitle"]}</p>
            <div class="ticket-barcode">{_barcode_html}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = [SystemMessage(content=final_system_prompt)]

# Render chat history
for idx, msg in enumerate(st.session_state.messages):
    if isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar=USER_AVATAR):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            st.write(msg.content)
            render_message_pdf_button(msg.content, key_suffix=f"history_{idx}")
            for block_idx, (lang, code) in enumerate(extract_code_blocks(msg.content)):
                render_code_tools(lang, code, key_suffix=f"history_{idx}_{block_idx}")
    # SystemMessage is intentionally not rendered in the chat UI

# Chat input
if user_input := st.chat_input(current_config["input_placeholder"]):
    with st.chat_message("user", avatar=USER_AVATAR):
        st.write(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
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
