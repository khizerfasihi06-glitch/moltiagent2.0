import os
import io
import re
import streamlit as st
import streamlit.components.v1 as components
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# 1. Page Configuration
st.set_page_config(page_title="Avatar AI Experts", page_icon="💧", layout="centered")

# 2. Optimized & Cached Groq / LangChain Initializers (Prevents 429 Errors)
@st.cache_resource
def init_groq_llm():
    """Initializes and caches the chat model.
       Implements automatic backoff retries on rate limits (429s)."""
    # Fallback to streamlit secrets or environment variables
    api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("Missing GROQ_API_KEY. Please set it in your environment variables or Streamlit secrets.")
        st.stop()

    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        max_retries=5,  # Automatically waits and backs off exponentially on 429s
        timeout=60
    )

@st.cache_resource
def init_embeddings():
    """Initializes and caches the text embedding layer.
       Groq does not currently offer an embeddings endpoint, so we use
       FastEmbed — a lightweight, ONNX-based local embedding model with no
       torch/transformers dependency. This keeps RAG fully functional
       without a second API key and avoids heavy ML-stack install issues
       (e.g. the torchvision import error some transformers versions hit)."""
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# Instantiate the cached singletons
llm = init_groq_llm()
embeddings = init_embeddings()

@st.cache_resource(show_spinner="Analyzing documents and generating vector space...")
def build_vector_store(text_content):
    """Chunks text and generates a ChromaDB collection, cached as a live resource
       (Chroma holds an active client connection, so cache_resource is the correct
       cache type here rather than cache_data)."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = [Document(page_content=x) for x in text_splitter.split_text(text_content) if x.strip()]
    if not docs:
        return None
    # In-memory, ephemeral Chroma collection scoped to this session's document set
    return Chroma.from_documents(docs, embeddings, collection_name="avatar_session_docs")


def generate_chat_pdf(persona_name, history):
    """Renders the conversation transcript (excluding the system prompt) into a
       PDF and returns it as raw bytes, ready for st.download_button."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph(f"Conversation Transcript — {persona_name}", styles["Title"]), Spacer(1, 14)]

    for msg in history:
        if isinstance(msg, SystemMessage):
            continue
        speaker = "You" if isinstance(msg, HumanMessage) else "Assistant"
        # Escape reportlab markup characters and preserve line breaks
        safe_content = (
            msg.content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
        )
        story.append(Paragraph(speaker, styles["Heading4"]))
        story.append(Paragraph(safe_content, styles["Normal"]))
        story.append(Spacer(1, 10))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# 3. CSS Styling System ("Boarding Pass")
DESIGN_CSS = """
<style>
@import url('https://googleapis.com');

:root {
    --ink: #1B1230;
    --marigold: #2E86FF;
    --coral: #FF5F87;
    --bg-app: var(--background-color, #1B1230);
    --panel: var(--secondary-background-color, #241A3D);
    --text-primary: var(--text-color, #F5F1FA);
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

.stApp, section[data-testid="stSidebar"], .ticket, div[data-testid="stChatMessage"],
[data-testid="stFileUploaderDropzone"], [data-testid="stChatInput"] {
    transition: background-color 0.25s ease, border-color 0.25s ease, color 0.25s ease, box-shadow 0.25s ease;
}

@media (prefers-reduced-motion: reduce) {
    .stApp, section[data-testid="stSidebar"], .ticket, div[data-testid="stChatMessage"],
    [data-testid="stFileUploaderDropzone"], [data-testid="stChatInput"] {
        transition: none !important;
        animation: none !important;
    }
}

section[data-testid="stSidebar"] {
    background-color: var(--panel);
    border-right: 1px dashed var(--border);
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-family: 'Fraunces', serif;
    font-weight: 700;
    letter-spacing: 0.01em;
    color: var(--text-primary) !important;
}
section[data-testid="stSidebar"] label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted) !important;
}

/* Boarding pass header layout */
.ticket {
    display: flex;
    margin-bottom: 1.6rem;
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 12px 30px color-mix(in srgb, var(--ink) 35%, transparent);
    background: var(--panel);
    position: relative;
    border: 1px solid var(--border);
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
@keyframes spotlight-pulse {
    0%, 100% { opacity: 0.5; transform: translate(-50%, -50%) scale(0.88); }
    50%      { opacity: 0.85; transform: translate(-50%, -50%) scale(1.05); }
}
.ticket-stub .no-label {
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    opacity: 0.75;
    position: relative;
    z-index: 1;
}
.ticket-stub .no-value {
    font-size: 1.6rem;
    font-weight: 600;
    line-height: 1.1;
    position: relative;
    z-index: 1;
}
.ticket-perf {
    flex: 0 0 0;
    border-left: 2px dashed var(--bg-app);
    position: relative;
    background: transparent;
}
.ticket-perf::before, .ticket-perf::after {
    content: "";
    position: absolute;
    width: 16px;
    height: 16px;
    background: var(--bg-app);
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

.topnav {
    width: 100%;
    background: var(--panel);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.7rem 1.4rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    border: 1px solid var(--border);
}
.topnav-brand {
    font-family: 'Fraunces', serif;
    font-weight: 700;
    font-size: 1.15rem;
    color: var(--text-primary);
}
.topnav-status {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--marigold);
    background: color-mix(in srgb, var(--marigold) 15%, transparent);
    padding: 4px 10px;
    border-radius: 999px;
}
</style>
"""
st.markdown(DESIGN_CSS, unsafe_allow_html=True)

# 4. Sidebar Elements & Configuration State
with st.sidebar:
    st.title("🛫 Departure Lounge")

    # Expert Selection Matrix — 1,000 personas, generated from 50 domains x
    # 20 role types (50 x 20 = 1000 unique combinations). Each domain and
    # each role has its own unique 3-letter code, so every "DOMAIN-ROLE"
    # stamp is guaranteed unique without listing 1,000 lines by hand.
    _DOMAINS = [
        ("Marketing", "MKT"), ("Finance", "FIN"), ("Healthcare", "HLT"), ("Legal", "LAW"),
        ("Technology", "TEC"), ("Education", "EDU"), ("Real Estate", "RES"), ("Hospitality", "HOS"),
        ("Retail", "RET"), ("Manufacturing", "MFG"), ("Agriculture", "AGR"), ("Energy", "ENR"),
        ("Transportation", "TRN"), ("Logistics", "LOG"), ("Construction", "CON"), ("Automotive", "AUT"),
        ("Aerospace", "AER"), ("Telecommunications", "TEL"), ("Insurance", "INS"), ("Banking", "BNK"),
        ("Nonprofit", "NPO"), ("Government", "GOV"), ("Media", "MED"), ("Entertainment", "ENT"),
        ("Sports", "SPT"), ("Fashion", "FSH"), ("Food & Beverage", "FNB"), ("Environmental", "ENV"),
        ("Cybersecurity", "CYB"), ("Artificial Intelligence", "AIX"), ("Blockchain", "BLK"),
        ("Biotechnology", "BIO"), ("Pharmaceuticals", "PHM"), ("Mental Health", "MHL"), ("Fitness", "FIT"),
        ("Nutrition", "NUT"), ("Travel", "TRV"), ("Music", "MUS"), ("Film", "FLM"), ("Publishing", "PUB"),
        ("Architecture", "ARC"), ("Interior Design", "INT"), ("Urban Planning", "URB"),
        ("Human Resources", "HRS"), ("Supply Chain", "SPC"), ("E-commerce", "ECM"), ("Gaming", "GAM"),
        ("Photography", "PHO"), ("Journalism", "JRN"), ("Political Science", "POL"),
    ]
    _ROLES = [
        ("Strategist", "STR", "framing high-leverage plans and trade-off analysis"),
        ("Analyst", "ANL", "breaking down data, trends, and metrics into clear takeaways"),
        ("Consultant", "CNS", "diagnosing problems and recommending practical fixes"),
        ("Coach", "COA", "building accountability, habits, and skill progression"),
        ("Specialist", "SPL", "bringing deep, technical, domain-specific know-how"),
        ("Architect", "ARC", "structuring systems, workflows, and long-term design"),
        ("Advisor", "ADV", "giving plain-language guidance grounded in best practices"),
        ("Researcher", "RSR", "synthesizing evidence and current thinking in the field"),
        ("Writer", "WRT", "producing clear, audience-tailored written content"),
        ("Planner", "PLN", "sequencing steps, logistics, and timelines"),
        ("Designer", "DSN", "shaping user-facing form, flow, and experience"),
        ("Engineer", "ENG", "building and troubleshooting technical systems"),
        ("Manager", "MGR", "coordinating people, priorities, and delivery"),
        ("Educator", "EDC", "explaining concepts clearly for learners at any level"),
        ("Mentor", "MTR", "offering experience-based guidance and encouragement"),
        ("Auditor", "AUD", "reviewing processes for accuracy, risk, and compliance"),
        ("Developer", "DEV", "building functional, maintainable technical solutions"),
        ("Curator", "CUR", "selecting, organizing, and contextualizing quality content"),
        ("Producer", "PRD", "coordinating end-to-end delivery of a finished output"),
        ("Facilitator", "FAC", "guiding discussions, workshops, and group decisions"),
    ]

    persona_configs = {}
    idx = 0
    for domain_name, domain_code in _DOMAINS:
        for role_name, role_code, role_blurb in _ROLES:
            idx += 1
            persona_name = f"{domain_name} {role_name}"
            persona_configs[persona_name] = {
                "channel": f"CH-{idx:04d}",
                "stamp": f"{domain_code}-{role_code}",
                "desc": f"{role_name} {role_blurb}, applied to {domain_name.lower()} contexts."
            }

    persona_option = st.selectbox(
        "Choose Your AI Guide:",
        list(persona_configs.keys())
    )

    config = persona_configs[persona_option]

    st.markdown("---")
    st.subheader("📁 Context Ingestion (RAG)")
    uploaded_file = st.file_uploader("Drop supporting files here to seed vector memory:", type=["txt", "md"])

# 5. Core Application Initialization (Session Memory Management)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(content=f"You are the {persona_option}. {config['desc']} Adapt your output behavior strictly to this identity framework.")
    ]

# If the expert selection changes, reset context safely
if st.session_state.chat_history and isinstance(st.session_state.chat_history[0], SystemMessage):
    if persona_option not in st.session_state.chat_history[0].content:
        st.session_state.chat_history = [
            SystemMessage(content=f"You are the {persona_option}. {config['desc']} Adapt your output behavior strictly to this identity framework.")
        ]

# Context Processing Block
vectorstore = None
if uploaded_file is not None:
    stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
    raw_text = stringio.read()
    if raw_text.strip():
        vectorstore = build_vector_store(raw_text)

# One-time PDF export of the conversation transcript, shown in the sidebar
if "pdf_download_used" not in st.session_state:
    st.session_state.pdf_download_used = False

def _mark_pdf_downloaded():
    st.session_state.pdf_download_used = True

with st.sidebar:
    st.markdown("---")
    st.subheader("📄 Export Transcript")
    has_conversation = any(isinstance(m, HumanMessage) for m in st.session_state.chat_history)

    if st.session_state.pdf_download_used:
        st.caption("✅ PDF already downloaded this session.")
    elif has_conversation:
        pdf_bytes = generate_chat_pdf(persona_option, st.session_state.chat_history)
        st.download_button(
            label="⬇️ Download Conversation as PDF",
            data=pdf_bytes,
            file_name=f"{persona_option.replace(' ', '_').replace('/', '-')}_transcript.pdf",
            mime="application/pdf",
            on_click=_mark_pdf_downloaded,
            key="pdf_download_once"
        )
    else:
        st.caption("Start chatting to enable PDF export.")

# 6. Header / Ticket UI
st.markdown(f"""
<div class="ticket">
    <div class="ticket-stub">
        <div class="no-label">Channel</div>
        <div class="no-value">{config['channel']}</div>
    </div>
    <div class="ticket-perf"></div>
    <div class="ticket-main">
        <div class="ticket-eyebrow">Boarding Pass · AI Expert Session</div>
        <h1 class="ticket-title">{persona_option}</h1>
        <p class="ticket-subtitle">{config['desc']}</p>
        <div class="ticket-stamp">{config['stamp']}</div>
    </div>
</div>
""", unsafe_allow_html=True)

if vectorstore is not None:
    st.markdown(
        '<div class="topnav"><span class="topnav-brand">📁 Context Loaded</span>'
        '<span class="topnav-status">RAG Active</span></div>',
        unsafe_allow_html=True
    )

# 7. Chat History Rendering
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# 8. Chat Input & Response Generation
user_prompt = st.chat_input("Ask your AI guide anything...")

if user_prompt:
    st.session_state.chat_history.append(HumanMessage(content=user_prompt))
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Retrieve relevant context from the vector store, if available
    context_snippet = ""
    if vectorstore is not None:
        relevant_docs = vectorstore.similarity_search(user_prompt, k=4)
        if relevant_docs:
            context_snippet = "\n\n".join(doc.page_content for doc in relevant_docs)

    messages_to_send = list(st.session_state.chat_history)
    if context_snippet:
        messages_to_send.insert(
            1,
            SystemMessage(content=f"Use the following retrieved context if relevant:\n\n{context_snippet}")
        )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = llm.invoke(messages_to_send)
        st.markdown(response.content)

    st.session_state.chat_history.append(AIMessage(content=response.content))
