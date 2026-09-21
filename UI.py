import os
import io
import re
import time
import httpx
import streamlit as st
import streamlit.components.v1 as components
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from pypdf import PdfReader

# 1. Page Configuration
st.set_page_config(page_title="Avatar AI Experts", page_icon="💧", layout="centered")

# 2. Optimized & Cached LLM Initializers, with a 4-provider Fallback Chain
#    (Gemini -> Groq -> OpenAI -> Mistral) so no single 429/quota error
#    takes the whole app down.
@st.cache_resource
def init_gemini_llm():
    """Initializes and caches the Gemini chat model.
       Returns None if no Gemini key is configured, so the app can still run
       on whichever other providers are configured."""
    api_key = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None

    # "gemini-flash-latest" is a rolling alias Google keeps pointed at its
    # newest GA Flash model (currently Gemini 3.5 Flash, released May 2026),
    # so this stays current without needing manual updates. Pin an explicit
    # version instead (e.g. "gemini-2.5-flash") via GEMINI_MODEL if you need
    # stable, unmoving behavior for production.
    model_name = st.secrets.get("GEMINI_MODEL") or os.environ.get("GEMINI_MODEL") or "gemini-flash-latest"

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        max_retries=5,  # Automatically waits and backs off exponentially on connection errors
        timeout=60
    )


@st.cache_resource
def init_groq_llm():
    """Initializes and caches the Groq chat model.
       Returns None if no Groq key is configured, so the app can still run
       on whichever other providers are configured."""
    api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None

    model_name = st.secrets.get("GROQ_MODEL") or os.environ.get("GROQ_MODEL") or "openai/gpt-oss-120b"

    return ChatGroq(
        model=model_name,
        api_key=api_key,
        max_retries=5,
        timeout=60
    )


@st.cache_resource
def init_openai_llm():
    """Initializes and caches the OpenAI chat model.
       Returns None if no OpenAI key is configured, so the app can still run
       on whichever other providers are configured."""
    api_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    # "gpt-5-mini" is OpenAI's current cost-efficient, general-purpose model.
    # Override via OPENAI_MODEL (e.g. "gpt-5.4-mini" or a flagship model) if
    # your account has access to something newer/larger.
    model_name = st.secrets.get("OPENAI_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-5-mini"

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        max_retries=5,
        timeout=60
    )


@st.cache_resource
def init_mistral_llm():
    """Initializes and caches the Mistral chat model.
       Returns None if no Mistral key is configured, so the app can still run
       on whichever other providers are configured."""
    api_key = st.secrets.get("MISTRAL_API_KEY") or os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        return None

    # "mistral-small-latest" is available on free/basic API tiers. Larger
    # models like "mistral-large-latest" return a 403 "tier_not_allowed"
    # error unless your account has billing/a higher tier enabled — bump
    # this via MISTRAL_MODEL once your account supports it.
    model_name = st.secrets.get("MISTRAL_MODEL") or os.environ.get("MISTRAL_MODEL") or "mistral-small-latest"

    return ChatMistralAI(
        model=model_name,
        api_key=api_key,
        max_retries=5,
        timeout=60
    )


def get_llm_chain():
    """Returns the configured LLMs in fallback order:
       Gemini -> Groq -> OpenAI -> Mistral.
       At least one of GOOGLE_API_KEY / GROQ_API_KEY / OPENAI_API_KEY /
       MISTRAL_API_KEY must be set; any subset works, in that priority order."""
    chain = [
        ("Gemini", init_gemini_llm()),
        ("Groq", init_groq_llm()),
        ("OpenAI", init_openai_llm()),
        ("Mistral", init_mistral_llm()),
    ]
    chain = [(name, model) for name, model in chain if model is not None]
    if not chain:
        st.error(
            "No LLM is configured. Set at least one of GOOGLE_API_KEY (Gemini), "
            "GROQ_API_KEY (Groq), OPENAI_API_KEY (OpenAI), or MISTRAL_API_KEY (Mistral) "
            "in your environment variables or Streamlit secrets."
        )
        st.stop()
    return chain

@st.cache_resource
def init_embeddings():
    """Initializes and caches the text embedding layer.
       We use FastEmbed — a lightweight, ONNX-based local embedding model with no
       torch/transformers dependency — so RAG works fully offline without
       relying on Mistral's embeddings endpoint or a second API key, and
       avoids heavy ML-stack install issues (e.g. the torchvision import
       error some transformers versions hit)."""
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# Instantiate the cached singletons
llm_chain = get_llm_chain()  # [("Gemini", llm_obj), ("Groq", llm_obj)] in fallback order
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


def extract_text_from_upload(uploaded):
    """Extracts plain text from an uploaded .txt, .md, or .pdf file."""
    name = uploaded.name.lower()
    if name.endswith(".pdf"):
        try:
            reader = PdfReader(uploaded)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            st.warning(f"Could not read '{uploaded.name}': {e}")
            return ""
    # txt / md
    try:
        return uploaded.getvalue().decode("utf-8")
    except UnicodeDecodeError:
        st.warning(f"Could not decode '{uploaded.name}' as UTF-8 text.")
        return ""


def count_chunks(text_content):
    """Quick, uncached chunk count for sidebar stats (splitting text is cheap;
       only the embedding step in build_vector_store needs caching)."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return len([c for c in splitter.split_text(text_content) if c.strip()])


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


def extract_response_text(content):
    """Normalizes an LLM response's .content into plain text.
       Gemini 3 models return content as a list of blocks (a text block plus
       an internal 'thinking' signature block) rather than a plain string,
       so this pulls out just the text parts regardless of provider shape."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def invoke_with_rate_limit_retry(llm, messages, max_attempts=4, base_delay=3):
    """Calls llm.invoke, retrying on rate-limit errors with exponential backoff.
       Provider client libraries' own max_retries typically only cover
       connection/timeout errors, not rate-limit errors, so that case is
       handled here instead. Different providers raise different exception
       types for 429s (e.g. Google's ResourceExhausted vs. an
       httpx.HTTPStatusError), so this detects it by message content to stay
       resilient across whichever provider is currently in use."""
    last_error = None
    for attempt in range(max_attempts):
        try:
            return llm.invoke(messages)
        except Exception as e:
            last_error = e
            if is_rate_limit_error(e) and attempt < max_attempts - 1:
                delay = base_delay * (2 ** attempt)
                st.toast(f"Rate limited — retrying in {delay}s…")
                time.sleep(delay)
                continue
            raise
    raise last_error


def is_rate_limit_error(e):
    return (
        "429" in str(e)
        or "ResourceExhausted" in type(e).__name__
        or "rate limit" in str(e).lower()
        or "rate_limited" in str(e).lower()
        or "quota" in str(e).lower()
    )


def invoke_with_fallback(chain, messages, max_attempts=4, base_delay=3):
    """Tries each (name, llm) in the fallback chain in order. Within a
       provider, rate-limit errors are retried with backoff via
       invoke_with_rate_limit_retry; if a provider's retries are exhausted,
       or it fails for any other reason, the next provider in the chain is
       tried instead. Raises the last error if every provider fails, and
       shows a toast when it actually falls over to a different provider so
       the switch isn't silent."""
    last_error = None
    for i, (name, model) in enumerate(chain):
        try:
            result = invoke_with_rate_limit_retry(model, messages, max_attempts=max_attempts, base_delay=base_delay)
            if i > 0:
                st.toast(f"Switched to {name} after the earlier provider failed.")
            return result
        except Exception as e:
            last_error = e
            continue
    raise last_error


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
        ("Cybersecurity", "CYB"),("Cloud Computing", "CC"),("Artificial Intelligence", "AIX"), ("Blockchain", "BLK"),
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
        ("Educator", "EDC", "explaining concepts clearly for learners at any level but just like a Q/Ans and maths explain me just like human writing"),
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
    uploaded_files = st.file_uploader(
        "Drop supporting files here to seed vector memory:",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True
    )
    if st.button("🗑️ Reset Knowledge Base"):
        build_vector_store.clear()
        st.rerun()

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
indexed_chunk_count = 0
if uploaded_files:
    combined_text = "\n\n".join(
        extract_text_from_upload(f) for f in uploaded_files
    ).strip()
    if combined_text:
        vectorstore = build_vector_store(combined_text)
        indexed_chunk_count = count_chunks(combined_text)
        with st.sidebar:
            st.caption(f"📚 {len(uploaded_files)} file(s) indexed · {indexed_chunk_count} chunks in memory")

# PDF export of the conversation transcript, shown in the sidebar — downloadable
# any number of times, and always reflects the latest chat history since the
# PDF bytes are regenerated fresh on each rerun.
with st.sidebar:
    st.markdown("---")
    st.subheader("📄 Export Transcript")
    has_conversation = any(isinstance(m, HumanMessage) for m in st.session_state.chat_history)

    if has_conversation:
        pdf_bytes = generate_chat_pdf(persona_option, st.session_state.chat_history)
        st.download_button(
            label="⬇️ Download Conversation as PDF",
            data=pdf_bytes,
            file_name=f"{persona_option.replace(' ', '_').replace('/', '-')}_transcript.pdf",
            mime="application/pdf",
            key="pdf_download_button"
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
    retrieved_docs = []
    if vectorstore is not None:
        retrieved_docs = vectorstore.similarity_search(user_prompt, k=4)
        if retrieved_docs:
            context_snippet = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # Build the outgoing message list with a SINGLE leading system message.
    # Some Mistral API versions reject requests containing more than one
    # system-role message, so any RAG context is merged into the existing
    # persona system message rather than inserted as a second one.
    messages_to_send = list(st.session_state.chat_history)
    if context_snippet and messages_to_send and isinstance(messages_to_send[0], SystemMessage):
        merged_system = SystemMessage(
            content=messages_to_send[0].content
            + f"\n\nUse the following retrieved context if relevant:\n\n{context_snippet}"
        )
        messages_to_send = [merged_system] + messages_to_send[1:]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = invoke_with_fallback(llm_chain, messages_to_send)
            except Exception as e:
                if is_rate_limit_error(e):
                    st.error(
                        "All configured providers hit their rate limit or quota, and retries were "
                        "exhausted. Wait a bit before trying again, or check usage at "
                        "aistudio.google.com (Gemini) / console.groq.com (Groq) / "
                        "platform.openai.com (OpenAI) / console.mistral.ai (Mistral)."
                    )
                else:
                    # Streamlit Cloud redacts exception details by default; surface
                    # the real error inline so it's actually debuggable.
                    st.error(f"The model call failed: {e}")
                st.stop()
        response_text = extract_response_text(response.content)
        st.markdown(response_text)
        if retrieved_docs:
            with st.expander(f"📚 Sources used from your documents ({len(retrieved_docs)})"):
                for i, doc in enumerate(retrieved_docs, start=1):
                    snippet = doc.page_content.strip().replace("\n", " ")
                    if len(snippet) > 300:
                        snippet = snippet[:300] + "…"
                    st.markdown(f"**Excerpt {i}:** {snippet}")

    st.session_state.chat_history.append(AIMessage(content=response_text))
