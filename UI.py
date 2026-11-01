import os
import io
import re
import json
import time
import hashlib
import html as html_lib

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
from pypdf import PdfReader

# 1. Page Configuration
st.set_page_config(page_title="Avatar AI Experts", page_icon="💧", layout="centered")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _secret(name, default=None):
    """Reads a key from st.secrets or the environment without crashing when
       no secrets.toml exists."""
    try:
        val = st.secrets.get(name)
    except Exception:
        val = None
    return val or os.environ.get(name) or default


# 2. Cached Groq LLM initializer
@st.cache_resource
def init_groq_llm():
    """Initializes and caches the Groq chat model (the only provider used)."""
    api_key = _secret("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY is not set. Add it to your environment variables or Streamlit secrets.")
        st.stop()
    model_name = _secret("GROQ_MODEL", "openai/gpt-oss-120b")
    return ChatGroq(model=model_name, api_key=api_key, max_retries=5, timeout=60), model_name


@st.cache_resource
def init_embeddings():
    """Local ONNX embeddings (FastEmbed) - no torch, no extra API key."""
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")


llm, groq_model_name = init_groq_llm()
embeddings = init_embeddings()


@st.cache_resource(show_spinner="Analyzing documents and generating vector space...")
def build_vector_store(text_content):
    """Chunks text and builds an in-memory Chroma collection. The collection
       name is derived from the content hash so different document sets never
       get mixed into the same collection."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = [Document(page_content=x) for x in text_splitter.split_text(text_content) if x.strip()]
    if not docs:
        return None
    name = "avatar_" + hashlib.md5(text_content.encode("utf-8", "ignore")).hexdigest()[:16]
    return Chroma.from_documents(docs, embeddings, collection_name=name)


def extract_text_from_upload(uploaded):
    name = uploaded.name.lower()
    if name.endswith(".pdf"):
        try:
            reader = PdfReader(uploaded)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            st.warning(f"Could not read '{uploaded.name}': {e}")
            return ""
    try:
        return uploaded.getvalue().decode("utf-8")
    except UnicodeDecodeError:
        st.warning(f"Could not decode '{uploaded.name}' as UTF-8 text.")
        return ""


def count_chunks(text_content):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return len([c for c in splitter.split_text(text_content) if c.strip()])


def generate_chat_pdf(persona_name, history):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph(f"Conversation Transcript — {html_lib.escape(persona_name)}", styles["Title"]), Spacer(1, 14)]

    for msg in history:
        if isinstance(msg, SystemMessage):
            continue
        speaker = "You" if isinstance(msg, HumanMessage) else "Assistant"
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
    """Normalizes an LLM response's .content into plain text (Gemini 3 returns
       a list of blocks)."""
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


def is_rate_limit_error(e):
    msg = str(e).lower()
    return (
        "429" in msg
        or "resourceexhausted" in type(e).__name__.lower()
        or "rate limit" in msg
        or "rate_limited" in msg
        or "quota" in msg
    )


def invoke_with_rate_limit_retry(llm, messages, max_attempts=4, base_delay=3):
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


# ---------------------------------------------------------------------------
# VOICE: speech-to-text (server-side Whisper) + text cleanup for speech
# ---------------------------------------------------------------------------
VOICE_LANGS = {
    "English (US)": "en-US",
    "English (UK)": "en-GB",
    "English (India)": "en-IN",
    "Urdu (Pakistan)": "ur-PK",
    "Hindi": "hi-IN",
    "Arabic": "ar-SA",
    "Spanish": "es-ES",
    "French": "fr-FR",
    "German": "de-DE",
    "Italian": "it-IT",
    "Portuguese (Brazil)": "pt-BR",
    "Turkish": "tr-TR",
    "Russian": "ru-RU",
    "Japanese": "ja-JP",
    "Korean": "ko-KR",
    "Chinese (Mandarin)": "zh-CN",
}


def transcribe_audio(audio_bytes, lang_code):
    """Transcribes recorded audio with Groq's Whisper. Returns (text, error_message)."""
    lang = lang_code.split("-")[0]
    try:
        from groq import Groq
        client = Groq(api_key=_secret("GROQ_API_KEY"))
        r = client.audio.transcriptions.create(
            file=("voice.wav", audio_bytes),
            model=_secret("GROQ_STT_MODEL", "whisper-large-v3-turbo"),
            language=lang,
        )
        text = (getattr(r, "text", None) or "").strip()
        return (text, None) if text else (None, "Groq returned an empty transcript.")
    except Exception as e:
        return None, str(e)


def clean_for_speech(text):
    """Strips markdown so the voice doesn't read symbols out loud."""
    text = re.sub(r"```.*?```", " (code block omitted) ", text, flags=re.S)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"^\s*[-*+•]\s+", "", text, flags=re.M)
    text = re.sub(r"^\s*\|?[-:| ]{3,}\|?\s*$", "", text, flags=re.M)
    text = re.sub(r"[*_~>|#]", "", text)
    text = re.sub(r"\s*\n\s*", ". ", text)
    text = re.sub(r"\.\s*\.", ".", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# VOICE: browser bridge (Web Speech API for mic dictation + speech synthesis)
# The bridge script is injected into the parent page once; later reruns only
# update its config and optionally tell it to speak / stop.
# ---------------------------------------------------------------------------
BRIDGE_VERSION = "5"

BRIDGE_JS = r"""
(function () {
  var V = "__V__";
  var D = document;
  if (window.__avatarVoice && window.__avatarVoice.destroy) {
    try { window.__avatarVoice.destroy(); } catch (e) {}
  }
  var SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  var synth = window.speechSynthesis;
  var S = {
    version: V,
    cfg: { lang: "en-US", rate: 1.0, autosend: true, handsfree: false },
    rec: null, listening: false, queueId: 0, utts: [], toastT: null, obs: null, obsT: null
  };

  /* ---------- styles ---------- */
  var style = D.createElement("style");
  style.id = "avatar-voice-style";
  style.textContent =
    "#avatar-mic{position:absolute;right:3.3rem;bottom:.55rem;width:2.1rem;height:2.1rem;border-radius:50%;" +
    "border:1px solid rgba(128,128,128,.45);background:transparent;cursor:pointer;font-size:1.05rem;line-height:1;" +
    "display:flex;align-items:center;justify-content:center;z-index:5;padding:0;transition:background .2s,border-color .2s}" +
    "#avatar-mic:hover{background:rgba(46,134,255,.2);border-color:#2E86FF}" +
    "#avatar-mic.listening{background:#FF5F87;border-color:#FF5F87;animation:avatar-pulse 1.1s infinite}" +
    "@keyframes avatar-pulse{0%{box-shadow:0 0 0 0 rgba(255,95,135,.6)}70%{box-shadow:0 0 0 12px rgba(255,95,135,0)}100%{box-shadow:0 0 0 0 rgba(255,95,135,0)}}" +
    "[data-testid='stChatInput'] textarea{padding-right:5.8rem !important}" +
    "#avatar-toast{position:fixed;left:50%;bottom:6rem;transform:translateX(-50%);background:#241A3D;color:#F5F1FA;" +
    "border:1px solid #FF5F87;padding:.55rem 1rem;border-radius:10px;font:500 .85rem Inter,sans-serif;z-index:99999;" +
    "opacity:0;transition:opacity .25s;pointer-events:none;max-width:90vw;text-align:center}" +
    "@media (prefers-reduced-motion:reduce){#avatar-mic.listening{animation:none}}";
  D.head.appendChild(style);

  function toast(msg) {
    var t = D.getElementById("avatar-toast");
    if (!t) { t = D.createElement("div"); t.id = "avatar-toast"; D.body.appendChild(t); }
    t.textContent = msg;
    t.style.opacity = "1";
    clearTimeout(S.toastT);
    S.toastT = setTimeout(function () { t.style.opacity = "0"; }, 3800);
  }

  /* ---------- chat input helpers ---------- */
  function getTextarea() {
    return D.querySelector('textarea[data-testid="stChatInputTextArea"]') ||
           D.querySelector('[data-testid="stChatInput"] textarea');
  }
  function setValue(ta, val) {
    var setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
    setter.call(ta, val);
    ta.dispatchEvent(new Event("input", { bubbles: true }));
  }
  function submit() {
    var btn = D.querySelector('[data-testid="stChatInputSubmitButton"]') ||
              D.querySelector('[data-testid="stChatInput"] button:not(#avatar-mic)');
    if (btn && !btn.disabled) { btn.click(); return; }
    var ta = getTextarea();
    if (ta) {
      ta.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", code: "Enter", keyCode: 13, which: 13, bubbles: true }));
    }
  }

  /* ---------- speech recognition (mic) ---------- */
  function setListening(on) {
    S.listening = on;
    var b = D.getElementById("avatar-mic");
    if (b) {
      b.classList.toggle("listening", on);
      b.title = on ? "Listening… click to stop" : "Speak your message";
      b.textContent = on ? "⏹" : "🎤";
    }
  }

  function startListening() {
    if (!SR) {
      toast("Speech recognition isn't supported in this browser. Use Chrome or Edge, or the 'Record audio' option in the sidebar.");
      return;
    }
    if (S.listening) { try { S.rec.stop(); } catch (e) {} return; }
    stopSpeaking();
    var r = new SR();
    r.lang = S.cfg.lang;
    r.interimResults = true;
    r.continuous = false;
    r.maxAlternatives = 1;
    var finalText = "";
    r.onstart = function () { setListening(true); };
    r.onresult = function (e) {
      var interim = "";
      for (var i = e.resultIndex; i < e.results.length; i++) {
        var t = e.results[i][0].transcript;
        if (e.results[i].isFinal) { finalText += t; } else { interim += t; }
      }
      var ta = getTextarea();
      if (ta) { setValue(ta, (finalText + " " + interim).trim()); }
    };
    r.onerror = function (e) {
      setListening(false);
      if (e.error === "not-allowed" || e.error === "service-not-allowed") {
        toast("Microphone access was blocked. Allow it in your browser's site settings.");
      } else if (e.error === "no-speech") {
        toast("I didn't hear anything. Try again.");
      } else if (e.error === "network") {
        toast("Speech service unreachable (network error).");
      } else if (e.error !== "aborted") {
        toast("Voice error: " + e.error);
      }
    };
    r.onend = function () {
      setListening(false);
      if (finalText.trim() && (S.cfg.autosend || S.cfg.handsfree)) {
        setTimeout(submit, 300);
      }
    };
    S.rec = r;
    try { r.start(); } catch (e) { setListening(false); }
  }

  /* ---------- speech synthesis (voice replies) ---------- */
  function pickVoice() {
    if (!synth) return null;
    var voices = synth.getVoices();
    if (!voices || !voices.length) return null;
    var lang = (S.cfg.lang || "en-US").toLowerCase();
    var prefix = lang.split("-")[0];
    var exact = voices.filter(function (v) { return v.lang && v.lang.toLowerCase().replace("_", "-") === lang; });
    var loose = voices.filter(function (v) { return v.lang && v.lang.toLowerCase().indexOf(prefix) === 0; });
    var pool = exact.length ? exact : loose;
    if (!pool.length) return null;
    var good = pool.filter(function (v) { return /natural|neural|google|online|premium|enhanced/i.test(v.name); });
    return (good.length ? good : pool)[0];
  }

  function splitText(text) {
    var sentences = text.match(/[^.!?\u06D4\u061F\n]+[.!?\u06D4\u061F]*\s*/g) || [text];
    var chunks = [], cur = "";
    sentences.forEach(function (s) {
      if ((cur + s).length > 200 && cur) { chunks.push(cur.trim()); cur = s; }
      else { cur += s; }
    });
    if (cur.trim()) chunks.push(cur.trim());
    return chunks;
  }

  function stopSpeaking() {
    S.queueId++;
    S.utts = [];
    if (synth) { try { synth.cancel(); } catch (e) {} }
  }

  function speak(text) {
    if (!synth || !text) return;
    stopSpeaking();
    var id = S.queueId;
    var chunks = splitText(text);
    setTimeout(function () {
      if (id !== S.queueId) return;
      var voice = pickVoice();
      chunks.forEach(function (c, i) {
        var u = new SpeechSynthesisUtterance(c);
        u.lang = S.cfg.lang;
        u.rate = S.cfg.rate;
        if (voice) u.voice = voice;
        u.onend = function () {
          if (i === chunks.length - 1 && id === S.queueId && S.cfg.handsfree) {
            setTimeout(startListening, 450);
          }
        };
        S.utts.push(u);  /* keep references (Chrome GC bug) */
        synth.speak(u);
      });
    }, 120);
  }

  function stopAll() {
    stopSpeaking();
    if (S.rec && S.listening) { try { S.rec.abort(); } catch (e) {} }
    setListening(false);
  }

  /* ---------- mic button injection ---------- */
  function ensureButton() {
    var box = D.querySelector('[data-testid="stChatInput"]');
    if (!box) return;
    var b = D.getElementById("avatar-mic");
    if (b && box.contains(b)) return;
    if (b) b.remove();
    b = D.createElement("button");
    b.id = "avatar-mic";
    b.type = "button";
    b.textContent = "🎤";
    b.title = "Speak your message";
    b.setAttribute("aria-label", "Voice input");
    b.addEventListener("click", function (ev) { ev.preventDefault(); ev.stopPropagation(); startListening(); });
    box.style.position = "relative";
    box.appendChild(b);
    setListening(S.listening);
  }

  S.obs = new MutationObserver(function () {
    if (S.obsT) return;
    S.obsT = setTimeout(function () { S.obsT = null; ensureButton(); }, 250);
  });
  S.obs.observe(D.body, { childList: true, subtree: true });
  ensureButton();

  /* stop the voice as soon as the user sends a new message */
  function onKey(ev) {
    if (ev.key === "Enter" && !ev.shiftKey && ev.target && ev.target.matches &&
        ev.target.matches("[data-testid='stChatInput'] textarea")) { stopSpeaking(); }
  }
  function onClick(ev) {
    if (ev.target && ev.target.closest && ev.target.closest("[data-testid='stChatInputSubmitButton']")) { stopSpeaking(); }
  }
  D.addEventListener("keydown", onKey, true);
  D.addEventListener("click", onClick, true);

  S.setConfig = function (c) { for (var k in c) { S.cfg[k] = c[k]; } };
  S.speak = speak;
  S.stop = stopAll;
  S.destroy = function () {
    stopAll();
    if (S.obs) S.obs.disconnect();
    D.removeEventListener("keydown", onKey, true);
    D.removeEventListener("click", onClick, true);
    ["avatar-mic", "avatar-voice-style", "avatar-toast"].forEach(function (id) {
      var el = D.getElementById(id); if (el) el.remove();
    });
    delete window.__avatarVoice;
  };
  window.__avatarVoice = S;
})();
"""

BRIDGE_LOADER = r"""
var P = window.parent, D = P.document;
try {
  if (!P.__avatarVoice || P.__avatarVoice.version !== EXPECTED) {
    var s = D.createElement("script");
    s.id = "avatar-voice-script";
    s.textContent = BRIDGE_CODE;
    D.head.appendChild(s);
  }
  var AV = P.__avatarVoice;
  if (AV) {
    AV.setConfig(CFG);
    if (ACTION === "speak") { AV.speak(TEXT); }
    else if (ACTION === "stop") { AV.stop(); }
  }
} catch (e) { console.error("Avatar voice bridge failed:", e); }
"""


def _js(value):
    """JSON-encodes a value so it is safe to embed inside an inline <script>."""
    return json.dumps(value).replace("</", "<\\/")


def voice_bridge(cfg, action="none", text=""):
    bridge = BRIDGE_JS.replace("__V__", BRIDGE_VERSION)
    page = (
        "<script>(function(){"
        f"var BRIDGE_CODE={_js(bridge)};"
        f"var EXPECTED={_js(BRIDGE_VERSION)};"
        f"var CFG={_js(cfg)};"
        f"var ACTION={_js(action)};"
        f"var TEXT={_js(text)};"
        f"{BRIDGE_LOADER}"
        "})();</script>"
    )
    components.html(page, height=0)


# 3. CSS Styling System ("Boarding Pass")
DESIGN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@400;500;600&display=swap');

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
if "audio_key" not in st.session_state:
    st.session_state.audio_key = 0

voice_prompt = None  # filled if the user records audio in the sidebar

with st.sidebar:
    st.title("🛫 Departure Lounge")

    _DOMAINS = [
        ("Marketing", "MKT"), ("Finance", "FIN"), ("Healthcare", "HLT"), ("Legal", "LAW"),
        ("Technology", "TEC"), ("Education", "EDU"), ("Real Estate", "RES"), ("Hospitality", "HOS"),
        ("Retail", "RET"), ("Manufacturing", "MFG"), ("Agriculture", "AGR"), ("Energy", "ENR"),
        ("Transportation", "TRN"), ("Logistics", "LOG"), ("Construction", "CON"), ("Automotive", "AUT"),
        ("Aerospace", "AER"), ("Telecommunications", "TEL"), ("Insurance", "INS"), ("Banking", "BNK"),
        ("Nonprofit", "NPO"), ("Government", "GOV"), ("Media", "MED"), ("Entertainment", "ENT"),
        ("Sports", "SPT"), ("Fashion", "FSH"), ("Food & Beverage", "FNB"), ("Environmental", "ENV"),
        ("Cybersecurity", "CYB"), ("Cloud Computing", "CLD"), ("Artificial Intelligence", "AIX"), ("Blockchain", "BLK"),
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
        ("Educator", "EDC", "explaining concepts clearly for learners at any level, in a simple Q&A style, with maths explained step by step in natural human writing"),
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
                "desc": f"{role_name} {role_blurb}, applied to {domain_name.lower()} contexts.",
            }

    persona_option = st.selectbox("Choose Your AI Guide:", list(persona_configs.keys()))
    config = persona_configs[persona_option]

    # ---- Voice controls ----
    st.markdown("---")
    st.subheader("🎙️ Voice")
    voice_lang_label = st.selectbox("Voice language:", list(VOICE_LANGS.keys()), index=0,
                                    help="Used for both listening (speech recognition) and speaking.")
    voice_lang = VOICE_LANGS[voice_lang_label]
    speak_replies = st.toggle("🔊 Speak replies aloud", value=True)
    speech_rate = st.slider("Speaking speed", 0.6, 1.6, 1.0, 0.05)
    auto_send = st.toggle("Auto-send when I stop talking", value=True)
    hands_free = st.toggle(
        "🗣️ Hands-free conversation", value=False,
        help="After the AI finishes speaking, the mic turns on again automatically. "
             "Click the 🎤 button once to begin."
    )
    c1, c2 = st.columns(2)
    read_clicked = c1.button("▶️ Read last", use_container_width=True)
    stop_clicked = c2.button("⏹ Stop", use_container_width=True)
    st.caption("Tip: click the 🎤 button inside the chat box to dictate. Works best in Chrome / Edge.")

    with st.expander("🎧 Record audio (works in any browser)"):
        if hasattr(st, "audio_input"):
            rec = st.audio_input("Record a voice message", key=f"audio_in_{st.session_state.audio_key}")
            if rec is not None:
                with st.spinner("Transcribing…"):
                    heard, err = transcribe_audio(rec.getvalue(), voice_lang)
                st.session_state.audio_key += 1  # resets the widget for next time
                if heard:
                    voice_prompt = heard
                else:
                    st.warning(f"Couldn't transcribe: {err}")
        else:
            st.caption("Update Streamlit (>= 1.39) to enable in-app audio recording.")

    # ---- Model ----
    st.markdown("---")
    st.subheader("🧠 Model")
    st.caption(f"Powered by Groq · `{groq_model_name}`")

    # ---- RAG ----
    st.markdown("---")
    st.subheader("📁 Context Ingestion (RAG)")
    uploaded_files = st.file_uploader(
        "Drop supporting files here to seed vector memory:",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
    )
    if st.button("🗑️ Reset Knowledge Base"):
        build_vector_store.clear()
        st.rerun()

# 5. Core Application Initialization (Session Memory Management)
def build_system_prompt(persona, desc):
    return (
        f"You are the {persona}. {desc} Adapt your output behavior strictly to this identity framework. "
        "Reply in the same language the user writes in."
    )


if "chat_history" not in st.session_state or st.session_state.get("persona") != persona_option:
    st.session_state.persona = persona_option
    st.session_state.chat_history = [SystemMessage(content=build_system_prompt(persona_option, config["desc"]))]

# Context Processing Block
vectorstore = None
indexed_chunk_count = 0
if uploaded_files:
    combined_text = "\n\n".join(extract_text_from_upload(f) for f in uploaded_files).strip()
    if combined_text:
        vectorstore = build_vector_store(combined_text)
        indexed_chunk_count = count_chunks(combined_text)
        with st.sidebar:
            st.caption(f"📚 {len(uploaded_files)} file(s) indexed · {indexed_chunk_count} chunks in memory")

# PDF export of the conversation transcript
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
            key="pdf_download_button",
        )
    else:
        st.caption("Start chatting to enable PDF export.")

# 6. Header / Ticket UI
st.markdown(f"""
<div class="ticket">
    <div class="ticket-stub">
        <div class="no-label">Channel</div>
        <div class="no-value">{html_lib.escape(config['channel'])}</div>
    </div>
    <div class="ticket-perf"></div>
    <div class="ticket-main">
        <div class="ticket-eyebrow">Boarding Pass · AI Expert Session</div>
        <h1 class="ticket-title">{html_lib.escape(persona_option)}</h1>
        <p class="ticket-subtitle">{html_lib.escape(config['desc'])}</p>
        <div class="ticket-stamp">{html_lib.escape(config['stamp'])}</div>
    </div>
</div>
""", unsafe_allow_html=True)

if vectorstore is not None:
    st.markdown(
        '<div class="topnav"><span class="topnav-brand">📁 Context Loaded</span>'
        '<span class="topnav-status">RAG Active</span></div>',
        unsafe_allow_html=True,
    )

# 7. Chat History Rendering
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Voice action to send to the browser at the end of this run
speak_text = ""
voice_action = "none"

if stop_clicked:
    voice_action = "stop"
elif read_clicked:
    last_ai = next((m for m in reversed(st.session_state.chat_history) if isinstance(m, AIMessage)), None)
    if last_ai:
        speak_text = clean_for_speech(last_ai.content)
        voice_action = "speak"
    else:
        st.toast("Nothing to read yet — send a message first.")

# 8. Chat Input & Response Generation
typed_prompt = st.chat_input("Ask your AI guide anything… (or click 🎤 to talk)")
user_prompt = typed_prompt or voice_prompt

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

    # Single leading system message (some providers reject more than one)
    extras = ""
    if speak_replies or hands_free:
        extras += (
            "\n\nThe user may hear your answer read aloud. Prefer natural, conversational prose; "
            "avoid tables, emojis, and long bulleted lists unless explicitly asked."
        )
    if context_snippet:
        extras += f"\n\nUse the following retrieved context if relevant:\n\n{context_snippet}"

    messages_to_send = list(st.session_state.chat_history)
    if extras and isinstance(messages_to_send[0], SystemMessage):
        messages_to_send = [SystemMessage(content=messages_to_send[0].content + extras)] + messages_to_send[1:]

    response = None
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = invoke_with_rate_limit_retry(llm, messages_to_send)
            except Exception as e:
                if is_rate_limit_error(e):
                    st.error(
                        "Groq rate limit or quota reached and retries were exhausted. "
                        "Wait a bit, or check usage at console.groq.com."
                    )
                else:
                    st.error(f"The model call failed: {e}")

        if response is not None:
            response_text = extract_response_text(response.content).strip() or "(The model returned an empty reply.)"
            st.markdown(response_text)
            st.caption(f"Answered by Groq · {groq_model_name}")
            if retrieved_docs:
                with st.expander(f"📚 Sources used from your documents ({len(retrieved_docs)})"):
                    for i, doc in enumerate(retrieved_docs, start=1):
                        snippet = doc.page_content.strip().replace("\n", " ")
                        if len(snippet) > 300:
                            snippet = snippet[:300] + "…"
                        st.markdown(f"**Excerpt {i}:** {snippet}")

    if response is not None:
        st.session_state.chat_history.append(AIMessage(content=response_text))
        if speak_replies or hands_free:
            speak_text = clean_for_speech(response_text)
            voice_action = "speak" if speak_text else "none"
    else:
        # Drop the unanswered question so history stays consistent
        st.session_state.chat_history.pop()

# 9. Voice bridge (mic button + speech synthesis). Rendered last on every run.
voice_bridge(
    cfg={
        "lang": voice_lang,
        "rate": float(speech_rate),
        "autosend": bool(auto_send),
        "handsfree": bool(hands_free),
    },
    action=voice_action,
    text=speak_text,
)
