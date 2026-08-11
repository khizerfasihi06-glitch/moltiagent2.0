import os
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI

# UI basics
st.set_page_config(page_title="Multi-personal AI Chatbot", layout="centered")
st.image("Image.png", width=150)

# Load Mistral API key from environment or Streamlit secrets (do NOT hardcode keys in source)

os.environ["MISTRAL_API_KEY"] = "EpgVFLlyQVkXWFmHv0lDwsMlZymmDoGP"

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
    "World History & Archaeology"
]

persona = st.sidebar.selectbox(
    label="Choose Which AI do you want:",
    options=persona_options
)

# Hide Streamlit branding (optional)
hide_streamlit_badge = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display:none;}
div[data-testid="stStatusWidget"] {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_badge, unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.title("Paper / Document")

uploaded_file = st.sidebar.file_uploader(
    "Upload a text document or paper context:",
    type=["txt", "pdf", "py", "csv", "json"]
)

document_context = ""
if uploaded_file is not None:
    try:
        # For simple text-like files, try decoding as utf-8
        raw = uploaded_file.getvalue()
        try:
            string = raw.decode("utf-8")
        except Exception:
            # If decoding fails, fall back to a simple representation
            string = str(raw)
        document_context = string
        st.sidebar.success(f'Loaded: {uploaded_file.name} ({len(document_context)} characters)')
    except Exception as e:
        st.sidebar.error("Error reading file: " + str(e))

# Predefined persona configs (you can extend this dict as needed)
PERSONA_CONFIGS = {
    "Motivation": {
        "title": "Mindset Master AI",
        "subtitle": "Your 24/7 personal pocket cheerleader.",
        "input_placeholder": "What goal are you tackling today?",
        "system_prompt": "You are an enthusiastic, high-energy Motivational AI agent. Boost user confidence, combat self-doubt, and give actionable productivity advice.",
        "spinner": "Channeling pure motivation..."
    },
    "Cooking": {
        "title": "Chef de Partie AI",
        "subtitle": "Your culinary guide, recipe creator, and kitchen assistant.",
        "input_placeholder": "What ingredients do you have, or what do you want to cook?",
        "system_prompt": "You are an expert culinary chef AI. Provide clear recipes, cooking techniques, ingredient substitutions, and kitchen tips.",
        "spinner": "Sharpening the knives..."
    },
    "Coding": {
        "title": "StackOverflow Companion",
        "subtitle": "Your expert software engineer and debugger.",
        "input_placeholder": "Paste your code error or ask a programming question...",
        "system_prompt": "You are an expert senior software engineer AI. Provide clean, secure, optimized code snippets and explain logic clearly.",
        "spinner": "Compiling thoughts..."
    },
    "Amazon": {
        "title": "Marketplace Navigator",
        "subtitle": "Your expert advisor for Amazon AWS, FBA, and e-commerce growth.",
        "input_placeholder": "Ask about AWS architecture, FBA listing optimization, or SEO...",
        "system_prompt": "You are an Amazon ecosystem expert AI. Offer clear, step-by-step guidance on AWS cloud infrastructure and Amazon Seller strategies.",
        "spinner": "Optimizing listings and servers..."
    },
    # add other specific configs as you prefer...
}

# Simple fallback generator for any persona not in PERSONA_CONFIGS
def make_default_config(name: str):
    return {
        "title": f"{name} Assistant",
        "subtitle": f"A helpful {name} assistant.",
        "input_placeholder": f"Ask the {name} assistant something...",
        "system_prompt": f"You are a helpful and knowledgeable assistant specialized in: {name}. Provide accurate and concise answers.",
        "spinner": "Thinking..."
    }

current_config = PERSONA_CONFIGS.get(persona, make_default_config(persona))

# Build final system prompt including uploaded document context if present
final_system_prompt = current_config["system_prompt"]
if document_context:
    final_system_prompt += (
        "\n\n[Context Document]\nThe user has provided the following reference document to help you answer questions:\n"
        + document_context
        + "\n[END OF CONTEXT DOCUMENT]\n"
    )

# Initialize session state
if "current_persona" not in st.session_state or st.session_state.current_persona != persona or \
   "last_doc_name" not in st.session_state or st.session_state.last_doc_name != (uploaded_file.name if uploaded_file else None):
    st.session_state.current_persona = persona
    st.session_state.last_doc_name = uploaded_file.name if uploaded_file else None
    st.session_state.messages = [
        SystemMessage(content=final_system_prompt)
    ]

st.title(current_config["title"])
st.subheader(current_config["subtitle"])

if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = [SystemMessage(content=final_system_prompt)]

# Render chat history
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)
    elif isinstance(msg, SystemMessage):
        # Optionally show system message in a small caption (or skip)
        pass

# Chat input
if user_input := st.chat_input(current_config["input_placeholder"]):
    with st.chat_message("user"):
        st.write(user_input)
        st.session_state.messages.append(HumanMessage(content=user_input))

    with st.chat_message("assistant"):
        with st.spinner(current_config.get("spinner", "Thinking...")):
            try:
                # model.invoke may expect a list of messages in the langchain message objects format
                response = model.invoke(st.session_state.messages)
                # response handling: try to read .content (langchain-like) else str(response)
                content = getattr(response, "content", None) or str(response)
                st.write(content)
                st.session_state.messages.append(AIMessage(content=content))
            except Exception as e:
                st.error("Failed to fetch response: " + str(e))
