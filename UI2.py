import os 
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI
st.image("Image.png", width=150)
st.set_page_config(page_title="Multi-personal AI Chatbot", layout="centered")
os.environ["MISTRAL_API_KEY"] = "GsoCCPZTCzSgOxV3Jx7pCN94g9hVxdJh"


@st.cache_resource
def get_model():
    return ChatMistralAI(model="mistral-small-2506", temperature=0.9)


model = get_model()
st.sidebar.title("Chat Setting")
persona = st.sidebar.selectbox(
    "Chose Which AI do you want:",
    ["Motivation", "Cooking", "Coding"]

)


st.sidebar.markdown("---")
st.sidebar.title("Paper / Document")

uploaded_file = st.sidebar.file_uploader(
    "Upload a text document or paper context:",
    type=["txt", "pdf", "py", "csv", "json"]

)
document_context = ""
if uploaded_file is not None:
    try:
        string = uploaded_file.getvalue().decode("utf-8")
        document_context = string
        st.sidebar.success(f'loaded:{uploaded_file.name} ({len(document_context)} charactor)')
    except Exception as e:
        st.sidebar.error("Error reading File.")


PERSONA_CONFIGS = {
    "Motivation": {
        "title": " Mindset Master AI",
        "subtitle": "Your 24/7 personal pocket cheerleader.",
        "input_placeholder": "What goal are you tackling today?",
        "system_prompt": "You are an enthusiastic, high-energy Motivational AI agent. Boost user confidence, combat self-doubt, and give actionable productivity advice.",
        "spinner": "Channeling pure motivation..."
    },
    "Cooking": {
        "title": " Chef de Partie AI",
        "subtitle": "Your culinary guide, recipe creator, and kitchen assistant.",
        "input_placeholder": "What ingredients do you have, or what do you want to cook?",
        "system_prompt": "You are an expert culinary chef AI. Provide clear recipes, cooking techniques, ingredient substitutions, and kitchen tips. Format recipes beautifully with bold text and lists.",
        "spinner": "Sharpening the knives..."
    },
    "Coding": {
        "title": " StackOverflow Companion",
        "subtitle": "Your expert software engineer and debugger.",
        "input_placeholder": "Paste your code error or ask a programming question...",
        "system_prompt": "You are an expert senior software engineer AI. Provide clean, secure, optimized code snippets. Explain logic clearly. Always wrap code blocks in proper markdown syntax with language identifiers.",
        "spinner": "Compiling thoughts..."
    }
}

current_config = PERSONA_CONFIGS[persona]

final_system_prompt = current_config["system_prompt"]
if document_context:
    final_system_prompt += f"\n\n[Context Document]\nThe user has provided the following reference document to help you answer questions:\n{document_context}\n[END OF CONTEXT DOCUMENT]\n\nRefer to this document content directly if the user asks you to explain, summarize, or extract details from it. "
if "current_persona" not in st.session_state or st.session_state.current_persona != persona or "last_doc_name" not in st.session_state or st.session_state.last_doc_name != (uploaded_file.name if uploaded_file else None):
    st.session_state.current_persona = persona
    st.session_state.last_doc_name =  uploaded_file.name if uploaded_file else None
    st.session_state.messages = [
        SystemMessage(content=final_system_prompt)
    ]

st.title(current_config['title'])
st.subheader(current_config["subtitle"])
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages =  [SystemMessage(config=final_system_prompt)]


for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistent"):
            st.write(msg.content)

if user_input := st.chat_input(current_config["input_placeholder"]):

    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))

    with st.chat_message("assistant"):
        with st.spinner(current_config["spinner"]):
            try:
                response = model.invoke(st.session_state.messages)
                st.write(response.content)
                st.session_state.messages.append(AIMessage(content=response.content))
            except Exception as e:
                st.error("Failed to fetch response.")
