import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import os 

st.set_page_config(page_title="Motivational", page_icon="")

os.environ["MISTRAL_API_KEY"] = "GsoCCPZTCzSgOxV3Jx7pCN94g9hVxdJh"

model = ChatMistralAI(model="mistral-small-2506", temperature=0.9)


if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a Motivational AI agent")
    ]

st.title("Mindset Master AI ")
st.subheader("Your 24/7 personal pocket cheerleader")

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message('user'):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistent"):
            st.write(msg.content)

if user_input := st.chat_input("What goal are you tackling today"):

    with st.chat_message('user'):
        st.write(user_input)
        st.session_state.messages.append(HumanMessage(content=user_input))

    with st.chat_message("assistent"):
        with st.spinner("Channeling"):
            response = model.invoke(st.session_state.messages)
            st.write(response.content)
            st.session_state.messages.append(AIMessage(content=response.content))


