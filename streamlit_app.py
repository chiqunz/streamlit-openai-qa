import streamlit as st
import json
from dify_client import ChatClient
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
import uuid

predefined_prompts = {
    "document_qa": "Here's a document: {0}",
    "polish_email": "Please help polish and rephrase the following email",
    "polish_paragraph": "Please help polish, rephrase, correct grammas for the following paragraphs"
}

my_tasks = ["polish_email", "polish_paragraph", "document_qa"]
backends = ["native", "dify"]
uploaded_file = None

def generate_messages(task, query, uploaded_file):
    messages = ""
    match task:
        case 'document_qa':
            document = uploaded_file.read().decode()
            messages = predefined_prompts[task].format(document) + f"\n\n---\n\n {query}"
        case None:
            messages = query
        case _:
            messages = predefined_prompts[task] + f"\n\n---\n\n {query}"
    return messages


def dify_generator(chat_response):
    for line in chat_response.iter_lines(decode_unicode=True):
        line = line.split('data:', 1)[-1]
        if line.strip():
            line = json.loads(line.strip())
            answer = line.get('answer')
            if answer:
                yield answer


openai_api_key = st.secrets["api"]['open_ai']
dify_key = st.secrets["api"]['dify']
if not openai_api_key:
    st.info("Please add your OpenAI API key in the env.", icon="🗝️")
if not dify_key:
    st.info("Please add your Dify App key in the env.", icon="🗝️")

st.title("🕸️Backends")
backend = st.selectbox("Select a backend:", backends, index=0)

chat_client = None
if backend == "native":
    st.title("📝Tasks")
    task = st.selectbox("Select a predefined task:", my_tasks, index=None)

    if task == "document_qa":
        # Show title and description.
        st.title("📄 Document question answering")
        st.write(
            "Upload a document below and ask a question about it – GPT will answer! "
        )

        # Let the user upload a file via `st.file_uploader`.
        uploaded_file = st.file_uploader(
            "Upload a document (.txt or .md)", type=("txt", "md")
        )
    
    llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=openai_api_key, streaming=True)
    tools = [DuckDuckGoSearchRun(name="Search")]
    agent_type = AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION
    chat_client = initialize_agent(
        tools, llm, agent=agent_type, handle_parsing_errors=True
    )
elif backend == "dify":
    task = None
    uploaded_file = None
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = 'chiqun_' + uuid.uuid4().hex
    chat_client = ChatClient(dify_key)
    conversation_id = None

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm baby Jasper. How can I help you?"}
    ]
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if query := st.chat_input(placeholder="Type here"):
    prompt = generate_messages(task, query, uploaded_file)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    task = None

    if backend == "native":
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = chat_client.run(st.session_state.messages, callbacks=[st_cb])
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    elif backend == "dify":
        if conversation_id is None:
            user_messages = [m for m in st.session_state.messages if m["role"] == "user"]
            if len(user_messages) > 1:
                conversations = chat_client.get_conversations(user=st.session_state.user_id)
                conversations = json.loads(conversations.text)['data']
                if len(conversations) > 0:
                    conversation_id = conversations[-1]['id']
        chat_response = chat_client.create_chat_message(inputs={}, query=prompt, user=st.session_state.user_id, response_mode="streaming", conversation_id=conversation_id)
        chat_response.raise_for_status()
        with st.chat_message("assistant"):
            msg = st.write_stream(dify_generator(chat_response))
            st.session_state.messages.append({"role": "assistant", "content": msg})


        
