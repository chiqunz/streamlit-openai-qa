import streamlit as st
from openai import OpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun

predefined_prompts = {
    "document_qa": "Here's a document: {0}",
    "polish_email": "Please help polish and rephrase the following email",
    "polish_paragraph": "Please help polish, rephrase, correct grammas for the following paragraphs"
}

my_tasks = ["polish_email", "polish_paragraph", "document_qa"]
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


# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = "" #st.secrets["api"]['open_ai']
if not openai_api_key:
    st.info("Please add your OpenAI API key in the env.", icon="🗝️")

st.title("📝Tasks")
st.selectbox("Select a predefined task:", my_tasks, index=None, key='selection')

if st.session_state.selection == "document_qa":
    # Show title and description.
    st.title("📄 Document question answering")
    st.write(
        "Upload a document below and ask a question about it – GPT will answer! "
    )

    # Let the user upload a file via `st.file_uploader`.
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .md)", type=("txt", "md")
    )

# Create an OpenAI client.
client = OpenAI(api_key=openai_api_key)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm baby Jasper. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if query := st.chat_input(placeholder="Type here"):
    prompt = generate_messages(st.session_state.selection, query, uploaded_file)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    st.session_state.selection = None

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=openai_api_key, streaming=True)
    search = DuckDuckGoSearchRun(name="Search")
    search_agent = initialize_agent(
        [search], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True
    )
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
