import streamlit as st
from openai import OpenAI

predefined_prompts = {
    "uploaded_file": "Here's a document: {0}",
    "polish_email": "Please help polish and rephrase the following email",
    "polish_paragraph": "Please help polish, rephrase, correct grammas for the following paragraphs"
}

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.secrets["api"]['open_ai']
if not openai_api_key:
    st.info("Please add your OpenAI API key in the env.", icon="🗝️")

st.title("📝Tasks")
my_tasks = ["polish_email", "polish_paragraph", "document_qa"]
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

# Create an OpenAI client.
client = OpenAI(api_key=openai_api_key)

# Ask the user for a question via `st.text_area`.
question = st.text_area(
    "Now ask a question",
    placeholder="I'm baby Jasper, what can I do for you?"
)

if question:
    messages = [
        {
            "role": "user",
            "content": f"{question}",
        }
    ]

    if task == "document_qa" and uploaded_file:

        # Process the uploaded file and question.
        document = uploaded_file.read().decode()
        messages = [
            {
                "role": "user",
                "content": predefined_prompts["uploaded_file"].format(document) + f"\n\n---\n\n {question}",
            }
        ]
    
    if task:
        messages = [
            {
                "role": "user",
                "content": predefined_prompts[task] + f"\n\n---\n\n {question}",
            }
        ]

    # Generate an answer using the OpenAI API.
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True,
    )

    # Stream the response to the app using `st.write_stream`.
    st.write_stream(stream)
