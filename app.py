import databutton as db
import streamlit as st
import openai
from core import get_index_for_pdf
from langchain.chains import retrieval_qa
from langchain.chat_models import ChatOpenAI
import os

# title for the streamlit app
st.title("Chatbot- RAG")

# setting up the OpenAI API key
os.environ["OPENAI_API_KEY"] = db.secrets.get("OPENAI_API_KEY")
openai.api_key = db.secrets.get("OPENAI_API_KEY")

# function to create a vector database for the pdf files
@st.cache_data
def create_vector_database(files,filenames):
    # showing a spinner to indicate that the vector database is being created
    with st.spinner("Vector database"):
        vectordb = get_index_for_pdf(
            [file.getvalue() for file in files], filenames, openai.api_key
        )
    return vectordb

# uploading the pdf files
pdfs = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if pdfs:
    pdf_file_names = [file.name for file in pdfs]
    st.session_state["vector_db"] = create_vector_database(pdfs, pdf_file_names)

prompt_pattern = """
    You are a helpful Assistant who answers to users questions based on multiple contexts given to you.

    Keep your answer short and to the point.
    
    The evidence are the context of the pdf extract with metadata. 
    
    Carefully focus on the metadata specially 'filename' and 'page' whenever answering.
    
    Make sure to add filename and page number at the end of sentence you are citing to.
        
    Reply "Not applicable" if text is irrelevant.
     
    The PDF content is:
    {pdf_extract}
"""

prompt = st.session_state.get("prompt", [{"role": "system", "content": "none"}])

# show past chat history
for chat in prompt:
    if chat["role"] != "system":
        with st.chat_message(chat["role"]):
            st.write(chat["content"])

# get the user input
user_input = st.text_input("Ask me anything")

if user_input:
    vectordb = st.session_state.get("vectordb", None)
    if not vectordb:
        with st.message("assistant"):
            st.write("You need to provide a PDF")
            st.stop()

    # Search the vectordb for similar content to the user's user input
    search_results = vectordb.similarity_search(user_input, k=3)
    # search_results
    pdf_extract = "/n ".join([result.page_content for result in search_results])

    # Update the prompt with the pdf extract
    prompt[0] = {
        "role": "system",
        "content": prompt_pattern.format(pdf_extract=pdf_extract),
    }

    # Add the user's user input to the prompt and display it
    prompt.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Display an empty assistant message while waiting for the response
    with st.chat_message("assistant"):
        botmsg = st.empty()

    # Call ChatGPT with streaming and display the response as it comes
    response = []
    result = ""
    for chunk in openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=prompt, stream=True
    ):
        text = chunk.choices[0].get("delta", {}).get("content")
        if text is not None:
            response.append(text)
            result = "".join(response).strip()
            botmsg.write(result)

    # Add the assistant's response to the prompt
    prompt.append({"role": "assistant", "content": result})

    # Store the updated prompt in the session state
    st.session_state["prompt"] = prompt
    prompt.append({"role": "assistant", "content": result})

    # Store the updated prompt in the session state
    st.session_state["prompt"] = prompt

