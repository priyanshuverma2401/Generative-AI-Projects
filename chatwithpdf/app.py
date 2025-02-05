import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

st.title("DocBot App")
llm = ChatGroq(groq_api_key = groq_api_key, model_name = "gemma2-9b-it")
print(llm)

prompt = ChatPromptTemplate.from_template(
  '''
  Answer the question based on the provided context only.
  please provide the most accurate response based on the question
  <context>
  {context}
  <context>
  Question : {input}
'''
)

# Sidebar for PDF upload
st.sidebar.title("Upload PDF")
uploaded_files = st.sidebar.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        save_path = os.path.join("myfiles", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success(f"Saved file: {uploaded_file.name} to myfiles")



def vector_embedding():
  if "vectors" not in st.session_state:
    st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    st.session_state.loader = PyPDFDirectoryLoader("./myfiles")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    st.session_state.final_document = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_document, st.session_state.embeddings)


prompt1 = st.text_input("Ask your Question: ")

if st.button("Process Files"):
  vector_embedding()
  st.write("File Processed successfully")

import time

if prompt1:
  document_chain = create_stuff_documents_chain(llm, prompt)
  retriever = st.session_state.vectors.as_retriever()
  retrieval_chain = create_retrieval_chain(retriever, document_chain)
  start = time.process_time()
  response = retrieval_chain.invoke({'input': prompt1})
  st.write(response['answer'])


  with st.expander("Show References"):
    for i, doc in enumerate(response["context"]):
      st.write(doc.page_content)
      st.write("-------------------------------------------------------------------------------------")