from dotenv import load_dotenv
load_dotenv()

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st
import tempfile
import os

st.title('ChatPDF')
st.write("---")

uploaded_file = st.file_uploader("PDF 업로드 하기")
st.write("---")

def pdf_to_document(uploaded_file):
    # save files
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    # loader
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()

    return pages

if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    # split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 100,
        chunk_overlap = 20,
        length_function = len,
        is_separator_regex= False,
    )
    texts = text_splitter.split_documents(pages)

    # embedding
    embeddings_model = OpenAIEmbeddings()

    # load it into Chroma
    vectordb = Chroma.from_documents(texts, embeddings_model)

    # questions
    st.header("PDF에게 질문해 보세요!")
    question = st.text_input("질문을 입력하세요.")
    if st.button("질문하기"):
        with st.spinner('답변 작성 중...'):
            llm = ChatOpenAI(temperature=0)
            qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectordb.as_retriever())
            result = qa_chain({"query": question})
            st.write(result["result"])
