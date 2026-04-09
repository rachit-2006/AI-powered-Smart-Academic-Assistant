import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader, TextLoader,PyMuPDFLoader

import tempfile
import os

from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain.schema import Document

from langchain.vectorstores import Chroma 
from dotenv import load_dotenv

load_dotenv()


# -------------------- Page Configuration --------------------
st.set_page_config(page_title="Smart Academic Assistant", layout="centered")

# -------------------- Title --------------------
st.title("📚 Smart Academic Assistant")
st.write("Upload your academic documents and ask questions to get structured answers.")

# -------------------- File Upload Section --------------------
uploaded_files = st.file_uploader(
    "Upload academic documents (PDF, DOCX, or TXT):",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)



# -------------------- Question Input --------------------
question = st.text_input("Enter your academic question:")

# -------------------- Submit Button --------------------
if st.button("Get Answer"):
    if not uploaded_files or not question:
        st.warning("Please upload at least one document and enter a question.")
    else:
        docs=[]
        for file in uploaded_files:

            file_suffix = os.path.splitext(file.name)[1]  # e.g., .pdf, .txt, .docx 
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
                tmp_file.write(file.read())
                tmp_file_path = tmp_file.name
            if(file.type=="application/pdf"):
                loader = PyMuPDFLoader(tmp_file_path)
                docs.extend(loader.load())
            elif(file.type=="text/plain"):
                loader=TextLoader(tmp_file_path)
                docs.extend(loader.load())
            elif(file.type=="application/vnd.openxmlformats-officedocument.wordprocessingml.document"):
                loader=UnstructuredFileLoader(tmp_file_path)
                docs.extend(loader.load())
            else:
                st.error("unsupported file type")
                continue
        # 2. Split documents using RecursiveCharacterTextSplitter or similar
        splitter=RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
        )
        chunks=splitter.split_documents(docs)
        # 3. Create embeddings and store in vector store (e.g., FAISS, Chroma)
        embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vector_store=Chroma.from_documents(
            documents=chunks,
            embedding=embeddings
        )

        retriever=vector_store.as_retriever(search_kwargs={'k':6})
        relevant_docs=retriever.invoke(question)

        model=ChatGroq(
            model="Llama-3.3-70b-Versatile",
        )
        
        prompt = PromptTemplate(
            template = """
                Use the context to answer the question.

                Rules:
                - You MAY infer and summarize based on the context
                - You MAY combine multiple parts of the text
                - DO NOT hallucinate unrelated facts

                If context is insufficient → say "I don't know"

                Context:
                {relevant}

                Question:
                {question}
                """,
            input_variables=['question','relevant'],
            
        )
        chain=prompt| model
        relevant_texts = "\n\n".join([doc.page_content for doc in relevant_docs])
    

        result = chain.invoke({'relevant': relevant_texts, 'question': question})

        st.subheader("📄 Answer:")
        st.write(result.content)


