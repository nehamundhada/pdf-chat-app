import streamlit as st
import os
import uuid
import boto3
import tempfile
import shutil
from langchain_community.vectorstores import FAISS
from langchain.embeddings import BedrockEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set up Bedrock embeddings
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1")

# Set up S3 client
s3 = boto3.client("s3")
bucket = os.environ.get("BUCKET_NAME")

st.set_page_config(page_title="Admin - PDF Upload", layout="wide")
st.header("Upload and Index PDF")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    if st.button("Upload and Index"):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_pdf_path = os.path.join(temp_dir, uploaded_file.name)
            
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load and split the PDF
            loader = PyPDFLoader(temp_pdf_path)
            pages = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
            docs = text_splitter.split_documents(pages)
            
            # Embed and store
            vectorstore = FAISS.from_documents(docs, bedrock_embeddings)

            doc_id = str(uuid.uuid4())  # Generate unique ID
            index_path = os.path.join(temp_dir, doc_id)
            vectorstore.save_local(index_path)

            # Upload to S3
            s3.upload_file(f"{index_path}/index.faiss", bucket, f"{doc_id}.faiss")
            s3.upload_file(f"{index_path}/index.pkl", bucket, f"{doc_id}.pkl")

            st.success(f"Document indexed and uploaded with ID: {doc_id}")
