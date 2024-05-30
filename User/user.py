import boto3
import streamlit as st
import os
import uuid


# s3_client
s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")

# Bedrock
from langchain_community.embeddings import BedrockEmbeddings

# Text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# PDF loader
from langchain_community.document_loaders import PyPDFLoader

# import FAISS
from langchain_community.vectorstores import FAISS

# bedrock client
bedrock_client = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(client=bedrock_client, model_id="amazon.titan-embed-text-v1")

def get_unique_id():
    return str(uuid.uuid4())

# Split pages into chunks
def split_text(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs

# Create vector store
def create_vector_store(request_id, splitted_docs):
    vectorstore_faiss=FAISS.from_documents(splitted_docs, bedrock_embeddings)
    file_name=f"{request_id}.bin"
    folder_path="/tmp/"
    vectorstore_faiss.save_local(index_name=file_name, folder_path=folder_path)
    # upload to s3
    s3_client.upload_file(Filename=folder_path+"/"+file_name+".faiss", Bucket=BUCKET_NAME, Key="my_faiss.faiss")
    s3_client.upload_file(Filename=folder_path+"/"+file_name+".pkl", Bucket=BUCKET_NAME, Key="my_faiss.pkl")
    return True

def main():
    st.write("This is Admin site for chat with PDF demo")
    uploaded_file = st.file_uploader("Choose a file", "pdf")
    if uploaded_file is not None:
        request_id = get_unique_id()
        st.write(f"Request ID: {request_id}")
        saved_file_name = f"{request_id}.pdf"
        with open(saved_file_name, mode="wb") as w:
            w.write(uploaded_file.getvalue())
        loader = PyPDFLoader(saved_file_name)
        pages = loader.load_and_split()

        st.write(f"Total pages of pdf: {len(pages)}")

        # Split text 
        splitted_docs = split_text(pages, 1000, 200) # 1000 characters and 200 overlaps
        st.write(f"Splitted doc length: {len(splitted_docs)}")
        #st.write(f"Splitted docs length: {len(splitted_docs)}")
        st.write("==============")
        st.write(splitted_docs[0])
        st.write("==============")
        st.write(splitted_docs[1])

        st.write("Create Vector store")
        result = create_vector_store(request_id, splitted_docs)

        if result:
            st.write("PDF processed successfully")
        else:
            st.write("ERROR")

if __name__ == "__main__":
    main()
