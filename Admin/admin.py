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


def main():
    st.write("This is Admin site for chat with PDF demo")
    uploaded_file = st.file_uploader("Choose a file", "pdf")
    if uploaded_file is not None:
        request_id = uuid()
        st.write(f"Request ID: {request_id}")
        saved_file_name = f"{request_id}.pdf"
        with open(saved_file_name, mode="wb") as w:
            w.write(uploaded_file.getvalue())
        loader = PyPDFLoader(saved_file_name)
        pages = loader.load_and_split()

        st.write(f"Total pages of pdf: {len(pages)}")


if __name__ == "__main__":
    main()
