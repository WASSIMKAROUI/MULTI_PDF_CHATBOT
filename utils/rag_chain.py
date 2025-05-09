import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()  # Loads OPENAI_API_KEY from .env

def chunk_and_embed(docs, persist_directory="vectordb"):
    # 1. Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    # 2. Convert to LangChain documents
    langchain_docs = []
    for doc in docs:
        splits = text_splitter.create_documents([doc["content"]])
        for s in splits:
            s.metadata["source"] = doc["filename"]
        langchain_docs.extend(splits)

    # 3. Create embeddings using standard OpenAI
    embeddings = OpenAIEmbeddings()

    # 4. Store in ChromaDB
    vectordb = Chroma.from_documents(
        documents=langchain_docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    vectordb.persist()
    return vectordb

def get_qa_chain(persist_directory="vectordb"):
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=OpenAIEmbeddings()
    )

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(),
        chain_type="stuff"
    )

    return qa_chain
