from dotenv import load_dotenv
import os
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
import streamlit as st

# Load environment variables
load_dotenv()

def setup_retrieval_chain(documents):


    vectorstore = add_documents_to_vectorstore(documents)
    

    retriever = vectorstore.as_retriever()
    

    rag_chain = create_rag_chain(retriever)
    
    return rag_chain


def add_documents_to_vectorstore(documents):
    vectorstore = Chroma(

    persist_directory="./chroma_db2",
    embedding_function=OpenAIEmbeddings(),
    collection_name="wassim")

    vectorstore.add_documents(documents)

    return vectorstore


def create_rag_chain(retriever):

    prompt = hub.pull("rlm/rag-prompt")
    

    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0
    )
    

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


def format_docs(docs):

    return "\n\n".join(doc.page_content for doc in docs)


def query_document(rag_chain, query):

    try:
        response = rag_chain.invoke(query)
        return response
    except Exception as e:
        return f"Error processing your query: {str(e)}"