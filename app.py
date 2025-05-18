import streamlit as st
import os
import tempfile
from pdf_reader import process_pdf
from rag_chain import setup_retrieval_chain, query_document

st.set_page_config(page_title="PDF Chat Assistant", layout="wide")

def main():
    st.title("PDF Chat Assistant")
    
    # Sidebar for PDF upload
    with st.sidebar:
        st.header("Upload Document")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                pdf_path = tmp_file.name
            
            with st.spinner("Processing PDF..."):
                # Process the PDF and create the vector store
                documents = process_pdf(pdf_path)
                st.session_state.documents = documents
                st.session_state.rag_chain = setup_retrieval_chain(documents)
                st.success(f"PDF processed successfully! ({len(documents)} chunks created)")
            
            # Clean up the temporary file
            os.unlink(pdf_path)
    
    # Main chat interface
    st.header("Chat with your PDF")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your document"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            if "rag_chain" not in st.session_state:
                response = "Please upload a PDF document first."
            else:
                with st.spinner("Thinking..."):
                    response = query_document(st.session_state.rag_chain, prompt)
            
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()