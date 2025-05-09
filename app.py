import streamlit as st
from utils.pdf_reader import extract_text_from_pdf
from utils.rag_chain import chunk_and_embed, get_qa_chain

st.set_page_config(page_title="Multi-PDF Chatbot", layout="wide")
st.title("📄 Multi-PDF Chatbot")

uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    st.subheader("Uploaded Files:")
    all_texts = []

    for file in uploaded_files:
        st.write(f"📘 {file.name}")
        text = extract_text_from_pdf(file)
        all_texts.append({"filename": file.name, "content": text})

    # Embed PDFs
    if st.button("📥 Embed and Index PDFs"):
        with st.spinner("Processing..."):
            chunk_and_embed(all_texts)
        st.success("✅ PDFs embedded successfully!")

# User Q&A Interface
st.subheader("Ask a Question About Your PDFs")
user_question = st.text_input("❓ Your Question:")

if user_question:
    with st.spinner("Thinking..."):
        qa_chain = get_qa_chain()
        answer = qa_chain.run(user_question)

    st.markdown("### 🤖 Answer:")
    st.write(answer)
