import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


def process_pdf(pdf_path):

   
    text = extract_text_from_pdf(pdf_path)
    
    documents = split_text_into_chunks(text)
    
    return documents


def extract_text_from_pdf(pdf_path):

    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    return text


def split_text_into_chunks(text):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
 
    chunks = text_splitter.split_text(text)
    

    documents = [Document(page_content=chunk) for chunk in chunks]
    
    return documents