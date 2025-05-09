from langchain.text_splitter import RecursiveCharacterTextSplitter

sample_text = """
France is a beautiful country with a rich cultural heritage. 
It is home to the Eiffel Tower, the Louvre Museum, and countless historic landmarks. 
French cuisine is known worldwide, especially for its pastries, cheeses, and wines. 
Tourists enjoy the diverse landscapes, from the French Riviera to the Alps. 
Each region offers a unique experience, blending history, nature, and modern life.
"""

# Step 1: Create the splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=30
)

# Step 2: Split the text
chunks = text_splitter.create_documents([sample_text])

# Step 3: Print the chunks
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk.page_content)
