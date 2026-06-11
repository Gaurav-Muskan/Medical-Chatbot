from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Load raw PDF(s)
DATA_PATH="data/"
def load_pdf_files(data):
    loader = DirectoryLoader(data,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents

documents = load_pdf_files(data = DATA_PATH)
# print("Length of PDF pages: ", len(documents))

# Stpe 2: Create Chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(documents)
print("Length of text chunks:- ", len(text_chunks))

# Stpe 3: Create Vector Embeddings
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model = get_embedding_model()

# Stpe 4: Store embeddings in FAISS
DB_FAISS_PATH="D:\VS Code\Medical-Chatbot/vectorstore/db_faiss"
# Generate embeddings for the text chunks and store them in a FAISS vector database
db=FAISS.from_documents(text_chunks, embedding_model) 
db.save_local(DB_FAISS_PATH)