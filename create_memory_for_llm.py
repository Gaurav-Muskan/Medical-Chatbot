from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Stpe 1: Load raw PDF(s)
DATA_PATH="data/"
def load_pdf_files(data):
    loader = DirectoryLoader(data,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents

documents = load_pdf_files(data = DATA_PATH)
print("Length of PDF pages: ", len(documents))

# Stpe 2: Create Chunks

# Stpe 3: Create Vector Embeddings

# Stpe 4: Store embeddings in FAISS