from dotenv import load_dotenv
import os

load_dotenv()


from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# Load API_KEY for GROQ
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

#  Setup  LLM.
def load_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0.5,
        max_tokens=512
    )
