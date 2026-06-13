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

#Step 2: Connect LLM with FAISS and Create chain
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer.
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context","question"])
    return prompt

# Load Database
DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Invoke with single query
user_query=input("Write Query Here: ")
response=qa_chain.invoke({'query':user_query})
print("RESULTS: ",response["result"])
print("SOURCE DOCUMENTS: ",response["source_documents"])