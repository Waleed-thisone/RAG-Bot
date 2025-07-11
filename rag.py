import os
import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

print("Loading files...")
loader = DirectoryLoader("docs/", glob="**/*.txt", loader_cls=TextLoader)
docs = loader.load()
print(f"✅ Loaded {len(docs)} files")

print("Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)
print(f"✅ Total chunks: {len(chunks)}")

print("Creating embeddings...")
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("Building FAISS vectorstore...")
vectordb = FAISS.from_documents(chunks, embedding)
retriever = vectordb.as_retriever(search_type="similarity", k=3)
print("Done.")

def get_answer(user_query: str) -> str:
    relevant_docs = retriever.get_relevant_documents(user_query)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""Use the context below to answer the question.Dont use words like based on provided context or anything just straight answer\n\nContext:\n{context}\n\nQuestion: {user_query}"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek/deepseek-r1-0528-qwen3-8b:free",
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload
    )

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"❌ Error: {response.status_code}\n{response.text}"
