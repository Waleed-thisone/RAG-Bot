import os
import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# Load prebuilt index
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = FAISS.load_local("faiss_index", embedding)
retriever = vectordb.as_retriever(search_type="similarity", k=3)

def get_answer(user_query: str) -> str:
    relevant_docs = retriever.get_relevant_documents(user_query)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""Use the context below to answer the question. Don't use words like based on provided context or anything, just give a straight answer.\n\nContext:\n{context}\n\nQuestion: {user_query}"""

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
