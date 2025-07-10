import os
import requests
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

def get_answer(user_query: str) -> str:
    # Load documents only when function is called
    loader = DirectoryLoader("docs/", glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embedding = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(chunks, embedding)
    retriever = vectordb.as_retriever(search_type="similarity", k=3)

    relevant_docs = retriever.get_relevant_documents(user_query)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""Use the context below to answer the question.\n\nContext:\n{context}\n\nQuestion: {user_query}"""

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
